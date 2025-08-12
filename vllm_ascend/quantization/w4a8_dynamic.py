#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context

import vllm_ascend.quantization.w8a8_dynamic as ascend_w8a8_dynamic
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import FusedMoEState
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe import select_experts
from vllm_ascend.quantization.w8a8_dynamic import (fused_experts_with_all2all,
                                                   fused_experts_with_mc2)
from vllm_ascend.torchair.utils import npu_stream_switch, npu_wait_tensor
from vllm_ascend.utils import dispose_tensor


def apply_mlp_decode(hidden_states: torch.Tensor,
                     w1: torch.Tensor,
                     w1_scale: torch.Tensor,
                     w2: torch.Tensor,
                     w2_scale: torch.Tensor,
                     group_list: torch.Tensor,
                     dynamic_scale: torch.Tensor = None,
                     group_list_type: int = 1) -> torch.Tensor:
    """
    apply MLP: gate_up_proj -> swiglu -> down_proj
    Args:
        hidden_states_wrapper: wrapper of input hidden states with shape (num_tokens, hidden_size).
        w1: expert weights1 with shape
            (num_experts, hidden_size, intermediate_size * 2)
        w1_scale: weights1 scale with shape (num_experts, intermediate_size * 2)
        w2: expert weights2 with shape
            (num_experts, intermediate_size, hidden_size)
        w2_scale: weights2 scale with shape (num_experts, hidden_size)
        group_list: number of tokens for each expert, follow cumsum mode, and
            with shape (num_experts).
        transpose_weight:
            w1: (num_experts, intermediate_size * 2, hidden_size) ->
                    (num_experts, hidden_size, intermediate_size * 2)
            w2: (num_experts, hidden_size, intermediate_size) ->
                    (num_experts, intermediate_size, hidden_size)
    Returns:
        hidden_states: output hidden states after MLP.
    """

    if dynamic_scale is None:
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(
            hidden_states)
        # Dispose the original unquantized hidden states
        # to save npu memory because they're no longer used.
        dispose_tensor(unquantized_hidden_states)
    else:
        pertoken_scale = dynamic_scale

    # gmm1: gate_up_proj
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        split_item=3,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.int32)[0]

    # act_fn: swiglu
    hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
        x=hidden_states,
        weight_scale=w1_scale,
        activation_scale=pertoken_scale,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=group_list,
        activate_left=True,
        quant_mode=1,
    )

    # gmm2: down_proj
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale],
        per_token_scale=[swiglu_out_scale],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=w2_scale.dtype)[0]
    return hidden_states


def apply_mlp(hidden_states: torch.Tensor,
              w1: torch.Tensor,
              w1_scale: torch.Tensor,
              w2: torch.Tensor,
              w2_scale: torch.Tensor,
              group_list: torch.Tensor,
              dynamic_scale: torch.Tensor = None,
              group_list_type: int = 1,
              w1_scale_bias: torch.Tensor = None,
              w2_scale_bias: torch.Tensor = None) -> torch.Tensor:
    """
    apply MLP: gate_up_proj -> swiglu -> down_proj

    Args:
        hidden_states: input hidden states with shape (num_tokens, hidden_size).
        w1: expert weights1 with shape
            (num_experts, hidden_size, intermediate_size * 2)
        w1_scale: weights1 scale with shape (num_experts, intermediate_size * 2)
        w2: expert weights2 with shape
            (num_experts, intermediate_size, hidden_size)
        w2_scale: weights2 scale with shape (num_experts, hidden_size)
        group_list: number of tokens for each expert, follow cumsum mode, and
            with shape (num_experts).
        transpose_weight:
            w1: (num_experts, intermediate_size * 2, hidden_size) ->
                    (num_experts, hidden_size, intermediate_size * 2)
            w2: (num_experts, hidden_size, intermediate_size) ->
                    (num_experts, intermediate_size, hidden_size)

    Returns:
        hidden_states: output hidden states after MLP.
    """

    if dynamic_scale is None:
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(
            hidden_states)
        # Dispose the original unquantized hidden states
        # to save npu memory because they're no longer used.
        dispose_tensor(unquantized_hidden_states)
    else:
        pertoken_scale = dynamic_scale

    bias1, bias2 = None, None
    _output_dtype = w2_scale.dtype

    if w1_scale_bias is not None:
        if group_list_type == 0:
            group_list = torch.cat(
                [group_list[:1], torch.diff(group_list, dim=0)])
            group_list_type = 1
        bias1 = [w1_scale_bias]
        bias2 = [w2_scale_bias]
        # TODO w4a8 scene: dynamic acquisition of dtype in the future
        _output_dtype = torch.bfloat16

    # gmm1: gate_up_proj
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        scale=[w1_scale],
        bias=bias1,
        per_token_scale=[pertoken_scale],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=_output_dtype)[0]

    # act_fn: swiglu
    hidden_states = torch_npu.npu_swiglu(hidden_states)
    hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(
        hidden_states)

    # gmm2: down_proj
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale],
        bias=bias2,
        per_token_scale=[swiglu_out_scale],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=_output_dtype)[0]

    return hidden_states


ascend_w8a8_dynamic.apply_mlp_decode = apply_mlp_decode
ascend_w8a8_dynamic.apply_mlp = apply_mlp


class AscendW4A8DynamicLinearMethod:
    """Linear method for Ascend W4A8_DYNAMIC
    """

    def __init__(self):
        self.transpose_weight = True
        try:
            self.group_size = get_current_vllm_config(
            ).quant_config.quant_description.get("group_size", 256)
        except AttributeError:
            self.group_size = 256

    @staticmethod
    def get_weight(input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_perchannel_param(output_size: int,
                             params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    def get_pergroup_param(self, input_size: int, output_size: int,
                           params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size,
                                                  1,
                                                  dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size,
                                                   1,
                                                   dtype=params_dtype)
        params_dict["weight_scale_second"] = torch.empty(output_size,
                                                         input_size //
                                                         self.group_size,
                                                         dtype=params_dtype)
        params_dict["weight_offset_second"] = torch.empty(output_size,
                                                          input_size //
                                                          self.group_size,
                                                          dtype=params_dtype)
        return params_dict

    @staticmethod
    def process_scale_second(weight: torch.Tensor, scale: torch.Tensor,
                             per_group_scale: torch.Tensor):
        k, n = weight.shape
        group_num, n = per_group_scale.shape
        weight_high = weight.to(torch.float32).reshape(
            group_num, -1, n) * per_group_scale.reshape(group_num, 1, n)
        weight_high = weight_high.reshape(k, n)
        bias = 8 * (weight_high.to(torch.float32) * scale).sum(dim=0)
        antiquant_scale = (scale * per_group_scale).reshape(group_num, n)
        return antiquant_scale.npu(), bias

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = None,
    ) -> torch.Tensor:
        return torch_npu.npu_weight_quant_batchmatmul(
            x,
            layer.weight,
            antiquant_scale=layer.weight_scale_second.to(x.dtype),
            antiquant_group_size=self.group_size,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module):
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten().to(
            torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()
        layer.weight_scale_second.data, scale_bias = self.process_scale_second(
            layer.weight.data,
            layer.weight_scale.data,
            layer.weight_scale_second.data.transpose(0, 1).contiguous(),
        )
        param = torch.nn.Parameter(scale_bias, requires_grad=False)
        layer.register_parameter("weight_scale_bias", param)
        layer.weight.data = torch_npu.npu_convert_weight_to_int4pack(
            layer.weight.data.to(torch.int32))


class AscendW4A8DynamicFusedMoEMethod:
    """FusedMoe method for Ascend W4A8_DYNAMIC.
    """

    def __init__(self):
        self.transpose_weight = True

        self.ep_group = get_ep_group()

        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled

        try:
            device_group = get_mc2_group().device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(
                local_rank)
        except AttributeError:
            self.moe_all_to_all_group_name = ""

    @staticmethod
    def get_weight(num_experts: int, intermediate_size_per_partition: int,
                   hidden_sizes: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(num_experts,
                                               2 *
                                               intermediate_size_per_partition,
                                               hidden_sizes,
                                               dtype=torch.int8)
        param_dict["w2_weight"] = torch.empty(num_experts,
                                              hidden_sizes,
                                              intermediate_size_per_partition,
                                              dtype=torch.int8)
        return param_dict

    @staticmethod
    def get_dynamic_quant_param(num_experts: int,
                                intermediate_size_per_partition: int,
                                hidden_sizes: int,
                                params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        config = get_current_vllm_config()
        group_size = config.quant_config.quant_description.get(
            "group_size", 256)

        param_dict["w13_weight_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=params_dtype)

        param_dict["w13_weight_offset"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=params_dtype)

        param_dict["w13_weight_scale_second"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // group_size,
            dtype=params_dtype)

        param_dict["w13_weight_offset_second"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // group_size,
            dtype=params_dtype)

        param_dict["w2_weight_scale"] = torch.empty(num_experts,
                                                    hidden_sizes,
                                                    1,
                                                    dtype=params_dtype)
        param_dict["w2_weight_offset"] = torch.empty(num_experts,
                                                     hidden_sizes,
                                                     1,
                                                     dtype=params_dtype)
        param_dict["w2_weight_scale_second"] = torch.empty(
            num_experts,
            hidden_sizes,
            intermediate_size_per_partition // group_size,
            dtype=params_dtype)
        param_dict["w2_weight_offset_second"] = torch.empty(
            num_experts,
            hidden_sizes,
            intermediate_size_per_partition // group_size,
            dtype=params_dtype)

        return param_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = True,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        shared_experts: Optional[Any] = None,
        quantized_x_for_share: Optional[Any] = None,
        dynamic_scale_for_share: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert router_logits.shape[
            1] == global_num_experts, "Number of global experts mismatch"

        # NOTE: now npu_moe_gating_top_k can only support `group_count=256` pattern
        if global_num_experts == 256:
            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                router_logits,
                k=top_k,  # topk currently is 8
                bias=e_score_correction_bias,
                k_group=topk_group,  # fix: 4
                group_count=num_expert_group,  # fix 8
                group_select_mode=
                1,  # 0: the maximum in the group; 1: topk2.sum(fix)
                renorm=0,  # 0: softmax->topk(fix); 1: topk->softmax
                norm_type=1,  # 0: softmax; 1: sigmoid(fix)
                # out_flag=False, # todo new api; should the third output be output
                # y2_flag=False, # old api; should the third output be output
                routed_scaling_factor=1,
                eps=float(1e-20))
        else:
            topk_weights, topk_ids = select_experts(
                hidden_states=x,
                router_logits=router_logits,
                top_k=top_k,
                use_grouped_topk=use_grouped_topk,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
            )

        fused_moe_state = get_forward_context().fused_moe_state
        shared_gate_up, shared_dequant_scale = None, None
        if shared_experts is not None and fused_moe_state == FusedMoEState.MC2:
            with npu_stream_switch("moe_secondary", 0):
                npu_wait_tensor(quantized_x_for_share, router_logits)
                share_up_out, _ = shared_experts.gate_up_proj(
                    (quantized_x_for_share, dynamic_scale_for_share))
                shared_gate_up, shared_dequant_scale = share_up_out[
                    0], share_up_out[1]

        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            topk_ids = torch.randint_like(topk_ids, 0, global_num_experts)

        topk_weights = topk_weights.to(x.dtype)
        if fused_moe_state == FusedMoEState.MC2:
            return fused_experts_with_mc2(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                w1_scale=layer.w13_weight_scale_second,
                w2_scale=layer.w2_weight_scale_second,
                w1_scale_bias=layer.w13_scale_bias,
                w2_scale_bias=layer.w2_scale_bias,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                top_k=top_k,
                expert_map=expert_map,
                moe_all_to_all_group_name=self.moe_all_to_all_group_name,
                log2phy=log2phy,
                global_redundant_expert_num=global_redundant_expert_num,
                shared_experts=shared_experts,
                is_torchair=self.torchair_graph_enabled,
                quantized_x_for_share=shared_gate_up,
                dynamic_scale_for_share=shared_dequant_scale,
                mc2_mask=kwargs.get("mc2_mask", None))
        else:
            # The current implementation of deepseek moe splits hidden_states
            # according to tp_size before they are feed into fused_moe module.
            # Therefore, all2all is needed no matter how dp/tp is set so as to
            # dispatch/combine tokens.
            return fused_experts_with_all2all(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                w1_scale=layer.w13_weight_scale_second,
                w2_scale=layer.w2_weight_scale_second,
                w1_scale_bias=layer.w13_scale_bias,
                w2_scale_bias=layer.w2_scale_bias,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                top_k=top_k,
                expert_map=expert_map,
                ep_group=self.ep_group,
                log2phy=log2phy,
                global_redundant_expert_num=global_redundant_expert_num,
            )

    def process_scale(self, weight: torch.Tensor, scale, per_group_scale):
        group_num, k, n = weight.shape
        per_group_scale = per_group_scale.reshape(group_num, -1, n)
        group_num, quantgroup_num, n = per_group_scale.shape
        weight_high = weight.to(torch.float32).reshape([group_num, quantgroup_num, -1, n]) * \
            per_group_scale.reshape([group_num, quantgroup_num, 1, n])
        weight_high = weight_high.reshape([group_num, k, n])
        bias = 8 * (weight_high.to(torch.float32) * scale).sum(axis=1)
        scale_fp32 = (scale * per_group_scale).to(torch.float16).to(
            torch.float32)
        scale_fp32_np = scale_fp32.cpu().numpy()
        scale_fp32_np.dtype = np.uint32
        sscale_uint64 = np.zeros((group_num, quantgroup_num, n * 2),
                                 dtype=np.uint32)

        sscale_uint64[..., ::2] = scale_fp32_np

        sscale_uint64_buffer = np.frombuffer(sscale_uint64.tobytes(),
                                             dtype=np.int64).copy()
        sscale_uint64_tensor = torch.from_numpy(sscale_uint64_buffer).reshape(
            group_num, quantgroup_num, n)
        sscale_uint64_tensor = sscale_uint64_tensor.npu()
        return sscale_uint64_tensor, bias

    def process_weights_after_loading(self, layer):
        if self.transpose_weight:
            layer.w13_weight.data = layer.w13_weight.data.transpose(
                1, 2).contiguous()
            layer.w2_weight.data = layer.w2_weight.data.transpose(
                1, 2).contiguous()
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(
            1, 2).contiguous()
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(
            1, 2).contiguous()
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.view(
            layer.w13_weight_offset.data.shape[0], -1)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(
            layer.w2_weight_offset.data.shape[0], -1)
        layer.w13_weight_scale_second.data = layer.w13_weight_scale_second.data.transpose(
            1, 2).contiguous()
        layer.w2_weight_scale_second.data = layer.w2_weight_scale_second.data.transpose(
            1, 2).contiguous()

        layer.w13_weight_scale_second.data, bias = self.process_scale(
            layer.w13_weight, layer.w13_weight_scale.data,
            layer.w13_weight_scale_second.data)
        param = torch.nn.Parameter(bias, requires_grad=False)
        layer.register_parameter("w13_scale_bias", param)
        layer.w2_weight_scale_second.data, bias1 = self.process_scale(
            layer.w2_weight, layer.w2_weight_scale.data,
            layer.w2_weight_scale_second.data)
        param = torch.nn.Parameter(bias1, requires_grad=False)
        layer.register_parameter("w2_scale_bias", param)

        layer.w13_weight.data = torch_npu.npu_quantize(
            layer.w13_weight.data.to(torch.float32),
            torch.tensor([1.]).npu(), None, torch.quint4x2, -1, False)
        layer.w2_weight.data = torch_npu.npu_quantize(
            layer.w2_weight.data.to(torch.float32),
            torch.tensor([1.]).npu(), None, torch.quint4x2, -1, False)
