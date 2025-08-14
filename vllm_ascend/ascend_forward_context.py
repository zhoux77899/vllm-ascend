import math
from contextlib import contextmanager
from enum import Enum
from typing import Any, Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed import (get_dp_group, get_ep_group,
                              get_tensor_model_parallel_world_size)
from vllm.forward_context import get_forward_context, set_forward_context

import vllm_ascend.envs as envs_ascend
from vllm_ascend.distributed.moe_comm_method import MoECommMethod


class FusedMoEState(Enum):
    AllGather = 0
    All2All = 1
    MC2 = 2
    AllGatherEP = 3
    NaiveMulticast = 4
    All2AllSeq = 5


# TODO(zzzzwwjj): add soc_version to choose branch
def _get_fused_moe_state(ep_size: int, with_prefill: bool,
                         is_deepseek_v3_r1: bool):
    # the fusion operator torch_npu.npu_grouped_matmul_finalize_routing called by allgather ep
    # only supports deepseek v3/r1
    if (envs_ascend.VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP and ep_size > 1
            and is_deepseek_v3_r1):
        return FusedMoEState.AllGatherEP
    elif ep_size == 1:
        if with_prefill:
            return FusedMoEState.NaiveMulticast
        else:
            return FusedMoEState.AllGather
    elif envs_ascend.VLLM_ASCEND_ENABLE_MOE_ALL2ALL_SEQ:
        # MC2 Dispatch/Combine performs better than alltoall_seq in decoding stage.
        return (FusedMoEState.All2AllSeq if
                (ep_size < 16 or with_prefill) else FusedMoEState.MC2)
    # NOTE: mc2 need ep_size >= 16 & all2all can't use in torchair graph.
    elif ep_size < 16 or with_prefill:
        return FusedMoEState.All2All
    else:
        return FusedMoEState.MC2


@contextmanager
def set_ascend_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: Optional[int] = None,
    num_tokens_across_dp: Optional[torch.Tensor] = None,
    with_prefill: bool = True,
    in_profile_run: bool = False,
    reserved_mc2_mask: Optional[torch.Tensor] = None,
    moe_comm_method: Optional[MoECommMethod] = None,
    num_actual_tokens: Optional[int] = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    We add some additional param into forward_context.
    """
    with set_forward_context(attn_metadata,
                             vllm_config,
                             virtual_engine=virtual_engine,
                             num_tokens=num_tokens,
                             num_tokens_across_dp=num_tokens_across_dp):
        forward_context = get_forward_context()
        forward_context.moe_comm_method = moe_comm_method
        forward_context.with_prefill = with_prefill
        ep_size = (get_ep_group().world_size if
                   vllm_config.parallel_config.enable_expert_parallel else 1)

        is_deepseek_v3_r1 = hasattr(
            vllm_config.model_config.hf_config, 'n_routed_experts'
        ) and vllm_config.model_config.hf_config.n_routed_experts == 256
        fused_moe_state = _get_fused_moe_state(ep_size, with_prefill,
                                               is_deepseek_v3_r1)
        forward_context.fused_moe_state = fused_moe_state
        forward_context.in_profile_run = in_profile_run

        # NOTE: This cannot be set using set_forward_context
        # due to multiple warmups before actual capturing
        forward_context.capturing = False

        if num_tokens is None and attn_metadata is not None:
            num_tokens = attn_metadata.num_actual_tokens

        dp_world_size = get_dp_group().world_size
        if dp_world_size > 1 and forward_context.dp_metadata is not None:
            max_tokens_across_dp = forward_context.dp_metadata.max_tokens_across_dp_cpu.item(
            )
        else:
            max_tokens_across_dp = num_tokens

        forward_context.max_tokens_across_dp = max_tokens_across_dp

        if num_tokens is not None:
            if num_actual_tokens is None:
                num_actual_tokens = num_tokens
            tp_world_size = get_tensor_model_parallel_world_size()
            # NOTE: token num which need to pad to when mc2
            forward_context.padded_num_tokens = math.ceil(
                max_tokens_across_dp / tp_world_size) * tp_world_size

            if reserved_mc2_mask is not None:
                mc2_mask = reserved_mc2_mask[:forward_context.
                                             padded_num_tokens]
                mc2_mask[:num_actual_tokens] = True
                mc2_mask[num_actual_tokens:] = False
                forward_context.mc2_mask = mc2_mask

        try:
            yield
        finally:
            pass
