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

"""Ascend implementation of upstream :class:`MMEncoderAttention`.

Eager path calls functional FIA; ACL-graph capture records ``.out`` FIA tasks
with ``graph_task_group_begin/end``, matching the LLM pattern in
:mod:`vllm_ascend.attention.attention_v1`.
"""

from __future__ import annotations

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention  # type: ignore
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_ascend.utils import weak_ref_tensors
from vllm_ascend.worker.encoder_acl_graph import (
    get_encoder_graph_params,
    get_encoder_graph_runtime_state,
    update_encoder_graph_workspace,
)

MIN_PAD_SIZE: int = 64
MAX_PAD_SIZE: int = 128
SWA_INT_MAX: int = 2147483647
FIA_BLOCK_SIZE: int = 128


class AscendMMEncoderAttention(MMEncoderAttention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )

        self.enable_pad = self.head_size > MIN_PAD_SIZE and self.head_size < MAX_PAD_SIZE
        self.scale_value = self.head_size**-0.5

    def ascend_attn_scale(self) -> float:
        return float(self.scale) if self.scale is not None else self.scale_value

    @classmethod
    def maybe_compute_seq_lens(
        cls,
        attn_backend: AttentionBackendEnum,
        cu_seqlens: np.ndarray,
        device: torch.device,
    ) -> np.ndarray | None:
        """Returns per-sequence lengths on CPU for ``prepare_encoder_metadata``.

        FIA uses cumulative ``actual_seq_lengths`` computed in ``forward_oot``
        via ``get_vit_fia_params``.
        """
        if cu_seqlens is None:
            return None

        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        seq_lens = torch.from_numpy(seq_lens).to("cpu", non_blocking=True)

        return seq_lens

    def reshape_qkv_to_3d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = query.view(bsz * q_len, self.num_heads, self.head_size)
        key = key.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if (num_repeat := self.num_queries_per_kv) > 1:
            key = torch.repeat_interleave(key, num_repeat, dim=1)
            value = torch.repeat_interleave(value, num_repeat, dim=1)

        return query, key, value

    def maybe_compute_cu_seqlens(
        self,
        bsz: int,
        q_len: int,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            if cu_seqlens.device.type != "cpu":
                cu_seqlens = cu_seqlens.to("cpu")
            return cu_seqlens

        cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device="cpu")
        return cu_seqlens

    def maybe_compute_actual_seq_lengths(
        self,
        bsz: int,
        q_len: int,
        cu_seqlens: torch.Tensor | None,
        sequence_lengths: torch.Tensor | None,
    ) -> tuple[list[int], list[int]]:
        """Build FIA ``actual_seq_lengths`` as cumulative host-side ``list[int]``."""
        if sequence_lengths is not None:
            seq_lens_cpu = sequence_lengths
            if seq_lens_cpu.device.type != "cpu":
                seq_lens_cpu = seq_lens_cpu.to("cpu")
            actual = seq_lens_cpu.cumsum(0).to(torch.int64).tolist()
        else:
            cu = self.maybe_compute_cu_seqlens(bsz, q_len, cu_seqlens)
            actual = cu[1:].to(torch.int64).tolist()

        return actual, actual

    def get_vit_fia_params(
        self,
        bsz: int,
        q_len: int,
        cu_seqlens: torch.Tensor | None,
        sequence_lengths: torch.Tensor | None,
    ) -> tuple[
        list[int],
        list[int],
        int,
        torch.Tensor | None,
        str,
        int,
        torch.Tensor | None,
        int,
        int,
    ]:
        """Build FIA metadata for dense ViT (PrefillNoCache + non-causal)."""
        actual_seq_lengths, actual_seq_lengths_kv = self.maybe_compute_actual_seq_lengths(
            bsz, q_len, cu_seqlens, sequence_lengths
        )
        return (
            actual_seq_lengths,
            actual_seq_lengths_kv,
            FIA_BLOCK_SIZE,
            None,
            "TND",
            0,
            None,
            SWA_INT_MAX,
            SWA_INT_MAX,
        )

    def forward_eager_vit_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() == 4

        (
            actual_seq_lengths,
            actual_seq_lengths_kv,
            block_size,
            block_table,
            input_layout,
            sparse_mode,
            attn_mask,
            pre_tokens,
            next_tokens,
        ) = self.get_vit_fia_params(bsz, q_len, cu_seqlens, sequence_lengths)

        q, k, v = self.reshape_qkv_to_3d(query, key, value, bsz, q_len, kv_len)

        origin_head_dim = q.shape[-1]
        if self.enable_pad:
            pad_len = MAX_PAD_SIZE - origin_head_dim
            q = F.pad(q, (0, pad_len), mode="constant", value=0)
            k = F.pad(k, (0, pad_len), mode="constant", value=0)
            v = F.pad(v, (0, pad_len), mode="constant", value=0)

        context_layer, _ = torch_npu.npu_fused_infer_attention_score(
            query=q,
            key=k,
            value=v,
            atten_mask=attn_mask,
            block_table=block_table,
            input_layout=input_layout,
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.ascend_attn_scale(),
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
        )

        if self.enable_pad:
            context_layer = context_layer[..., :origin_head_dim]

        if is_reshaped:
            context_layer = einops.rearrange(context_layer, "(b s) h d -> b s h d", b=bsz).contiguous()
        else:
            context_layer = einops.rearrange(context_layer, "(b s) h d -> b s (h d)", b=bsz).contiguous()
        return context_layer

    def full_graph_vit_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        runtime = get_encoder_graph_runtime_state()
        token_budget = runtime.token_budget
        params = get_encoder_graph_params()
        if token_budget is None or params is None:
            raise RuntimeError("Encoder graph capture state was not initialized (missing token_budget).")

        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() == 4

        (
            actual_seq_lengths,
            actual_seq_lengths_kv,
            block_size,
            block_table,
            input_layout,
            sparse_mode,
            attn_mask,
            pre_tokens,
            next_tokens,
        ) = self.get_vit_fia_params(bsz, q_len, cu_seqlens, sequence_lengths)

        q, k, v = self.reshape_qkv_to_3d(query, key, value, bsz, q_len, kv_len)

        origin_head_dim = q.shape[-1]
        if self.enable_pad:
            pad_len = MAX_PAD_SIZE - origin_head_dim
            q = F.pad(q, (0, pad_len), mode="constant", value=0)
            k = F.pad(k, (0, pad_len), mode="constant", value=0)
            v = F.pad(v, (0, pad_len), mode="constant", value=0)

        out = torch.empty_like(q)
        softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)

        vit_layer_idx = runtime.capture_layer_cursor
        runtime.capture_layer_cursor = vit_layer_idx + 1
        uses_sequence_lengths_host = sequence_lengths is not None

        workspace = params.workspaces.get(token_budget)
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=q,
                key=k,
                value=v,
                atten_mask=attn_mask,
                block_table=block_table,
                input_layout=input_layout,
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                scale=self.ascend_attn_scale(),
            )
            update_encoder_graph_workspace(token_budget, workspace)

        stream = torch_npu.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        params.events[token_budget].append(event)

        packed = (
            weak_ref_tensors(q),
            weak_ref_tensors(k),
            weak_ref_tensors(v),
            block_table,
            attn_mask,
            block_size,
            uses_sequence_lengths_host,
            vit_layer_idx,
            self.num_kv_heads,
            self.num_heads,
            self.ascend_attn_scale(),
            weak_ref_tensors(out),
            weak_ref_tensors(softmax_lse),
        )
        params.attn_params[token_budget].append(packed)

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=q,
            key=k,
            value=v,
            atten_mask=attn_mask,
            block_table=block_table,
            input_layout=input_layout,
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.ascend_attn_scale(),
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            workspace=workspace,
            out=[out, softmax_lse],
        )
        handle = torch.npu.graph_task_group_end(stream)
        params.handles[token_budget].append(handle)

        context_layer = out
        if self.enable_pad:
            context_layer = context_layer[..., :origin_head_dim]

        if is_reshaped:
            context_layer = einops.rearrange(context_layer, "(b s) h d -> b s h d", b=bsz).contiguous()
        else:
            context_layer = einops.rearrange(context_layer, "(b s) h d -> b s (h d)", b=bsz).contiguous()
        return context_layer

    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ):
        if get_encoder_graph_runtime_state().capturing:
            return self.full_graph_vit_fia(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                sequence_lengths=sequence_lengths,
            )

        return self.forward_eager_vit_fia(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
