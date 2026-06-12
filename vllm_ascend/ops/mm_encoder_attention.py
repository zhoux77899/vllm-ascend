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

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention  # type: ignore
from vllm.v1.attention.backends.registry import AttentionBackendEnum

MIN_PAD_SIZE: int = 64  # min_size to pad weight
MAX_PAD_SIZE: int = 128  # max_size to pad weight
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
        """
        Args:
            num_heads: number of attention heads per partition.
            head_size: hidden_size per attention head.
            scale: scale factor.
            num_kv_heads: number of kv heads.
            prefix: This has no effect, it is only here to make it easier to
                    swap between Attention and MMEncoderAttention.
            multimodal_config: configs for multi-modal.
        """
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )

        self.enable_pad = self.head_size > MIN_PAD_SIZE and self.head_size < MAX_PAD_SIZE
        self.scale_value = self.head_size**-0.5

    @classmethod
    def maybe_compute_seq_lens(
        cls,
        attn_backend: AttentionBackendEnum,
        cu_seqlens: np.ndarray,
        device: torch.device,
    ) -> np.ndarray | None:
        """Upstream contract helper for ``prepare_encoder_metadata``.

        Returns per-sequence lengths on CPU. FIA uses cumulative
        ``actual_seq_lengths`` computed in ``forward_oot`` via
        ``_get_vit_fia_params``.
        """
        if cu_seqlens is None:
            return None

        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        seq_lens = torch.from_numpy(seq_lens).to("cpu", non_blocking=True)

        return seq_lens

    def _reshape_qkv_to_3d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape query, key, value to 3D tensors:
        (batch_size * seq_len, num_heads, head_size)
        """
        query = query.view(bsz * q_len, self.num_heads, self.head_size)
        key = key.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=1)
            value = torch.repeat_interleave(value, num_repeat, dim=1)

        return query, key, value

    def _maybe_compute_cu_seqlens(
        self,
        bsz: int,
        q_len: int,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            if cu_seqlens.device.type != "cpu":
                cu_seqlens = cu_seqlens.to("cpu")
            return cu_seqlens

        # If cu_seqlens is not provided, we create a default one assuming all sequences have the same length.
        # This is used by models such as Hunyuan-OCR, which always pass None as cu_seqlens and rely on the operator to
        # compute it internally.
        cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device="cpu")
        return cu_seqlens

    def _maybe_compute_actual_seq_lengths(
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
            cu = self._maybe_compute_cu_seqlens(bsz, q_len, cu_seqlens)
            actual = cu[1:].to(torch.int64).tolist()

        return actual, actual

    def _get_vit_fia_params(
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
        """Build FIA inputs for dense ViT (PrefillNoCache + non-causal).

        Stage 1 uses fixed dense constants; Stage 2 capture/replay will reuse
        this helper as the single source of FIA metadata.
        """
        actual_seq_lengths, actual_seq_lengths_kv = self._maybe_compute_actual_seq_lengths(
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

    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # kept for upstream sig parity
        sequence_lengths: torch.Tensor | None = None,
    ):
        return self._forward_eager_fia(
            query, key, value, cu_seqlens, max_seqlen, sequence_lengths
        )

    def _forward_eager_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ):
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
        ) = self._get_vit_fia_params(bsz, q_len, cu_seqlens, sequence_lengths)

        # q, k, v: [b, s, head, head_dim] -> [b * s, head, head_dim]
        q, k, v = self._reshape_qkv_to_3d(query, key, value, bsz, q_len, kv_len)

        origin_head_dim = q.shape[-1]
        if self.enable_pad:
            pad_len = MAX_PAD_SIZE - origin_head_dim
            # [b * s, head, head_dim] -> [b * s, head, MAX_PAD_SIZE]
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
            scale=self.scale_value,
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
