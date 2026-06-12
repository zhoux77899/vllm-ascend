#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Unit tests for AscendMMEncoderAttention after Stage 1 FIA op replacement."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Avoid pulling in vllm / torch_npu on CI without full stack.
if "torch_npu" not in sys.modules:
    sys.modules["torch_npu"] = types.ModuleType("torch_npu")

_vllm_mm_encoder = types.ModuleType("vllm.model_executor.layers.attention.mm_encoder_attention")


class _MMEncoderAttention:
    pass


_vllm_mm_encoder.MMEncoderAttention = _MMEncoderAttention
sys.modules["vllm.model_executor.layers.attention.mm_encoder_attention"] = _vllm_mm_encoder

_vllm_registry = types.ModuleType("vllm.v1.attention.backends.registry")


class _AttentionBackendEnum:
    pass


_vllm_registry.AttentionBackendEnum = _AttentionBackendEnum
sys.modules["vllm.v1.attention.backends.registry"] = _vllm_registry

_MM_ENCODER_ATTENTION_PATH = (
    Path(__file__).resolve().parents[3] / "vllm_ascend" / "ops" / "mm_encoder_attention.py"
)
_spec = importlib.util.spec_from_file_location(
    "vllm_ascend.ops.mm_encoder_attention",
    _MM_ENCODER_ATTENTION_PATH,
)
_mm_encoder_attention = importlib.util.module_from_spec(_spec)
sys.modules["vllm_ascend.ops.mm_encoder_attention"] = _mm_encoder_attention
_spec.loader.exec_module(_mm_encoder_attention)

AscendMMEncoderAttention = _mm_encoder_attention.AscendMMEncoderAttention
MAX_PAD_SIZE = _mm_encoder_attention.MAX_PAD_SIZE
SWA_INT_MAX = _mm_encoder_attention.SWA_INT_MAX
FIA_BLOCK_SIZE = _mm_encoder_attention.FIA_BLOCK_SIZE


def _make_layer(num_heads=4, num_kv_heads=4, head_size=72):
    layer = AscendMMEncoderAttention.__new__(AscendMMEncoderAttention)
    layer.num_heads = num_heads
    layer.num_kv_heads = num_kv_heads
    layer.num_queries_per_kv = num_heads // num_kv_heads
    layer.head_size = head_size
    layer.enable_pad = 64 < head_size < 128
    layer.scale_value = head_size**-0.5
    return layer


def _ref_attention(query, key, value, seq_lens):
    """CPU fp32 reference. Inputs [B, S, H, D]; seq_lens: per-sample valid length."""
    bsz, _, num_heads, head_dim = query.shape
    out = torch.zeros_like(query, dtype=torch.float32)
    scale = head_dim**-0.5
    for b in range(bsz):
        s = seq_lens[b]
        q_b = query[b, :s].permute(1, 0, 2).to(torch.float32)
        k_b = key[b, :s].permute(1, 0, 2).to(torch.float32)
        v_b = value[b, :s].permute(1, 0, 2).to(torch.float32)
        out_b = F.scaled_dot_product_attention(
            q_b.unsqueeze(0),
            k_b.unsqueeze(0),
            v_b.unsqueeze(0),
            scale=scale,
            dropout_p=0.0,
            is_causal=False,
        ).squeeze(0)
        out[b, :s] = out_b.permute(1, 0, 2).to(query.dtype)
    return out


def _compute_fia_output(query, key, value, actual_seq_lengths, scale):
    """Emulate functional FIA by segment-wise CPU SDPA."""
    output = torch.zeros_like(query)
    starts = [0] + list(actual_seq_lengths[:-1])
    for i, end in enumerate(actual_seq_lengths):
        beg = starts[i]
        if end <= beg:
            continue
        q_seg = query[beg:end].permute(1, 0, 2).to(torch.float32)
        k_seg = key[beg:end].permute(1, 0, 2).to(torch.float32)
        v_seg = value[beg:end].permute(1, 0, 2).to(torch.float32)
        attn = F.scaled_dot_product_attention(
            q_seg.unsqueeze(0),
            k_seg.unsqueeze(0),
            v_seg.unsqueeze(0),
            scale=scale,
            dropout_p=0.0,
            is_causal=False,
        ).squeeze(0)
        output[beg:end] = attn.permute(1, 0, 2).to(query.dtype)
    return output


@pytest.fixture
def fake_fia():
    captured = {}

    def fake_fia(
        *,
        query,
        key,
        value,
        atten_mask,
        block_table,
        input_layout,
        block_size,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        num_key_value_heads,
        num_heads,
        scale,
        sparse_mode,
        pre_tokens=None,
        next_tokens=None,
    ):
        captured["q_shape"] = query.shape
        captured["k_shape"] = key.shape
        captured["v_shape"] = value.shape
        captured["input_layout"] = input_layout
        captured["block_size"] = block_size
        captured["actual_seq_lengths"] = actual_seq_lengths
        captured["actual_seq_lengths_kv"] = actual_seq_lengths_kv
        captured["num_heads"] = num_heads
        captured["num_key_value_heads"] = num_key_value_heads
        captured["scale"] = scale
        captured["sparse_mode"] = sparse_mode
        captured["block_table"] = block_table
        captured["atten_mask"] = atten_mask
        captured["pre_tokens"] = pre_tokens
        captured["next_tokens"] = next_tokens
        output = _compute_fia_output(query, key, value, actual_seq_lengths, scale)
        return output, None

    _mm_encoder_attention.torch_npu.npu_fused_infer_attention_score = fake_fia
    yield captured
    if hasattr(_mm_encoder_attention.torch_npu, "npu_fused_infer_attention_score"):
        delattr(_mm_encoder_attention.torch_npu, "npu_fused_infer_attention_score")


def test_shape_basic(fake_fia):
    layer = _make_layer(num_heads=4, num_kv_heads=4, head_size=128)
    bsz, q_len = 2, 4
    query = torch.randn(bsz, q_len, layer.num_heads * layer.head_size)
    key = query.clone()
    value = query.clone()
    cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32)

    out = layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

    assert out.shape == (bsz, q_len, layer.num_heads * layer.head_size)
    assert fake_fia["input_layout"] == "TND"
    assert fake_fia["sparse_mode"] == 0
    assert fake_fia["block_size"] == FIA_BLOCK_SIZE


def test_shape_4d_output(fake_fia):
    layer = _make_layer(num_heads=4, num_kv_heads=4, head_size=128)
    bsz, q_len = 2, 4
    query = torch.randn(bsz, q_len, layer.num_heads, layer.head_size)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32)

    out = layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

    assert out.shape == query.shape


def test_head_dim_pad(fake_fia):
    layer = _make_layer(num_heads=4, num_kv_heads=4, head_size=72)
    bsz, q_len = 2, 4
    query = torch.randn(bsz, q_len, layer.num_heads, layer.head_size, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32)

    out = layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

    assert fake_fia["q_shape"] == (bsz * q_len, 4, MAX_PAD_SIZE)
    assert out.shape == query.shape


def test_variable_seqlens(fake_fia):
    layer = _make_layer(num_heads=4, num_kv_heads=4, head_size=72)
    seq_lens = [3, 7, 2]
    cu_seqlens = torch.tensor([0, 3, 10, 12], dtype=torch.int32, device="cpu")
    max_q_len = max(seq_lens)
    torch.manual_seed(0)
    query = torch.randn(len(seq_lens), max_q_len, layer.num_heads, 72, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    out = layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

    assert out.shape == query.shape
    assert fake_fia["actual_seq_lengths"] == [3, 10, 12]
    assert fake_fia["actual_seq_lengths_kv"] == [3, 10, 12]
    assert fake_fia["q_shape"] == (len(seq_lens) * max_q_len, 4, MAX_PAD_SIZE)


def test_precomputed_sequence_lengths(fake_fia):
    layer = _make_layer(num_heads=4, num_kv_heads=4, head_size=72)
    seq_lens = [3, 7, 2]
    sequence_lengths = torch.tensor(seq_lens, dtype=torch.int64, device="cpu")
    max_q_len = max(seq_lens)
    query = torch.randn(len(seq_lens), max_q_len, layer.num_heads, 72, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seqlens = torch.tensor([0, 3, 10, 12], dtype=torch.int32, device="cpu")

    out = layer.forward_oot(
        query,
        key,
        value,
        cu_seqlens=cu_seqlens,
        sequence_lengths=sequence_lengths,
    )

    assert fake_fia["actual_seq_lengths"] == [3, 10, 12]
    assert out.shape == query.shape


def test_mqa_gqa_repeat(fake_fia):
    layer = _make_layer(num_heads=8, num_kv_heads=2, head_size=72)
    bsz, q_len = 2, 4
    query = torch.randn(bsz, q_len, layer.num_heads, layer.head_size, dtype=torch.bfloat16)
    key = torch.randn(bsz, q_len, layer.num_kv_heads, layer.head_size, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32)

    layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

    assert fake_fia["k_shape"][1] == layer.num_heads
    assert fake_fia["v_shape"][1] == layer.num_heads
    assert fake_fia["num_heads"] == 8
    assert fake_fia["num_key_value_heads"] == 2


def test_get_vit_fia_params_dense_constants():
    layer = _make_layer(num_heads=4, num_kv_heads=4, head_size=72)
    bsz, q_len = 2, 4
    cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32)

    params = layer._get_vit_fia_params(bsz, q_len, cu_seqlens, None)

    assert params[0] == [4, 8]
    assert params[1] == [4, 8]
    assert params[2] == FIA_BLOCK_SIZE
    assert params[3] is None
    assert params[4] == "TND"
    assert params[5] == 0
    assert params[6] is None
    assert params[7] == SWA_INT_MAX
    assert params[8] == SWA_INT_MAX


def test_fia_call_args(fake_fia):
    layer = _make_layer(num_heads=16, num_kv_heads=16, head_size=72)
    bsz, q_len = 4, 4
    query = torch.randn(bsz, q_len, layer.num_heads, layer.head_size, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32)

    layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

    assert fake_fia["input_layout"] == "TND"
    assert fake_fia["sparse_mode"] == 0
    assert fake_fia["block_size"] == FIA_BLOCK_SIZE
    assert fake_fia["block_table"] is None
    assert fake_fia["atten_mask"] is None
    assert fake_fia["pre_tokens"] == SWA_INT_MAX
    assert fake_fia["next_tokens"] == SWA_INT_MAX
    assert fake_fia["actual_seq_lengths"] == [4, 8, 12, 16]
    assert fake_fia["scale"] == pytest.approx(layer.scale_value)


def test_dtype_bf16(fake_fia):
    layer = _make_layer(num_heads=16, num_kv_heads=16, head_size=72)
    bsz, q_len = 2, 4
    query = torch.randn(bsz, q_len, layer.num_heads, layer.head_size, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32)

    out = layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

    assert out.dtype == torch.bfloat16
