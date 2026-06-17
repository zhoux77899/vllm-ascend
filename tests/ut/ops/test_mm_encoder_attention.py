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

"""Unit tests for AscendMMEncoderAttention FIA eager + capture paths."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

if "torch_npu" not in sys.modules:
    _torch_npu = types.ModuleType("torch_npu")
    _torch_npu.npu = MagicMock()
    _torch_npu.npu.current_stream = MagicMock(return_value=MagicMock())
    sys.modules["torch_npu"] = _torch_npu

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

# Stub encoder graph module before loading mm_encoder_attention.
_encoder_graph_state = types.SimpleNamespace(
    token_budget=None,
    capturing=False,
    capture_layer_cursor=0,
    host_cu_seqlens_ends=None,
    host_cu_window_seqlens_ends=None,
    host_sequence_lengths=None,
)
_encoder_graph_params = types.SimpleNamespace(
    events={2048: []},
    workspaces={2048: None},
    handles={2048: []},
    attn_params={2048: []},
)


def _get_encoder_graph_runtime_state():
    return _encoder_graph_state


def _get_encoder_graph_params():
    return _encoder_graph_params


def _update_encoder_graph_workspace(token_budget, workspace):
    _encoder_graph_params.workspaces[token_budget] = workspace


def _weak_ref_tensors(x):
    return x


_encoder_acl_graph = types.ModuleType("vllm_ascend.worker.encoder_acl_graph")
_encoder_acl_graph.get_encoder_graph_runtime_state = _get_encoder_graph_runtime_state
_encoder_acl_graph.get_encoder_graph_params = _get_encoder_graph_params
_encoder_acl_graph.update_encoder_graph_workspace = _update_encoder_graph_workspace
_encoder_acl_graph.encoder_graph_capture_scope = MagicMock()
sys.modules["vllm_ascend.worker.encoder_acl_graph"] = _encoder_acl_graph

_vllm_ascend_utils = types.ModuleType("vllm_ascend.utils")
_vllm_ascend_utils.weak_ref_tensors = _weak_ref_tensors
sys.modules["vllm_ascend.utils"] = _vllm_ascend_utils
sys.modules.setdefault("vllm_ascend", types.ModuleType("vllm_ascend"))
sys.modules.setdefault("vllm_ascend.worker", types.ModuleType("vllm_ascend.worker"))

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


def _reset_encoder_graph_state():
    _encoder_graph_state.token_budget = None
    _encoder_graph_state.capturing = False
    _encoder_graph_state.capture_layer_cursor = 0
    _encoder_graph_params.events = {2048: []}
    _encoder_graph_params.workspaces = {2048: None}
    _encoder_graph_params.handles = {2048: []}
    _encoder_graph_params.attn_params = {2048: []}


def _make_layer(num_heads=4, num_kv_heads=4, head_size=72):
    layer = AscendMMEncoderAttention.__new__(AscendMMEncoderAttention)
    layer.num_heads = num_heads
    layer.num_kv_heads = num_kv_heads
    layer.num_queries_per_kv = num_heads // num_kv_heads
    layer.head_size = head_size
    layer.enable_pad = 64 < head_size < 128
    layer.scale_value = head_size**-0.5
    layer.scale = None
    return layer


def _compute_fia_output(query, key, value, actual_seq_lengths, scale):
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
        captured["mode"] = "functional"
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

    def fake_fia_out(*, workspace, out, **kwargs):
        captured["mode"] = "out"
        captured["workspace"] = workspace
        captured["softmax_lse"] = out[1]
        captured["q_shape"] = kwargs["query"].shape
        captured["actual_seq_lengths"] = kwargs["actual_seq_lengths"]
        captured["block_size"] = kwargs["block_size"]
        captured["sparse_mode"] = kwargs["sparse_mode"]
        captured["pre_tokens"] = kwargs.get("pre_tokens")
        captured["next_tokens"] = kwargs.get("next_tokens")
        out[0][...] = _compute_fia_output(
            kwargs["query"],
            kwargs["key"],
            kwargs["value"],
            kwargs["actual_seq_lengths"],
            kwargs["scale"],
        )

    def fake_get_workspace(**kwargs):
        return torch.zeros(1)

    _mm_encoder_attention.torch_npu.npu_fused_infer_attention_score = fake_fia
    _mm_encoder_attention.torch_npu.npu_fused_infer_attention_score.out = fake_fia_out
    _mm_encoder_attention.torch_npu._npu_fused_infer_attention_score_get_max_workspace = fake_get_workspace
    _mm_encoder_attention.torch.npu = MagicMock()
    _mm_encoder_attention.torch.npu.ExternalEvent = MagicMock
    _mm_encoder_attention.torch.npu.graph_task_group_begin = MagicMock()
    _mm_encoder_attention.torch.npu.graph_task_group_end = MagicMock(return_value=42)
    yield captured
    _reset_encoder_graph_state()


def test_shape_basic(fake_fia):
    layer = _make_layer(num_heads=4, num_kv_heads=4, head_size=128)
    bsz, q_len = 2, 4
    query = torch.randn(bsz, q_len, layer.num_heads * layer.head_size)
    key = query.clone()
    value = query.clone()
    cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32)

    out = layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

    assert out.shape == (bsz, q_len, layer.num_heads * layer.head_size)
    assert fake_fia["mode"] == "functional"
    assert fake_fia["input_layout"] == "TND"
    assert fake_fia["sparse_mode"] == 0
    assert fake_fia["block_size"] == FIA_BLOCK_SIZE


def test_maybe_compute_actual_seq_lengths_from_sequence_lengths():
    layer = _make_layer()
    actual_q, actual_kv = layer.maybe_compute_actual_seq_lengths(
        bsz=2,
        q_len=4,
        cu_seqlens=None,
        sequence_lengths=torch.tensor([4, 4], dtype=torch.int64),
    )
    assert actual_q == [4, 8]
    assert actual_kv == [4, 8]


def test_get_vit_fia_params_from_cu_seqlens():
    layer = _make_layer()
    cu_seqlens = torch.arange(0, 9, step=4, dtype=torch.int32)
    actual_q, actual_kv, block_size, block_table, layout, sparse_mode, attn_mask, pre, nxt = (
        layer.get_vit_fia_params(
            bsz=2,
            q_len=4,
            cu_seqlens=cu_seqlens,
            sequence_lengths=None,
        )
    )
    assert actual_q == [4, 8]
    assert actual_kv == [4, 8]
    assert block_size == FIA_BLOCK_SIZE
    assert block_table is None
    assert layout == "TND"
    assert sparse_mode == 0
    assert attn_mask is None
    assert pre == SWA_INT_MAX
    assert nxt == SWA_INT_MAX


def test_variable_seqlens(fake_fia):
    layer = _make_layer(num_heads=4, num_kv_heads=4, head_size=72)
    seq_lens = [3, 7, 2]
    cu_seqlens = torch.tensor([0, 3, 10, 12], dtype=torch.int32, device="cpu")
    max_q_len = max(seq_lens)
    query = torch.randn(len(seq_lens), max_q_len, layer.num_heads, 72, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    out = layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

    assert out.shape == query.shape
    assert fake_fia["actual_seq_lengths"] == [3, 10, 12]
    assert fake_fia["q_shape"] == (len(seq_lens) * max_q_len, 4, MAX_PAD_SIZE)


def test_capture_appends_attn_params(fake_fia):
    layer = _make_layer(num_heads=4, num_kv_heads=4, head_size=72)
    _encoder_graph_state.capturing = True
    _encoder_graph_state.token_budget = 2048
    _encoder_graph_state.capture_layer_cursor = 0

    bsz, q_len = 2, 4
    query = torch.randn(bsz, q_len, layer.num_heads, 72, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32)

    layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

    assert len(_encoder_graph_params.attn_params[2048]) == 1
    assert len(_encoder_graph_params.handles[2048]) == 1
    assert _encoder_graph_params.attn_params[2048][0][6] is False
    assert fake_fia["mode"] == "out"
    assert fake_fia["softmax_lse"].numel() == 1
    _mm_encoder_attention.torch.npu.graph_task_group_begin.assert_called_once()
    _mm_encoder_attention.torch.npu.graph_task_group_end.assert_called_once()


def test_capture_uses_sequence_lengths_host(fake_fia):
    layer = _make_layer(num_heads=4, num_kv_heads=4, head_size=72)
    _encoder_graph_state.capturing = True
    _encoder_graph_state.token_budget = 2048

    seq_lens = [3, 5]
    sequence_lengths = torch.tensor(seq_lens, dtype=torch.int64)
    query = torch.randn(2, 5, layer.num_heads, 72, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    layer.forward_oot(query, key, value, sequence_lengths=sequence_lengths)

    assert _encoder_graph_params.attn_params[2048][0][6] is True
    assert fake_fia["actual_seq_lengths"] == [3, 8]
