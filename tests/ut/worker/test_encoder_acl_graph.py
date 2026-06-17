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

"""Unit tests for ViT encoder ACL graph helpers and manager hooks."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Minimal torch_npu / vllm stubs for import without NPU hardware.
if "torch_npu" not in sys.modules:
    _torch_npu = types.ModuleType("torch_npu")
    _torch_npu.npu = MagicMock()
    sys.modules["torch_npu"] = _torch_npu
else:
    _torch_npu = sys.modules["torch_npu"]

if not hasattr(torch, "npu"):
    torch.npu = _torch_npu.npu

_vllm_logger = types.ModuleType("vllm.logger")
_vllm_logger.init_logger = lambda name: MagicMock()
sys.modules["vllm.logger"] = _vllm_logger

_vllm_encoder_cudagraph = types.ModuleType("vllm.v1.worker.encoder_cudagraph")


class _BudgetGraphMetadata:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _EncoderCudaGraphManager:
    def __init__(self, *args, **kwargs):
        self.token_budgets = [2048, 4096]
        self.max_batch_size = 4
        self.max_frames_per_batch = 0
        self.budget_graphs = {}
        self.model = args[3] if len(args) > 3 else kwargs.get("model") or MagicMock()
        self.config = self.model.get_encoder_cudagraph_config()
        self.device = args[1] if len(args) > 1 else kwargs.get("device", "npu")
        self.dtype = args[2] if len(args) > 2 else kwargs.get("dtype", "bfloat16")

    def capture(self):
        return None


_vllm_encoder_cudagraph.BudgetGraphMetadata = _BudgetGraphMetadata
_vllm_encoder_cudagraph.EncoderCudaGraphManager = _EncoderCudaGraphManager
sys.modules["vllm.v1.worker.encoder_cudagraph"] = _vllm_encoder_cudagraph

_vllm_ascend_utils = types.ModuleType("vllm_ascend.utils")
_vllm_ascend_utils.weak_ref_tensors = lambda x: x
sys.modules["vllm_ascend.utils"] = _vllm_ascend_utils
sys.modules.setdefault("vllm_ascend", types.ModuleType("vllm_ascend"))

_ENCODER_ACL_GRAPH_PATH = (
    Path(__file__).resolve().parents[3] / "vllm_ascend" / "worker" / "encoder_acl_graph.py"
)
_spec = importlib.util.spec_from_file_location(
    "vllm_ascend.worker.encoder_acl_graph",
    _ENCODER_ACL_GRAPH_PATH,
)
_encoder_acl_graph = importlib.util.module_from_spec(_spec)
sys.modules["vllm_ascend.worker.encoder_acl_graph"] = _encoder_acl_graph
_spec.loader.exec_module(_encoder_acl_graph)

align_fia = _encoder_acl_graph._align_fia_endpoints_to_num_tokens
resolve_lengths = _encoder_acl_graph._resolve_vit_actual_lengths
update_encoder_full_graph_params = _encoder_acl_graph.update_encoder_full_graph_params
EncoderAclGraphManager = _encoder_acl_graph.EncoderAclGraphManager
reset_encoder_graph_params_for_testing = _encoder_acl_graph.reset_encoder_graph_params_for_testing
reset_encoder_graph_runtime_state = _encoder_acl_graph.reset_encoder_graph_runtime_state
set_encoder_graph_params = _encoder_acl_graph.set_encoder_graph_params
get_encoder_graph_params = _encoder_acl_graph.get_encoder_graph_params


@pytest.fixture(autouse=True)
def _reset_state():
    reset_encoder_graph_params_for_testing()
    reset_encoder_graph_runtime_state()
    yield
    reset_encoder_graph_params_for_testing()
    reset_encoder_graph_runtime_state()


def test_align_fia_endpoints_to_num_tokens():
    assert align_fia([4, 8, 0], 8) == [4, 8]
    assert align_fia([4, 16], 8) == [4, 8]
    assert align_fia([], 8) == [8]


def test_fullatt_block_indexes_routing():
    runtime = _encoder_acl_graph.get_encoder_graph_runtime_state()
    runtime.host_cu_seqlens_ends = [4, 8]
    runtime.host_cu_window_seqlens_ends = [2, 6]
    runtime.host_sequence_lengths = [3, 7]

    full_q, _ = resolve_lengths(
        num_query_tokens=8,
        uses_sequence_lengths_host=False,
        vit_layer_idx=0,
        fullatt_block_indexes=frozenset({0, 2}),
    )
    window_q, _ = resolve_lengths(
        num_query_tokens=8,
        uses_sequence_lengths_host=False,
        vit_layer_idx=1,
        fullatt_block_indexes=frozenset({0, 2}),
    )
    assert full_q == [4, 8]
    assert window_q == [2, 6, 8]

    seq_q, _ = resolve_lengths(
        num_query_tokens=8,
        uses_sequence_lengths_host=True,
        vit_layer_idx=0,
        fullatt_block_indexes=frozenset({0, 2}),
    )
    assert seq_q == [3, 7, 8]


def test_update_encoder_full_graph_params_routes_host_lengths():
    set_encoder_graph_params([2048])
    params = get_encoder_graph_params()
    query = MagicMock()
    query.shape = [8, 4, 72]
    packed = (
        query,
        MagicMock(),
        MagicMock(),
        None,
        None,
        128,
        False,
        0,
        4,
        4,
        0.125,
        MagicMock(),
        MagicMock(),
    )
    params.handles[2048] = [1]
    params.events[2048] = [MagicMock()]
    params.attn_params[2048] = [packed]
    params.workspaces[2048] = MagicMock()

    runtime = _encoder_acl_graph.get_encoder_graph_runtime_state()
    runtime.host_cu_seqlens_ends = [4, 8]
    runtime.host_cu_window_seqlens_ends = [2, 6]

    captured = {}

    def fake_out(**kwargs):
        captured["actual_seq_lengths"] = kwargs["actual_seq_lengths"]

    with patch.object(_encoder_acl_graph.torch.npu, "stream"), patch.object(
        _encoder_acl_graph.torch.npu, "graph_task_update_begin"
    ), patch.object(_encoder_acl_graph.torch.npu, "graph_task_update_end"), patch.object(
        _encoder_acl_graph.torch_npu,
        "npu_fused_infer_attention_score",
        types.SimpleNamespace(out=fake_out),
    ):
        update_encoder_full_graph_params(
            MagicMock(),
            2048,
            fullatt_block_indexes=frozenset({0}),
        )

    assert captured["actual_seq_lengths"] == [4, 8]


def _make_manager():
    model = MagicMock()
    model.get_encoder_cudagraph_config.return_value = MagicMock(
        input_key_by_modality={"image": "pixel_values"},
        buffer_keys=["cu_seqlens"],
    )
    return EncoderAclGraphManager(MagicMock(), "npu", "bfloat16", model), model


def test_manager_update_stream_defaults_none():
    mgr, _ = _make_manager()
    assert mgr.update_stream is None


def test_manager_capture_registers_graph_params():
    mgr, _ = _make_manager()
    mgr.token_budgets = [2048]

    with patch.object(_EncoderCudaGraphManager, "capture", return_value=None):
        mgr.capture()

    params = get_encoder_graph_params()
    assert params is not None
    assert 2048 in params.events


def test_manager_uses_npu_graph_in_capture_budget_graph():
    mgr, model = _make_manager()
    mgr.max_batch_size = 2
    mgr.max_frames_per_batch = 0
    model.prepare_encoder_cudagraph_capture_inputs.return_value = MagicMock(
        mm_kwargs={"pixel_values": torch.zeros(2, 3, 224, 224)},
        buffers={"cu_seqlens": torch.zeros(3, dtype=torch.int32)},
    )
    model.encoder_cudagraph_forward.return_value = torch.zeros(2, 64)

    fake_graph = MagicMock()
    with patch.object(_encoder_acl_graph.torch.npu, "NPUGraph", return_value=fake_graph), patch.object(
        _encoder_acl_graph.torch.npu, "graph"
    ), patch.object(_encoder_acl_graph, "encoder_graph_capture_scope"):
        mgr._capture_budget_graph(2048)

    assert 2048 in mgr.budget_graphs
    assert mgr.budget_graphs[2048].graph is fake_graph
