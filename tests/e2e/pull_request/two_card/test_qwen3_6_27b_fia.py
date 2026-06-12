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

"""Stage 1 E2E for Qwen/Qwen3.6-27B after ViT FIA op replacement."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from vllm.assets.image import ImageAsset

from tests.e2e.conftest import VllmRunner

GOLDEN_PATH = Path(__file__).parent / "golden" / "qwen3_6_27b_image_tp2.json"
MODEL = "Qwen/Qwen3.6-27B"


def _qwen3_vl_prompt(question: str) -> str:
    placeholder = "<|image_pad|>"
    return (
        "<|im_start|>system\nYou are a helpful assistant.\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
        f"{question}\n"
        f"<|im_start|>assistant\n"
    )


def _load_golden():
    assert GOLDEN_PATH.exists(), (
        f"Golden fixture missing: {GOLDEN_PATH}. "
        "Generate it on NPU hardware per docs/plans/aclgraph-vit-01-stage1-fia.md §6.2.4."
    )
    with GOLDEN_PATH.open(encoding="utf-8") as fp:
        golden = json.load(fp)
    if golden and golden[0].get("_placeholder"):
        pytest.skip(
            "Golden fixture is a placeholder; record real token_ids on NPU "
            "(see docs/plans/aclgraph-vit-01-stage1-fia.md §6.2.4)."
        )
    return golden


@patch.dict(os.environ, {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
def test_qwen3_6_27b_image_fia_accuracy_tp2():
    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
    questions = [
        "What is the content of this image?",
        "Describe the dominant color of this image in one word.",
        "Is there any text in this image?",
    ]
    prompts = [_qwen3_vl_prompt(q) for q in questions]
    images = [image] * len(prompts)

    with VllmRunner(
        MODEL,
        tensor_parallel_size=2,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            prompts=prompts,
            images=images,
            max_tokens=64,
        )

    assert len(outputs) == len(prompts)
    for _token_ids, text in outputs:
        assert text, "Generated output should not be empty."

    golden = _load_golden()
    assert len(golden) == len(outputs)
    for i, ((tok_ids, _), exp) in enumerate(zip(outputs, golden)):
        exp_ids = exp["token_ids"]
        compare_len = min(len(tok_ids), len(exp_ids))
        matched = sum(1 for j in range(compare_len) if tok_ids[j] == exp_ids[j])
        ratio = matched / max(compare_len, 1)
        assert ratio >= 0.99, (
            f"prompt[{i}] top-1 agreement {ratio:.4f} < 0.99 (matched {matched}/{compare_len})"
        )


@patch.dict(os.environ, {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
def test_qwen3_6_27b_video_fia_smoke_tp2():
    fake_frames = [
        Image.new("RGB", (224, 224), color=(r, g, b))
        for (r, g, b) in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (200, 200, 200)]
    ]
    question = "Describe what happens in this video."
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.\n"
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
        f"{question}\n"
        "<|im_start|>assistant\n"
    )
    whitelist = ["video", "color", "frame", "red", "green", "blue", "white", "gray", "image"]

    with VllmRunner(
        MODEL,
        tensor_parallel_size=2,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"video": 1},
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            prompts=[prompt],
            videos=[fake_frames],
            max_tokens=64,
        )

    assert len(outputs) == 1
    _, text = outputs[0]
    assert text, "Generated output should not be empty."
    text_lc = text.lower()
    assert any(k in text_lc for k in whitelist), f"video smoke output lacks whitelist token: {text!r}"
