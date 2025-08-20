import copy
from unittest.mock import Mock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.w4a8_dynamic import (
    AscendW4A8DynamicFusedMoEMethod, AscendW4A8DynamicLinearMethod, apply_mlp,
    apply_mlp_decode)


class TestAscendW4A8DynamicLinearMethod(TestBase):

    def setUp(self):
        self.method = AscendW4A8DynamicLinearMethod()
        self.method.group_size = 8

    def test_get_weight(self):
        weight = self.method.get_weight(8, 32, torch.bfloat16)
        self.assertEqual(weight["weight"].dtype, torch.int8)
        self.assertEqual(weight["weight"].shape, (32, 8))

    def test_get_pergroup_param(self):
        params = self.method.get_pergroup_param(8, 32, torch.bfloat16)
        self.assertEqual(params["weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(params["weight_scale"].shape, (32, 1))
        self.assertEqual(params["weight_offset"].dtype, torch.bfloat16)
        self.assertEqual(params["weight_offset"].shape, (32, 1))
        self.assertEqual(params["weight_scale_second"].dtype, torch.bfloat16)
        self.assertEqual(params["weight_scale_second"].shape, (32, 1))
        self.assertEqual(params["weight_offset_second"].dtype, torch.bfloat16)
        self.assertEqual(params["weight_offset_second"].shape, (32, 1))


class TestAscendW4A8DynamicFusedMoEMethod(TestBase):
    experts = 8
    input_size = 16
    output_size = 56
    group_size = 2

    @patch('vllm_ascend.quantization.w4a8_dynamic.get_current_vllm_config')
    @patch('vllm_ascend.quantization.w4a8_dynamic.get_ep_group')
    @patch("vllm_ascend.ascend_config.get_ascend_config")
    @patch('vllm_ascend.quantization.w4a8_dynamic.get_mc2_group')
    @patch('torch.distributed.get_rank', return_value=0)
    def setUp(self, mock_get_rank, mock_get_mc2_group, mock_get_ascend_config,
              mock_get_ep_group, get_current_vllm_config):
        mock_ascend_config = Mock()
        mock_ascend_config.torchair_graph_config = Mock(enabled=False)
        mock_get_ascend_config.return_value = mock_ascend_config
        mock_vllm_config = Mock()
        mock_vllm_config.quant_config = Mock(quant_description={
            "group_size": self.group_size,
            "version": "0.0.0"
        })
        mock_vllm_config.parallel_config = Mock(enable_expert_parallel=True)
        get_current_vllm_config.return_value = mock_vllm_config
        self.quant_method = AscendW4A8DynamicFusedMoEMethod()

    def test_get_weight(self):
        # old quant version w4a8 weight
        param_dict = self.quant_method.get_weight(self.experts,
                                                  self.input_size,
                                                  self.output_size,
                                                  torch.bfloat16)
        self.assertEqual(param_dict["w13_weight"].dtype, torch.int8)
        self.assertEqual(param_dict["w13_weight"].shape,
                         (self.experts, 2 * self.input_size, self.output_size))
        # new quant version weight
        self.quant_method.new_quant_version = True
        param_dict = self.quant_method.get_weight(self.experts,
                                                  self.input_size,
                                                  self.output_size,
                                                  torch.bfloat16)
        self.assertEqual(param_dict["w13_weight"].dtype, torch.int8)
        self.assertEqual(param_dict["w13_weight"].shape,
                         (self.experts, self.input_size, self.output_size))

    def test_get_dynamic_quant_param(self):
        # old quant version weight
        param_dict = self.quant_method.get_dynamic_quant_param(
            self.experts, self.input_size, self.output_size, torch.bfloat16)
        self.assertEqual(param_dict["w13_weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(param_dict["w13_weight_scale"].shape,
                         (self.experts, 2 * self.input_size, 1))
        self.assertEqual(param_dict["w13_weight_scale_second"].dtype,
                         torch.bfloat16)
        self.assertEqual(param_dict["w13_weight_scale_second"].shape,
                         (self.experts, 2 * self.input_size,
                          self.output_size // self.group_size))
        self.assertEqual(param_dict["w2_weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(param_dict["w2_weight_scale"].shape,
                         (self.experts, self.output_size, 1))
        self.assertEqual(param_dict["w2_weight_scale_second"].dtype,
                         torch.bfloat16)
        self.assertEqual(param_dict["w2_weight_scale_second"].shape,
                         (self.experts, self.output_size,
                          self.input_size // self.group_size))
        # new quant version weight
        self.quant_method.new_quant_version = True
        param_dict = self.quant_method.get_dynamic_quant_param(
            self.experts, self.input_size, self.output_size, torch.bfloat16)
        self.assertEqual(param_dict["w2_scale_bias"].dtype, torch.float32)
        self.assertEqual(
            param_dict["w2_scale_bias"].shape,
            (self.experts, self.output_size, 16 // self.quant_method.tp_size))

    @patch('torch_npu.npu_format_cast_')
    @patch('torch_npu.npu_quantize')
    @patch('torch.Tensor.npu')
    def test_process_weights_after_loading(self, mock_npu_format_cast, mock_npu,
                                           mock_npu_quantize):
        # old quant version weight
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(torch.zeros(
            (self.experts, 2 * self.input_size, self.output_size),
            dtype=torch.int8),
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.zeros(
            (self.experts, self.output_size, self.input_size),
            dtype=torch.int8),
                                             requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(torch.ones(
            (self.experts, 2 * self.input_size, 1), dtype=torch.bfloat16),
                                                    requires_grad=False)
        layer.w13_weight_scale_second = torch.nn.Parameter(torch.ones(
            (self.experts, 2 * self.input_size,
             self.output_size // self.group_size),
            dtype=torch.bfloat16),
                                                           requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(torch.ones(
            (self.experts, self.output_size, 1), dtype=torch.bfloat16),
                                                   requires_grad=False)
        layer.w2_weight_scale_second = torch.nn.Parameter(torch.ones(
            (self.experts, self.output_size,
             self.input_size // self.group_size),
            dtype=torch.bfloat16),
                                                          requires_grad=False)
        new_layer = copy.deepcopy(layer)

        mock_npu.return_value = torch.Tensor()
        mock_npu_quantize.return_value = torch.Tensor()
        mock_npu_format_cast.return_value = torch.Tensor()
        self.quant_method.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "w13_scale_bias"))
        self.assertEqual(layer.w13_scale_bias.data.shape,
                         (self.experts, 2 * self.input_size))
        self.assertEqual(layer.w13_scale_bias.data.dtype, torch.float32)
        self.assertTrue(hasattr(layer, "w2_scale_bias"))
        self.assertEqual(layer.w2_scale_bias.data.shape,
                         (self.experts, self.output_size))
        self.assertEqual(layer.w2_scale_bias.data.dtype, torch.float32)
        # new quant version weight
        self.quant_method.new_quant_version = True
        new_layer.w13_weight.data = torch.zeros(
            (self.experts, self.input_size, self.output_size),
            dtype=torch.int8)
        new_layer.w2_weight.data = torch.zeros(
            (self.experts, self.output_size // 2, self.input_size),
            dtype=torch.int8)
        w13_scale_bias = torch.zeros((self.experts, 2 * self.input_size, 1),
                                     dtype=torch.float32)
        new_layer.w13_scale_bias = torch.nn.Parameter(w13_scale_bias,
                                                      requires_grad=False)
        w2_scale_bias = torch.zeros(
            (self.experts, self.output_size, 16 // self.quant_method.tp_size),
            dtype=torch.float32)
        new_layer.w2_scale_bias = torch.nn.Parameter(w2_scale_bias,
                                                     requires_grad=False)
        self.quant_method.process_weights_after_loading(new_layer)
        self.assertEqual(new_layer.w13_scale_bias.data.shape,
                         (self.experts, 2 * self.input_size))
        self.assertEqual(new_layer.w2_scale_bias.data.shape,
                         (self.experts, self.output_size))

    @patch("torch_npu.npu_swiglu")
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_dynamic_quant")
    def test_apply_mlp(
        self,
        mock_dynamic_quant,
        mock_grouped_matmul,
        mock_swiglu,
    ):
        placeholder = torch.randn(128, 128, dtype=torch.bfloat16)
        placeholder_int8 = torch.randint(0, 100, (128, 128), dtype=torch.int8)
        placeholder_ones = torch.ones(128, dtype=torch.int32)

        mock_dynamic_quant.return_value = (
            placeholder_int8,
            placeholder_ones,
        )
        mock_grouped_matmul.return_value = [placeholder]
        mock_swiglu.return_value = placeholder

        result = apply_mlp(
            hidden_states=placeholder,
            w1=placeholder,
            w1_scale=placeholder,
            w2=placeholder,
            w2_scale=placeholder,
            group_list=placeholder,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, torch.bfloat16)

    @patch("torch_npu.npu_dequant_swiglu_quant")
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_dynamic_quant")
    def test_apply_mlp_decode(
        self,
        mock_dynamic_quant,
        mock_grouped_matmul,
        mock_dequant_swiglu_quant,
    ):
        placeholder = torch.randn(128, 128, dtype=torch.bfloat16)
        placeholder_int8 = torch.randint(0, 100, (128, 128), dtype=torch.int8)
        placeholder_ones = torch.ones(128, dtype=torch.int32)

        mock_dynamic_quant.return_value = (
            placeholder_int8,
            placeholder_ones,
        )
        mock_grouped_matmul.return_value = [placeholder]
        mock_dequant_swiglu_quant.return_value = (
            placeholder_int8,
            placeholder_ones,
        )

        result = apply_mlp_decode(
            hidden_states=placeholder,
            w1=placeholder,
            w1_scale=placeholder,
            w2=placeholder,
            w2_scale=placeholder,
            group_list=placeholder,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, torch.bfloat16)
