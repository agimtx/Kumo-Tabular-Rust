"""Unit tests for the lossy INT4 weight encoding."""

import unittest

import numpy as np

from export_onnx import (
    REGRESSION_HEAD_WEIGHT,
    int4_group_size_for_task,
    keep_int4_weight_fp32,
    pack_int4,
    quantize_int4,
    unpack_int4,
)
from check_quantized_weights import decode_int4


class Int4QuantizationTests(unittest.TestCase):
    def test_default_groups_only_change_regression(self) -> None:
        self.assertEqual(int4_group_size_for_task("regression", None), 16)
        self.assertEqual(int4_group_size_for_task("classification", None), 0)
        self.assertEqual(int4_group_size_for_task("regression", 0), 0)
        with self.assertRaises(ValueError):
            int4_group_size_for_task("regression", -1)

    def test_only_regression_quantile_head_keeps_fp32(self) -> None:
        assert keep_int4_weight_fp32(REGRESSION_HEAD_WEIGHT, "regression")
        assert not keep_int4_weight_fp32(REGRESSION_HEAD_WEIGHT, "classification")
        assert not keep_int4_weight_fp32("model.icl_block.layers.0.attn.out_lin.weight", "regression")

    def test_pack_unpack_round_trip_signed_nibbles_and_odd_length(self) -> None:
        source = np.array([-8, -7, -1, 0, 1, 7, -3], dtype=np.int8)
        np.testing.assert_array_equal(unpack_int4(pack_int4(source), source.size), source)

    def test_per_output_row_scales_reduce_error_with_outlier_channel(self) -> None:
        source = np.array([[1000.0, 0.2, -0.3], [0.1, 0.2, -0.3]], dtype=np.float32)
        quantized, scales = quantize_int4(source)
        restored = quantized.astype(np.float32) * scales[:, None]
        assert scales.shape == (2,)
        # Each output row keeps its own range instead of inheriting row 0's outlier.
        assert np.max(np.abs(restored[1] - source[1])) < 0.03

    def test_zero_row_has_finite_scale_and_zero_values(self) -> None:
        quantized, scales = quantize_int4(np.zeros((2, 8), dtype=np.float32))
        assert np.isfinite(scales).all()
        np.testing.assert_array_equal(quantized, 0)
        np.testing.assert_array_equal(scales, 1)

    def test_group_scales_follow_input_channels(self) -> None:
        source = np.array([[100., 1., .2, -.3], [-20., 2., -.1, .3]], dtype=np.float32)
        quantized, scales = quantize_int4(source, group_size=2)
        self.assertEqual(scales.shape, (2, 2))
        restored = quantized.reshape(2, 2, 2) * scales[..., None]
        self.assertLess(np.max(np.abs(restored[:, 1] - source.reshape(2, 2, 2)[:, 1])), .05)
        np.testing.assert_allclose(decode_int4(pack_int4(quantized), scales[..., None], source.shape),
                                   restored.reshape(source.shape))

    def test_nondivisible_group_uses_channel_scale(self) -> None:
        source = np.array([[1., 2., 3.]], dtype=np.float32)
        quantized, scales = quantize_int4(source, group_size=2)
        self.assertEqual(scales.shape, (1,))
        self.assertEqual(quantized.shape, source.shape)

    def test_grouping_preserves_odd_length_and_higher_rank(self) -> None:
        source = np.arange(30, dtype=np.float32).reshape(3, 2, 5) / 7
        quantized, scales = quantize_int4(source, group_size=1)
        self.assertEqual(scales.shape, (3, 2, 5))
        np.testing.assert_allclose(decode_int4(pack_int4(quantized), scales[..., None], source.shape), source)


if __name__ == "__main__":
    unittest.main()
