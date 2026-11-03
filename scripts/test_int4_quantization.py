"""Unit tests for the lossy INT4 weight encoding."""

import unittest

import numpy as np

from export_onnx import pack_int4, quantize_int4, unpack_int4


class Int4QuantizationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
