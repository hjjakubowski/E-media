import os
import sys
import unittest
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.fft_models import ChannelMetrics
from src.fft_services import NumpyRoundTripFftAnalyzer


class FftRoundTripTest(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = NumpyRoundTripFftAnalyzer(tolerance=1e-6)

    def test_grayscale_fft_round_trip_reconstructs_original_pixels(self) -> None:
        image = np.array(
            [
                [0, 32, 64, 96],
                [16, 48, 80, 112],
                [128, 160, 192, 224],
                [144, 176, 208, 255],
            ],
            dtype=np.uint8,
        )

        result = self.analyzer.analyze(image)

        self.assertEqual(result.mode, "gray")
        np.testing.assert_array_equal(result.reconstructed, image)
        self.assertTrue(result.summary.passed)
        self.assertLessEqual(result.summary.max_abs_error, result.summary.tolerance)
        self.assertEqual(len(result.channel_metrics), 1)
        self.assertEqual(result.spectrum_log_display.shape, image.shape)
        self.assertEqual(result.phase_display.shape, image.shape)

    def test_bgr_input_fft_round_trip_reconstructs_rgb_channels(self) -> None:
        red = np.arange(16, dtype=np.uint8).reshape(4, 4) * 7
        green = np.flipud(red)
        blue = np.rot90(red)
        bgr_image = np.stack((blue, green, red), axis=2)
        expected_rgb_image = np.stack((red, green, blue), axis=2)

        result = self.analyzer.analyze(bgr_image)

        self.assertEqual(result.mode, "rgb")
        np.testing.assert_array_equal(result.original, expected_rgb_image)
        np.testing.assert_array_equal(result.reconstructed, expected_rgb_image)
        self.assertTrue(result.summary.passed)
        self.assertLessEqual(result.summary.max_abs_error, result.summary.tolerance)
        self.assertEqual(len(result.channel_metrics), 3)
        self.assertEqual(result.spectrum_log_display.shape, expected_rgb_image.shape)
        self.assertEqual(result.phase_display.shape, expected_rgb_image.shape)
        self.assertEqual(result.error_map.shape, expected_rgb_image.shape)

    def test_bgra_input_preserves_alpha_channel(self) -> None:
        blue = np.arange(16, dtype=np.uint8).reshape(4, 4)
        green = np.flipud(blue)
        red = np.rot90(blue)
        alpha = np.full((4, 4), 127, dtype=np.uint8)
        bgra_image = np.stack((blue, green, red, alpha), axis=2)
        expected_rgba_image = np.stack((red, green, blue, alpha), axis=2)

        result = self.analyzer.analyze(bgra_image)

        self.assertEqual(result.mode, "rgba")
        np.testing.assert_array_equal(result.original, expected_rgba_image)
        np.testing.assert_array_equal(result.reconstructed, expected_rgba_image)
        self.assertEqual(len(result.channel_metrics), 4)
        self.assertEqual(result.spectrum_log_display.shape, expected_rgba_image.shape)
        self.assertEqual(result.phase_display.shape, expected_rgba_image.shape)
        self.assertEqual(result.error_map.shape, expected_rgba_image.shape)

    def test_uint16_input_reconstructs_without_clipping_to_uint8(self) -> None:
        image = np.array(
            [
                [0, 256],
                [1024, 65535],
            ],
            dtype=np.uint16,
        )

        result = self.analyzer.analyze(image)

        self.assertEqual(result.reconstructed.dtype, np.uint16)
        np.testing.assert_array_equal(result.reconstructed, image)
        self.assertTrue(result.summary.passed)

    def test_fft_metrics_report_round_trip_error_within_tolerance(self) -> None:
        image = np.arange(25, dtype=np.uint8).reshape(5, 5)

        result = self.analyzer.analyze(image)

        self.assertTrue(result.summary.passed)
        self.assertLessEqual(result.summary.max_abs_error, result.summary.tolerance)
        self.assertLessEqual(result.summary.mse_mean, result.summary.tolerance)
        self.assertLessEqual(result.summary.rmse_mean, result.summary.tolerance)
        self.assertLessEqual(result.summary.mae_mean, result.summary.tolerance)

    def test_psnr_summary_uses_finite_values_when_only_some_channels_are_perfect(self) -> None:
        metrics = [
            ChannelMetrics(0, 0.0, 0.0, 0.0, 0.0, float("inf")),
            ChannelMetrics(1, 1.0, 1.0, 1.0, 1.0, 48.0),
        ]

        summary = self.analyzer._build_summary(metrics)

        self.assertEqual(summary.psnr_mean_db, 48.0)


if __name__ == "__main__":
    unittest.main()
