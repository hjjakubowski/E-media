import os
import sys
import unittest
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
        np.testing.assert_array_equal(result.reconstructed_uint8, image)
        self.assertTrue(result.summary.passed)
        self.assertLessEqual(result.summary.max_abs_error, result.summary.tolerance)
        self.assertEqual(len(result.channel_metrics), 1)
        self.assertEqual(result.spectrum_log_display.shape, image.shape)

    def test_bgr_input_fft_round_trip_reconstructs_rgb_channels(self) -> None:
        red = np.arange(16, dtype=np.uint8).reshape(4, 4) * 7
        green = np.flipud(red)
        blue = np.rot90(red)
        bgr_image = np.stack((blue, green, red), axis=2)
        expected_rgb_image = np.stack((red, green, blue), axis=2)

        result = self.analyzer.analyze(bgr_image)

        self.assertEqual(result.mode, "rgb")
        np.testing.assert_array_equal(result.original, expected_rgb_image)
        np.testing.assert_array_equal(result.reconstructed_uint8, expected_rgb_image)
        self.assertTrue(result.summary.passed)
        self.assertLessEqual(result.summary.max_abs_error, result.summary.tolerance)
        self.assertEqual(len(result.channel_metrics), 3)
        self.assertEqual(result.spectrum_log_display.shape, expected_rgb_image.shape)
        self.assertEqual(result.error_map.shape, expected_rgb_image.shape)

    def test_fft_metrics_report_round_trip_error_within_tolerance(self) -> None:
        image = np.arange(25, dtype=np.uint8).reshape(5, 5)

        result = self.analyzer.analyze(image)

        self.assertTrue(result.summary.passed)
        self.assertLessEqual(result.summary.max_abs_error, result.summary.tolerance)
        self.assertLessEqual(result.summary.mse_mean, result.summary.tolerance)
        self.assertLessEqual(result.summary.rmse_mean, result.summary.tolerance)
        self.assertLessEqual(result.summary.mae_mean, result.summary.tolerance)


if __name__ == "__main__":
    unittest.main()
