import cv2 as cv
import numpy as np
from typing import List

from .fft_models import ChannelMetrics, FftAnalysisResult, FftMetricsSummary
from .fft_contracts import ImageLoader, FftAnalyzer

def to_rgb_or_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    elif image.ndim == 3 and image.shape[2] == 4:
        return cv.cvtColor(image, cv.COLOR_BGRA2RGBA)
    elif image.ndim == 3 and image.shape[2] == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image format: {image.shape}")

class Cv2ImageLoader:
    def load(self, image_path: str) -> np.ndarray:
        image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        return image

class NumpyRoundTripFftAnalyzer:
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    @staticmethod
    def _channel_analysis(channel: np.ndarray, channel_index: int) -> tuple:
        channel_org = channel.astype(np.float64)

        freq = np.fft.fft2(channel_org)
        freq_shift = np.fft.fftshift(freq)
        spectrum_log = np.log10(np.abs(freq_shift))

        channel_rec = np.fft.ifft2(freq).real
        err = channel_rec - channel_org

        mse = float(np.mean(err**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(abs(err)))
        max_abs = float(np.max(abs(err)))
        psnr = float('inf') if mse == 0.0 else float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))

        metrics = ChannelMetrics(
            channel_index=channel_index,
            mse=mse,
            rmse=rmse,
            mae=mae,
            max_abs_error=max_abs,
            psnr_db=psnr,
        )
        return spectrum_log, channel_rec, err, metrics

    def _build_summary(self, channel_metrics: List[ChannelMetrics]) -> FftMetricsSummary:
        mse_mean = float(np.mean([m.mse for m in channel_metrics]))
        rmse_mean = float(np.mean([m.rmse for m in channel_metrics]))
        mae_mean = float(np.mean([m.mae for m in channel_metrics]))
        max_abs_error = float(np.max([m.max_abs_error for m in channel_metrics]))
        psnr_values = [m.psnr_db for m in channel_metrics]
        psnr_mean_db = float(np.mean(psnr_values)) if all(np.isfinite(psnr_values)) else float("inf")
        passed = max_abs_error <= self.tolerance

        return FftMetricsSummary(
            mse_mean=mse_mean,
            rmse_mean=rmse_mean,
            mae_mean=mae_mean,
            max_abs_error=max_abs_error,
            psnr_mean_db=psnr_mean_db,
            passed=passed,
            tolerance=self.tolerance,
        )

    def analyze(self, image: np.ndarray) -> FftAnalysisResult:
        image = to_rgb_or_gray(image)

        if image.ndim == 2:
            spectrum, rec, err, metrics  = self._channel_analysis(image, 0)
            reconstructed_uint8 = np.clip(np.rint(rec), 0, 255).astype(np.uint8)
            summary = self._build_summary([metrics])

            return FftAnalysisResult(
                mode='gray',
                original=image,
                spectrum_log_display=spectrum,
                reconstructed_uint8=reconstructed_uint8,
                error_map=np.abs(err),
                channel_metrics=[metrics],
                summary=summary,
            )

        spectrums = []
        reconstructed_channels = []
        errors = []
        per_channel = []

        for i in range(3):
            spectrum, rec, err, metrics  = self._channel_analysis(image[:, :, i], i)
            spectrums.append(spectrum)
            reconstructed_uint8_ch = np.clip(np.rint(rec), 0, 255).astype(np.uint8)
            reconstructed_channels.append(reconstructed_uint8_ch)
            errors.append(np.abs(err))
            per_channel.append(metrics)

        spectrum_log_display = np.stack(spectrums, axis=2)
        max_val = float(np.max(spectrum_log_display))

        if max_val > 0:
            spectrum_log_display = spectrum_log_display / max_val

        reconstructed_uint8 = np.stack(reconstructed_channels, axis=2)
        error_map = np.stack(errors, axis=2)
        summary = self._build_summary(per_channel)

        return FftAnalysisResult(
            mode='rgb',
            original=image,
            spectrum_log_display=spectrum_log_display,
            reconstructed_uint8=reconstructed_uint8,
            error_map=error_map,
            channel_metrics=per_channel,
            summary=summary,
        )

class FftVerificationService:
    def __init__(self, loader: ImageLoader, analyzer: FftAnalyzer):
        self.loader = loader
        self.analyzer = analyzer

    def analyze_image(self, image_path: str) -> FftAnalysisResult:
        image = self.loader.load(image_path)
        return self.analyzer.analyze(image)