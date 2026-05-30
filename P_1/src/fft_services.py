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


def image_mode(image: np.ndarray) -> str:
    if image.ndim == 2:
        return "gray"
    if image.ndim == 3 and image.shape[2] == 3:
        return "rgb"
    if image.ndim == 3 and image.shape[2] == 4:
        return "rgba"
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
    def _data_range(channel: np.ndarray) -> float:
        if np.issubdtype(channel.dtype, np.integer):
            return float(np.iinfo(channel.dtype).max)
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            return 1.0
        min_value = float(np.min(finite))
        max_value = float(np.max(finite))
        return max(max_value - min_value, 1.0)

    @staticmethod
    def _reconstruct_to_source_dtype(reconstructed: np.ndarray, dtype: np.dtype) -> np.ndarray:
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return np.clip(np.rint(reconstructed), info.min, info.max).astype(dtype)
        return reconstructed.astype(dtype, copy=False)

    @staticmethod
    def _normalize_frequency_stack(values: np.ndarray) -> np.ndarray:
        max_value = float(np.max(values))
        if max_value > 0:
            return values / max_value
        return values

    @staticmethod
    def _channel_analysis(channel: np.ndarray, channel_index: int) -> tuple:
        channel_org = channel.astype(np.float64)

        freq = np.fft.fft2(channel_org)
        freq_shift = np.fft.fftshift(freq)
        spectrum_log = np.log10(np.abs(freq_shift) + 1.0)
        phase = np.angle(freq_shift)

        channel_rec = np.fft.ifft2(freq).real
        err = channel_rec - channel_org

        mse = float(np.mean(err**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(err)))
        max_abs = float(np.max(np.abs(err)))
        data_range = NumpyRoundTripFftAnalyzer._data_range(channel)
        psnr = float("inf") if mse == 0.0 else float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))

        metrics = ChannelMetrics(
            channel_index=channel_index,
            mse=mse,
            rmse=rmse,
            mae=mae,
            max_abs_error=max_abs,
            psnr_db=psnr,
        )
        return spectrum_log, phase, channel_rec, err, metrics

    def _build_summary(self, channel_metrics: List[ChannelMetrics]) -> FftMetricsSummary:
        mse_mean = float(np.mean([m.mse for m in channel_metrics]))
        rmse_mean = float(np.mean([m.rmse for m in channel_metrics]))
        mae_mean = float(np.mean([m.mae for m in channel_metrics]))
        max_abs_error = float(np.max([m.max_abs_error for m in channel_metrics]))
        psnr_values = [m.psnr_db for m in channel_metrics]
        finite_psnr_values = [value for value in psnr_values if np.isfinite(value)]
        psnr_mean_db = float(np.mean(finite_psnr_values)) if finite_psnr_values else float("inf")
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
        mode = image_mode(image)

        if image.ndim == 2:
            spectrum, phase, rec, err, metrics = self._channel_analysis(image, 0)
            reconstructed = self._reconstruct_to_source_dtype(rec, image.dtype)
            summary = self._build_summary([metrics])

            return FftAnalysisResult(
                mode=mode,
                original=image,
                spectrum_log_display=spectrum,
                phase_display=phase,
                reconstructed=reconstructed,
                error_map=np.abs(err),
                channel_metrics=[metrics],
                summary=summary,
            )

        spectrums = []
        phases = []
        reconstructed_channels = []
        errors = []
        per_channel = []

        for i in range(image.shape[2]):
            spectrum, phase, rec, err, metrics = self._channel_analysis(image[:, :, i], i)
            spectrums.append(spectrum)
            phases.append(phase)
            reconstructed_channels.append(self._reconstruct_to_source_dtype(rec, image.dtype))
            errors.append(np.abs(err))
            per_channel.append(metrics)

        spectrum_log_display = np.stack(spectrums, axis=2)
        spectrum_log_display = self._normalize_frequency_stack(spectrum_log_display)
        phase_display = np.stack(phases, axis=2)

        reconstructed = np.stack(reconstructed_channels, axis=2)
        error_map = np.stack(errors, axis=2)
        summary = self._build_summary(per_channel)

        return FftAnalysisResult(
            mode=mode,
            original=image,
            spectrum_log_display=spectrum_log_display,
            phase_display=phase_display,
            reconstructed=reconstructed,
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
