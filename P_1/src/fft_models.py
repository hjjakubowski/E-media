from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass(frozen=True)
class ChannelMetrics:
    channel_index: int
    mse: float
    rmse: float
    mae: float
    max_abs_error: float
    psnr_db: float

@dataclass(frozen=True)
class FftMetricsSummary:
    mse_mean: float
    rmse_mean: float
    mae_mean: float
    max_abs_error: float
    psnr_mean_db: float
    passed: bool
    tolerance: float

@dataclass(frozen=True)
class FftAnalysisResult:
    mode: str
    original: np.ndarray
    spectrum_log_display: np.ndarray
    reconstructed_uint8: np.ndarray
    error_map: np.ndarray
    channel_metrics: List[ChannelMetrics]
    summary: FftMetricsSummary

