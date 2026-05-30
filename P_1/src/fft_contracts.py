import numpy as np
from typing import Protocol

from .fft_models import FftAnalysisResult


class ImageLoader(Protocol):
    def load(self, image_path: str) -> np.ndarray:
        ...

class FftAnalyzer(Protocol):
    def analyze(self, image: np.ndarray) -> FftAnalysisResult:
        ...

class FftPresenter(Protocol):
    def show(self, result: FftAnalysisResult) -> None:
        ...
