from typing import Protocol
from .fft_models import FftAnalysisResult
import numpy as np


class ImageLoader(Protocol):
    def load(self, image_path: str) -> np.ndarray:
        ...

class FftAnalyzer(Protocol):
    def analyze(self, image_path: str) -> FftAnalysisResult:
        ...

class FftPresenter(Protocol):
    def show(self, result: FftAnalysisResult) -> None:
        ...

