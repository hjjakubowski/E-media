import numpy as np
import matplotlib.pyplot as plt
from .fft_models import FftAnalysisResult


class MatplotlibFftPresenter:
    def show(self, result: FftAnalysisResult) -> None:
        plt.figure(figsize=(11, 8))

        plt.subplot(2, 2, 1)
        plt.title("Original image")
        plt.imshow(result.original, cmap="gray" if result.mode == "gray" else None)
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.title("FFT log-spectrum")
        plt.imshow(result.spectrum_log_display, cmap="gray" if result.mode == "gray" else None)
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.title("IFFT reconstructed")
        plt.imshow(result.reconstructed_uint8, cmap="gray" if result.mode == "gray" else None)
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.title("Absolute error map")
        err = result.error_map
        if err.ndim == 3:
            err = np.mean(err, axis=2)
        im = plt.imshow(err, cmap="hot")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.axis("off")

        s = result.summary
        plt.suptitle(
            f"passed={s.passed} | tol={s.tolerance:.1e} | "
            f"mse={s.mse_mean:.3e} | rmse={s.rmse_mean:.3e} | "
            f"max_abs={s.max_abs_error:.3e} | psnr={s.psnr_mean_db:.2f} dB"
        )
        plt.tight_layout()
        plt.show()