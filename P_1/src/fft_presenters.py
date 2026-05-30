import numpy as np
import matplotlib.pyplot as plt
from .fft_models import FftAnalysisResult


class MatplotlibFftPresenter:
    def show(self, result: FftAnalysisResult) -> None:
        plt.figure(figsize=(13, 8))

        plt.subplot(2, 3, 1)
        plt.title("Original image")
        plt.imshow(result.original, cmap="gray" if result.mode == "gray" else None)
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.title("FFT log-spectrum")
        spectrum = result.spectrum_log_display
        spectrum_cmap = "gray" if result.mode == "gray" else None
        if spectrum.ndim == 3 and spectrum.shape[2] == 4:
            spectrum = np.mean(spectrum, axis=2)
            spectrum_cmap = "gray"
        plt.imshow(spectrum, cmap=spectrum_cmap)
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.title("FFT phase")
        phase = result.phase_display
        if phase.ndim == 3:
            phase = np.mean(phase, axis=2)
        im_phase = plt.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im_phase, fraction=0.046, pad=0.04)
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.title("IFFT reconstructed")
        plt.imshow(result.reconstructed, cmap="gray" if result.mode == "gray" else None)
        plt.axis("off")

        plt.subplot(2, 3, 5)
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
