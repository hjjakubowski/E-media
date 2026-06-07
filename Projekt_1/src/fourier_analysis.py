from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ModuleNotFoundError as error:
    missing_name = error.name
    raise SystemExit(
        f"Missing dependency: {missing_name}. "
        "Install project dependencies with: python3 -m pip install -r requirements.txt"
    ) from error


SHOW_IMAGES = False


def save_image_preview(source_path: Path, output_dir: Path) -> Path:
    preview_path = output_dir / f"{source_path.stem}_preview.png"

    with Image.open(source_path) as image:
        image.save(preview_path)

        if SHOW_IMAGES:
            image.show(title="Original PNG image")

    return preview_path


def save_fourier_spectrum(source_path: Path, output_dir: Path) -> tuple[Path, Path]:
    magnitude_path = output_dir / f"{source_path.stem}_fft_magnitude.png"
    phase_path = output_dir / f"{source_path.stem}_fft_phase.png"

    grayscale_pixels = read_grayscale_pixels(source_path)

    spectrum = np.fft.fft2(grayscale_pixels)
    centered_spectrum = np.fft.fftshift(spectrum)

    magnitude = np.abs(centered_spectrum)
    phase = np.angle(centered_spectrum)

    visible_magnitude = np.log1p(magnitude)

    save_grayscale_image(visible_magnitude, magnitude_path)
    save_grayscale_image(phase, phase_path)

    if SHOW_IMAGES:
        Image.open(magnitude_path).show(title="FFT magnitude")
        Image.open(phase_path).show(title="FFT phase")

    return magnitude_path, phase_path


def read_grayscale_pixels(source_path: Path) -> np.ndarray:
    with Image.open(source_path) as image:
        rgb_pixels = np.asarray(image.convert("RGB"), dtype=np.float32)

    red = rgb_pixels[:, :, 0]
    green = rgb_pixels[:, :, 1]
    blue = rgb_pixels[:, :, 2]

    return 0.299 * red + 0.587 * green + 0.114 * blue


def save_grayscale_image(values: np.ndarray, path: Path) -> None:
    lowest = float(values.min())
    highest = float(values.max())

    if highest == lowest:
        image_bytes = np.zeros(values.shape, dtype=np.uint8)
    else:
        normalized = (values - lowest) / (highest - lowest)
        image_bytes = (normalized * 255).astype(np.uint8)

    Image.fromarray(image_bytes).save(path)


def run_fft_self_test() -> None:
    constant_image = np.ones((8, 8), dtype=np.float32)
    constant_spectrum = np.fft.fft2(constant_image)

    expected_dc_value = constant_image.size
    real_dc_value = abs(constant_spectrum[0, 0])

    if not np.isclose(real_dc_value, expected_dc_value):
        raise AssertionError("FFT self-test failed for constant image DC value.")

    spectrum_without_dc = constant_spectrum.copy()
    spectrum_without_dc[0, 0] = 0

    if not np.allclose(spectrum_without_dc, 0):
        raise AssertionError("FFT self-test failed for constant image frequencies.")

    impulse_image = np.zeros((8, 8), dtype=np.float32)
    impulse_image[0, 0] = 1
    impulse_spectrum = np.fft.fft2(impulse_image)

    if not np.allclose(np.abs(impulse_spectrum), 1):
        raise AssertionError("FFT self-test failed for impulse image.")

    print("FFT self-test passed.")
