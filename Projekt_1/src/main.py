from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.fourier_analysis import (
    run_fft_self_test,
    save_fourier_spectrum,
    save_image_preview,
)
from src.png_analysis import (
    anonymize_png,
    print_ancillary_chunks,
    print_critical_chunks,
    read_png_chunks,
)

SOURCE_PNG = PROJECT_DIR / "data" / "Black_metadata.png"
OUTPUT_DIR = PROJECT_DIR / "output"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    chunks = read_png_chunks(SOURCE_PNG)

    print(f"Input file: {SOURCE_PNG}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    print_critical_chunks(chunks)
    print()

    print_ancillary_chunks(chunks)
    print()

    preview_path = save_image_preview(SOURCE_PNG, OUTPUT_DIR)
    print(f"Image preview saved to: {preview_path}")

    magnitude_path, phase_path = save_fourier_spectrum(SOURCE_PNG, OUTPUT_DIR)
    print(f"FFT magnitude image saved to: {magnitude_path}")
    print(f"FFT phase image saved to: {phase_path}")

    print()
    # run_fft_self_test()

    print()
    anonymized_path = anonymize_png(SOURCE_PNG, chunks, OUTPUT_DIR)
    print(f"Anonymized PNG saved to: {anonymized_path}")


if __name__ == "__main__":
    main()
