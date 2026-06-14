from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from Projekt_2.src.analysis_report import generate_visibility_report
from Projekt_2.src.compression_compare import compare_compression_methods
from Projekt_2.src.png_format import Ihdr, idat_data, make_chunk, make_ihdr, read_png, write_png
from Projekt_2.src.png_pixels import encode_pixels
from Projekt_2.src.rsa_core import generate_keypair


class ReportsTest(unittest.TestCase):
    def test_visibility_report_generates_images_and_text(self) -> None:
        public_key, private_key = generate_keypair(256)

        with TemporaryDirectory() as temporary_directory:
            temp_dir = Path(temporary_directory)
            source = temp_dir / "source.png"
            write_test_png(source, 64, 1, bytes([7] * 64))

            result = generate_visibility_report(
                source,
                temp_dir,
                public_key,
                private_key,
            )

            self.assertTrue(result.ecb_encrypted_path.is_file())
            self.assertTrue(result.chain_encrypted_path.is_file())
            self.assertTrue(result.ecb_decrypted_path.is_file())
            self.assertTrue(result.chain_decrypted_path.is_file())
            self.assertIn("ECB ciphertext blocks", result.report_path.read_text())

    def test_compression_compare_generates_report_and_roundtrips_idat(self) -> None:
        public_key, private_key = generate_keypair(256)

        with TemporaryDirectory() as temporary_directory:
            temp_dir = Path(temporary_directory)
            source = temp_dir / "source.png"
            write_test_png(source, 64, 1, bytes(range(64)))

            result = compare_compression_methods(
                source,
                temp_dir,
                public_key,
                private_key,
                "chain",
            )

            self.assertTrue(result.pixel_encrypted_path.is_file())
            self.assertTrue(result.compressed_encrypted_path.is_file())
            self.assertTrue(result.report_path.is_file())
            self.assertIn("not equivalent", result.report_path.read_text())
            self.assertEqual(
                idat_data(read_png(source)),
                idat_data(read_png(result.compressed_decrypted_path)),
            )


def write_test_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    ihdr = Ihdr(
        width=width,
        height=height,
        bit_depth=8,
        color_type=0,
        compression_method=0,
        filter_method=0,
        interlace_method=0,
    )
    chunks = [
        make_ihdr(ihdr),
        make_chunk("IDAT", encode_pixels(ihdr, pixels)),
        make_chunk("IEND", b""),
    ]
    write_png(path, chunks)


if __name__ == "__main__":
    unittest.main()
