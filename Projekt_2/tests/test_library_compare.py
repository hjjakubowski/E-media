from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util
import unittest

from Projekt_2.src.library_compare import compare_with_library_rsa
from Projekt_2.src.png_format import Ihdr, make_chunk, make_ihdr, write_png
from Projekt_2.src.png_pixels import encode_pixels
from Projekt_2.src.rsa_core import generate_keypair


HAS_PYCRYPTODOME = importlib.util.find_spec("Crypto") is not None


@unittest.skipUnless(HAS_PYCRYPTODOME, "pycryptodome is not installed")
class LibraryCompareTest(unittest.TestCase):
    def test_library_comparison_uses_same_key_and_roundtrips(self) -> None:
        _, private_key = generate_keypair(512)

        with TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source.png"
            pixels = bytes((index * 37) % 256 for index in range(64))
            write_test_png(source, 64, 1, 0, pixels)

            result = compare_with_library_rsa(source, private_key)

            self.assertGreater(result.sample_length, 0)
            self.assertTrue(result.own_roundtrip_ok)
            self.assertTrue(result.library_roundtrip_ok)
            self.assertFalse(result.ciphertexts_equal)
            self.assertTrue(result.library_is_probabilistic)


def write_test_png(path: Path, width: int, height: int, color_type: int, pixels: bytes) -> None:
    ihdr = Ihdr(
        width=width,
        height=height,
        bit_depth=8,
        color_type=color_type,
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
