from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from Projekt_2.src.png_crypto import (
    decrypt_compressed_idat_png,
    decrypt_png,
    encrypt_compressed_idat_png,
    encrypt_png,
    read_metadata,
)
from Projekt_2.src.png_format import (
    Ihdr,
    idat_data,
    make_chunk,
    make_ihdr,
    parse_ihdr,
    read_png,
    write_png,
)
from Projekt_2.src.png_pixels import decode_pixels, encode_pixels, samples_per_pixel
from Projekt_2.src.rsa_core import generate_keypair


class PngCryptoTest(unittest.TestCase):
    def test_encrypt_decrypt_roundtrip_for_supported_color_types(self) -> None:
        public_key, private_key = generate_keypair(256)

        cases = [
            (0, 64, 1),
            (2, 22, 1),
            (6, 16, 1),
        ]

        with TemporaryDirectory() as temporary_directory:
            temp_dir = Path(temporary_directory)

            for color_type, width, height in cases:
                source = temp_dir / f"source_color_{color_type}.png"
                pixels = self.make_pixels(width, height, color_type)
                self.write_test_png(source, width, height, color_type, pixels)

                for mode in ("ecb", "chain"):
                    with self.subTest(color_type=color_type, mode=mode):
                        encrypted = temp_dir / f"encrypted_{color_type}_{mode}.png"
                        decrypted = temp_dir / f"decrypted_{color_type}_{mode}.png"

                        encrypt_png(source, encrypted, public_key, mode)
                        decrypt_png(encrypted, decrypted, private_key)

                        encrypted_chunks = read_png(encrypted)
                        encrypted_ihdr = parse_ihdr(encrypted_chunks)
                        metadata = read_metadata(encrypted_chunks)
                        decrypted_pixels = self.read_pixels(decrypted)

                        self.assertEqual(16, encrypted_ihdr.bit_depth)
                        self.assertEqual(color_type, encrypted_ihdr.color_type)
                        self.assertEqual(mode, metadata["mode"])
                        self.assertEqual(pixels, decrypted_pixels)

    def test_palette_png_is_rejected(self) -> None:
        public_key, _ = generate_keypair(256)
        source = Path("Projekt_2/data/PWr.png")

        with TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                encrypt_png(
                    source,
                    Path(temporary_directory) / "encrypted.png",
                    public_key,
                    "ecb",
                )

    def test_png_with_multiple_idat_chunks_roundtrips(self) -> None:
        public_key, private_key = generate_keypair(256)

        with TemporaryDirectory() as temporary_directory:
            temp_dir = Path(temporary_directory)
            source = temp_dir / "multi_idat.png"
            encrypted = temp_dir / "multi_idat_encrypted.png"
            decrypted = temp_dir / "multi_idat_decrypted.png"
            pixels = self.make_pixels(width=64, height=1, color_type=0)
            self.write_test_png(
                source,
                width=64,
                height=1,
                color_type=0,
                pixels=pixels,
                split_idat=True,
            )

            self.assertEqual(
                2,
                sum(1 for chunk in read_png(source) if chunk.chunk_type == "IDAT"),
            )

            encrypt_png(source, encrypted, public_key, "chain")
            decrypt_png(encrypted, decrypted, private_key)

            self.assertEqual(pixels, self.read_pixels(decrypted))

    def test_compressed_idat_encryption_roundtrips_original_idat_bytes(self) -> None:
        public_key, private_key = generate_keypair(256)

        with TemporaryDirectory() as temporary_directory:
            temp_dir = Path(temporary_directory)
            source = temp_dir / "compressed_source.png"
            encrypted = temp_dir / "compressed_encrypted.png"
            decrypted = temp_dir / "compressed_decrypted.png"
            pixels = self.make_pixels(width=64, height=1, color_type=0)
            self.write_test_png(source, 64, 1, 0, pixels, split_idat=True)

            encrypt_compressed_idat_png(source, encrypted, public_key, "ecb")
            decrypt_compressed_idat_png(encrypted, decrypted, private_key)

            source_chunks = read_png(source)
            encrypted_metadata = read_metadata(read_png(encrypted))
            decrypted_chunks = read_png(decrypted)

            self.assertEqual("compressed_idat", encrypted_metadata["payload"])
            self.assertEqual(idat_data(source_chunks), idat_data(decrypted_chunks))
            self.assertEqual(pixels, self.read_pixels(decrypted))

    @staticmethod
    def make_pixels(width: int, height: int, color_type: int) -> bytes:
        size = width * height * samples_per_pixel(color_type)
        return bytes((index * 37 + color_type) % 256 for index in range(size))

    @staticmethod
    def write_test_png(
        path: Path,
        width: int,
        height: int,
        color_type: int,
        pixels: bytes,
        split_idat: bool = False,
    ) -> None:
        ihdr = Ihdr(
            width=width,
            height=height,
            bit_depth=8,
            color_type=color_type,
            compression_method=0,
            filter_method=0,
            interlace_method=0,
        )
        compressed_pixels = encode_pixels(ihdr, pixels)

        if split_idat:
            split_at = max(1, len(compressed_pixels) // 2)
            idat_chunks = [
                make_chunk("IDAT", compressed_pixels[:split_at]),
                make_chunk("IDAT", compressed_pixels[split_at:]),
            ]
        else:
            idat_chunks = [make_chunk("IDAT", compressed_pixels)]

        chunks = [make_ihdr(ihdr), *idat_chunks, make_chunk("IEND", b"")]
        write_png(path, chunks)

    @staticmethod
    def read_pixels(path: Path) -> bytes:
        chunks = read_png(path)
        return decode_pixels(parse_ihdr(chunks), idat_data(chunks))


if __name__ == "__main__":
    unittest.main()
