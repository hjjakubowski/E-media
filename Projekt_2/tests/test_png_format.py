from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from Projekt_2.src.png_format import (
    PNG_SIGNATURE,
    chunk_to_bytes,
    crc_is_valid,
    make_chunk,
    read_png,
    write_png,
)


class PngFormatTest(unittest.TestCase):
    def test_chunk_metadata_matches_project_1_model(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "minimal.png"
            chunks = [
                make_chunk(
                    "IHDR",
                    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00",
                ),
                make_chunk("IDAT", b"\x78\x9c\x63\x60\x00\x00\x00\x02\x00\x01"),
                make_chunk("IEND", b""),
            ]

            write_png(path, chunks)
            read_chunks = read_png(path)

            self.assertEqual(8, read_chunks[0].offset)
            self.assertEqual(13, read_chunks[0].length)
            self.assertTrue(crc_is_valid(read_chunks[0]))
            self.assertEqual(
                PNG_SIGNATURE + b"".join(chunk_to_bytes(chunk) for chunk in read_chunks),
                path.read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
