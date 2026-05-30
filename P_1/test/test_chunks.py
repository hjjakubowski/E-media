import os
import struct
import sys
import tempfile
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.chunks import (  # noqa: E402
    PNG_SIGNATURE,
    anonymize_png_chunks,
    describe_chunk,
    load_all_chunks,
    write_chunk,
)


class PngChunksTest(unittest.TestCase):
    def _write_png_like_file(self, path: str) -> None:
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        with open(path, "wb") as f:
            f.write(PNG_SIGNATURE)
            write_chunk(f, b"IHDR", ihdr)
            write_chunk(f, b"tEXt", b"Author\x00Jan")
            write_chunk(f, b"IDAT", b"compressed-pixels")
            write_chunk(f, b"IEND", b"")
            f.write(b"hidden-after-iend")

    def test_load_all_chunks_reports_trailing_data_after_iend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample.png")
            self._write_png_like_file(path)

            chunks = load_all_chunks(path)

            self.assertEqual(chunks[-1].chunk_type, b"IEND")
            self.assertEqual(chunks[-1].trailing_data, b"hidden-after-iend")

    def test_anonymize_removes_ancillary_chunks_and_trailing_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = os.path.join(tmpdir, "sample.png")
            output = os.path.join(tmpdir, "sample_anon.png")
            self._write_png_like_file(source)

            report = anonymize_png_chunks(load_all_chunks(source), output)
            output_chunks = load_all_chunks(output)

            self.assertEqual([chunk.chunk_type for chunk in output_chunks], [b"IHDR", b"IDAT", b"IEND"])
            self.assertEqual(output_chunks[-1].trailing_data, b"")
            self.assertEqual(report["removed_types"], {"tEXt": 1})
            self.assertEqual(report["trailing_removed"], len(b"hidden-after-iend"))
            self.assertTrue(report["idat_preserved"])

    def test_anonymize_keeps_rendering_relevant_ancillary_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = os.path.join(tmpdir, "gamma.png")
            output = os.path.join(tmpdir, "gamma_anon.png")
            ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            with open(source, "wb") as f:
                f.write(PNG_SIGNATURE)
                write_chunk(f, b"IHDR", ihdr)
                write_chunk(f, b"gAMA", struct.pack(">I", 45455))
                write_chunk(f, b"tEXt", b"Author\x00Jan")
                write_chunk(f, b"IDAT", b"pixels")
                write_chunk(f, b"IEND", b"")

            report = anonymize_png_chunks(load_all_chunks(source), output)
            output_chunks = load_all_chunks(output)

            self.assertEqual([chunk.chunk_type for chunk in output_chunks], [b"IHDR", b"gAMA", b"IDAT", b"IEND"])
            self.assertEqual(report["kept_rendering_ancillary_types"], {"gAMA": 1})
            self.assertEqual(report["removed_types"], {"tEXt": 1})

    def test_describe_plte_contains_all_palette_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "palette.png")
            ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 3, 0, 0, 0)
            with open(path, "wb") as f:
                f.write(PNG_SIGNATURE)
                write_chunk(f, b"IHDR", ihdr)
                write_chunk(f, b"PLTE", bytes([1, 2, 3, 4, 5, 6]))
                write_chunk(f, b"IDAT", b"pixels")
                write_chunk(f, b"IEND", b"")

            plte = load_all_chunks(path)[1]
            description = describe_chunk(plte)

            self.assertIn("colors_rgb=[(1, 2, 3), (4, 5, 6)]", description)
            self.assertIn("data_hex=01 02 03 04 05 06", description)


if __name__ == "__main__":
    unittest.main()
