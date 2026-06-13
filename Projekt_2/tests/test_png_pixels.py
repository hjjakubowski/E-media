from __future__ import annotations

import unittest

from Projekt_2.src.png_pixels import paeth_predictor, unfilter_row


class PngPixelsTest(unittest.TestCase):
    def test_unfilter_row_supports_filters_0_to_4(self) -> None:
        raw = bytes([10, 20, 30, 40, 50, 60, 70, 80])
        previous = bytes([1, 2, 3, 4, 5, 6, 7, 8])
        bpp = 2

        cases = {
            0: raw,
            1: self.make_sub_filter(raw, bpp),
            2: self.make_up_filter(raw, previous),
            3: self.make_average_filter(raw, previous, bpp),
            4: self.make_paeth_filter(raw, previous, bpp),
        }

        for filter_type, filtered in cases.items():
            with self.subTest(filter_type=filter_type):
                self.assertEqual(raw, unfilter_row(filter_type, filtered, previous, bpp))

    @staticmethod
    def make_sub_filter(raw: bytes, bpp: int) -> bytes:
        output = bytearray()

        for index, value in enumerate(raw):
            left = raw[index - bpp] if index >= bpp else 0
            output.append((value - left) & 0xFF)

        return bytes(output)

    @staticmethod
    def make_up_filter(raw: bytes, previous: bytes) -> bytes:
        return bytes((value - previous[index]) & 0xFF for index, value in enumerate(raw))

    @staticmethod
    def make_average_filter(raw: bytes, previous: bytes, bpp: int) -> bytes:
        output = bytearray()

        for index, value in enumerate(raw):
            left = raw[index - bpp] if index >= bpp else 0
            up = previous[index]
            output.append((value - ((left + up) // 2)) & 0xFF)

        return bytes(output)

    @staticmethod
    def make_paeth_filter(raw: bytes, previous: bytes, bpp: int) -> bytes:
        output = bytearray()

        for index, value in enumerate(raw):
            left = raw[index - bpp] if index >= bpp else 0
            up = previous[index]
            up_left = previous[index - bpp] if index >= bpp else 0
            predictor = paeth_predictor(left, up, up_left)
            output.append((value - predictor) & 0xFF)

        return bytes(output)


if __name__ == "__main__":
    unittest.main()
