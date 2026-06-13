from __future__ import annotations

import zlib

from .png_format import Ihdr, PngChunk, idat_data


SAMPLES_PER_PIXEL = {
    0: 1,
    2: 3,
    6: 4,
}


def validate_supported_png(ihdr: Ihdr, expected_bit_depth: int | None = None) -> None:
    if ihdr.color_type not in SAMPLES_PER_PIXEL:
        raise ValueError(
            "Only PNG color types 0 (grayscale), 2 (RGB), and 6 (RGBA) are supported."
        )

    if ihdr.bit_depth not in (8, 16):
        raise ValueError("Only 8-bit and 16-bit PNG images are supported.")

    if expected_bit_depth is not None and ihdr.bit_depth != expected_bit_depth:
        raise ValueError(f"Expected {expected_bit_depth}-bit PNG image.")

    if ihdr.compression_method != 0:
        raise ValueError("Unsupported PNG compression method.")

    if ihdr.filter_method != 0:
        raise ValueError("Unsupported PNG filter method.")

    if ihdr.interlace_method != 0:
        raise ValueError("Interlaced PNG images are not supported.")


def samples_per_pixel(color_type: int) -> int:
    try:
        return SAMPLES_PER_PIXEL[color_type]
    except KeyError as error:
        raise ValueError(f"Unsupported PNG color type: {color_type}") from error


def bytes_per_sample(bit_depth: int) -> int:
    if bit_depth == 8:
        return 1

    if bit_depth == 16:
        return 2

    raise ValueError(f"Unsupported PNG bit depth: {bit_depth}")


def row_bytes(ihdr: Ihdr) -> int:
    return ihdr.width * samples_per_pixel(ihdr.color_type) * bytes_per_sample(
        ihdr.bit_depth
    )


def filter_bpp(ihdr: Ihdr) -> int:
    return samples_per_pixel(ihdr.color_type) * bytes_per_sample(ihdr.bit_depth)


def decode_pixels(ihdr: Ihdr, compressed_idat: bytes) -> bytes:
    validate_supported_png(ihdr)
    return unfilter_scanlines(ihdr, zlib.decompress(compressed_idat))


def decode_png_pixels(ihdr: Ihdr, chunks: list[PngChunk]) -> bytes:
    return decode_pixels(ihdr, idat_data(chunks))


def encode_pixels(ihdr: Ihdr, pixels: bytes) -> bytes:
    validate_supported_png(ihdr)
    return zlib.compress(make_filter_zero_scanlines(ihdr, pixels))


def unfilter_scanlines(ihdr: Ihdr, scanlines: bytes) -> bytes:
    validate_supported_png(ihdr)

    line_size = row_bytes(ihdr)
    expected_size = ihdr.height * (line_size + 1)

    if len(scanlines) != expected_size:
        raise ValueError(
            f"Invalid decompressed image data size: expected {expected_size}, "
            f"got {len(scanlines)}."
        )

    bpp = filter_bpp(ihdr)
    output = bytearray()
    previous_row = bytes(line_size)
    offset = 0

    for _ in range(ihdr.height):
        filter_type = scanlines[offset]
        offset += 1
        filtered = scanlines[offset : offset + line_size]
        offset += line_size

        row = unfilter_row(filter_type, filtered, previous_row, bpp)
        output.extend(row)
        previous_row = row

    return bytes(output)


def unfilter_row(
    filter_type: int,
    filtered: bytes,
    previous_row: bytes,
    bpp: int,
) -> bytes:
    row = bytearray(len(filtered))

    for index, value in enumerate(filtered):
        left = row[index - bpp] if index >= bpp else 0
        up = previous_row[index]
        up_left = previous_row[index - bpp] if index >= bpp else 0

        if filter_type == 0:
            predictor = 0
        elif filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = up
        elif filter_type == 3:
            predictor = (left + up) // 2
        elif filter_type == 4:
            predictor = paeth_predictor(left, up, up_left)
        else:
            raise ValueError(f"Unsupported PNG scanline filter: {filter_type}")

        row[index] = (value + predictor) & 0xFF

    return bytes(row)


def make_filter_zero_scanlines(ihdr: Ihdr, pixels: bytes) -> bytes:
    line_size = row_bytes(ihdr)
    expected_size = ihdr.height * line_size

    if len(pixels) != expected_size:
        raise ValueError(
            f"Invalid raw pixel data size: expected {expected_size}, got {len(pixels)}."
        )

    output = bytearray()

    for row_start in range(0, len(pixels), line_size):
        output.append(0)
        output.extend(pixels[row_start : row_start + line_size])

    return bytes(output)


def paeth_predictor(left: int, up: int, up_left: int) -> int:
    prediction = left + up - up_left
    distance_left = abs(prediction - left)
    distance_up = abs(prediction - up)
    distance_up_left = abs(prediction - up_left)

    if distance_left <= distance_up and distance_left <= distance_up_left:
        return left

    if distance_up <= distance_up_left:
        return up

    return up_left
