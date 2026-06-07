from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import binascii


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
HEX_BYTES_PER_LINE = 16

COLOR_TYPES = {
    0: "grayscale",
    2: "truecolor RGB",
    3: "indexed color with palette",
    4: "grayscale with alpha",
    6: "truecolor RGB with alpha",
}


@dataclass(frozen=True)
class PngChunk:
    offset: int
    length: int
    chunk_type: str
    data: bytes
    crc: int


def read_png_chunks(path: Path) -> list[PngChunk]:
    file_bytes = path.read_bytes()

    if file_bytes[:8] != PNG_SIGNATURE:
        raise ValueError("This is not a PNG file: invalid PNG signature.")

    chunks: list[PngChunk] = []
    offset = len(PNG_SIGNATURE)

    while offset < len(file_bytes):
        length = int.from_bytes(file_bytes[offset : offset + 4], "big")
        chunk_type = file_bytes[offset + 4 : offset + 8].decode("ascii")

        data_start = offset + 8
        data_end = data_start + length
        crc_start = data_end
        crc_end = crc_start + 4

        if crc_end > len(file_bytes):
            raise ValueError(f"Broken PNG chunk at byte offset {offset}.")

        chunk_data = file_bytes[data_start:data_end]
        crc = int.from_bytes(file_bytes[crc_start:crc_end], "big")

        chunks.append(
            PngChunk(
                offset=offset,
                length=length,
                chunk_type=chunk_type,
                data=chunk_data,
                crc=crc,
            )
        )

        offset = crc_end

        if chunk_type == "IEND":
            break

    return chunks


def print_critical_chunks(chunks: list[PngChunk]) -> None:
    print("Critical PNG chunks:")

    for index, chunk in enumerate(chunks, start=1):
        if not is_critical_chunk(chunk):
            continue

        print("-" * 72)
        print(f"Chunk #{index}: {chunk.chunk_type}")
        print(f"Offset in file: {chunk.offset}")
        print(f"Data length: {chunk.length} bytes")
        print(f"CRC from file: 0x{chunk.crc:08x}")
        print(f"CRC is valid: {crc_is_valid(chunk)}")

        if chunk.chunk_type == "IHDR":
            print_ihdr(chunk)
        elif chunk.chunk_type == "PLTE":
            print_plte(chunk)
        elif chunk.chunk_type == "IDAT":
            print_idat(chunk)
        elif chunk.chunk_type == "IEND":
            print("IEND has no data. It only marks the end of the PNG file.")


def is_critical_chunk(chunk: PngChunk) -> bool:
    return chunk.chunk_type[0].isupper()


def crc_is_valid(chunk: PngChunk) -> bool:
    crc_input = chunk.chunk_type.encode("ascii") + chunk.data
    calculated_crc = binascii.crc32(crc_input) & 0xFFFFFFFF
    return calculated_crc == chunk.crc


def print_ihdr(chunk: PngChunk) -> None:
    data = chunk.data

    width = int.from_bytes(data[0:4], "big")
    height = int.from_bytes(data[4:8], "big")
    bit_depth = data[8]
    color_type = data[9]
    compression_method = data[10]
    filter_method = data[11]
    interlace_method = data[12]

    print("IHDR fields:")
    print(f"  width: {width} px")
    print(f"  height: {height} px")
    print(f"  bit depth: {bit_depth}")
    print(f"  color type: {color_type} ({COLOR_TYPES.get(color_type, 'unknown')})")
    print(f"  compression method: {compression_method}")
    print(f"  filter method: {filter_method}")
    print(f"  interlace method: {interlace_method}")
    print("IHDR raw bytes:")
    print_hex_dump(data)


def print_plte(chunk: PngChunk) -> None:
    print("PLTE palette entries:")

    for color_index in range(0, chunk.length, 3):
        red = chunk.data[color_index]
        green = chunk.data[color_index + 1]
        blue = chunk.data[color_index + 2]
        palette_index = color_index // 3
        print(f"  {palette_index:3d}: R={red:3d}, G={green:3d}, B={blue:3d}")

    print("PLTE raw bytes:")
    print_hex_dump(chunk.data)


def print_idat(chunk: PngChunk) -> None:
    print("IDAT stores compressed image data.")
    print("IDAT raw compressed bytes:")
    print_hex_dump(chunk.data)


def print_hex_dump(data: bytes) -> None:
    for offset in range(0, len(data), HEX_BYTES_PER_LINE):
        line = data[offset : offset + HEX_BYTES_PER_LINE]
        print(f"  {offset:04x}: {line.hex(' ')}")


def anonymize_png(source_path: Path, chunks: list[PngChunk], output_dir: Path) -> Path:
    anonymized_path = output_dir / f"{source_path.stem}_anonymized.png"
    removed_chunks: list[str] = []

    with anonymized_path.open("wb") as output_file:
        output_file.write(PNG_SIGNATURE)

        for chunk in chunks:
            if is_critical_chunk(chunk):
                output_file.write(chunk_to_bytes(chunk))
            else:
                removed_chunks.append(chunk.chunk_type)

    print(f"Removed ancillary chunks: {removed_chunks}")
    return anonymized_path


def chunk_to_bytes(chunk: PngChunk) -> bytes:
    length = chunk.length.to_bytes(4, "big")
    chunk_type = chunk.chunk_type.encode("ascii")
    crc = chunk.crc.to_bytes(4, "big")

    return length + chunk_type + chunk.data + crc
