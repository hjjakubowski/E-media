from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import binascii

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
HEX_BYTES_PER_LINE = 16
MAX_PREVIEW_BYTES = 64

COLOR_TYPES = {
    0: "grayscale",
    2: "truecolor RGB",
    3: "indexed color with palette",
    4: "grayscale with alpha",
    6: "truecolor RGB with alpha",
}

PHYSICAL_UNITS = {
    0: "unit unknown",
    1: "pixels per meter",
}

TIFF_TYPES = {
    1: "BYTE",
    2: "ASCII",
    3: "SHORT",
    4: "LONG",
    5: "RATIONAL",
    7: "UNDEFINED",
    9: "SLONG",
    10: "SRATIONAL",
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


def print_chunk_table(chunks: list[PngChunk], title: str) -> None:
    print(title)
    print("  offset  type  length  kind")

    for chunk in chunks:
        kind = "critical" if is_critical_chunk(chunk) else "ancillary"
        print(f"  {chunk.offset:6d}  {chunk.chunk_type:4s}  {chunk.length:6d}  {kind}")


def print_critical_chunks(chunks: list[PngChunk]) -> None:
    print("Critical PNG chunks:")

    for index, chunk in enumerate(chunks, start=1):
        if not is_critical_chunk(chunk):
            continue

        print("-" * 72)
        print_basic_chunk_info(index, chunk)

        if chunk.chunk_type == "IHDR":
            print_ihdr(chunk)
        elif chunk.chunk_type == "PLTE":
            print_plte(chunk)
        elif chunk.chunk_type == "IDAT":
            print_idat(chunk)
        elif chunk.chunk_type == "IEND":
            print("IEND has no data. It only marks the end of the PNG file.")


def print_ancillary_chunks(chunks: list[PngChunk]) -> None:
    ancillary_chunks = [chunk for chunk in chunks if not is_critical_chunk(chunk)]

    print("Ancillary PNG chunks:")

    if not ancillary_chunks:
        print("No ancillary chunks found.")
        return

    color_type = get_ihdr_color_type(chunks)

    for index, chunk in enumerate(chunks, start=1):
        if is_critical_chunk(chunk):
            continue

        print("-" * 72)
        print_basic_chunk_info(index, chunk)

        if chunk.chunk_type == "pHYs":
            print_phys(chunk)
        elif chunk.chunk_type == "tIME":
            print_time(chunk)
        elif chunk.chunk_type == "tEXt":
            print_text(chunk)
        elif chunk.chunk_type == "eXIf":
            print_exif(chunk)
        elif chunk.chunk_type == "bKGD":
            print_bkgd(chunk, color_type)
        else:
            print_unknown_ancillary(chunk)


def print_basic_chunk_info(index: int, chunk: PngChunk) -> None:
    print(f"Chunk #{index}: {chunk.chunk_type}")
    print(f"Offset in file: {chunk.offset}")
    print(f"Data length: {chunk.length} bytes")
    print(f"CRC from file: 0x{chunk.crc:08x}")
    print(f"CRC is valid: {crc_is_valid(chunk)}")


def is_critical_chunk(chunk: PngChunk) -> bool:
    return chunk.chunk_type[0].isupper()


def crc_is_valid(chunk: PngChunk) -> bool:
    crc_input = chunk.chunk_type.encode("ascii") + chunk.data
    calculated_crc = binascii.crc32(crc_input) & 0xFFFFFFFF
    return calculated_crc == chunk.crc


def get_ihdr_color_type(chunks: list[PngChunk]) -> int | None:
    for chunk in chunks:
        if chunk.chunk_type == "IHDR" and len(chunk.data) == 13:
            return chunk.data[9]
    return None


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


def print_phys(chunk: PngChunk) -> None:
    if chunk.length != 9:
        print("Invalid pHYs chunk length. Expected 9 bytes.")
        print_hex_preview(chunk.data)
        return

    pixels_per_unit_x = int.from_bytes(chunk.data[0:4], "big")
    pixels_per_unit_y = int.from_bytes(chunk.data[4:8], "big")
    unit = chunk.data[8]

    print("pHYs fields:")
    print(f"  pixels per unit, X axis: {pixels_per_unit_x}")
    print(f"  pixels per unit, Y axis: {pixels_per_unit_y}")
    print(f"  unit: {unit} ({PHYSICAL_UNITS.get(unit, 'unknown')})")

    if unit == 1:
        dpi_x = pixels_per_unit_x * 0.0254
        dpi_y = pixels_per_unit_y * 0.0254
        print(f"  approximate DPI, X axis: {dpi_x:.2f}")
        print(f"  approximate DPI, Y axis: {dpi_y:.2f}")


def print_time(chunk: PngChunk) -> None:
    if chunk.length != 7:
        print("Invalid tIME chunk length. Expected 7 bytes.")
        print_hex_preview(chunk.data)
        return

    year = int.from_bytes(chunk.data[0:2], "big")
    month = chunk.data[2]
    day = chunk.data[3]
    hour = chunk.data[4]
    minute = chunk.data[5]
    second = chunk.data[6]

    print("tIME fields:")
    print(
        f"  last modification time: {year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d} UTC"
    )


def print_text(chunk: PngChunk) -> None:
    if b"\x00" not in chunk.data:
        print("Invalid tEXt chunk: missing null separator between keyword and text.")
        print_hex_preview(chunk.data)
        return

    keyword_bytes, text_bytes = chunk.data.split(b"\x00", 1)
    keyword = keyword_bytes.decode("latin-1", errors="replace")
    text = text_bytes.decode("latin-1", errors="replace")

    print("tEXt fields:")
    print(f"  keyword: {keyword}")
    print(f"  text: {text}")


def print_exif(chunk: PngChunk) -> None:
    print("eXIf fields:")
    print(f"  EXIF data length: {chunk.length} bytes")

    tiff_data = chunk.data

    if tiff_data.startswith(b"Exif\x00\x00"):
        print("  optional Exif header: present")
        tiff_data = tiff_data[6:]
    else:
        print("  optional Exif header: not present")

    if len(tiff_data) < 8:
        print("  TIFF header: too short")
        print_hex_preview(chunk.data)
        return

    byte_order_marker = tiff_data[0:2]

    if byte_order_marker == b"II":
        byte_order = "little"
        print("  TIFF byte order: little endian (II)")
    elif byte_order_marker == b"MM":
        byte_order = "big"
        print("  TIFF byte order: big endian (MM)")
    else:
        print("  TIFF byte order: unknown")
        print_hex_preview(chunk.data)
        return

    magic_number = int.from_bytes(tiff_data[2:4], byte_order)
    first_ifd_offset = int.from_bytes(tiff_data[4:8], byte_order)

    print(f"  TIFF magic number: {magic_number}")
    print(f"  first IFD offset: {first_ifd_offset}")

    if magic_number != 42:
        print("  warning: TIFF magic number should be 42")

    print_ifd_preview(tiff_data, byte_order, first_ifd_offset)
    print("  first EXIF bytes:")
    print_hex_preview(chunk.data)


def print_ifd_preview(tiff_data: bytes, byte_order: str, ifd_offset: int) -> None:
    if ifd_offset + 2 > len(tiff_data):
        print("  IFD preview: offset outside EXIF data")
        return

    entry_count = int.from_bytes(tiff_data[ifd_offset : ifd_offset + 2], byte_order)
    print(f"  IFD entries: {entry_count}")

    entries_start = ifd_offset + 2

    for entry_index in range(min(entry_count, 3)):
        start = entries_start + entry_index * 12
        end = start + 12

        if end > len(tiff_data):
            print(f"  IFD entry #{entry_index + 1}: outside EXIF data")
            return

        entry = tiff_data[start:end]
        tag = int.from_bytes(entry[0:2], byte_order)
        value_type = int.from_bytes(entry[2:4], byte_order)
        value_count = int.from_bytes(entry[4:8], byte_order)
        value_or_offset = int.from_bytes(entry[8:12], byte_order)

        print(f"  IFD entry #{entry_index + 1}:")
        print(f"    tag: 0x{tag:04x}")
        print(f"    type: {value_type} ({TIFF_TYPES.get(value_type, 'unknown')})")
        print(f"    count: {value_count}")
        print(f"    value or offset: {value_or_offset}")


def print_bkgd(chunk: PngChunk, color_type: int | None) -> None:
    print("bKGD fields:")

    if color_type == 3 and chunk.length == 1:
        print(f"  palette index: {chunk.data[0]}")
    elif color_type in (0, 4) and chunk.length == 2:
        gray = int.from_bytes(chunk.data[0:2], "big")
        print(f"  grayscale background value: {gray}")
    elif color_type in (2, 6) and chunk.length == 6:
        red = int.from_bytes(chunk.data[0:2], "big")
        green = int.from_bytes(chunk.data[2:4], "big")
        blue = int.from_bytes(chunk.data[4:6], "big")
        print(f"  red background value: {red}")
        print(f"  green background value: {green}")
        print(f"  blue background value: {blue}")
    else:
        print("  cannot interpret bKGD for this color type or chunk length")
        print_hex_preview(chunk.data)


def print_unknown_ancillary(chunk: PngChunk) -> None:
    print("No detailed parser for this ancillary chunk.")
    print("Raw byte preview:")
    print_hex_preview(chunk.data)


def print_hex_dump(data: bytes) -> None:
    for offset in range(0, len(data), HEX_BYTES_PER_LINE):
        line = data[offset : offset + HEX_BYTES_PER_LINE]
        print(f"  {offset:04x}: {line.hex(' ')}")


def print_hex_preview(data: bytes) -> None:
    preview = data[:MAX_PREVIEW_BYTES]
    print_hex_dump(preview)

    if len(data) > MAX_PREVIEW_BYTES:
        hidden_bytes = len(data) - MAX_PREVIEW_BYTES
        print(f"  ... {hidden_bytes} more bytes")


def anonymize_png(source_path: Path, chunks: list[PngChunk], output_dir: Path) -> Path:
    anonymized_path = output_dir / f"{source_path.stem}_anonymized.png"
    removed_chunks: list[str] = []

    print_chunk_table(chunks, "Chunks before anonymization:")

    with anonymized_path.open("wb") as output_file:
        output_file.write(PNG_SIGNATURE)

        for chunk in chunks:
            if is_critical_chunk(chunk):
                output_file.write(chunk_to_bytes(chunk))
            else:
                removed_chunks.append(chunk.chunk_type)

    anonymized_chunks = read_png_chunks(anonymized_path)

    print(f"Removed ancillary chunks: {removed_chunks}")
    print_chunk_table(anonymized_chunks, "Chunks after anonymization:")
    print(
        "Anonymization rewrites the PNG sequentially, so removed chunks leave no empty byte ranges."
    )

    return anonymized_path


def chunk_to_bytes(chunk: PngChunk) -> bytes:
    length = chunk.length.to_bytes(4, "big")
    chunk_type = chunk.chunk_type.encode("ascii")
    crc = chunk.crc.to_bytes(4, "big")

    return length + chunk_type + chunk.data + crc
