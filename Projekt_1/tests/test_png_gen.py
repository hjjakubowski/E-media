from __future__ import annotations

from pathlib import Path
import binascii

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SOURCE_PNG = DATA_DIR / "Black.png"
OUTPUT_PNG = DATA_DIR / "Black_metadata.png"


def main() -> None:
    create_png_with_metadata(SOURCE_PNG, OUTPUT_PNG)

    print(f"Input PNG: {SOURCE_PNG}")
    print(f"Generated PNG: {OUTPUT_PNG}")
    print("Generated chunks:")

    for chunk_type in read_chunk_types(OUTPUT_PNG):
        print(f"  {chunk_type}")


def create_png_with_metadata(input_path: Path, output_path: Path) -> None:
    file_bytes = input_path.read_bytes()

    if file_bytes[:8] != PNG_SIGNATURE:
        raise ValueError(f"{input_path} is not a PNG file.")

    chunks_to_add = [
        make_text_chunk("Comment", "Demo PNG with ancillary metadata for grade 4.0"),
        make_time_chunk(year=2026, month=6, day=7, hour=12, minute=0, second=0),
        make_exif_chunk(),
    ]

    output_bytes = bytearray(PNG_SIGNATURE)
    offset = len(PNG_SIGNATURE)
    chunks_added = False

    while offset < len(file_bytes):
        chunk_length = int.from_bytes(file_bytes[offset : offset + 4], "big")
        chunk_type = file_bytes[offset + 4 : offset + 8].decode("ascii")
        chunk_end = offset + 12 + chunk_length

        if chunk_type == "IDAT" and not chunks_added:
            for chunk in chunks_to_add:
                output_bytes.extend(chunk)
            chunks_added = True

        output_bytes.extend(file_bytes[offset:chunk_end])
        offset = chunk_end

        if chunk_type == "IEND":
            break

    output_path.write_bytes(output_bytes)


def make_text_chunk(keyword: str, text: str) -> bytes:
    chunk_data = keyword.encode("latin-1") + b"\x00" + text.encode("latin-1")
    return make_chunk("tEXt", chunk_data)


def make_time_chunk(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
) -> bytes:
    chunk_data = year.to_bytes(2, "big") + bytes([month, day, hour, minute, second])
    return make_chunk("tIME", chunk_data)


def make_exif_chunk() -> bytes:
    description = b"E-media EXIF demo\x00"
    byte_order = "little"

    tiff_header = b"II" + (42).to_bytes(2, byte_order) + (8).to_bytes(4, byte_order)
    ifd_entry_count = (1).to_bytes(2, byte_order)

    value_offset = 8 + 2 + 12 + 4

    image_description_tag = (0x010E).to_bytes(2, byte_order)
    ascii_type = (2).to_bytes(2, byte_order)
    value_count = len(description).to_bytes(4, byte_order)
    value_pointer = value_offset.to_bytes(4, byte_order)
    ifd_entry = image_description_tag + ascii_type + value_count + value_pointer

    next_ifd_offset = (0).to_bytes(4, byte_order)

    exif_data = (
        tiff_header + ifd_entry_count + ifd_entry + next_ifd_offset + description
    )
    return make_chunk("eXIf", exif_data)


def make_chunk(chunk_type: str, chunk_data: bytes) -> bytes:
    chunk_type_bytes = chunk_type.encode("ascii")
    crc = binascii.crc32(chunk_type_bytes + chunk_data) & 0xFFFFFFFF

    return (
        len(chunk_data).to_bytes(4, "big")
        + chunk_type_bytes
        + chunk_data
        + crc.to_bytes(4, "big")
    )


def read_chunk_types(path: Path) -> list[str]:
    file_bytes = path.read_bytes()

    if file_bytes[:8] != PNG_SIGNATURE:
        raise ValueError(f"{path} is not a PNG file.")

    chunk_types: list[str] = []
    offset = len(PNG_SIGNATURE)

    while offset < len(file_bytes):
        chunk_length = int.from_bytes(file_bytes[offset : offset + 4], "big")
        chunk_type = file_bytes[offset + 4 : offset + 8].decode("ascii")

        chunk_types.append(chunk_type)
        offset += 12 + chunk_length

        if chunk_type == "IEND":
            break

    return chunk_types


if __name__ == "__main__":
    main()
