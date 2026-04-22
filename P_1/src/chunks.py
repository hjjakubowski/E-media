import struct as st
import zlib
from dataclasses import dataclass
from typing import BinaryIO, Iterable

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass
class PngChunk:
    chunk_type: bytes
    data: bytes
    length: int
    crc_read: int
    crc_calc: int
    is_critical: bool
    index: int


def read_chunk(f: BinaryIO) -> tuple[bytes, bytes, int, int, int]:
    length_raw = f.read(4)
    if len(length_raw) != 4:
        raise EOFError("Brak danych na dlugosc chunka")
    (chunk_length,) = st.unpack(">I", length_raw)

    chunk_type = f.read(4)
    if len(chunk_type) != 4:
        raise EOFError("Brak danych na typ chunka")

    chunk_data = f.read(chunk_length)
    if len(chunk_data) != chunk_length:
        raise EOFError("Brak danych chunka")

    crc_raw = f.read(4)
    if len(crc_raw) != 4:
        raise EOFError("Brak CRC")
    (chunk_crc,) = st.unpack(">I", crc_raw)

    calc_crc = zlib.crc32(chunk_type)
    calc_crc = zlib.crc32(chunk_data, calc_crc) & 0xFFFFFFFF

    if chunk_crc != calc_crc:
        raise ValueError(f"Invalid CRC: read=0x{chunk_crc:08X}, calc=0x{calc_crc:08X}")

    return chunk_type, chunk_data, chunk_length, chunk_crc, calc_crc


def parse_ihdr(chunk_data: bytes) -> dict:
    if len(chunk_data) != 13:
        raise ValueError(f"IHDR should have 13 bytes, got {len(chunk_data)} bytes")

    width, height, bit_depth, color_type, compression, filter_method, interlace_method = st.unpack(
        ">IIBBBBB", chunk_data
    )
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "compression": compression,
        "filter_method": filter_method,
        "interlace_method": interlace_method,
    }


def display_IHDR_chunks_info(image: str) -> None:
    with open(image, "rb") as f:
        if f.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            raise ValueError("Invalid PNG signature")

        chunk_type, chunk_data, _, _, _ = read_chunk(f)
        if chunk_type != b"IHDR":
            raise ValueError(f"The first chunk is not IHDR - {chunk_type!r}")

        ihdr = parse_ihdr(chunk_data)

    print(
        f"Width: {ihdr['width']},\n Height: {ihdr['height']},\n Bit depth: {ihdr['bit_depth']},\n "
        f"Color: {ihdr['color_type']},\n Compression: {ihdr['compression']},\n "
        f"Filter method: {ihdr['filter_method']},\n interlace method: {ihdr['interlace_method']}"
    )


def load_all_chunks(image: str) -> list[PngChunk]:
    with open(image, "rb") as f:
        signature = f.read(len(PNG_SIGNATURE))
        if signature != PNG_SIGNATURE:
            raise ValueError("Invalid PNG signature")

        chunks: list[PngChunk] = []
        index = 0

        while True:
            chunk_type, chunk_data, length, crc_read, crc_calc = read_chunk(f)
            is_critical = not (chunk_type[0] & 32)
            chunks.append(
                PngChunk(
                    chunk_type=chunk_type,
                    data=chunk_data,
                    length=length,
                    crc_read=crc_read,
                    crc_calc=crc_calc,
                    is_critical=is_critical,
                    index=index,
                )
            )

            if chunk_type == b"IEND":
                break
            index += 1

    return chunks


def decode_time(data: bytes) -> str | None:
    if len(data) != 7:
        return None
    year, month, day, hour, minute, second = st.unpack(">HBBBBBB", data)
    return f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"


def decode_text(data: bytes) -> tuple[str, str] | None:
    parts = data.split(b"\x00", 1)
    if len(parts) != 2:
        return None
    key = parts[0].decode("latin-1", errors="replace")
    text = parts[1].decode("latin-1", errors="replace")
    return key, text


def decode_exif_summary(data: bytes) -> dict | None:
    if len(data) < 8:
        return None

    endian = data[0:2]
    if endian == b"II":
        byte_order = "<"
    elif endian == b"MM":
        byte_order = ">"
    else:
        return None

    magic = st.unpack(f"{byte_order}H", data[2:4])[0]
    if magic != 42:
        return None

    offset = st.unpack(f"{byte_order}I", data[4:8])[0]
    if offset + 2 > len(data):
        return None

    num_tags = st.unpack(f"{byte_order}H", data[offset : offset + 2])[0]
    return {"endian": endian.decode("ascii"), "num_tags": num_tags}


def describe_chunk(chunk: PngChunk) -> str:
    chunk_name = chunk.chunk_type.decode("ascii", errors="replace")
    role = "critical" if chunk.is_critical else "ancillary"
    base = f"[{chunk.index}] {chunk_name} ({role}, {chunk.length} B)"

    if chunk.chunk_type == b"tIME":
        decoded = decode_time(chunk.data)
        if decoded:
            return f"{base} - time={decoded}"
    elif chunk.chunk_type == b"tEXt":
        decoded = decode_text(chunk.data)
        if decoded:
            key, text = decoded
            return f"{base} - text[{key}]={text}"
    elif chunk.chunk_type == b"eXIf":
        decoded = decode_exif_summary(chunk.data)
        if decoded:
            return f"{base} - EXIF endian={decoded['endian']}, tags={decoded['num_tags']}"

    return base


def write_chunk(f_out: BinaryIO, chunk_type: bytes, chunk_data: bytes) -> None:
    f_out.write(st.pack(">I", len(chunk_data)))
    f_out.write(chunk_type)
    f_out.write(chunk_data)
    calc_crc = zlib.crc32(chunk_type)
    calc_crc = zlib.crc32(chunk_data, calc_crc) & 0xFFFFFFFF
    f_out.write(st.pack(">I", calc_crc))


def anonymize_png_chunks(chunks: Iterable[PngChunk], output_image: str) -> dict:
    kept = 0
    removed = 0
    removed_types: dict[str, int] = {}

    with open(output_image, "wb") as f_out:
        f_out.write(PNG_SIGNATURE)

        for chunk in chunks:
            if chunk.is_critical:
                write_chunk(f_out, chunk.chunk_type, chunk.data)
                kept += 1
            else:
                removed += 1
                name = chunk.chunk_type.decode("ascii", errors="replace")
                removed_types[name] = removed_types.get(name, 0) + 1

    return {"kept": kept, "removed": removed, "removed_types": removed_types}


def load_all_chunks_and_anonimize(image: str, output_image: str) -> dict:
    chunks = load_all_chunks(image)
    return anonymize_png_chunks(chunks, output_image)


