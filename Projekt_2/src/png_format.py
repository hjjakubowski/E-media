from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import binascii


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
RSA_METADATA_CHUNK = "rsAP"


@dataclass(frozen=True)
class PngChunk:
    offset: int
    length: int
    chunk_type: str
    data: bytes
    crc: int


@dataclass(frozen=True)
class Ihdr:
    width: int
    height: int
    bit_depth: int
    color_type: int
    compression_method: int
    filter_method: int
    interlace_method: int


def read_png_chunks(path: Path) -> list[PngChunk]:
    file_bytes = path.read_bytes()

    if file_bytes[:8] != PNG_SIGNATURE:
        raise ValueError("This is not a PNG file: invalid PNG signature.")

    chunks: list[PngChunk] = []
    offset = len(PNG_SIGNATURE)

    while offset < len(file_bytes):
        if offset + 8 > len(file_bytes):
            raise ValueError(f"Broken PNG chunk at byte offset {offset}.")

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

    if not chunks or chunks[-1].chunk_type != "IEND":
        raise ValueError("PNG file does not contain IEND chunk.")

    return chunks


def read_png(path: Path) -> list[PngChunk]:
    chunks = read_png_chunks(path)

    for chunk in chunks:
        if not crc_is_valid(chunk):
            raise ValueError(f"Invalid CRC for PNG chunk {chunk.chunk_type}.")

    return chunks


def write_png(path: Path, chunks: list[PngChunk]) -> None:
    output = bytearray(PNG_SIGNATURE)

    for chunk in chunks:
        output.extend(chunk_to_bytes(chunk))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output)


def make_chunk(chunk_type: str, chunk_data: bytes, offset: int = 0) -> PngChunk:
    return PngChunk(
        offset=offset,
        length=len(chunk_data),
        chunk_type=chunk_type,
        data=chunk_data,
        crc=chunk_crc(chunk_type, chunk_data),
    )


def chunk_to_bytes(chunk: PngChunk) -> bytes:
    length = chunk.length.to_bytes(4, "big")
    chunk_type = chunk.chunk_type.encode("ascii")
    crc = chunk.crc.to_bytes(4, "big")

    return length + chunk_type + chunk.data + crc


def chunk_crc(chunk_type: str, data: bytes) -> int:
    return binascii.crc32(chunk_type.encode("ascii") + data) & 0xFFFFFFFF


def crc_is_valid(chunk: PngChunk) -> bool:
    crc_input = chunk.chunk_type.encode("ascii") + chunk.data
    calculated_crc = binascii.crc32(crc_input) & 0xFFFFFFFF
    return calculated_crc == chunk.crc


def is_critical_chunk(chunk: PngChunk) -> bool:
    return chunk.chunk_type[0].isupper()


def parse_ihdr(chunks: list[PngChunk]) -> Ihdr:
    if not chunks or chunks[0].chunk_type != "IHDR":
        raise ValueError("PNG file must start with IHDR chunk.")

    data = chunks[0].data

    if len(data) != 13:
        raise ValueError("IHDR chunk must have exactly 13 bytes.")

    return Ihdr(
        width=int.from_bytes(data[0:4], "big"),
        height=int.from_bytes(data[4:8], "big"),
        bit_depth=data[8],
        color_type=data[9],
        compression_method=data[10],
        filter_method=data[11],
        interlace_method=data[12],
    )


def make_ihdr(ihdr: Ihdr) -> PngChunk:
    data = (
        ihdr.width.to_bytes(4, "big")
        + ihdr.height.to_bytes(4, "big")
        + bytes(
            [
                ihdr.bit_depth,
                ihdr.color_type,
                ihdr.compression_method,
                ihdr.filter_method,
                ihdr.interlace_method,
            ]
        )
    )
    return make_chunk("IHDR", data)


def idat_data(chunks: list[PngChunk]) -> bytes:
    return b"".join(chunk.data for chunk in chunks if chunk.chunk_type == "IDAT")


def without_chunks(chunks: list[PngChunk], chunk_type: str) -> list[PngChunk]:
    return [chunk for chunk in chunks if chunk.chunk_type != chunk_type]


def replace_ihdr(chunks: list[PngChunk], ihdr: Ihdr) -> list[PngChunk]:
    return [make_ihdr(ihdr)] + chunks[1:]


def replace_idat(chunks: list[PngChunk], compressed_data: bytes) -> list[PngChunk]:
    output: list[PngChunk] = []
    inserted = False

    for chunk in chunks:
        if chunk.chunk_type == "IDAT":
            if not inserted:
                output.append(make_chunk("IDAT", compressed_data))
                inserted = True
            continue

        output.append(chunk)

    if not inserted:
        raise ValueError("PNG file does not contain IDAT chunk.")

    return output


def insert_before_first_idat(
    chunks: list[PngChunk],
    new_chunk: PngChunk,
) -> list[PngChunk]:
    output: list[PngChunk] = []
    inserted = False

    for chunk in chunks:
        if chunk.chunk_type == "IDAT" and not inserted:
            output.append(new_chunk)
            inserted = True

        output.append(chunk)

    if not inserted:
        raise ValueError("PNG file does not contain IDAT chunk.")

    return output
