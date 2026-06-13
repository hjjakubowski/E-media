from __future__ import annotations

from pathlib import Path
import json

from .png_format import (
    Ihdr,
    PngChunk,
    RSA_METADATA_CHUNK,
    idat_data,
    insert_before_first_idat,
    make_chunk,
    parse_ihdr,
    read_png,
    replace_idat,
    replace_ihdr,
    without_chunks,
    write_png,
)
from .png_pixels import decode_pixels, encode_pixels, row_bytes, validate_supported_png
from .rsa_core import (
    RsaPrivateKey,
    RsaPublicKey,
    key_byte_size,
    plain_block_size,
    public_from_private,
)
from .rsa_modes import SUPPORTED_MODES, decrypt_bytes, encrypt_bytes, encrypted_length


METADATA_VERSION = 1


def encrypt_png(
    source_path: Path,
    output_path: Path,
    public_key: RsaPublicKey,
    mode: str,
) -> Path:
    normalized_mode = mode.lower()

    if normalized_mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported RSA mode: {mode}")

    chunks = read_png(source_path)
    source_ihdr = parse_ihdr(chunks)
    validate_supported_png(source_ihdr, expected_bit_depth=8)

    pixels = decode_pixels(source_ihdr, idat_data(chunks))
    cipher_length = encrypted_length(len(pixels), public_key)

    encrypted_ihdr = encrypted_png_ihdr(source_ihdr)
    capacity = encrypted_capacity(encrypted_ihdr)

    if cipher_length > capacity:
        raise ValueError(
            "RSA ciphertext does not fit in the 16-bit PNG carrier. "
            f"Need {cipher_length} bytes, capacity is {capacity} bytes."
        )

    ciphertext, iv = encrypt_bytes(pixels, public_key, normalized_mode)
    carrier_pixels = ciphertext.ljust(capacity, b"\x00")
    encrypted_idat = encode_pixels(encrypted_ihdr, carrier_pixels)

    metadata = {
        "version": METADATA_VERSION,
        "mode": normalized_mode,
        "width": source_ihdr.width,
        "height": source_ihdr.height,
        "color_type": source_ihdr.color_type,
        "original_bit_depth": source_ihdr.bit_depth,
        "encrypted_bit_depth": encrypted_ihdr.bit_depth,
        "plain_length": len(pixels),
        "cipher_length": len(ciphertext),
        "key_bytes": key_byte_size(public_key.n),
        "plain_block_size": plain_block_size(public_key),
        "iv": iv.hex(),
    }

    output_chunks = without_chunks(chunks, RSA_METADATA_CHUNK)
    output_chunks = replace_ihdr(output_chunks, encrypted_ihdr)
    output_chunks = replace_idat(output_chunks, encrypted_idat)
    output_chunks = insert_before_first_idat(
        output_chunks,
        make_chunk(RSA_METADATA_CHUNK, encode_metadata(metadata)),
    )

    write_png(output_path, output_chunks)
    return output_path


def decrypt_png(
    source_path: Path,
    output_path: Path,
    private_key: RsaPrivateKey,
) -> Path:
    chunks = read_png(source_path)
    encrypted_ihdr = parse_ihdr(chunks)
    validate_supported_png(encrypted_ihdr, expected_bit_depth=16)

    metadata = read_metadata(chunks)
    validate_metadata(metadata, encrypted_ihdr, private_key)

    carrier_pixels = decode_pixels(encrypted_ihdr, idat_data(chunks))
    cipher_length = int(metadata["cipher_length"])
    ciphertext = carrier_pixels[:cipher_length]
    iv = bytes.fromhex(str(metadata.get("iv", "")))

    pixels = decrypt_bytes(
        ciphertext,
        private_key,
        int(metadata["plain_length"]),
        str(metadata["mode"]),
        iv,
    )

    output_ihdr = Ihdr(
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        bit_depth=int(metadata["original_bit_depth"]),
        color_type=int(metadata["color_type"]),
        compression_method=0,
        filter_method=0,
        interlace_method=0,
    )
    output_idat = encode_pixels(output_ihdr, pixels)

    output_chunks = without_chunks(chunks, RSA_METADATA_CHUNK)
    output_chunks = replace_ihdr(output_chunks, output_ihdr)
    output_chunks = replace_idat(output_chunks, output_idat)

    write_png(output_path, output_chunks)
    return output_path


def encrypted_png_ihdr(source_ihdr: Ihdr) -> Ihdr:
    return Ihdr(
        width=source_ihdr.width,
        height=source_ihdr.height,
        bit_depth=16,
        color_type=source_ihdr.color_type,
        compression_method=source_ihdr.compression_method,
        filter_method=source_ihdr.filter_method,
        interlace_method=source_ihdr.interlace_method,
    )


def encrypted_capacity(encrypted_ihdr: Ihdr) -> int:
    return encrypted_ihdr.height * row_bytes(encrypted_ihdr)


def encode_metadata(metadata: dict[str, object]) -> bytes:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_metadata(chunks: list[PngChunk]) -> dict[str, object]:
    metadata_chunks = [
        chunk for chunk in chunks if chunk.chunk_type == RSA_METADATA_CHUNK
    ]

    if not metadata_chunks:
        raise ValueError(f"PNG file does not contain {RSA_METADATA_CHUNK} metadata.")

    if len(metadata_chunks) > 1:
        raise ValueError(f"PNG file contains multiple {RSA_METADATA_CHUNK} chunks.")

    return json.loads(metadata_chunks[0].data.decode("utf-8"))


def validate_metadata(
    metadata: dict[str, object],
    encrypted_ihdr: Ihdr,
    private_key: RsaPrivateKey,
) -> None:
    if int(metadata.get("version", 0)) != METADATA_VERSION:
        raise ValueError("Unsupported RSA PNG metadata version.")

    if str(metadata.get("mode", "")).lower() not in SUPPORTED_MODES:
        raise ValueError("Unsupported RSA PNG mode in metadata.")

    if int(metadata["width"]) != encrypted_ihdr.width:
        raise ValueError("Metadata width does not match encrypted PNG.")

    if int(metadata["height"]) != encrypted_ihdr.height:
        raise ValueError("Metadata height does not match encrypted PNG.")

    if int(metadata["color_type"]) != encrypted_ihdr.color_type:
        raise ValueError("Metadata color type does not match encrypted PNG.")

    if int(metadata["encrypted_bit_depth"]) != encrypted_ihdr.bit_depth:
        raise ValueError("Metadata bit depth does not match encrypted PNG.")

    if int(metadata["key_bytes"]) != key_byte_size(private_key.n):
        raise ValueError("Private key size does not match encrypted PNG metadata.")

    expected_plain_size = plain_block_size(public_from_private(private_key))

    if int(metadata["plain_block_size"]) != expected_plain_size:
        raise ValueError("Private key block size does not match encrypted PNG metadata.")
