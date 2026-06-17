from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .png_crypto import (
    decrypt_compressed_idat_png,
    decrypt_png,
    encrypt_compressed_idat_png,
    encrypt_png,
    read_metadata,
)
from .png_format import idat_data, read_png
from .rsa_core import RsaPrivateKey, RsaPublicKey


@dataclass(frozen=True)
class CompressionCompareResult:
    pixel_encrypted_path: Path
    compressed_encrypted_path: Path
    pixel_decrypted_path: Path
    compressed_decrypted_path: Path
    report_path: Path


def compare_compression_methods(source_path: Path, output_dir: Path, public_key: RsaPublicKey, private_key: RsaPrivateKey, mode: str) -> CompressionCompareResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem

    pixel_encrypted_path = output_dir / f"{stem}_{mode}_pixels_encrypted.png"
    compressed_encrypted_path = output_dir / f"{stem}_{mode}_compressed_idat_encrypted.png"
    pixel_decrypted_path = output_dir / f"{stem}_{mode}_pixels_decrypted.png"
    compressed_decrypted_path = output_dir / f"{stem}_{mode}_compressed_idat_decrypted.png"
    report_path = output_dir / f"{stem}_{mode}_compression_compare.txt"

    encrypt_png(source_path, pixel_encrypted_path, public_key, mode)
    encrypt_compressed_idat_png(source_path, compressed_encrypted_path, public_key, mode)
    decrypt_png(pixel_encrypted_path, pixel_decrypted_path, private_key)
    decrypt_compressed_idat_png(
        compressed_encrypted_path,
        compressed_decrypted_path,
        private_key,
    )

    source_idat_len = len(idat_data(read_png(source_path)))
    pixel_metadata = read_metadata(read_png(pixel_encrypted_path))
    compressed_metadata = read_metadata(read_png(compressed_encrypted_path))

    report_path.write_text(
        build_compression_report_text(
            source_path,
            pixel_encrypted_path,
            compressed_encrypted_path,
            source_idat_len,
            int(pixel_metadata["plain_length"]),
            int(pixel_metadata["cipher_length"]),
            int(compressed_metadata["plain_length"]),
            int(compressed_metadata["cipher_length"]),
        ),
        encoding="utf-8",
    )

    return CompressionCompareResult(
        pixel_encrypted_path=pixel_encrypted_path,
        compressed_encrypted_path=compressed_encrypted_path,
        pixel_decrypted_path=pixel_decrypted_path,
        compressed_decrypted_path=compressed_decrypted_path,
        report_path=report_path,
    )


def build_compression_report_text(source_path: Path, pixel_encrypted_path: Path, compressed_encrypted_path: Path, source_idat_len: int, pixel_plain_len: int, pixel_cipher_len: int, compressed_plain_len: int, compressed_cipher_len: int) -> str:
    return "\n".join(
        [
            "PNG compression method comparison",
            f"Source: {source_path}",
            f"Decompressed-pixels encrypted PNG: {pixel_encrypted_path}",
            f"Compressed-IDAT encrypted PNG: {compressed_encrypted_path}",
            "",
            f"Original compressed IDAT bytes: {source_idat_len}",
            f"Decompressed pixel payload bytes: {pixel_plain_len}",
            f"Decompressed pixel RSA ciphertext bytes: {pixel_cipher_len}",
            f"Compressed IDAT payload bytes: {compressed_plain_len}",
            f"Compressed IDAT RSA ciphertext bytes: {compressed_cipher_len}",
            "",
            "Conclusion:",
            (
                "The methods are not equivalent. Encrypting decompressed pixels "
                "changes the image sample bytes and then creates a new valid zlib "
                "stream. Encrypting compressed IDAT treats the zlib stream itself "
                "as the RSA payload and restores that exact compressed stream after "
                "decryption."
            ),
        ]
    )
