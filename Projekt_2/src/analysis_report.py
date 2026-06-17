from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .png_crypto import decrypt_png, encrypt_png, read_metadata
from .png_format import idat_data, parse_ihdr, read_png
from .png_pixels import decode_pixels, validate_supported_png
from .rsa_core import RsaPrivateKey, RsaPublicKey, key_byte_size, plain_block_size


@dataclass(frozen=True)
class BlockStats:
    block_size: int
    total_blocks: int
    unique_blocks: int
    repeated_blocks: int
    repeated_percent: float


@dataclass(frozen=True)
class VisibilityReportResult:
    ecb_encrypted_path: Path
    chain_encrypted_path: Path
    ecb_decrypted_path: Path
    chain_decrypted_path: Path
    report_path: Path


def generate_visibility_report(source_path: Path, output_dir: Path, public_key: RsaPublicKey, private_key: RsaPrivateKey) -> VisibilityReportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem

    ecb_encrypted_path = output_dir / f"{stem}_ecb_encrypted.png"
    chain_encrypted_path = output_dir / f"{stem}_chain_encrypted.png"
    ecb_decrypted_path = output_dir / f"{stem}_ecb_decrypted.png"
    chain_decrypted_path = output_dir / f"{stem}_chain_decrypted.png"
    report_path = output_dir / f"{stem}_visibility_report.txt"

    encrypt_png(source_path, ecb_encrypted_path, public_key, "ecb")
    encrypt_png(source_path, chain_encrypted_path, public_key, "chain")
    decrypt_png(ecb_encrypted_path, ecb_decrypted_path, private_key)
    decrypt_png(chain_encrypted_path, chain_decrypted_path, private_key)

    source_pixels = read_pixels(source_path)
    source_stats = count_repeated_blocks(source_pixels, plain_block_size(public_key))
    ecb_stats = encrypted_ciphertext_stats(ecb_encrypted_path, public_key)
    chain_stats = encrypted_ciphertext_stats(chain_encrypted_path, public_key)

    report_path.write_text(
        build_visibility_report_text(
            source_path,
            ecb_encrypted_path,
            chain_encrypted_path,
            source_stats,
            ecb_stats,
            chain_stats,
        ),
        encoding="utf-8",
    )

    return VisibilityReportResult(
        ecb_encrypted_path=ecb_encrypted_path,
        chain_encrypted_path=chain_encrypted_path,
        ecb_decrypted_path=ecb_decrypted_path,
        chain_decrypted_path=chain_decrypted_path,
        report_path=report_path,
    )


def read_pixels(path: Path) -> bytes:
    chunks = read_png(path)
    ihdr = parse_ihdr(chunks)
    validate_supported_png(ihdr, expected_bit_depth=8)
    return decode_pixels(ihdr, idat_data(chunks))


def encrypted_ciphertext_stats(path: Path, public_key: RsaPublicKey) -> BlockStats:
    chunks = read_png(path)
    ihdr = parse_ihdr(chunks)
    metadata = read_metadata(chunks)
    carrier_pixels = decode_pixels(ihdr, idat_data(chunks))
    cipher_length = int(metadata["cipher_length"])
    ciphertext = carrier_pixels[:cipher_length]
    return count_repeated_blocks(ciphertext, key_byte_size(public_key.n))


def count_repeated_blocks(data: bytes, block_size: int) -> BlockStats:
    if block_size <= 0:
        raise ValueError("Block size must be positive.")

    blocks = [
        data[offset : offset + block_size]
        for offset in range(0, len(data), block_size)
    ]
    total_blocks = len(blocks)
    unique_blocks = len(set(blocks))
    repeated_blocks = total_blocks - unique_blocks
    repeated_percent = (
        (repeated_blocks / total_blocks) * 100.0 if total_blocks else 0.0
    )

    return BlockStats(
        block_size=block_size,
        total_blocks=total_blocks,
        unique_blocks=unique_blocks,
        repeated_blocks=repeated_blocks,
        repeated_percent=repeated_percent,
    )


def build_visibility_report_text(source_path: Path, ecb_encrypted_path: Path, chain_encrypted_path: Path, source_stats: BlockStats, ecb_stats: BlockStats, chain_stats: BlockStats) -> str:
    return "\n".join(
        [
            "PNG RSA visibility report",
            f"Source: {source_path}",
            f"ECB encrypted PNG: {ecb_encrypted_path}",
            f"CHAIN encrypted PNG: {chain_encrypted_path}",
            "",
            format_stats("Plain pixel blocks", source_stats),
            format_stats("ECB ciphertext blocks", ecb_stats),
            format_stats("CHAIN ciphertext blocks", chain_stats),
            "",
            "Conclusion:",
            (
                "ECB is deterministic: equal input blocks produce equal ciphertext "
                "blocks, so repeated visual structures can remain easier to detect."
            ),
            (
                "CHAIN mixes each block with the previous ciphertext block and IV, "
                "so repeated blocks are reduced and the image content is harder to infer."
            ),
        ]
    )


def format_stats(label: str, stats: BlockStats) -> str:
    return (
        f"{label}: block_size={stats.block_size}, "
        f"total={stats.total_blocks}, unique={stats.unique_blocks}, "
        f"repeated={stats.repeated_blocks}, "
        f"repeated_percent={stats.repeated_percent:.2f}%"
    )
