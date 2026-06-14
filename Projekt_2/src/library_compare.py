from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

from .png_format import idat_data, parse_ihdr, read_png
from .png_pixels import decode_pixels, validate_supported_png
from .rsa_core import RsaPrivateKey, plain_block_size, public_from_private
from .rsa_modes import decrypt_bytes_ecb, encrypt_bytes_ecb


@dataclass(frozen=True)
class LibraryCompareResult:
    sample_length: int
    own_cipher_length: int
    library_cipher_length: int
    own_sha256: str
    library_sha256: str
    ciphertexts_equal: bool
    own_roundtrip_ok: bool
    library_roundtrip_ok: bool
    library_is_probabilistic: bool
    explanation: str


def compare_with_library_rsa(
    source_path: Path,
    private_key: RsaPrivateKey,
) -> LibraryCompareResult:
    try:
        from Crypto.Cipher import PKCS1_v1_5
        from Crypto.PublicKey import RSA
        from Crypto.Random import get_random_bytes
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Missing dependency: pycryptodome. Install project dependencies from "
            "requirements.txt before using compare-library."
        ) from error

    public_key = public_from_private(private_key)
    chunks = read_png(source_path)
    ihdr = parse_ihdr(chunks)
    validate_supported_png(ihdr, expected_bit_depth=8)
    pixels = decode_pixels(ihdr, idat_data(chunks))

    key_bytes = (private_key.n.bit_length() + 7) // 8
    library_max_plaintext = key_bytes - 11
    sample_length = min(len(pixels), plain_block_size(public_key), library_max_plaintext)

    if sample_length <= 0:
        raise ValueError("RSA key is too small for PKCS#1 v1.5 library comparison.")

    sample = pixels[:sample_length]
    own_ciphertext = encrypt_bytes_ecb(sample, public_key)
    own_plaintext = decrypt_bytes_ecb(own_ciphertext, private_key, len(sample))

    library_key = RSA.construct(
        (
            private_key.n,
            private_key.e,
            private_key.d,
            private_key.p,
            private_key.q,
        )
    )
    library_cipher = PKCS1_v1_5.new(library_key.publickey())
    library_decipher = PKCS1_v1_5.new(library_key)
    library_ciphertext = library_cipher.encrypt(sample)
    library_ciphertext_second = library_cipher.encrypt(sample)
    sentinel = get_random_bytes(sample_length)
    library_plaintext = library_decipher.decrypt(library_ciphertext, sentinel)

    return LibraryCompareResult(
        sample_length=sample_length,
        own_cipher_length=len(own_ciphertext),
        library_cipher_length=len(library_ciphertext),
        own_sha256=sha256_hex(own_ciphertext),
        library_sha256=sha256_hex(library_ciphertext),
        ciphertexts_equal=own_ciphertext == library_ciphertext,
        own_roundtrip_ok=own_plaintext == sample,
        library_roundtrip_ok=library_plaintext == sample,
        library_is_probabilistic=library_ciphertext != library_ciphertext_second,
        explanation=(
            "The ciphertexts should differ: this project uses raw deterministic RSA "
            "blocks, while the library call uses PKCS#1 v1.5 padding with random bytes."
        ),
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
