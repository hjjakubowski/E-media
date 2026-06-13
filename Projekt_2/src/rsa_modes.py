from __future__ import annotations

import secrets

from .rsa_core import (
    RsaPrivateKey,
    RsaPublicKey,
    cipher_block_size,
    decrypt_int,
    encrypt_int,
    key_byte_size,
    plain_block_size,
    public_from_private,
)


MODE_ECB = "ecb"
MODE_CHAIN = "chain"
SUPPORTED_MODES = {MODE_ECB, MODE_CHAIN}


def encrypt_bytes_ecb(data: bytes, public_key: RsaPublicKey) -> bytes:
    plain_size = plain_block_size(public_key)
    cipher_size = cipher_block_size(public_key)
    output = bytearray()

    for block in split_and_pad(data, plain_size):
        message = int.from_bytes(block, "big")
        ciphertext = encrypt_int(message, public_key)
        output.extend(ciphertext.to_bytes(cipher_size, "big"))

    return bytes(output)


def decrypt_bytes_ecb(
    ciphertext: bytes,
    private_key: RsaPrivateKey,
    original_length: int,
) -> bytes:
    public_key = public_from_private(private_key)
    plain_size = plain_block_size(public_key)
    cipher_size = cipher_block_size(public_key)

    if len(ciphertext) % cipher_size != 0:
        raise ValueError("ECB ciphertext length is not a multiple of RSA block size.")

    output = bytearray()

    for offset in range(0, len(ciphertext), cipher_size):
        block = ciphertext[offset : offset + cipher_size]
        decrypted = decrypt_int(int.from_bytes(block, "big"), private_key)
        output.extend(decrypted.to_bytes(plain_size, "big"))

    return bytes(output[:original_length])


def encrypt_bytes_chain(
    data: bytes,
    public_key: RsaPublicKey,
    iv: bytes | None = None,
) -> tuple[bytes, bytes]:
    plain_size = plain_block_size(public_key)
    cipher_size = cipher_block_size(public_key)

    if iv is None:
        iv = secrets.token_bytes(plain_size)

    if len(iv) != plain_size:
        raise ValueError(f"CHAIN IV must have exactly {plain_size} bytes.")

    previous = iv
    output = bytearray()

    for block in split_and_pad(data, plain_size):
        mixed = xor_bytes(block, previous)
        message = int.from_bytes(mixed, "big")
        ciphertext = encrypt_int(message, public_key).to_bytes(cipher_size, "big")
        output.extend(ciphertext)
        previous = ciphertext[-plain_size:]

    return bytes(output), iv


def decrypt_bytes_chain(
    ciphertext: bytes,
    private_key: RsaPrivateKey,
    original_length: int,
    iv: bytes,
) -> bytes:
    public_key = public_from_private(private_key)
    plain_size = plain_block_size(public_key)
    cipher_size = cipher_block_size(public_key)

    if len(iv) != plain_size:
        raise ValueError(f"CHAIN IV must have exactly {plain_size} bytes.")

    if len(ciphertext) % cipher_size != 0:
        raise ValueError("CHAIN ciphertext length is not a multiple of RSA block size.")

    previous = iv
    output = bytearray()

    for offset in range(0, len(ciphertext), cipher_size):
        block = ciphertext[offset : offset + cipher_size]
        mixed_int = decrypt_int(int.from_bytes(block, "big"), private_key)
        mixed = mixed_int.to_bytes(plain_size, "big")
        output.extend(xor_bytes(mixed, previous))
        previous = block[-plain_size:]

    return bytes(output[:original_length])


def encrypt_bytes(
    data: bytes,
    public_key: RsaPublicKey,
    mode: str,
) -> tuple[bytes, bytes]:
    normalized_mode = mode.lower()

    if normalized_mode == MODE_ECB:
        return encrypt_bytes_ecb(data, public_key), b""

    if normalized_mode == MODE_CHAIN:
        return encrypt_bytes_chain(data, public_key)

    raise ValueError(f"Unsupported RSA mode: {mode}")


def decrypt_bytes(
    ciphertext: bytes,
    private_key: RsaPrivateKey,
    original_length: int,
    mode: str,
    iv: bytes = b"",
) -> bytes:
    normalized_mode = mode.lower()

    if normalized_mode == MODE_ECB:
        return decrypt_bytes_ecb(ciphertext, private_key, original_length)

    if normalized_mode == MODE_CHAIN:
        return decrypt_bytes_chain(ciphertext, private_key, original_length, iv)

    raise ValueError(f"Unsupported RSA mode: {mode}")


def encrypted_length(plain_length: int, public_key: RsaPublicKey) -> int:
    plain_size = plain_block_size(public_key)
    cipher_size = key_byte_size(public_key.n)

    if plain_length == 0:
        return cipher_size

    block_count = (plain_length + plain_size - 1) // plain_size
    return block_count * cipher_size


def split_and_pad(data: bytes, block_size: int) -> list[bytes]:
    if block_size <= 0:
        raise ValueError("Block size must be positive.")

    if not data:
        return [bytes(block_size)]

    blocks: list[bytes] = []

    for offset in range(0, len(data), block_size):
        block = data[offset : offset + block_size]
        blocks.append(block.ljust(block_size, b"\x00"))

    return blocks


def xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("Cannot XOR byte strings with different lengths.")

    return bytes(a ^ b for a, b in zip(left, right, strict=True))
