from __future__ import annotations

import unittest

from Projekt_2.src.rsa_core import generate_keypair, plain_block_size
from Projekt_2.src.rsa_modes import decrypt_bytes, encrypt_bytes


class RsaModesTest(unittest.TestCase):
    def test_ecb_and_chain_roundtrip(self) -> None:
        public_key, private_key = generate_keypair(256)
        message = (b"PNG RSA mode test " * 20) + b"end"

        for mode in ("ecb", "chain"):
            with self.subTest(mode=mode):
                ciphertext, iv = encrypt_bytes(message, public_key, mode)
                plaintext = decrypt_bytes(
                    ciphertext,
                    private_key,
                    len(message),
                    mode,
                    iv,
                )

                self.assertEqual(message, plaintext)

    def test_ecb_repeats_ciphertext_for_repeated_plaintext_blocks(self) -> None:
        public_key, _ = generate_keypair(256)
        plain_size = plain_block_size(public_key)
        repeated = b"A" * plain_size * 4

        ciphertext, _ = encrypt_bytes(repeated, public_key, "ecb")
        cipher_size = (public_key.n.bit_length() + 7) // 8
        blocks = [
            ciphertext[offset : offset + cipher_size]
            for offset in range(0, len(ciphertext), cipher_size)
        ]

        self.assertEqual(1, len(set(blocks)))


if __name__ == "__main__":
    unittest.main()
