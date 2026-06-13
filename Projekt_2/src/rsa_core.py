from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import secrets


DEFAULT_PUBLIC_EXPONENT = 65537
SMALL_PRIMES = (
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
)


@dataclass(frozen=True)
class RsaPublicKey:
    n: int
    e: int


@dataclass(frozen=True)
class RsaPrivateKey:
    n: int
    d: int
    e: int
    p: int
    q: int


def key_byte_size(n: int) -> int:
    return (n.bit_length() + 7) // 8


def plain_block_size(public_key: RsaPublicKey) -> int:
    return key_byte_size(public_key.n) - 1


def cipher_block_size(public_key: RsaPublicKey) -> int:
    return key_byte_size(public_key.n)


def public_from_private(private_key: RsaPrivateKey) -> RsaPublicKey:
    return RsaPublicKey(n=private_key.n, e=private_key.e)


def generate_keypair(
    bits: int = 512,
    public_exponent: int = DEFAULT_PUBLIC_EXPONENT,
) -> tuple[RsaPublicKey, RsaPrivateKey]:
    if bits < 128:
        raise ValueError("RSA key size must be at least 128 bits.")

    half_bits = bits // 2

    while True:
        p = generate_prime(half_bits)
        q = generate_prime(bits - half_bits)

        if p == q:
            continue

        n = p * q
        phi = (p - 1) * (q - 1)

        if math.gcd(public_exponent, phi) != 1:
            continue

        d = pow(public_exponent, -1, phi)
        public_key = RsaPublicKey(n=n, e=public_exponent)
        private_key = RsaPrivateKey(n=n, d=d, e=public_exponent, p=p, q=q)
        return public_key, private_key


def generate_prime(bits: int) -> int:
    if bits < 8:
        raise ValueError("Prime size must be at least 8 bits.")

    while True:
        candidate = secrets.randbits(bits)
        candidate |= 1
        candidate |= 1 << (bits - 1)

        if is_probable_prime(candidate):
            return candidate


def is_probable_prime(number: int, rounds: int = 16) -> bool:
    if number == 2:
        return True

    if number < 2 or number % 2 == 0:
        return False

    for prime in SMALL_PRIMES:
        if number == prime:
            return True
        if number % prime == 0:
            return False

    odd_part = number - 1
    power_of_two = 0

    while odd_part % 2 == 0:
        odd_part //= 2
        power_of_two += 1

    for _ in range(rounds):
        witness = secrets.randbelow(number - 3) + 2
        value = pow(witness, odd_part, number)

        if value in (1, number - 1):
            continue

        for _ in range(power_of_two - 1):
            value = pow(value, 2, number)

            if value == number - 1:
                break
        else:
            return False

    return True


def encrypt_int(message: int, public_key: RsaPublicKey) -> int:
    if not 0 <= message < public_key.n:
        raise ValueError("RSA message integer must be in range 0 <= m < n.")

    return pow(message, public_key.e, public_key.n)


def decrypt_int(ciphertext: int, private_key: RsaPrivateKey) -> int:
    if not 0 <= ciphertext < private_key.n:
        raise ValueError("RSA ciphertext integer must be in range 0 <= c < n.")

    return pow(ciphertext, private_key.d, private_key.n)


def save_public_key(path: Path, public_key: RsaPublicKey) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"n": str(public_key.n), "e": public_key.e}, indent=2),
        encoding="utf-8",
    )


def save_private_key(path: Path, private_key: RsaPrivateKey) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "n": str(private_key.n),
                "d": str(private_key.d),
                "e": private_key.e,
                "p": str(private_key.p),
                "q": str(private_key.q),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_public_key(path: Path) -> RsaPublicKey:
    data = json.loads(path.read_text(encoding="utf-8"))
    return RsaPublicKey(n=int(data["n"]), e=int(data["e"]))


def load_private_key(path: Path) -> RsaPrivateKey:
    data = json.loads(path.read_text(encoding="utf-8"))
    return RsaPrivateKey(
        n=int(data["n"]),
        d=int(data["d"]),
        e=int(data["e"]),
        p=int(data["p"]),
        q=int(data["q"]),
    )
