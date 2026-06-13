from __future__ import annotations

from pathlib import Path
import argparse
import sys


if __package__ in (None, ""):
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    from src.png_crypto import decrypt_png, encrypt_png, read_metadata
    from src.png_format import (
        crc_is_valid,
        idat_data,
        is_critical_chunk,
        parse_ihdr,
        read_png,
    )
    from src.rsa_core import (
        generate_keypair,
        load_private_key,
        load_public_key,
        save_private_key,
        save_public_key,
    )
else:
    PROJECT_DIR = Path(__file__).resolve().parents[1]

    from .png_crypto import decrypt_png, encrypt_png, read_metadata
    from .png_format import (
        crc_is_valid,
        idat_data,
        is_critical_chunk,
        parse_ihdr,
        read_png,
    )
    from .rsa_core import (
        generate_keypair,
        load_private_key,
        load_public_key,
        save_private_key,
        save_public_key,
    )


DEFAULT_PUBLIC_KEY = PROJECT_DIR / "data" / "keys" / "public_key.json"
DEFAULT_PRIVATE_KEY = PROJECT_DIR / "data" / "keys" / "private_key.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PNG RSA encryption project. Supports 8-bit color types 0, 2, 6."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    keygen = subparsers.add_parser("keygen", help="generate an RSA key pair")
    keygen.add_argument("--bits", type=int, default=512)
    keygen.add_argument("--public-key", type=Path, default=DEFAULT_PUBLIC_KEY)
    keygen.add_argument("--private-key", type=Path, default=DEFAULT_PRIVATE_KEY)

    encrypt = subparsers.add_parser("encrypt", help="encrypt an 8-bit PNG as 16-bit PNG")
    encrypt.add_argument("--input", type=Path, required=True)
    encrypt.add_argument("--output", type=Path, required=True)
    encrypt.add_argument("--public-key", type=Path, default=DEFAULT_PUBLIC_KEY)
    encrypt.add_argument("--mode", choices=("ecb", "chain"), required=True)

    decrypt = subparsers.add_parser("decrypt", help="decrypt a PNG encrypted by this tool")
    decrypt.add_argument("--input", type=Path, required=True)
    decrypt.add_argument("--output", type=Path, required=True)
    decrypt.add_argument("--private-key", type=Path, default=DEFAULT_PRIVATE_KEY)

    inspect = subparsers.add_parser("inspect", help="print basic PNG structure")
    inspect.add_argument("--input", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "keygen":
        public_key, private_key = generate_keypair(args.bits)
        save_public_key(args.public_key, public_key)
        save_private_key(args.private_key, private_key)
        print(f"Public key saved to: {args.public_key}")
        print(f"Private key saved to: {args.private_key}")
        print(f"Key size: {args.bits} bits")
        return 0

    if args.command == "encrypt":
        public_key = load_public_key(args.public_key)
        encrypt_png(args.input, args.output, public_key, args.mode)
        print(f"Encrypted PNG saved to: {args.output}")
        return 0

    if args.command == "decrypt":
        private_key = load_private_key(args.private_key)
        decrypt_png(args.input, args.output, private_key)
        print(f"Decrypted PNG saved to: {args.output}")
        return 0

    if args.command == "inspect":
        inspect_png(args.input)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def inspect_png(path: Path) -> None:
    chunks = read_png(path)
    ihdr = parse_ihdr(chunks)
    idat_count = sum(1 for chunk in chunks if chunk.chunk_type == "IDAT")

    print(f"File: {path}")
    print(f"Size: {ihdr.width}x{ihdr.height}")
    print(f"Bit depth: {ihdr.bit_depth}")
    print(f"Color type: {ihdr.color_type}")
    print(f"Interlace method: {ihdr.interlace_method}")
    print(f"IDAT chunks: {idat_count}")
    print(f"Compressed IDAT bytes: {len(idat_data(chunks))}")
    print("Chunks:")
    print("  offset  type  length  kind       crc-valid")

    for chunk in chunks:
        kind = "critical" if is_critical_chunk(chunk) else "ancillary"
        print(
            f"  {chunk.offset:6d}  {chunk.chunk_type:4s}  "
            f"{chunk.length:6d}  {kind:9s}  {crc_is_valid(chunk)}"
        )

    try:
        metadata = read_metadata(chunks)
    except ValueError:
        return

    print("RSA metadata:")

    for key, value in metadata.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
