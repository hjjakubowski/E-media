from __future__ import annotations

from pathlib import Path
import argparse
import sys


if __package__ in (None, ""):
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    from src.analysis_report import generate_visibility_report
    from src.compression_compare import compare_compression_methods
    from src.library_compare import compare_with_library_rsa
    from src.png_crypto import (
        decrypt_compressed_idat_png,
        decrypt_png,
        encrypt_compressed_idat_png,
        encrypt_png,
        read_metadata,
    )
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

    from .analysis_report import generate_visibility_report
    from .compression_compare import compare_compression_methods
    from .library_compare import compare_with_library_rsa
    from .png_crypto import (
        decrypt_compressed_idat_png,
        decrypt_png,
        encrypt_compressed_idat_png,
        encrypt_png,
        read_metadata,
    )
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
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"


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

    encrypt_compressed = subparsers.add_parser(
        "encrypt-compressed",
        help="encrypt compressed IDAT bytes directly as a 16-bit PNG carrier",
    )
    encrypt_compressed.add_argument("--input", type=Path, required=True)
    encrypt_compressed.add_argument("--output", type=Path, required=True)
    encrypt_compressed.add_argument("--public-key", type=Path, default=DEFAULT_PUBLIC_KEY)
    encrypt_compressed.add_argument("--mode", choices=("ecb", "chain"), required=True)

    decrypt_compressed = subparsers.add_parser(
        "decrypt-compressed",
        help="decrypt a PNG created by encrypt-compressed",
    )
    decrypt_compressed.add_argument("--input", type=Path, required=True)
    decrypt_compressed.add_argument("--output", type=Path, required=True)
    decrypt_compressed.add_argument("--private-key", type=Path, default=DEFAULT_PRIVATE_KEY)

    compare_library = subparsers.add_parser(
        "compare-library",
        help="compare project RSA with pycryptodome RSA using the same key",
    )
    compare_library.add_argument("--input", type=Path, required=True)
    compare_library.add_argument("--private-key", type=Path, default=DEFAULT_PRIVATE_KEY)

    report = subparsers.add_parser(
        "report",
        help="generate ECB vs CHAIN visibility report and output images",
    )
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    report.add_argument("--public-key", type=Path, default=DEFAULT_PUBLIC_KEY)
    report.add_argument("--private-key", type=Path, default=DEFAULT_PRIVATE_KEY)

    compare_compression = subparsers.add_parser(
        "compare-compression",
        help="compare encrypted decompressed pixels with encrypted compressed IDAT",
    )
    compare_compression.add_argument("--input", type=Path, required=True)
    compare_compression.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    compare_compression.add_argument("--public-key", type=Path, default=DEFAULT_PUBLIC_KEY)
    compare_compression.add_argument("--private-key", type=Path, default=DEFAULT_PRIVATE_KEY)
    compare_compression.add_argument("--mode", choices=("ecb", "chain"), default="ecb")

    subparsers.add_parser("gui", help="run the Tkinter GUI")

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

    if args.command == "encrypt-compressed":
        public_key = load_public_key(args.public_key)
        encrypt_compressed_idat_png(args.input, args.output, public_key, args.mode)
        print(f"Compressed-IDAT encrypted PNG saved to: {args.output}")
        return 0

    if args.command == "decrypt-compressed":
        private_key = load_private_key(args.private_key)
        decrypt_compressed_idat_png(args.input, args.output, private_key)
        print(f"Compressed-IDAT decrypted PNG saved to: {args.output}")
        return 0

    if args.command == "compare-library":
        private_key = load_private_key(args.private_key)
        try:
            result = compare_with_library_rsa(args.input, private_key)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1
        print(f"Sample length: {result.sample_length}")
        print(f"Own RSA ciphertext length: {result.own_cipher_length}")
        print(f"Library RSA ciphertext length: {result.library_cipher_length}")
        print(f"Own RSA SHA-256: {result.own_sha256}")
        print(f"Library RSA SHA-256: {result.library_sha256}")
        print(f"Ciphertexts equal: {result.ciphertexts_equal}")
        print(f"Own roundtrip OK: {result.own_roundtrip_ok}")
        print(f"Library roundtrip OK: {result.library_roundtrip_ok}")
        print(f"Library probabilistic: {result.library_is_probabilistic}")
        print(result.explanation)
        return 0

    if args.command == "report":
        public_key = load_public_key(args.public_key)
        private_key = load_private_key(args.private_key)
        result = generate_visibility_report(
            args.input,
            args.output_dir,
            public_key,
            private_key,
        )
        print(f"ECB encrypted PNG: {result.ecb_encrypted_path}")
        print(f"CHAIN encrypted PNG: {result.chain_encrypted_path}")
        print(f"ECB decrypted PNG: {result.ecb_decrypted_path}")
        print(f"CHAIN decrypted PNG: {result.chain_decrypted_path}")
        print(f"Report saved to: {result.report_path}")
        return 0

    if args.command == "compare-compression":
        public_key = load_public_key(args.public_key)
        private_key = load_private_key(args.private_key)
        result = compare_compression_methods(
            args.input,
            args.output_dir,
            public_key,
            private_key,
            args.mode,
        )
        print(f"Pixels encrypted PNG: {result.pixel_encrypted_path}")
        print(f"Compressed-IDAT encrypted PNG: {result.compressed_encrypted_path}")
        print(f"Pixels decrypted PNG: {result.pixel_decrypted_path}")
        print(f"Compressed-IDAT decrypted PNG: {result.compressed_decrypted_path}")
        print(f"Report saved to: {result.report_path}")
        return 0

    if args.command == "gui":
        if __package__ in (None, ""):
            from src.gui import main as gui_main
        else:
            from .gui import main as gui_main

        gui_main()
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
