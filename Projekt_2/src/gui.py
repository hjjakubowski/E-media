from __future__ import annotations

from pathlib import Path
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    from PIL import Image, ImageOps
except ModuleNotFoundError as error:
    missing_name = error.name
    raise SystemExit(
        f"Missing dependency: {missing_name}. "
        "Install project dependencies with: python3 -m pip install -r requirements.txt"
    ) from error


PROJECT_DIR = Path(__file__).resolve().parents[1]

if __package__ in (None, "") and str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

if __package__ in (None, ""):
    from src.analysis_report import generate_visibility_report  # noqa: E402
    from src.compression_compare import compare_compression_methods  # noqa: E402
    from src.library_compare import compare_with_library_rsa  # noqa: E402
    from src.png_crypto import (  # noqa: E402
        decrypt_compressed_idat_png,
        decrypt_png,
        encrypt_compressed_idat_png,
        encrypt_png,
    )
    from src.rsa_core import (  # noqa: E402
        generate_keypair,
        load_private_key,
        load_public_key,
        save_private_key,
        save_public_key,
    )
else:
    from .analysis_report import generate_visibility_report
    from .compression_compare import compare_compression_methods
    from .library_compare import compare_with_library_rsa
    from .png_crypto import (
        decrypt_compressed_idat_png,
        decrypt_png,
        encrypt_compressed_idat_png,
        encrypt_png,
    )
    from .rsa_core import (
        generate_keypair,
        load_private_key,
        load_public_key,
        save_private_key,
        save_public_key,
    )


DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
PUBLIC_KEY = DATA_DIR / "keys" / "public_key.json"
PRIVATE_KEY = DATA_DIR / "keys" / "private_key.json"
IMAGE_SIZE = (320, 220)


class PngRsaGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.selected_png: Path | None = None
        self.image_labels: dict[str, ttk.Label] = {}
        self.image_references: list[tk.PhotoImage] = []
        self.buttons: list[ttk.Button] = []

        self.root.title("PNG RSA")
        self.root.geometry("1200x850")
        self.root.minsize(950, 700)

        self.build_layout()

    def build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.grid(row=0, column=0, sticky="ew")
        top_frame.columnconfigure(1, weight=1)

        choose_button = self.add_button(top_frame, "Wybierz PNG", self.choose_file)
        choose_button.grid(row=0, column=0, sticky="w")

        self.selected_file_label = ttk.Label(top_frame, text="Nie wybrano pliku")
        self.selected_file_label.grid(row=0, column=1, padx=10, sticky="ew")

        actions_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        actions_frame.grid(row=1, column=0, sticky="ew")

        actions = [
            ("Generuj klucze", self.generate_keys),
            ("Szyfruj ECB", lambda: self.encrypt_selected("ecb")),
            ("Szyfruj CHAIN", lambda: self.encrypt_selected("chain")),
            ("Deszyfruj", self.decrypt_selected),
            ("Szyfruj IDAT", lambda: self.encrypt_compressed_selected("ecb")),
            ("Deszyfruj IDAT", self.decrypt_compressed_selected),
            ("Porównaj kompresję", self.run_compression_compare),
            ("Raport ECB/CHAIN", self.run_visibility_report),
            ("Porównaj bibliotekę", self.run_library_compare),
        ]

        for index, (text, command) in enumerate(actions):
            button = self.add_button(actions_frame, text, command)
            button.grid(row=index // 5, column=index % 5, padx=4, pady=4, sticky="ew")
            actions_frame.columnconfigure(index % 5, weight=1)

        images_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        images_frame.grid(row=2, column=0, sticky="nsew")
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        images_frame.rowconfigure(0, weight=1)
        images_frame.rowconfigure(1, weight=1)

        self.add_image_panel(images_frame, "original", "Oryginał", 0, 0)
        self.add_image_panel(images_frame, "ecb", "ECB", 0, 1)
        self.add_image_panel(images_frame, "chain", "CHAIN", 1, 0)
        self.add_image_panel(images_frame, "decrypted", "Odszyfrowany", 1, 1)

        output_frame = ttk.LabelFrame(self.root, text="Log", padding=10)
        output_frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="nsew")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            height=13,
            wrap="word",
        )
        self.output_text.grid(row=0, column=0, sticky="nsew")
        self.output_text.configure(state="disabled")

        self.status_label = ttk.Label(self.root, text="Gotowe", anchor="w")
        self.status_label.grid(row=4, column=0, padx=10, pady=(0, 8), sticky="ew")

    def add_button(
        self,
        parent: ttk.Frame,
        text: str,
        command: object,
    ) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command)
        self.buttons.append(button)
        return button

    def add_image_panel(
        self,
        parent: ttk.Frame,
        key: str,
        title: str,
        row: int,
        column: int,
    ) -> None:
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.grid(row=row, column=column, padx=5, pady=5, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        label = ttk.Label(frame, text="Brak obrazu", anchor="center")
        label.grid(row=0, column=0, sticky="nsew")
        self.image_labels[key] = label

    def choose_file(self) -> None:
        file_name = filedialog.askopenfilename(
            title="Wybierz plik PNG",
            initialdir=DATA_DIR,
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )

        if not file_name:
            return

        self.selected_png = Path(file_name)
        self.selected_file_label.configure(text=str(self.selected_png))
        self.show_image("original", self.selected_png)
        self.set_output_text(f"Wybrano plik:\n{self.selected_png}\n")
        self.set_status("Gotowe")

    def generate_keys(self) -> None:
        def task() -> dict[str, object]:
            public_key, private_key = generate_keypair(512)
            save_public_key(PUBLIC_KEY, public_key)
            save_private_key(PRIVATE_KEY, private_key)
            return {
                "text": (
                    "Wygenerowano klucze RSA 512 bit.\n"
                    f"Klucz publiczny: {PUBLIC_KEY}\n"
                    f"Klucz prywatny: {PRIVATE_KEY}\n"
                )
            }

        self.run_in_background("Generowanie kluczy...", task, self.show_text_result)

    def encrypt_selected(self, mode: str) -> None:
        source_path = self.require_selected_png()

        if source_path is None:
            return

        def task() -> dict[str, object]:
            public_key = load_public_key(PUBLIC_KEY)
            output_path = OUTPUT_DIR / f"{source_path.stem}_{mode}_encrypted.png"
            encrypt_png(source_path, output_path, public_key, mode)
            return {
                "text": f"Zaszyfrowano obraz trybem {mode.upper()}:\n{output_path}\n",
                "image_key": mode,
                "image_path": output_path,
            }

        self.run_in_background(f"Szyfrowanie {mode.upper()}...", task, self.show_image_result)

    def encrypt_compressed_selected(self, mode: str) -> None:
        source_path = self.require_selected_png()

        if source_path is None:
            return

        def task() -> dict[str, object]:
            public_key = load_public_key(PUBLIC_KEY)
            output_path = OUTPUT_DIR / f"{source_path.stem}_{mode}_compressed_idat.png"
            encrypt_compressed_idat_png(source_path, output_path, public_key, mode)
            return {
                "text": (
                    "Zaszyfrowano bezpośrednio skompresowany IDAT "
                    f"trybem {mode.upper()}:\n{output_path}\n"
                ),
                "image_key": "ecb",
                "image_path": output_path,
            }

        self.run_in_background("Szyfrowanie skompresowanego IDAT...", task, self.show_image_result)

    def decrypt_selected(self) -> None:
        file_name = filedialog.askopenfilename(
            title="Wybierz zaszyfrowany PNG",
            initialdir=OUTPUT_DIR,
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )

        if not file_name:
            return

        encrypted_path = Path(file_name)

        def task() -> dict[str, object]:
            private_key = load_private_key(PRIVATE_KEY)
            output_path = OUTPUT_DIR / f"{encrypted_path.stem}_decrypted.png"
            decrypt_png(encrypted_path, output_path, private_key)
            return {
                "text": f"Odszyfrowano PNG:\n{output_path}\n",
                "image_key": "decrypted",
                "image_path": output_path,
            }

        self.run_in_background("Deszyfrowanie...", task, self.show_image_result)

    def decrypt_compressed_selected(self) -> None:
        file_name = filedialog.askopenfilename(
            title="Wybierz PNG z zaszyfrowanym IDAT",
            initialdir=OUTPUT_DIR,
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )

        if not file_name:
            return

        encrypted_path = Path(file_name)

        def task() -> dict[str, object]:
            private_key = load_private_key(PRIVATE_KEY)
            output_path = OUTPUT_DIR / f"{encrypted_path.stem}_decrypted.png"
            decrypt_compressed_idat_png(encrypted_path, output_path, private_key)
            return {
                "text": f"Odszyfrowano wariant compressed-IDAT:\n{output_path}\n",
                "image_key": "decrypted",
                "image_path": output_path,
            }

        self.run_in_background("Deszyfrowanie skompresowanego IDAT...", task, self.show_image_result)

    def run_visibility_report(self) -> None:
        source_path = self.require_selected_png()

        if source_path is None:
            return

        def task() -> dict[str, object]:
            public_key = load_public_key(PUBLIC_KEY)
            private_key = load_private_key(PRIVATE_KEY)
            result = generate_visibility_report(
                source_path,
                OUTPUT_DIR,
                public_key,
                private_key,
            )
            return {
                "text": result.report_path.read_text(encoding="utf-8"),
                "ecb": result.ecb_encrypted_path,
                "chain": result.chain_encrypted_path,
                "decrypted": result.ecb_decrypted_path,
            }

        self.run_in_background("Generowanie raportu ECB/CHAIN...", task, self.show_report_result)

    def run_compression_compare(self) -> None:
        source_path = self.require_selected_png()

        if source_path is None:
            return

        def task() -> dict[str, object]:
            public_key = load_public_key(PUBLIC_KEY)
            private_key = load_private_key(PRIVATE_KEY)
            result = compare_compression_methods(
                source_path,
                OUTPUT_DIR,
                public_key,
                private_key,
                "ecb",
            )
            return {
                "text": result.report_path.read_text(encoding="utf-8"),
                "ecb": result.pixel_encrypted_path,
                "chain": result.compressed_encrypted_path,
                "decrypted": result.pixel_decrypted_path,
            }

        self.run_in_background("Porównywanie metod kompresji...", task, self.show_report_result)

    def run_library_compare(self) -> None:
        source_path = self.require_selected_png()

        if source_path is None:
            return

        def task() -> dict[str, object]:
            private_key = load_private_key(PRIVATE_KEY)
            result = compare_with_library_rsa(source_path, private_key)
            return {
                "text": (
                    f"Sample length: {result.sample_length}\n"
                    f"Own RSA ciphertext length: {result.own_cipher_length}\n"
                    f"Library RSA ciphertext length: {result.library_cipher_length}\n"
                    f"Own RSA SHA-256: {result.own_sha256}\n"
                    f"Library RSA SHA-256: {result.library_sha256}\n"
                    f"Ciphertexts equal: {result.ciphertexts_equal}\n"
                    f"Own roundtrip OK: {result.own_roundtrip_ok}\n"
                    f"Library roundtrip OK: {result.library_roundtrip_ok}\n"
                    f"Library probabilistic: {result.library_is_probabilistic}\n"
                    f"{result.explanation}\n"
                )
            }

        self.run_in_background("Porównywanie z biblioteką RSA...", task, self.show_text_result)

    def run_in_background(
        self,
        status: str,
        task: object,
        on_success: object,
    ) -> None:
        self.set_status(status)
        self.set_buttons_state("disabled")
        self.set_output_text(status + "\n")

        def worker() -> None:
            try:
                result = task()
            except Exception as error:
                self.root.after(0, self.show_error, error)
                return

            self.root.after(0, on_success, result)

        threading.Thread(target=worker, daemon=True).start()

    def show_text_result(self, result: dict[str, object]) -> None:
        self.set_buttons_state("normal")
        self.set_status("Gotowe")
        self.set_output_text(str(result["text"]))

    def show_image_result(self, result: dict[str, object]) -> None:
        self.show_text_result(result)
        self.show_image(str(result["image_key"]), Path(result["image_path"]))

    def show_report_result(self, result: dict[str, object]) -> None:
        self.set_buttons_state("normal")
        self.set_status("Gotowe")
        self.set_output_text(str(result["text"]))

        for key in ("ecb", "chain", "decrypted"):
            if key in result:
                self.show_image(key, Path(result[key]))

    def show_error(self, error: Exception) -> None:
        self.set_buttons_state("normal")
        self.set_status("Błąd")
        self.set_output_text(f"Błąd:\n{error}\n")
        messagebox.showerror("Błąd", str(error))

    def show_image(self, key: str, image_path: Path) -> None:
        OUTPUT_DIR.mkdir(exist_ok=True)

        with Image.open(image_path) as image:
            image = ImageOps.contain(image.convert("RGB"), IMAGE_SIZE)
            thumbnail_path = OUTPUT_DIR / f"_gui_{key}.png"
            image.save(thumbnail_path)

        photo = tk.PhotoImage(file=str(thumbnail_path))
        self.image_references.append(photo)
        self.image_labels[key].configure(image=photo, text="")

    def require_selected_png(self) -> Path | None:
        if self.selected_png is None:
            messagebox.showerror("Błąd", "Najpierw wybierz plik PNG.")
            return None

        if not self.selected_png.is_file():
            messagebox.showerror("Błąd", f"Plik nie istnieje:\n{self.selected_png}")
            return None

        return self.selected_png

    def set_output_text(self, text: str) -> None:
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.configure(state="disabled")

    def set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def set_buttons_state(self, state: str) -> None:
        for button in self.buttons:
            button.configure(state=state)


def create_root() -> tk.Tk:
    return tk.Tk()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    root = create_root()
    PngRsaGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
