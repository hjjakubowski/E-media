from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import io
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

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.fourier_analysis import (  # noqa: E402
    run_fft_self_test,
    save_fourier_spectrum,
    save_image_preview,
)
from src.png_analysis import (  # noqa: E402
    anonymize_png,
    print_ancillary_chunks,
    print_critical_chunks,
    read_png_chunks,
)


DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
IMAGE_SIZE = (320, 220)


class PngAnalyzerGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.image_labels: dict[str, ttk.Label] = {}
        self.image_references: list[tk.PhotoImage] = []

        self.root.title("PNG Analyzer")
        self.root.geometry("1200x850")
        self.root.minsize(900, 650)

        self.build_layout()

    def build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.grid(row=0, column=0, sticky="ew")
        top_frame.columnconfigure(1, weight=1)

        choose_button = ttk.Button(top_frame, text="Wybierz PNG", command=self.choose_file)
        choose_button.grid(row=0, column=0, sticky="w")
        self.choose_button = choose_button

        self.selected_file_label = ttk.Label(top_frame, text="Nie wybrano pliku")
        self.selected_file_label.grid(row=0, column=1, padx=10, sticky="ew")

        images_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        images_frame.grid(row=1, column=0, sticky="nsew")
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        images_frame.rowconfigure(0, weight=1)
        images_frame.rowconfigure(1, weight=1)

        self.add_image_panel(images_frame, "original", "Oryginalny obraz", 0, 0)
        self.add_image_panel(images_frame, "anonymized", "Obraz po anonimizacji", 0, 1)
        self.add_image_panel(images_frame, "magnitude", "Widmo amplitudowe FFT", 1, 0)
        self.add_image_panel(images_frame, "phase", "Widmo fazowe FFT", 1, 1)

        output_frame = ttk.LabelFrame(self.root, text="Informacje z analizy", padding=10)
        output_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="nsew")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        self.output_text = scrolledtext.ScrolledText(output_frame, height=14, wrap="none")
        self.output_text.grid(row=0, column=0, sticky="nsew")
        self.output_text.configure(state="disabled")

        self.status_label = ttk.Label(self.root, text="Gotowe", anchor="w")
        self.status_label.grid(row=3, column=0, padx=10, pady=(0, 8), sticky="ew")

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

        if file_name:
            self.start_analysis(Path(file_name))

    def start_analysis(self, source_path: Path) -> None:
        if not source_path.is_file():
            messagebox.showerror("Blad", f"Plik nie istnieje:\n{source_path}")
            return

        self.selected_file_label.configure(text=str(source_path))
        self.set_status("Analiza w toku...")
        self.set_output_text("Analiza w toku...\n")
        self.choose_button.configure(state="disabled")

        worker = threading.Thread(
            target=self.analyze_in_background,
            args=(source_path,),
            daemon=True,
        )
        worker.start()

    def analyze_in_background(self, source_path: Path) -> None:
        try:
            result = run_analysis(source_path)
        except Exception as error:
            self.root.after(0, self.show_error, error)
            return

        self.root.after(0, self.show_result, result)

    def show_result(self, result: dict[str, object]) -> None:
        self.choose_button.configure(state="normal")
        self.set_status("Gotowe")
        self.set_output_text(str(result["text"]))

        self.show_image("original", Path(result["original_path"]))
        self.show_image("anonymized", Path(result["anonymized_path"]))
        self.show_image("magnitude", Path(result["magnitude_path"]))
        self.show_image("phase", Path(result["phase_path"]))

    def show_error(self, error: Exception) -> None:
        self.choose_button.configure(state="normal")
        self.set_status("Blad analizy")
        self.set_output_text(f"Blad analizy:\n{error}\n")
        messagebox.showerror("Blad analizy", str(error))

    def show_image(self, key: str, image_path: Path) -> None:
        with Image.open(image_path) as image:
            image = ImageOps.contain(image.convert("RGB"), IMAGE_SIZE)
            thumbnail_path = OUTPUT_DIR / f"_gui_{key}.png"
            image.save(thumbnail_path)

        photo = tk.PhotoImage(file=str(thumbnail_path))
        self.image_references.append(photo)
        self.image_labels[key].configure(image=photo, text="")

    def set_output_text(self, text: str) -> None:
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.configure(state="disabled")

    def set_status(self, text: str) -> None:
        self.status_label.configure(text=text)


def run_analysis(source_path: Path) -> dict[str, object]:
    OUTPUT_DIR.mkdir(exist_ok=True)

    output = io.StringIO()

    with redirect_stdout(output):
        chunks = read_png_chunks(source_path)

        print(f"Input file: {source_path}")
        print(f"Output directory: {OUTPUT_DIR}")
        print()

        print_critical_chunks(chunks)
        print()

        print_ancillary_chunks(chunks)
        print()

        preview_path = save_image_preview(source_path, OUTPUT_DIR)
        print(f"Image preview saved to: {preview_path}")

        magnitude_path, phase_path = save_fourier_spectrum(source_path, OUTPUT_DIR)
        print(f"FFT magnitude image saved to: {magnitude_path}")
        print(f"FFT phase image saved to: {phase_path}")

        print()
        run_fft_self_test()

        print()
        anonymized_path = anonymize_png(source_path, chunks, OUTPUT_DIR)
        print(f"Anonymized PNG saved to: {anonymized_path}")

    return {
        "text": output.getvalue(),
        "original_path": preview_path,
        "anonymized_path": anonymized_path,
        "magnitude_path": magnitude_path,
        "phase_path": phase_path,
    }


def create_root() -> tk.Tk:
    return tk.Tk()


def main() -> None:
    root = create_root()
    PngAnalyzerGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
