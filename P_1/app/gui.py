from __future__ import annotations

import contextlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QPlainTextEdit,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.chunks import (  # noqa: E402
    anonymize_png_chunks,
    describe_chunk,
    display_IHDR_chunks_info,
    load_all_chunks,
)
from src.fft_services import (  # noqa: E402
    Cv2ImageLoader,
    FftVerificationService,
    NumpyRoundTripFftAnalyzer,
)
from src.fft_models import FftAnalysisResult  # noqa: E402


@dataclass(frozen=True)
class PngAnalysis:
    image_path: Path
    anonymized_path: Path
    fft: FftAnalysisResult
    console_output: str


def _unique_anonymized_path(image_path: Path) -> Path:
    candidate = image_path.with_name(f"{image_path.stem}_anon.png")
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = image_path.with_name(f"{image_path.stem}_anon_{index}.png")
        if not candidate.exists():
            return candidate
        index += 1


def analyze_png_file(image_path: str | os.PathLike[str], tolerance: float = 1e-6) -> PngAnalysis:
    path = Path(image_path).expanduser().resolve()
    if path.suffix.lower() != ".png":
        raise ValueError(f"Plik nie jest PNG: {path}")

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        print(f"Plik: {path}")
        print()
        print("Naglowek IHDR:")
        display_IHDR_chunks_info(str(path))
        print()

        service = FftVerificationService(
            loader=Cv2ImageLoader(),
            analyzer=NumpyRoundTripFftAnalyzer(tolerance=tolerance),
        )
        fft_result = service.analyze_image(str(path))
        summary = fft_result.summary
        print("Weryfikacja FFT/IFFT:")
        print(
            f"passed={summary.passed}, tolerance={summary.tolerance:.1e}, "
            f"mse={summary.mse_mean:.6e}, rmse={summary.rmse_mean:.6e}, "
            f"mae={summary.mae_mean:.6e}, max_abs={summary.max_abs_error:.6e}, "
            f"psnr={summary.psnr_mean_db:.6f} dB"
        )
        for metric in fft_result.channel_metrics:
            print(
                f"kanal {metric.channel_index}: mse={metric.mse:.6e}, "
                f"rmse={metric.rmse:.6e}, mae={metric.mae:.6e}, "
                f"max_abs={metric.max_abs_error:.6e}, psnr={metric.psnr_db:.6f} dB"
            )
        print()

        print("Chunki PNG:")
        chunks = load_all_chunks(str(path))
        for chunk in chunks:
            print(describe_chunk(chunk))
        print()

        anonymized_path = _unique_anonymized_path(path)
        report = anonymize_png_chunks(chunks, str(anonymized_path))
        print(
            "Anonimizacja: "
            f"kept={report['kept']}, removed={report['removed']}, "
            f"removed_types={report['removed_types']}, output={anonymized_path}"
        )

    return PngAnalysis(
        image_path=path,
        anonymized_path=anonymized_path,
        fft=fft_result,
        console_output=output.getvalue(),
    )


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(data.shape, dtype=np.uint8)

    clean = np.zeros(data.shape, dtype=np.float64)
    clean[finite] = data[finite]
    min_value = float(clean[finite].min())
    max_value = float(clean[finite].max())
    if max_value <= min_value:
        return np.zeros(data.shape, dtype=np.uint8)

    normalized = (clean - min_value) / (max_value - min_value)
    return np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)


def _hot_colormap(array: np.ndarray) -> np.ndarray:
    values = _normalize_to_uint8(array).astype(np.float32) / 255.0
    red = np.clip(values * 3.0, 0.0, 1.0)
    green = np.clip(values * 3.0 - 1.0, 0.0, 1.0)
    blue = np.clip(values * 3.0 - 2.0, 0.0, 1.0)
    return np.dstack((red, green, blue)).astype(np.float32)


def numpy_to_qimage(array: np.ndarray, force_gray: bool = False, hot: bool = False) -> QImage:
    data = np.asarray(array)
    if hot:
        data = _hot_colormap(data)

    if data.ndim == 2 or force_gray:
        if data.ndim == 3:
            data = np.mean(data[:, :, :3], axis=2)
        image = np.ascontiguousarray(_normalize_to_uint8(data))
        height, width = image.shape
        return QImage(image.data, width, height, width, QImage.Format_Grayscale8).copy()

    if data.ndim == 3:
        if data.shape[2] == 4:
            image = data[:, :, :3]
        else:
            image = data[:, :, :3]

        if image.dtype != np.uint8:
            finite = np.isfinite(image)
            if not finite.any():
                image = np.zeros(image.shape, dtype=np.uint8)
            elif np.nanmax(image[finite]) <= 1.0 and np.nanmin(image[finite]) >= 0.0:
                image = np.clip(np.rint(np.nan_to_num(image) * 255.0), 0, 255).astype(np.uint8)
            else:
                image = _normalize_to_uint8(image)
        image = np.ascontiguousarray(image)
        height, width, _ = image.shape
        return QImage(image.data, width, height, width * 3, QImage.Format_RGB888).copy()

    raise ValueError(f"Nieobslugiwany format tablicy obrazu: {data.shape}")


class AnalysisSignals(QObject):
    result_ready = Signal(object)
    error = Signal(str, str)
    finished = Signal()


class AnalysisWorker(QRunnable):
    def __init__(self, paths: list[str]):
        super().__init__()
        self.paths = paths
        self.signals = AnalysisSignals()

    @Slot()
    def run(self) -> None:
        for path in self.paths:
            try:
                self.signals.result_ready.emit(analyze_png_file(path))
            except Exception as exc:
                self.signals.error.emit(path, str(exc))
        self.signals.finished.emit()


class DropArea(QFrame):
    files_selected = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setObjectName("dropArea")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(18)

        title = QLabel("Przeciagnij pliki PNG do analizy")
        title.setObjectName("dropTitle")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("albo wybierz je z exploratora plikow")
        subtitle.setObjectName("dropSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)

        button = QPushButton("Otworz pliki PNG")
        button.setObjectName("primaryButton")
        button.clicked.connect(self._open_dialog)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(button, alignment=Qt.AlignCenter)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._event_png_paths(event):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = self._event_png_paths(event)
        if paths:
            self.files_selected.emit(paths)
            event.acceptProposedAction()

    def _event_png_paths(self, event: QDragEnterEvent | QDropEvent) -> list[str]:
        urls = event.mimeData().urls()
        paths = [url.toLocalFile() for url in urls if url.isLocalFile()]
        return [path for path in paths if path.lower().endswith(".png")]

    def _open_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Wybierz pliki PNG",
            str(PROJECT_DIR / "data"),
            "PNG (*.png)",
        )
        if paths:
            self.files_selected.emit(paths)


class UploadPage(QWidget):
    files_selected = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 48, 48, 48)
        layout.setAlignment(Qt.AlignCenter)

        heading = QLabel("Analiza plikow PNG")
        heading.setObjectName("pageHeading")
        heading.setAlignment(Qt.AlignCenter)

        self.status = QLabel("")
        self.status.setObjectName("statusText")
        self.status.setAlignment(Qt.AlignCenter)

        drop_area = DropArea()
        drop_area.files_selected.connect(self.files_selected)

        layout.addStretch(1)
        layout.addWidget(heading)
        layout.addWidget(self.status)
        layout.addSpacing(24)
        layout.addWidget(drop_area)
        layout.addStretch(1)

    def set_status(self, text: str) -> None:
        self.status.setText(text)


class PlotPanel(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self._source_pixmap: QPixmap | None = None
        self.setObjectName("plotPanel")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setObjectName("plotTitle")
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(220, 160)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(title_label)
        layout.addWidget(self.image_label, 1)

    def set_image(self, image: QImage) -> None:
        self._source_pixmap = QPixmap.fromImage(image)
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._source_pixmap is None:
            return
        self.image_label.setPixmap(
            self._source_pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class AnalysisPage(QWidget):
    select_more = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.results: list[PngAnalysis | None] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 18)
        root.setSpacing(12)

        top = QHBoxLayout()
        self.title = QLabel("Analiza w toku")
        self.title.setObjectName("analysisTitle")
        self.next_button = QPushButton("Analizuj kolejne zdjecie")
        self.next_button.setObjectName("secondaryButton")
        self.next_button.hide()
        self.next_button.clicked.connect(self.select_more)
        top.addWidget(self.title)
        top.addStretch(1)
        top.addWidget(self.next_button)
        root.addLayout(top)

        content = QHBoxLayout()
        content.setSpacing(12)
        self.file_list = QListWidget()
        self.file_list.setObjectName("fileList")
        self.file_list.setMaximumWidth(280)
        self.file_list.currentRowChanged.connect(self.show_result)
        content.addWidget(self.file_list)

        plots = QWidget()
        plots_layout = QGridLayout(plots)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(10)
        self.original = PlotPanel("Original image")
        self.spectrum = PlotPanel("FFT log-spectrum")
        self.reconstructed = PlotPanel("IFFT reconstructed")
        self.error_map = PlotPanel("Absolute error map")
        plots_layout.addWidget(self.original, 0, 0)
        plots_layout.addWidget(self.spectrum, 0, 1)
        plots_layout.addWidget(self.reconstructed, 1, 0)
        plots_layout.addWidget(self.error_map, 1, 1)
        content.addWidget(plots, 1)
        root.addLayout(content, 4)

        self.console = QPlainTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(170)
        root.addWidget(self.console, 1)

    def reset(self, paths: list[str]) -> None:
        self.results = [None] * len(paths)
        self.file_list.clear()
        self.console.clear()
        self.title.setText(f"Analiza w toku: 0/{len(paths)}")
        self.next_button.hide()
        for path in paths:
            item = QListWidgetItem(Path(path).name)
            item.setToolTip(str(Path(path).expanduser().resolve()))
            self.file_list.addItem(item)

    def add_result(self, result: PngAnalysis) -> None:
        row = self._row_for_path(result.image_path)
        if row < 0:
            row = len(self.results)
            self.results.append(None)
            self.file_list.addItem(QListWidgetItem(result.image_path.name))
        self.results[row] = result
        item = self.file_list.item(row)
        if item is not None:
            item.setText(f"{result.image_path.name}")
            item.setToolTip(str(result.image_path))
        if self.file_list.currentRow() < 0:
            self.file_list.setCurrentRow(row)
        completed = sum(1 for item_result in self.results if item_result is not None)
        self.title.setText(f"Analiza w toku: {completed}/{self.file_list.count()}")
        self.show_result(row)

    def add_error(self, path: str, message: str) -> None:
        row = self._row_for_path(Path(path))
        if row >= 0:
            item = self.file_list.item(row)
            if item is not None:
                item.setText(f"{Path(path).name} - blad")
        self.console.appendPlainText(f"Blad analizy pliku {path}:\n{message}\n")

    def finish(self) -> None:
        self.title.setText("Analiza zakonczona")
        self.next_button.show()

    def show_result(self, row: int) -> None:
        if row < 0 or row >= len(self.results):
            return
        result = self.results[row]
        if result is None:
            return
        fft = result.fft
        is_gray = fft.mode == "gray"
        self.original.set_image(numpy_to_qimage(fft.original, force_gray=is_gray))
        self.spectrum.set_image(numpy_to_qimage(fft.spectrum_log_display, force_gray=is_gray))
        self.reconstructed.set_image(numpy_to_qimage(fft.reconstructed_uint8, force_gray=is_gray))
        err = fft.error_map
        if err.ndim == 3:
            err = np.mean(err, axis=2)
        self.error_map.set_image(numpy_to_qimage(err, hot=True))
        self.console.setPlainText(result.console_output)

    def _row_for_path(self, path: Path) -> int:
        resolved = str(path.expanduser().resolve())
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item and item.toolTip() == resolved:
                return row
        return -1


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("E-media - analiza PNG")
        self.resize(1280, 860)
        self.thread_pool = QThreadPool.globalInstance()
        self._workers: list[AnalysisWorker] = []

        self.stack = QStackedWidget()
        self.upload_page = UploadPage()
        self.analysis_page = AnalysisPage()
        self.upload_page.files_selected.connect(self.start_analysis)
        self.analysis_page.select_more.connect(self.show_upload)
        self.stack.addWidget(self.upload_page)
        self.stack.addWidget(self.analysis_page)
        self.setCentralWidget(self.stack)

    @Slot()
    def show_upload(self) -> None:
        self.stack.setCurrentWidget(self.upload_page)

    @Slot(list)
    def start_analysis(self, paths: list[str]) -> None:
        png_paths = [path for path in paths if path.lower().endswith(".png")]
        if not png_paths:
            self.upload_page.set_status("Wybierz przynajmniej jeden plik PNG.")
            return

        self.upload_page.set_status("")
        self.analysis_page.reset(png_paths)
        self.stack.setCurrentWidget(self.analysis_page)

        worker = AnalysisWorker(png_paths)
        worker.signals.result_ready.connect(self.analysis_page.add_result)
        worker.signals.error.connect(self.analysis_page.add_error)
        worker.signals.finished.connect(self.analysis_page.finish)
        worker.signals.finished.connect(lambda: self._workers.remove(worker) if worker in self._workers else None)
        self._workers.append(worker)
        self.thread_pool.start(worker)


def apply_style(app: QApplication) -> None:
    app.setStyleSheet(
        """
        QWidget {
            background: #f5f7f8;
            color: #1f2933;
            font-family: Inter, Segoe UI, Arial, sans-serif;
            font-size: 14px;
        }
        #pageHeading {
            font-size: 30px;
            font-weight: 700;
        }
        #dropArea {
            background: #ffffff;
            border: 2px dashed #748494;
            border-radius: 8px;
            min-width: 560px;
            min-height: 280px;
        }
        #dropTitle {
            font-size: 22px;
            font-weight: 700;
        }
        #dropSubtitle {
            color: #52616f;
            font-size: 15px;
        }
        #statusText {
            color: #9f1239;
            min-height: 20px;
        }
        QPushButton {
            border: 1px solid #7b8794;
            border-radius: 6px;
            padding: 9px 16px;
            background: #ffffff;
        }
        QPushButton:hover {
            background: #edf2f7;
        }
        #primaryButton {
            background: #146c94;
            color: #ffffff;
            border-color: #146c94;
            font-weight: 700;
        }
        #primaryButton:hover {
            background: #0f5e82;
        }
        #secondaryButton {
            background: #2f6f4e;
            color: #ffffff;
            border-color: #2f6f4e;
            font-weight: 700;
        }
        #secondaryButton:hover {
            background: #285f43;
        }
        #analysisTitle {
            font-size: 20px;
            font-weight: 700;
        }
        #fileList {
            background: #ffffff;
            border: 1px solid #d3dae0;
            border-radius: 6px;
            padding: 4px;
        }
        #plotPanel {
            background: #ffffff;
            border: 1px solid #d3dae0;
            border-radius: 6px;
        }
        #plotTitle {
            color: #2f3a45;
            font-weight: 700;
        }
        #console {
            background: #111827;
            color: #e5e7eb;
            border: 1px solid #273244;
            border-radius: 6px;
            font-family: JetBrains Mono, Consolas, monospace;
            font-size: 12px;
        }
        """
    )


def main() -> int:
    app = QApplication(sys.argv)
    apply_style(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
