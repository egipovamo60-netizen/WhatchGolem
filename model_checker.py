"""
model_checker.py — Проверка наличия моделей при запуске приложения.

Показывает диалог со статусом каждой модели и кнопками скачивания.
Вызывается из main.py до создания главного окна.

Использование:
    from model_checker import check_and_show
    check_and_show()   # покажет диалог если есть отсутствующие модели
"""

import os
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QScrollArea, QWidget, QFrame, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

_ROOT = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════
#  КАТАЛОГ МОДЕЛЕЙ
# ═══════════════════════════════════════════════════════════

def _p(*parts: str) -> str:
    """Строит абсолютный путь относительно корня проекта."""
    return os.path.join(_ROOT, *parts)


@dataclass
class ModelSpec:
    name: str               # Отображаемое имя
    description: str        # Для какого плагина используется
    path: str               # Путь к файлу/директории для проверки
    size_hint: str          # Человекочитаемый размер ("~4.5 ГБ")
    min_size: int = 0       # Минимальный размер файла в байтах (0 = только существование)
    download_type: str = "" # "url" | "hf_snapshot" | "hf_file" | "trainable"
    download_url: str = ""  # для download_type="url"
    hf_repo: str = ""       # для hf_snapshot / hf_file
    hf_filename: str = ""   # для hf_file
    hf_local_dir: str = ""  # для hf_snapshot / hf_file
    required: bool = True   # Обязательная для базовой работы приложения


MODELS: list[ModelSpec] = [
    ModelSpec(
        name="YOLO v11n",
        description="Детекция объектов — VehicleAnalyzer, PersonAnalyzer, OpenSetAnalyzer",
        path=_p("yolo11n.pt"),
        size_hint="~6 МБ",
        download_type="url",
        download_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        required=True,
    ),
    ModelSpec(
        name="Qwen2-VL-2B-Instruct",
        description="Языковая VL-модель — Qwen2VLAnalyzer",
        path=_p("Models", "Qwen2-VL-2B-Instruct", "config.json"),
        size_hint="~4.5 ГБ",
        download_type="hf_snapshot",
        hf_repo="Qwen/Qwen2-VL-2B-Instruct",
        hf_local_dir=_p("Models", "Qwen2-VL-2B-Instruct"),
        required=False,
    ),
    ModelSpec(
        name="Qwen2.5-VL-3B GGUF (Q4_K_M)",
        description="Квантизованная GGUF-модель — QwenVehicleAnalyzer",
        path=_p("Models", "Qwen2.5-VL-3B-GGUF", "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"),
        size_hint="~2.6 ГБ",
        min_size=1_800_000_000,
        download_type="hf_file",
        hf_repo="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
        hf_filename="Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        hf_local_dir=_p("Models", "Qwen2.5-VL-3B-GGUF"),
        required=False,
    ),
    ModelSpec(
        name="Qwen2.5-VL-3B GGUF (mmproj)",
        description="Мультимодальный проектор — QwenVehicleAnalyzer",
        path=_p("Models", "Qwen2.5-VL-3B-GGUF", "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"),
        size_hint="~900 МБ",
        min_size=800_000_000,
        download_type="hf_file",
        hf_repo="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
        hf_filename="mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
        hf_local_dir=_p("Models", "Qwen2.5-VL-3B-GGUF"),
        required=False,
    ),
    ModelSpec(
        name="Qwen3.5-4B-AWQ",
        description="Квантизованная AWQ-модель — QwenVehicleAnalyzer (требует autoawq + CUDA)",
        path=_p("Models", "Qwen3.5-4B-AWQ", "config.json"),
        size_hint="~2.5 ГБ",
        download_type="hf_snapshot",
        hf_repo="cyankiwi/Qwen3.5-4B-AWQ-4bit",
        hf_local_dir=_p("Models", "Qwen3.5-4B-AWQ"),
        required=False,
    ),
    ModelSpec(
        name="Parking Blocker (.pt)",
        description="Детектор парковочных блокираторов — обучается в разделе «Настройки»",
        path=_p("Models", "640 parking_blocker.pt"),
        size_hint="—",
        download_type="trainable",
        required=False,
    ),
]


def _is_present(spec: ModelSpec) -> bool:
    """Проверяет наличие и минимальный размер модели на диске."""
    if not os.path.exists(spec.path):
        return False
    if spec.min_size > 0 and os.path.isfile(spec.path):
        return os.path.getsize(spec.path) >= spec.min_size
    return True


def any_missing() -> bool:
    """Возвращает True, если хотя бы одна модель отсутствует."""
    return any(not _is_present(s) for s in MODELS)


# ═══════════════════════════════════════════════════════════
#  ПОТОК ЗАГРУЗКИ
# ═══════════════════════════════════════════════════════════

class _DownloadThread(QThread):
    progress    = pyqtSignal(int)   # 0-100; -1 = неопределённый
    status_msg  = pyqtSignal(str)   # текст прогресса
    finished_dl = pyqtSignal(bool, str)  # успех, сообщение об ошибке

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec

    def run(self) -> None:
        try:
            dt = self.spec.download_type
            if dt == "url":
                self._dl_url()
            elif dt == "hf_snapshot":
                self._dl_hf_snapshot()
            elif dt == "hf_file":
                self._dl_hf_file()
            else:
                self.finished_dl.emit(False, "Тип загрузки не поддерживается")
                return
            self.finished_dl.emit(True, "")
        except Exception as exc:
            self.finished_dl.emit(False, str(exc))

    # ── конкретные загрузчики ────────────────────────────────

    def _dl_url(self) -> None:
        url  = self.spec.download_url
        dest = self.spec.path
        dest_dir = os.path.dirname(os.path.abspath(dest))
        os.makedirs(dest_dir, exist_ok=True)

        self.status_msg.emit("Подключение…")

        def _cb(blocks: int, block_size: int, total: int) -> None:
            if total > 0:
                pct = min(100, int(blocks * block_size * 100 / total))
                self.progress.emit(pct)
                done_mb  = blocks * block_size / 1024 / 1024
                total_mb = total / 1024 / 1024
                self.status_msg.emit(f"{done_mb:.1f} / {total_mb:.1f} МБ")

        urllib.request.urlretrieve(url, dest, _cb)
        self.progress.emit(100)

    def _dl_hf_snapshot(self) -> None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise RuntimeError(
                "Установите huggingface_hub:\n  pip install huggingface_hub"
            )
        os.makedirs(self.spec.hf_local_dir, exist_ok=True)
        self.progress.emit(-1)
        self.status_msg.emit(f"Скачивание {self.spec.hf_repo} с HuggingFace…")
        snapshot_download(
            repo_id=self.spec.hf_repo,
            local_dir=self.spec.hf_local_dir,
            ignore_patterns=["*.bin"],
        )
        self.progress.emit(100)

    def _dl_hf_file(self) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError(
                "Установите huggingface_hub:\n  pip install huggingface_hub"
            )
        os.makedirs(self.spec.hf_local_dir, exist_ok=True)
        self.progress.emit(-1)
        self.status_msg.emit(f"Скачивание {self.spec.hf_filename}…")
        hf_hub_download(
            repo_id=self.spec.hf_repo,
            filename=self.spec.hf_filename,
            local_dir=self.spec.hf_local_dir,
        )
        self.progress.emit(100)


# ═══════════════════════════════════════════════════════════
#  ВИДЖЕТ ОДНОЙ СТРОКИ
# ═══════════════════════════════════════════════════════════

class _ModelRow(QFrame):
    download_clicked = pyqtSignal(object)  # передаёт ModelSpec

    def __init__(self, spec: ModelSpec, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        self.setFrameShape(QFrame.StyledPanel)
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 8, 10, 8)
        root_layout.setSpacing(4)

        # ── верхняя строка: имя + описание + размер + статус + кнопка ──
        top = QHBoxLayout()
        top.setSpacing(8)

        info_col = QVBoxLayout()
        info_col.setSpacing(2)
        self._name_lbl = QLabel(self.spec.name)
        self._name_lbl.setFont(QFont("", -1, QFont.Bold))
        self._desc_lbl = QLabel(self.spec.description)
        self._desc_lbl.setStyleSheet("color: gray; font-size: 11px;")
        info_col.addWidget(self._name_lbl)
        info_col.addWidget(self._desc_lbl)
        top.addLayout(info_col, stretch=4)

        self._size_lbl = QLabel(self.spec.size_hint)
        self._size_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._size_lbl.setStyleSheet("color: gray; min-width: 70px;")
        top.addWidget(self._size_lbl)

        self._status_lbl = QLabel()
        self._status_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._status_lbl.setMinimumWidth(130)
        top.addWidget(self._status_lbl)

        self._btn = QPushButton()
        self._btn.setFixedWidth(120)
        self._btn.clicked.connect(lambda: self.download_clicked.emit(self.spec))
        top.addWidget(self._btn)

        root_layout.addLayout(top)

        # ── прогресс-строка (скрыта по умолчанию) ──
        prog_row = QHBoxLayout()
        prog_row.setSpacing(8)
        self._progress = QProgressBar()
        self._progress.setFixedHeight(14)
        self._progress.setTextVisible(False)
        self._progress_lbl = QLabel()
        self._progress_lbl.setStyleSheet("font-size: 11px; color: #555;")
        prog_row.addWidget(self._progress, stretch=3)
        prog_row.addWidget(self._progress_lbl, stretch=2)
        root_layout.addLayout(prog_row)

        self._progress.hide()
        self._progress_lbl.hide()

    # ── публичный API ────────────────────────────────────────

    def refresh(self) -> None:
        """Перечитывает состояние с диска и обновляет UI."""
        present = _is_present(self.spec)
        if present:
            self._status_lbl.setText("✅  Найдена")
            self._status_lbl.setStyleSheet("color: green;")
            self._btn.setVisible(False)
        elif self.spec.download_type == "trainable":
            self._status_lbl.setText("⚠️  Не обучена")
            self._status_lbl.setStyleSheet("color: orange;")
            self._btn.setText("Обучить в настройках")
            self._btn.setEnabled(False)
            self._btn.setVisible(True)
        elif not self.spec.download_type:
            self._status_lbl.setText("❌  Не найдена")
            self._status_lbl.setStyleSheet("color: red;")
            self._btn.setText("—")
            self._btn.setEnabled(False)
            self._btn.setVisible(True)
        else:
            self._status_lbl.setText("❌  Не найдена")
            self._status_lbl.setStyleSheet("color: red;")
            self._btn.setText("Скачать")
            self._btn.setEnabled(True)
            self._btn.setVisible(True)

    def set_downloading(self, active: bool) -> None:
        if active:
            self._progress.setRange(0, 0)   # анимированный «неопределённый» режим
            self._progress.show()
            self._progress_lbl.setText("Подготовка…")
            self._progress_lbl.show()
            self._btn.setEnabled(False)
            self._btn.setText("Загрузка…")
            self._status_lbl.setText("⏳  Загрузка…")
            self._status_lbl.setStyleSheet("color: #2a7dd4;")
        else:
            self._progress.hide()
            self._progress_lbl.hide()

    def on_progress(self, pct: int) -> None:
        if pct < 0:
            self._progress.setRange(0, 0)
        else:
            self._progress.setRange(0, 100)
            self._progress.setValue(pct)

    def on_status(self, msg: str) -> None:
        self._progress_lbl.setText(msg)

    def set_error(self, msg: str) -> None:
        self._status_lbl.setText("❌  Ошибка")
        self._status_lbl.setStyleSheet("color: red;")
        self._progress_lbl.setText(f"Ошибка: {msg[:100]}")
        self._progress_lbl.show()
        self._btn.setEnabled(True)
        self._btn.setText("Повторить")

    def set_btn_enabled(self, enabled: bool) -> None:
        """Используется для блокировки кнопки во время чужой загрузки."""
        if self._btn.text() not in ("Загрузка…", "Обучить в настройках", "—"):
            self._btn.setEnabled(enabled)

    @property
    def is_present(self) -> bool:
        return _is_present(self.spec)

    @property
    def can_download(self) -> bool:
        return self.spec.download_type not in ("trainable", "")


# ═══════════════════════════════════════════════════════════
#  ГЛАВНЫЙ ДИАЛОГ
# ═══════════════════════════════════════════════════════════

class ModelCheckerDialog(QDialog):
    """Диалог проверки и загрузки моделей."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Watch Golem — Проверка моделей")
        self.setMinimumSize(720, 460)
        self.setModal(True)

        self._rows: list[_ModelRow] = []
        self._active_thread: Optional[_DownloadThread] = None
        self._queue: list[ModelSpec] = []

        self._build_ui()

    def _build_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setSpacing(10)
        main.setContentsMargins(16, 16, 16, 12)

        # Заголовок
        title = QLabel("Проверка установленных моделей")
        title.setFont(QFont("", 14, QFont.Bold))
        main.addWidget(title)

        hint = QLabel(
            "Модели, отмеченные ❌, отсутствуют на диске. "
            "Нажмите «Скачать», чтобы загрузить, или «Продолжить» для запуска без них."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        main.addWidget(hint)

        # Список моделей в прокручиваемой области
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        rows_vbox = QVBoxLayout(container)
        rows_vbox.setSpacing(6)
        rows_vbox.setContentsMargins(4, 4, 4, 4)

        for spec in MODELS:
            row = _ModelRow(spec)
            row.download_clicked.connect(self._enqueue)
            rows_vbox.addWidget(row)
            self._rows.append(row)

        rows_vbox.addStretch()
        scroll.setWidget(container)
        main.addWidget(scroll)

        # Нижняя панель
        footer = QHBoxLayout()

        self._dl_all_btn = QPushButton("⬇  Скачать все отсутствующие")
        self._dl_all_btn.clicked.connect(self._download_all_missing)
        footer.addWidget(self._dl_all_btn)

        footer.addStretch()

        continue_btn = QPushButton("Продолжить →")
        continue_btn.setDefault(True)
        continue_btn.setStyleSheet("font-weight: bold; padding: 6px 22px;")
        continue_btn.clicked.connect(self.accept)
        footer.addWidget(continue_btn)

        main.addLayout(footer)

        self._update_dl_all_btn()

    # ── очередь загрузки ─────────────────────────────────────

    def _enqueue(self, spec: ModelSpec) -> None:
        """Добавляет модель в очередь и запускает, если поток свободен."""
        if spec not in self._queue:
            self._queue.append(spec)
        self._process_queue()

    def _download_all_missing(self) -> None:
        for row in self._rows:
            spec = row.spec
            if not row.is_present and row.can_download and spec not in self._queue:
                self._queue.append(spec)
        self._dl_all_btn.setEnabled(False)
        self._process_queue()

    def _process_queue(self) -> None:
        if self._active_thread and self._active_thread.isRunning():
            return
        if not self._queue:
            self._set_all_buttons_enabled(True)
            self._update_dl_all_btn()
            return
        spec = self._queue.pop(0)
        self._start_download(spec)

    def _start_download(self, spec: ModelSpec) -> None:
        row = self._find_row(spec)
        if row is None:
            self._process_queue()
            return

        thread = _DownloadThread(spec)
        thread.progress.connect(row.on_progress)
        thread.status_msg.connect(row.on_status)
        thread.finished_dl.connect(lambda ok, msg, r=row: self._on_done(r, ok, msg))

        self._active_thread = thread
        row.set_downloading(True)
        self._set_all_buttons_enabled(False)
        thread.start()

    def _on_done(self, row: _ModelRow, ok: bool, msg: str) -> None:
        self._active_thread = None
        row.set_downloading(False)
        if ok:
            row.refresh()
        else:
            row.set_error(msg)
        self._process_queue()

    # ── вспомогательные методы ───────────────────────────────

    def _set_all_buttons_enabled(self, enabled: bool) -> None:
        for row in self._rows:
            row.set_btn_enabled(enabled)

    def _update_dl_all_btn(self) -> None:
        has_downloadable = any(
            not r.is_present and r.can_download for r in self._rows
        )
        self._dl_all_btn.setVisible(has_downloadable)
        self._dl_all_btn.setEnabled(has_downloadable)

    def _find_row(self, spec: ModelSpec) -> Optional[_ModelRow]:
        for row in self._rows:
            if row.spec is spec:
                return row
        return None


# ═══════════════════════════════════════════════════════════
#  ПУБЛИЧНЫЙ API
# ═══════════════════════════════════════════════════════════

def check_and_show(parent: Optional[QWidget] = None) -> None:
    """
    Показывает диалог проверки моделей, если хотя бы одна отсутствует.
    Блокирует выполнение до тех пор, пока пользователь не нажмёт «Продолжить».

    Вызывать из main.py после создания QApplication, но до создания главного окна.
    """
    if not any_missing():
        return
    dlg = ModelCheckerDialog(parent)
    dlg.exec_()
