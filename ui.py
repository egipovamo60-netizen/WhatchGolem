import os
import time
import subprocess
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QCheckBox, QMessageBox,
    QSlider, QHBoxLayout, QProgressDialog, QGroupBox, QDoubleSpinBox, QComboBox,
    QGridLayout, QTabWidget, QListWidget, QPlainTextEdit, QDialog, QDialogButtonBox)
from PyQt5.QtCore import QTimer, QThread, Qt, pyqtSignal
from video_processing import (
    process_videos_in_folder, process_single_video, 
    process_single_image, process_images_in_folder
)
from duplicate_detector import detect_and_remove_duplicates
from image_grouper import group_images_by_similarity
from Plugins.VehicleAnalyzer import analyze_vehicle, analyze_vehicle_folder, generate_report
from Plugins.QwenVehicleAnalyzer import analyze_vehicle_qwen, analyze_vehicle_folder_qwen, generate_report_qwen, generate_report_word_qwen
from Plugins.Qwen2VLAnalyzer import analyze_vehicle_qwen2vl, analyze_vehicle_folder_qwen2vl, generate_report_qwen2vl, generate_report_word_qwen2vl
from Plugins.OpenALPRAnalyzer import analyze_vehicle_openalpr, analyze_vehicle_folder_openalpr, generate_report_openalpr
from Plugins.PersonAnalyzer import analyze_person, analyze_person_folder, generate_report_person, generate_report_word_person, reload_face_database
from Plugins.PhoneAnalyzer import analyze_phone, analyze_phone_folder, generate_report_phone, generate_report_word_phone
from Plugins.OpenSetAnalyzer import analyze_open_set, analyze_open_set_video, analyze_open_set_multi_video, generate_report_open_set, generate_report_word_open_set
from Plugins.UnknownClassMerger import analyze_unknown_class_folder, consolidate_unknown_class_database
from Plugins.ModelInspector import (
    get_model_info, format_model_info, detect_image, detect_image_tiled,
    format_detections, unload_model, save_detection_stats, save_detection_stats_batch,
)


class TrainBlockerWorker(QThread):
    """
    Запускает тренировку как отдельный subprocess (изолирован от Qt/CUDA GUI-процесса).
    Читает stdout построчно и транслирует в лог.
    """
    log_message  = pyqtSignal(str)
    train_done   = pyqtSignal(str)
    train_error  = pyqtSignal(str)

    def __init__(self, source_dir, epochs, imgsz, batch, weights, output_path,
                 overwrite_dataset, device, use_synthetic=False, n_train=2000, n_val=300,
                 patience=50, data_yaml=None, class_name="parking_blocker", parent=None):
        super().__init__(parent)
        self.source_dir        = source_dir
        self.epochs            = epochs
        self.imgsz             = imgsz
        self.batch             = batch
        self.weights           = weights
        self.output_path       = output_path
        self.overwrite_dataset = overwrite_dataset
        self.device            = device
        self.use_synthetic     = use_synthetic
        self.n_train           = n_train
        self.n_val             = n_val
        self.patience          = patience
        self.data_yaml         = data_yaml
        self.class_name        = class_name
        self._process          = None
        self._cancelled        = False

    def cancel(self):
        self._cancelled = True
        if self._process and self._process.poll() is None:
            self._process.terminate()

    def run(self):
        import sys, json as _json
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Аргументы передаём через временный JSON-файл
        args_file = os.path.join(base_dir, "_train_blocker_args.json")
        args_data = {
            "source_dir":        self.source_dir,
            "epochs":            self.epochs,
            "imgsz":             self.imgsz,
            "batch":             self.batch,
            "weights":           self.weights or "",
            "output_path":       self.output_path,
            "overwrite_dataset": self.overwrite_dataset,
            "device":            self.device,
            "use_synthetic":     self.use_synthetic,
            "n_train":           self.n_train,
            "n_val":             self.n_val,
            "patience":          self.patience,
            "data_yaml":         self.data_yaml or "",
            "class_name":        self.class_name,
        }
        try:
            with open(args_file, "w", encoding="utf-8") as f:
                _json.dump(args_data, f)

            script = os.path.join(base_dir, "_run_train_blocker.py")
            cmd = [sys.executable, script, args_file]

            env = {
                **os.environ,
                "PYTHONUNBUFFERED":     "1",
                "KMP_DUPLICATE_LIB_OK": "TRUE",
                "OMP_NUM_THREADS":      "1",
                "MKL_NUM_THREADS":      "1",
            }
            # Блокируем CUDA только если пользователь явно выбрал CPU
            if str(self.device).lower() == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = "-1"
                env["NO_CUDA"] = "1"
                env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,          # line-buffered
                cwd=base_dir,
                env=env,
            )

            # readline() вместо итератора — не накапливает внутренний чанк-буфер
            for line in iter(self._process.stdout.readline, ""):
                line = line.rstrip()
                if line:
                    self.log_message.emit(line)
                if self._cancelled:
                    break

            self._process.wait()
            rc = self._process.returncode

            if self._cancelled:
                self.log_message.emit("Тренировка отменена.")
                return

            if rc != 0:
                self.train_error.emit(f"Процесс завершился с кодом {rc}")
                return

            self.train_done.emit(self.output_path)

        except Exception as exc:
            self.train_error.emit(str(exc))
        finally:
            try:
                os.remove(args_file)
            except Exception:
                pass


class VideoTestWorker(QThread):
    """Фоновый поток: прогон YOLO-модели по видеофайлу, сохранение результата."""
    log_message  = pyqtSignal(str)
    test_done    = pyqtSignal(str)   # путь к выходному видео
    test_error   = pyqtSignal(str)
    progress     = pyqtSignal(int, int)  # (текущий_кадр, всего_кадров)
    stats_ready  = pyqtSignal(list)      # список записей статистики по кадрам

    def __init__(self, model_path: str, video_path: str, output_path: str,
                 conf: float = 0.25, iou: float = 0.45,
                 imgsz: int = 640, skip_frames: int = 0,
                 parent=None):
        super().__init__(parent)
        self.model_path  = model_path
        self.video_path  = video_path
        self.output_path = output_path
        self.conf        = conf
        self.iou         = iou
        self.imgsz       = imgsz
        self.skip_frames = skip_frames
        self._cancelled  = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            import cv2 as _cv2
            from ultralytics import YOLO as _YOLO

            model = _YOLO(self.model_path)
            cap   = _cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.test_error.emit(f"Не удалось открыть видео: {self.video_path}")
                return

            fps    = cap.get(_cv2.CAP_PROP_FPS) or 25.0
            width  = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
            total  = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))

            fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
            out    = _cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

            frame_idx  = 0
            written    = 0
            det_total  = 0
            skip       = max(0, self.skip_frames)
            _stat_entries: list = []
            import datetime as _dt

            self.log_message.emit(
                f"Видео: {width}×{height} @ {fps:.1f} fps, кадров: {total}"
            )

            while True:
                if self._cancelled:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Пропуск кадров для ускорения (дублируем предыдущий результат)
                if skip > 0 and (frame_idx - 1) % (skip + 1) != 0:
                    out.write(frame)
                    written += 1
                    continue

                results = model(
                    frame,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    verbose=False,
                )
                annotated = results[0].plot()
                out.write(annotated)
                written += 1

                n_det = len(results[0].boxes) if results[0].boxes is not None else 0
                det_total += n_det

                # накапливаем статистику для текущего кадра
                _confs = []
                if results[0].boxes is not None:
                    for _box in results[0].boxes:
                        _confs.append(round(float(_box.conf[0]), 3))
                _stat_entries.append({
                    "ts":           _dt.datetime.now().isoformat(timespec="seconds"),
                    "model":        os.path.basename(self.model_path),
                    "image":        f"{os.path.basename(self.video_path)}#frame{frame_idx}",
                    "frame_w":      int(width),
                    "frame_h":      int(height),
                    "n_detections": n_det,
                    "confs":        _confs,
                })

                if frame_idx % 50 == 0 or frame_idx == total:
                    self.progress.emit(frame_idx, total)
                    self.log_message.emit(
                        f"  Кадр {frame_idx}/{total}  детекций в кадре: {n_det}"
                    )

            cap.release()
            out.release()

            if self._cancelled:
                self.log_message.emit("Прервано пользователем.")
                if _stat_entries:
                    self.stats_ready.emit(_stat_entries)
                return

            self.log_message.emit(
                f"Готово. Кадров записано: {written}, "
                f"суммарных детекций: {det_total}"
            )
            if _stat_entries:
                self.stats_ready.emit(_stat_entries)
            self.test_done.emit(self.output_path)

        except Exception as exc:
            import traceback
            self.test_error.emit(traceback.format_exc())


class AugmentCropsWorker(QThread):
    log_message  = pyqtSignal(str)
    aug_done     = pyqtSignal(int)   # итоговое число файлов
    aug_error    = pyqtSignal(str)

    def __init__(self, crops_dir: str, target_count: int, parent=None):
        super().__init__(parent)
        self.crops_dir    = crops_dir
        self.target_count = target_count

    def run(self):
        try:
            import sys as _sys
            base_dir = os.path.dirname(os.path.abspath(__file__))
            _sys.path.insert(0, base_dir)
            from Plugins.ParkingBlockerTrainer.synthetic_dataset_generator import augment_crops
            total = augment_crops(
                crops_dir    = self.crops_dir,
                target_count = self.target_count,
                log_callback = lambda msg: self.log_message.emit(msg),
            )
            self.aug_done.emit(total)
        except Exception as exc:
            self.aug_error.emit(str(exc))


class PhoneAnalysisWorker(QThread):
    """Фоновый поток для анализа мобильных телефонов."""
    progress_update = pyqtSignal(str, int, int)
    analysis_done = pyqtSignal(list)
    analysis_error = pyqtSignal(str)

    def __init__(self, file_path=None, folder_path=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.folder_path = folder_path
        self._cancelled = False
        self.elapsed_seconds = 0

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            _t0 = time.time()
            if self.file_path:
                self.progress_update.emit("Загрузка моделей...", 0, 4)

                def step_cb(step_name, step_num):
                    if self._cancelled:
                        return False
                    self.progress_update.emit(step_name, step_num, 4)
                    return True

                results = [analyze_phone(self.file_path, progress_callback=step_cb)]
                if not self._cancelled:
                    self.progress_update.emit("Готово", 4, 4)

            elif self.folder_path:
                extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
                files = sorted(
                    f for f in os.listdir(self.folder_path)
                    if f.lower().endswith(extensions)
                )
                if not files:
                    self.analysis_done.emit([])
                    return

                total_steps = len(files) * 4 + 1
                self.progress_update.emit(f"Подготовка (0/{len(files)})...", 0, total_steps)
                results = []

                for idx, fname in enumerate(files):
                    if self._cancelled:
                        break
                    base_step = idx * 4

                    def step_cb(step_name, step_num, _base=base_step, _idx=idx,
                                _total=len(files), _fname=fname):
                        if self._cancelled:
                            return False
                        self.progress_update.emit(
                            f"[{_idx + 1}/{_total}] {_fname}\n{step_name}",
                            _base + step_num, total_steps
                        )
                        return True

                    full_path = os.path.join(self.folder_path, fname)
                    result = analyze_phone(full_path, progress_callback=step_cb)
                    results.append(result)

                if not self._cancelled:
                    self.progress_update.emit("Формирование отчёта...", total_steps, total_steps)
            else:
                results = []

            self.elapsed_seconds = time.time() - _t0
            if not self._cancelled:
                self.analysis_done.emit(results)

        except Exception as e:
            self.analysis_error.emit(str(e))


class OpenSetAnalysisWorker(QThread):
    """Фоновый поток для open-set детекции объектов."""
    progress_update = pyqtSignal(str, int, int)
    analysis_done = pyqtSignal(list)
    analysis_error = pyqtSignal(str)

    def __init__(self, file_path=None, folder_path=None, sample_interval_seconds=1.0,
                 analysis_options=None, multi_video_output_dir=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.folder_path = folder_path
        self.multi_video_output_dir = multi_video_output_dir
        self.sample_interval_seconds = sample_interval_seconds
        self.analysis_options = dict(analysis_options or {})
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v')

            if self.file_path:
                self.progress_update.emit("Подготовка open-set анализа...", 0, 4)

                def step_cb(step_name, step_num, maximum=None):
                    if self._cancelled:
                        return False
                    self.progress_update.emit(step_name, step_num, maximum or 4)
                    return True

                lower_path = self.file_path.lower()
                if lower_path.endswith(video_extensions):
                    results = [
                        analyze_open_set_video(
                            self.file_path,
                            progress_callback=step_cb,
                            sample_interval_seconds=self.sample_interval_seconds,
                            **self.analysis_options,
                        )
                    ]
                else:
                    results = [
                        analyze_open_set(
                            self.file_path,
                            progress_callback=step_cb,
                            **self.analysis_options,
                        )
                    ]
                if not self._cancelled:
                    self.progress_update.emit("Готово", 4, 4)

            elif self.folder_path:
                files = sorted(
                    f for f in os.listdir(self.folder_path)
                    if f.lower().endswith(image_extensions + video_extensions)
                )
                if not files:
                    self.analysis_done.emit([])
                    return

                all_paths = [os.path.join(self.folder_path, f) for f in files]
                total_phases = len(files) * 4 + 2

                def step_cb(step_name, step_num, maximum=None):
                    if self._cancelled:
                        return False
                    self.progress_update.emit(step_name, step_num, maximum or total_phases)
                    return True

                result = analyze_open_set_multi_video(
                    all_paths,
                    self.multi_video_output_dir,
                    progress_callback=step_cb,
                    sample_interval_seconds=self.sample_interval_seconds,
                    **self.analysis_options,
                )
                if not self._cancelled:
                    self.progress_update.emit("Формирование отчёта...", total_phases, total_phases)
                    results = [result]
            else:
                results = []

            if not self._cancelled:
                self.analysis_done.emit(results)

        except Exception as e:
            self.analysis_error.emit(str(e))


class UnknownClassMergeWorker(QThread):
    """Фоновый поток для пополнения единой базы UnknownClass."""
    progress_update = pyqtSignal(str, int, int)
    analysis_done = pyqtSignal(dict)
    analysis_error = pyqtSignal(str)

    def __init__(self, folder_path=None, similarity_threshold=0.88, parent=None):
        super().__init__(parent)
        self.folder_path = folder_path
        self.similarity_threshold = float(similarity_threshold)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if not self.folder_path:
                self.analysis_done.emit({})
                return

            def step_cb(step_name, step_num):
                if self._cancelled:
                    return False
                self.progress_update.emit(step_name, step_num, 100)
                return True

            result = analyze_unknown_class_folder(
                self.folder_path,
                similarity_threshold=self.similarity_threshold,
                progress_callback=step_cb,
            )

            if not self._cancelled:
                self.analysis_done.emit(result)

        except Exception as e:
            self.analysis_error.emit(str(e))


class ConsolidateUnknownClassWorker(QThread):
    """Фоновый поток для объединения похожих классов в базе UnknownClassDB."""
    progress_update = pyqtSignal(str, int, int)
    analysis_done = pyqtSignal(dict)
    analysis_error = pyqtSignal(str)

    def __init__(self, similarity_threshold=0.88, db_root=None, parent=None):
        super().__init__(parent)
        self.similarity_threshold = float(similarity_threshold)
        self.db_root = db_root
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            def step_cb(step_name, step_num):
                if self._cancelled:
                    return False
                self.progress_update.emit(step_name, step_num, 100)
                return True

            result = consolidate_unknown_class_database(
                similarity_threshold=self.similarity_threshold,
                db_root=self.db_root,
                progress_callback=step_cb,
            )
            if not self._cancelled:
                self.analysis_done.emit(result)
        except Exception as e:
            self.analysis_error.emit(str(e))


class PersonAnalysisWorker(QThread):
    """Фоновый поток для анализа людей (пол, одежда, лицо)."""
    progress_update = pyqtSignal(str, int, int)
    analysis_done = pyqtSignal(list)
    analysis_error = pyqtSignal(str)

    def __init__(self, file_path=None, folder_path=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.folder_path = folder_path
        self._cancelled = False
        self.elapsed_seconds = 0

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            _t0 = time.time()
            if self.file_path:
                self.progress_update.emit("Загрузка моделей...", 0, 4)

                def step_cb(step_name, step_num):
                    if self._cancelled:
                        return False
                    self.progress_update.emit(step_name, step_num, 4)
                    return True

                results = [analyze_person(self.file_path, progress_callback=step_cb)]
                if not self._cancelled:
                    self.progress_update.emit("Готово", 4, 4)

            elif self.folder_path:
                extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
                files = sorted(
                    f for f in os.listdir(self.folder_path)
                    if f.lower().endswith(extensions)
                )
                if not files:
                    self.analysis_done.emit([])
                    return

                total_steps = len(files) * 4 + 1
                self.progress_update.emit(f"Подготовка (0/{len(files)})...", 0, total_steps)
                results = []

                for idx, fname in enumerate(files):
                    if self._cancelled:
                        break
                    base_step = idx * 4

                    def step_cb(step_name, step_num, _base=base_step, _idx=idx,
                                _total=len(files), _fname=fname):
                        if self._cancelled:
                            return False
                        self.progress_update.emit(
                            f"[{_idx + 1}/{_total}] {_fname}\n{step_name}",
                            _base + step_num, total_steps
                        )
                        return True

                    full_path = os.path.join(self.folder_path, fname)
                    result = analyze_person(full_path, progress_callback=step_cb)
                    results.append(result)

                if not self._cancelled:
                    self.progress_update.emit("Формирование отчёта...", total_steps, total_steps)
            else:
                results = []

            self.elapsed_seconds = time.time() - _t0
            if not self._cancelled:
                self.analysis_done.emit(results)

        except Exception as e:
            self.analysis_error.emit(str(e))


class QwenVehicleAnalysisWorker(QThread):
    """Фоновый поток для анализа транспорта через Qwen2.5-VL."""
    progress_update = pyqtSignal(str, int, int)
    analysis_done = pyqtSignal(list)
    analysis_error = pyqtSignal(str)

    def __init__(self, file_path=None, folder_path=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.folder_path = folder_path
        self._cancelled = False
        self.elapsed_seconds = 0

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            _t0 = time.time()
            if self.file_path:
                self.progress_update.emit("Загрузка Qwen2.5-VL...", 0, 4)

                def step_cb(step_name, step_num):
                    if self._cancelled:
                        return False
                    self.progress_update.emit(step_name, step_num, 4)
                    return True

                results = [analyze_vehicle_qwen(self.file_path, progress_callback=step_cb)]
                if not self._cancelled:
                    self.progress_update.emit("Готово", 4, 4)

            elif self.folder_path:
                extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
                files = sorted(
                    f for f in os.listdir(self.folder_path)
                    if f.lower().endswith(extensions)
                )
                if not files:
                    self.analysis_done.emit([])
                    return

                total_steps = len(files) * 4 + 1
                self.progress_update.emit(f"Подготовка (0/{len(files)})...", 0, total_steps)
                results = []

                for idx, fname in enumerate(files):
                    if self._cancelled:
                        break
                    base_step = idx * 4

                    def step_cb(step_name, step_num, _base=base_step, _idx=idx,
                                _total=len(files), _fname=fname):
                        if self._cancelled:
                            return False
                        self.progress_update.emit(
                            f"[{_idx + 1}/{_total}] {_fname}\n{step_name}",
                            _base + step_num, total_steps
                        )
                        return True

                    full_path = os.path.join(self.folder_path, fname)
                    result = analyze_vehicle_qwen(full_path, progress_callback=step_cb)
                    results.append(result)

                if not self._cancelled:
                    self.progress_update.emit("Формирование отчёта...", total_steps, total_steps)
            else:
                results = []

            self.elapsed_seconds = time.time() - _t0
            if not self._cancelled:
                self.analysis_done.emit(results)

        except Exception as e:
            self.analysis_error.emit(str(e))


class Qwen2VLAnalysisWorker(QThread):
    """Фоновый поток для анализа транспорта через Qwen2-VL-2B."""
    progress_update = pyqtSignal(str, int, int)
    analysis_done = pyqtSignal(list)
    analysis_error = pyqtSignal(str)

    def __init__(self, file_path=None, folder_path=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.folder_path = folder_path
        self._cancelled = False
        self.elapsed_seconds = 0

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            _t0 = time.time()
            if self.file_path:
                self.progress_update.emit("Загрузка Qwen2-VL-2B...", 0, 4)

                def step_cb(step_name, step_num):
                    if self._cancelled:
                        return False
                    self.progress_update.emit(step_name, step_num, 4)
                    return True

                results = [analyze_vehicle_qwen2vl(self.file_path, progress_callback=step_cb)]
                if not self._cancelled:
                    self.progress_update.emit("Готово", 4, 4)

            elif self.folder_path:
                extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
                files = sorted(
                    f for f in os.listdir(self.folder_path)
                    if f.lower().endswith(extensions)
                )
                if not files:
                    self.analysis_done.emit([])
                    return

                total_steps = len(files) * 4 + 1
                self.progress_update.emit(f"Подготовка (0/{len(files)})...", 0, total_steps)
                results = []

                for idx, fname in enumerate(files):
                    if self._cancelled:
                        break
                    base_step = idx * 4

                    def step_cb(step_name, step_num, _base=base_step, _idx=idx,
                                _total=len(files), _fname=fname):
                        if self._cancelled:
                            return False
                        self.progress_update.emit(
                            f"[{_idx + 1}/{_total}] {_fname}\n{step_name}",
                            _base + step_num, total_steps
                        )
                        return True

                    full_path = os.path.join(self.folder_path, fname)
                    result = analyze_vehicle_qwen2vl(full_path, progress_callback=step_cb)
                    results.append(result)

                if not self._cancelled:
                    self.progress_update.emit("Формирование отчёта...", total_steps, total_steps)
            else:
                results = []

            self.elapsed_seconds = time.time() - _t0
            if not self._cancelled:
                self.analysis_done.emit(results)

        except Exception as e:
            self.analysis_error.emit(str(e))


class OpenALPRAnalysisWorker(QThread):
    """Фоновый поток для анализа транспорта через OpenALPR (локально, без интернета)."""
    progress_update = pyqtSignal(str, int, int)
    analysis_done = pyqtSignal(list)
    analysis_error = pyqtSignal(str)

    def __init__(self, file_path=None, folder_path=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.folder_path = folder_path
        self._cancelled = False
        self.elapsed_seconds = 0

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            _t0 = time.time()
            if self.file_path:
                self.progress_update.emit("Подготовка OpenALPR...", 0, 4)

                def step_cb(step_name, step_num):
                    if self._cancelled:
                        return False
                    self.progress_update.emit(step_name, step_num, 4)
                    return True

                results = [analyze_vehicle_openalpr(self.file_path, progress_callback=step_cb)]
                if not self._cancelled:
                    self.progress_update.emit("Готово", 4, 4)

            elif self.folder_path:
                extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
                files = sorted(
                    f for f in os.listdir(self.folder_path)
                    if f.lower().endswith(extensions)
                )
                if not files:
                    self.analysis_done.emit([])
                    return

                total_steps = len(files) * 4 + 1
                self.progress_update.emit(f"Подготовка (0/{len(files)})...", 0, total_steps)
                results = []

                for idx, fname in enumerate(files):
                    if self._cancelled:
                        break
                    base_step = idx * 4

                    def step_cb(step_name, step_num, _base=base_step, _idx=idx,
                                _total=len(files), _fname=fname):
                        if self._cancelled:
                            return False
                        self.progress_update.emit(
                            f"[{_idx + 1}/{_total}] {_fname}\n{step_name}",
                            _base + step_num, total_steps
                        )
                        return True

                    full_path = os.path.join(self.folder_path, fname)
                    result = analyze_vehicle_openalpr(full_path, progress_callback=step_cb)
                    results.append(result)

                if not self._cancelled:
                    self.progress_update.emit("Формирование отчёта...", total_steps, total_steps)
            else:
                results = []

            self.elapsed_seconds = time.time() - _t0
            if not self._cancelled:
                self.analysis_done.emit(results)

        except Exception as e:
            self.analysis_error.emit(str(e))


class VehicleAnalysisWorker(QThread):
    """Фоновый поток для анализа транспорта."""
    progress_update = pyqtSignal(str, int, int)   # label, value, maximum
    analysis_done = pyqtSignal(list)               # results
    analysis_error = pyqtSignal(str)               # error message

    def __init__(self, file_path=None, folder_path=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.folder_path = folder_path
        self._cancelled = False
        self.elapsed_seconds = 0

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            _t0 = time.time()
            if self.file_path:
                self.progress_update.emit("Загрузка модели...", 0, 4)

                def step_cb(step_name, step_num):
                    if self._cancelled:
                        return False
                    self.progress_update.emit(step_name, step_num, 4)
                    return True

                results = [analyze_vehicle(self.file_path, progress_callback=step_cb)]
                if not self._cancelled:
                    self.progress_update.emit("Готово", 4, 4)

            elif self.folder_path:
                extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
                files = sorted(
                    f for f in os.listdir(self.folder_path)
                    if f.lower().endswith(extensions)
                )
                if not files:
                    self.analysis_done.emit([])
                    return

                total_steps = len(files) * 3 + 1
                self.progress_update.emit(f"Подготовка (0/{len(files)})...", 0, total_steps)
                results = []

                for idx, fname in enumerate(files):
                    if self._cancelled:
                        break
                    base_step = idx * 3

                    def step_cb(step_name, step_num, _base=base_step, _idx=idx,
                                _total=len(files), _fname=fname):
                        if self._cancelled:
                            return False
                        self.progress_update.emit(
                            f"[{_idx + 1}/{_total}] {_fname}\n{step_name}",
                            _base + step_num, total_steps
                        )
                        return True

                    full_path = os.path.join(self.folder_path, fname)
                    result = analyze_vehicle(full_path, progress_callback=step_cb)
                    results.append(result)

                if not self._cancelled:
                    self.progress_update.emit("Формирование отчёта...", total_steps, total_steps)
            else:
                results = []

            self.elapsed_seconds = time.time() - _t0

        except Exception as e:
            self.analysis_error.emit(str(e))


class VideoObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Watch Golem")
        self.setGeometry(50, 50, 420, 780)

        self.folder_path = None
        self.file_path = None
        self.file_type = None  # 'video' или 'image'
        self.duplicates_folder = None  # Папка для поиска дубликатов
        self.grouping_folder = None  # Папка для группировки изображений
        self.vehicle_folder = None  # Папка/файл для анализа транспорта
        self.vehicle_file = None
        self.person_folder = None   # Папка/файл для анализа людей
        self.person_file = None
        self.phone_folder = None    # Папка/файл для анализа телефонов
        self.phone_file = None
        self.openset_folder = None  # Папка/файл для open-set анализа
        self.openset_file = None
        self.openset_sample_interval_seconds = 1.0
        self.openset_unknown_crop_margin_scale = 0.12
        self.openset_unknown_dir = None
        self.openset_unknown_classes_dir = None
        self.openset_unknown_class_similarity_threshold = 0.88
        self.openset_use_hdbscan = False
        self.unknown_merge_folder = None
        self.unknown_merge_similarity_threshold = 0.88
        self.unknown_merge_output_dir = None
        self.unknown_merge_classes_dir = None
        self.unknown_merge_report_path = None
        self.unknown_merge_json_report_path = None
        self.unknown_merge_word_report_path = None
        self.unknown_consolidate_threshold = 0.92
        self.openset_presets = {
            "Стандартный": {
                "det_confidence": 0.15,
                "known_similarity_threshold": 0.24,
                "strong_similarity_threshold": 0.30,
                "margin_threshold": 0.015,
                "yolo_support_confidence": 0.35,
                "generic_top_k": 12,
                "max_proposals": 24,
            },
            "Чувствительный": {
                "det_confidence": 0.10,
                "known_similarity_threshold": 0.29,
                "strong_similarity_threshold": 0.34,
                "margin_threshold": 0.022,
                "yolo_support_confidence": 0.45,
                "generic_top_k": 18,
                "max_proposals": 36,
            },
            "Агрессивный": {
                "det_confidence": 0.08,
                "known_similarity_threshold": 0.32,
                "strong_similarity_threshold": 0.38,
                "margin_threshold": 0.030,
                "yolo_support_confidence": 0.55,
                "generic_top_k": 24,
                "max_proposals": 48,
            },
            "Максимальный": {
                "det_confidence": 0.05,
                "known_similarity_threshold": 0.36,
                "strong_similarity_threshold": 0.43,
                "margin_threshold": 0.040,
                "yolo_support_confidence": 0.65,
                "generic_top_k": 32,
                "max_proposals": 64,
            },
        }
        self.openset_preset_name = "Стандартный"
        self.openset_analysis_options = self._get_open_set_analysis_options(self.openset_preset_name)
        
        # Параметры чувствительности для обнаружения дубликатов
        self.hamming_threshold = 10  # Расстояние Хэмминга (0-64, меньше = строже)
        self.ssim_threshold = 0.75   # SSIM порог (0-1, больше = строже)
        self.confidence_threshold = 0.9  # Порог уверенности для детекции (0-1, больше = строже)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_files)

        self.tracking_objects = {}  # Словарь для отслеживаемых объектов
        self.trail_points = {}  # Следы объектов

        self.log_file = "object_detection_log.txt"  # Имя файла для сохранения логов

        # Очищаем лог-файл при запуске
        with open(self.log_file, "w", encoding='utf-8') as f:
            f.write("Обнаруженные объекты:\n")

        self.init_ui()

    def _create_group_box(self, title):
        box = QGroupBox(title)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(8)
        box.setLayout(layout)
        return box, layout

    def _create_button_row(self, *buttons):
        layout = QHBoxLayout()
        layout.setSpacing(8)
        for button in buttons:
            layout.addWidget(button)
        return layout

    def _create_status_label(self, text):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #5b6570;")
        label.setMinimumHeight(24)
        return label

    def _get_open_set_analysis_options(self, preset_name):
        options = dict(self.openset_presets[preset_name])
        options["unknown_crop_margin_scale"] = self.openset_unknown_crop_margin_scale
        options["unknown_class_similarity_threshold"] = self.openset_unknown_class_similarity_threshold
        options["use_hdbscan"] = self.openset_use_hdbscan
        return options

    def _format_open_set_preset_details(self, preset_name):
        preset = self.openset_presets[preset_name]
        return (
            f"Режим: {preset_name}\n"
            f"YOLO confidence: {preset['det_confidence']:.2f}\n"
            f"Known similarity: {preset['known_similarity_threshold']:.3f}\n"
            f"Strong similarity: {preset['strong_similarity_threshold']:.3f}\n"
            f"Margin: {preset['margin_threshold']:.3f}\n"
            f"YOLO support confidence: {preset['yolo_support_confidence']:.2f}\n"
            f"Generic proposals: {preset['generic_top_k']}\n"
            f"Max proposals: {preset['max_proposals']}\n"
            f"Unknown crop margin: {self.openset_unknown_crop_margin_scale:.2f}\n"
            f"Unknown class similarity: {self.openset_unknown_class_similarity_threshold:.2f}"
        )

    def _update_open_set_preset_details(self):
        if hasattr(self, 'openset_preset_details_label'):
            self.openset_preset_details_label.setText(
                self._format_open_set_preset_details(self.openset_preset_name)
            )

    def _set_open_set_unknown_dir(self, folder_path):
        self.openset_unknown_dir = folder_path if folder_path and os.path.isdir(folder_path) else None
        if hasattr(self, 'btn_open_openset_unknown_folder'):
            self.btn_open_openset_unknown_folder.setEnabled(bool(self.openset_unknown_dir))

    def _set_open_set_unknown_classes_dir(self, folder_path):
        self.openset_unknown_classes_dir = folder_path if folder_path and os.path.isdir(folder_path) else None
        if hasattr(self, 'btn_open_openset_unknown_classes_folder'):
            self.btn_open_openset_unknown_classes_folder.setEnabled(bool(self.openset_unknown_classes_dir))

    def _set_unknown_merge_output_dir(self, folder_path):
        self.unknown_merge_output_dir = folder_path if folder_path and os.path.isdir(folder_path) else None
        if hasattr(self, 'btn_open_unknown_merge_output_folder'):
            self.btn_open_unknown_merge_output_folder.setEnabled(bool(self.unknown_merge_output_dir))

    def _set_unknown_merge_classes_dir(self, folder_path):
        self.unknown_merge_classes_dir = folder_path if folder_path and os.path.isdir(folder_path) else None
        if hasattr(self, 'btn_open_unknown_merge_classes_folder'):
            self.btn_open_unknown_merge_classes_folder.setEnabled(bool(self.unknown_merge_classes_dir))

    def _clear_unknown_merge_preview(self, message=None):
        if hasattr(self, 'unknown_merge_preview_list'):
            self.unknown_merge_preview_list.clear()
        if hasattr(self, 'unknown_merge_preview_label'):
            self.unknown_merge_preview_label.setText(message or "База UnknownClass ещё не сформирована")

    def _update_unknown_merge_preview(self, result):
        if not hasattr(self, 'unknown_merge_preview_list'):
            return

        self.unknown_merge_preview_list.clear()
        clusters = list(result.get("clusters", [])) if result else []
        if not clusters:
            self._clear_unknown_merge_preview("Классы базы UnknownClass не найдены")
            return

        for cluster in clusters:
            item_text = (
                f"{cluster.get('class_name', 'MergedUnknownClass')}"
                f" | {cluster.get('size', 0)} изображ."
                f" | sources: {len(cluster.get('source_dirs', []))}"
            )
            self.unknown_merge_preview_list.addItem(item_text)

        self.unknown_merge_preview_label.setText(
            f"Классов в базе: {len(clusters)} | Всего изображений: {result.get('total_image_count', 0)} | Порог: {result.get('similarity_threshold', self.unknown_merge_similarity_threshold):.2f}"
        )

    def _open_folder_in_explorer(self, folder_path, title):
        if not folder_path or not os.path.isdir(folder_path):
            QMessageBox.warning(self, "Ошибка", f"{title} недоступна")
            return False

        try:
            if hasattr(os, "startfile"):
                os.startfile(folder_path)
            elif os.name == "posix":
                subprocess.Popen(["xdg-open", folder_path])
            else:
                subprocess.Popen(["open", folder_path])
            return True
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть {title}:\n{str(e)}")
            return False

    def _build_detection_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        source_box, source_layout = self._create_group_box("Источник")

        self.btn_open = QPushButton("Папка")
        self.btn_open.clicked.connect(self.open_folder)
        self.btn_open_file = QPushButton("Видеофайл")
        self.btn_open_file.clicked.connect(self.open_video_file)
        self.btn_open_image = QPushButton("Изображение")
        self.btn_open_image.clicked.connect(self.open_image_file)

        source_layout.addLayout(
            self._create_button_row(
                self.btn_open,
                self.btn_open_file,
                self.btn_open_image,
            )
        )

        action_box, action_layout = self._create_group_box("Запуск")

        self.btn_start = QPushButton("Анализ папки")
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_start.setEnabled(False)

        self.btn_start_file = QPushButton("Анализ файла")
        self.btn_start_file.clicked.connect(self.start_analysis_file)
        self.btn_start_file.setEnabled(False)

        self.btn_stop = QPushButton("Стоп")
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.btn_stop.setEnabled(False)

        action_layout.addLayout(
            self._create_button_row(
                self.btn_start,
                self.btn_start_file,
                self.btn_stop,
            )
        )

        self.show_video_checkbox = QCheckBox("Показывать видео во время обработки")
        self.show_video_checkbox.setChecked(False)
        action_layout.addWidget(self.show_video_checkbox)

        detection_box, detection_layout = self._create_group_box("Точность детекции")
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Порог:"))

        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(10)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(90)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        confidence_layout.addWidget(self.confidence_slider)

        self.confidence_value_label = QLabel("0.90")
        self.confidence_value_label.setMinimumWidth(36)
        confidence_layout.addWidget(self.confidence_value_label)

        detection_layout.addLayout(confidence_layout)
        detection_hint = QLabel("0.10-0.30 мягко | 0.30-0.60 средне | 0.60-1.00 строго")
        detection_hint.setStyleSheet("color: gray; font-size: 10px;")
        detection_layout.addWidget(detection_hint)

        preview_box, preview_layout = self._create_group_box("Предпросмотр")
        self.label = QLabel("Выберите папку, видео или изображение для анализа")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)
        self.label.setMinimumHeight(260)
        self.label.setStyleSheet(
            "background: #f4f6f8; border: 1px solid #d5d9de; border-radius: 6px; color: #667085;"
        )
        preview_layout.addWidget(self.label)

        layout.addWidget(source_box)
        layout.addWidget(action_box)
        layout.addWidget(detection_box)
        layout.addWidget(preview_box, 1)
        return tab

    def _build_duplicates_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        params_box, params_layout = self._create_group_box("Параметры сравнения")
        params_grid = QGridLayout()
        params_grid.setHorizontalSpacing(8)
        params_grid.setVerticalSpacing(6)

        params_grid.addWidget(QLabel("Хэмминг:"), 0, 0)
        self.hamming_slider = QSlider(Qt.Horizontal)
        self.hamming_slider.setMinimum(5)
        self.hamming_slider.setMaximum(25)
        self.hamming_slider.setValue(10)
        self.hamming_slider.setTickPosition(QSlider.TicksBelow)
        self.hamming_slider.setTickInterval(5)
        self.hamming_slider.valueChanged.connect(self.on_hamming_changed)
        params_grid.addWidget(self.hamming_slider, 0, 1)

        self.hamming_value_label = QLabel("10")
        self.hamming_value_label.setMinimumWidth(30)
        params_grid.addWidget(self.hamming_value_label, 0, 2)

        hamming_info = QLabel("5-8 строго | 8-12 средне | 12-25 мягко")
        hamming_info.setStyleSheet("color: gray; font-size: 10px;")
        params_grid.addWidget(hamming_info, 1, 1, 1, 2)

        params_grid.addWidget(QLabel("SSIM:"), 2, 0)
        self.ssim_slider = QSlider(Qt.Horizontal)
        self.ssim_slider.setMinimum(60)
        self.ssim_slider.setMaximum(100)
        self.ssim_slider.setValue(75)
        self.ssim_slider.setTickPosition(QSlider.TicksBelow)
        self.ssim_slider.setTickInterval(10)
        self.ssim_slider.valueChanged.connect(self.on_ssim_changed)
        params_grid.addWidget(self.ssim_slider, 2, 1)

        self.ssim_value_label = QLabel("0.75")
        self.ssim_value_label.setMinimumWidth(36)
        params_grid.addWidget(self.ssim_value_label, 2, 2)

        ssim_info = QLabel("0.60-0.75 слегка похожие | 0.75-0.90 похожие | 0.90-1.00 идентичные")
        ssim_info.setStyleSheet("color: gray; font-size: 10px;")
        params_grid.addWidget(ssim_info, 3, 1, 1, 2)

        params_layout.addLayout(params_grid)

        duplicates_box, duplicates_layout = self._create_group_box("Удаление дубликатов")
        self.btn_select_duplicates_folder = QPushButton("Выбрать папку")
        self.btn_select_duplicates_folder.clicked.connect(self.open_duplicates_folder)
        duplicates_layout.addWidget(self.btn_select_duplicates_folder)

        self.duplicates_folder_label = self._create_status_label("Папка не выбрана")
        duplicates_layout.addWidget(self.duplicates_folder_label)

        self.btn_remove_duplicates = QPushButton("Запустить удаление")
        self.btn_remove_duplicates.clicked.connect(self.remove_duplicates)
        self.btn_remove_duplicates.setEnabled(False)
        duplicates_layout.addWidget(self.btn_remove_duplicates)

        grouping_box, grouping_layout = self._create_group_box("Группировка похожих")
        self.btn_select_grouping_folder = QPushButton("Выбрать папку")
        self.btn_select_grouping_folder.clicked.connect(self.open_grouping_folder)
        grouping_layout.addWidget(self.btn_select_grouping_folder)

        self.grouping_folder_label = self._create_status_label("Папка не выбрана")
        grouping_layout.addWidget(self.grouping_folder_label)

        self.btn_group_images = QPushButton("Запустить группировку")
        self.btn_group_images.clicked.connect(self.group_images)
        self.btn_group_images.setEnabled(False)
        grouping_layout.addWidget(self.btn_group_images)

        layout.addWidget(params_box)
        layout.addWidget(duplicates_box)
        layout.addWidget(grouping_box)
        layout.addStretch(1)
        return tab

    def _build_vehicle_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        source_box, source_layout = self._create_group_box("Источник")
        self.btn_vehicle_file = QPushButton("Изображение")
        self.btn_vehicle_file.clicked.connect(self.open_vehicle_file)
        self.btn_vehicle_folder = QPushButton("Папка")
        self.btn_vehicle_folder.clicked.connect(self.open_vehicle_folder)
        source_layout.addLayout(self._create_button_row(self.btn_vehicle_file, self.btn_vehicle_folder))

        self.vehicle_path_label = self._create_status_label("Файл или папка не выбраны")
        source_layout.addWidget(self.vehicle_path_label)

        action_box, action_layout = self._create_group_box("Режим анализа")
        action_grid = QGridLayout()
        action_grid.setHorizontalSpacing(8)
        action_grid.setVerticalSpacing(8)

        self.btn_analyze_vehicle = QPushButton("Базовый")
        self.btn_analyze_vehicle.clicked.connect(self.run_vehicle_analysis)
        self.btn_analyze_vehicle.setEnabled(False)

        self.btn_analyze_vehicle_qwen = QPushButton("Qwen2.5-VL")
        self.btn_analyze_vehicle_qwen.clicked.connect(self.run_qwen_vehicle_analysis)
        self.btn_analyze_vehicle_qwen.setEnabled(False)

        self.btn_analyze_vehicle_qwen2vl = QPushButton("Qwen2-VL-2B")
        self.btn_analyze_vehicle_qwen2vl.clicked.connect(self.run_qwen2vl_vehicle_analysis)
        self.btn_analyze_vehicle_qwen2vl.setEnabled(False)

        self.btn_analyze_vehicle_openalpr = QPushButton("OpenALPR")
        self.btn_analyze_vehicle_openalpr.clicked.connect(self.run_openalpr_vehicle_analysis)
        self.btn_analyze_vehicle_openalpr.setEnabled(False)

        action_grid.addWidget(self.btn_analyze_vehicle, 0, 0)
        action_grid.addWidget(self.btn_analyze_vehicle_qwen, 0, 1)
        action_grid.addWidget(self.btn_analyze_vehicle_qwen2vl, 1, 0)
        action_grid.addWidget(self.btn_analyze_vehicle_openalpr, 1, 1)
        action_layout.addLayout(action_grid)

        layout.addWidget(source_box)
        layout.addWidget(action_box)
        layout.addStretch(1)
        return tab

    def _build_people_phone_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        person_box, person_layout = self._create_group_box("Анализ людей")
        self.btn_person_file = QPushButton("Изображение человека")
        self.btn_person_file.clicked.connect(self.open_person_file)
        self.btn_person_folder = QPushButton("Папка с людьми")
        self.btn_person_folder.clicked.connect(self.open_person_folder)
        person_layout.addLayout(self._create_button_row(self.btn_person_file, self.btn_person_folder))

        self.person_path_label = self._create_status_label("Файл или папка не выбраны")
        person_layout.addWidget(self.person_path_label)

        self.btn_analyze_person = QPushButton("Запустить анализ людей")
        self.btn_analyze_person.clicked.connect(self.run_person_analysis)
        self.btn_analyze_person.setEnabled(False)
        person_layout.addWidget(self.btn_analyze_person)

        phone_box, phone_layout = self._create_group_box("Анализ телефонов")
        self.btn_phone_file = QPushButton("Изображение телефона")
        self.btn_phone_file.clicked.connect(self.open_phone_file)
        self.btn_phone_folder = QPushButton("Папка с телефонами")
        self.btn_phone_folder.clicked.connect(self.open_phone_folder)
        phone_layout.addLayout(self._create_button_row(self.btn_phone_file, self.btn_phone_folder))

        self.phone_path_label = self._create_status_label("Файл или папка не выбраны")
        phone_layout.addWidget(self.phone_path_label)

        self.btn_analyze_phone = QPushButton("Запустить анализ телефона")
        self.btn_analyze_phone.clicked.connect(self.run_phone_analysis)
        self.btn_analyze_phone.setEnabled(False)
        phone_layout.addWidget(self.btn_analyze_phone)

        layout.addWidget(person_box)
        layout.addWidget(phone_box)
        layout.addStretch(1)
        return tab

    def _build_open_set_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        source_box, source_layout = self._create_group_box("Источник")
        self.btn_openset_file = QPushButton("Изображение")
        self.btn_openset_file.clicked.connect(self.open_openset_file)
        self.btn_openset_video = QPushButton("Видео")
        self.btn_openset_video.clicked.connect(self.open_openset_video)
        self.btn_openset_folder = QPushButton("Папка")
        self.btn_openset_folder.clicked.connect(self.open_openset_folder)
        source_layout.addLayout(self._create_button_row(
            self.btn_openset_file, self.btn_openset_video, self.btn_openset_folder,
        ))

        self.openset_path_label = self._create_status_label("Файл или папка не выбраны")
        source_layout.addWidget(self.openset_path_label)

        preset_box, preset_layout = self._create_group_box("Чувствительность")
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Режим:"))

        self.openset_preset_combo = QComboBox()
        self.openset_preset_combo.addItems(list(self.openset_presets.keys()))
        self.openset_preset_combo.setCurrentText(self.openset_preset_name)
        self.openset_preset_combo.currentTextChanged.connect(self.on_open_set_preset_changed)
        preset_row.addWidget(self.openset_preset_combo)
        preset_row.addStretch(1)
        preset_layout.addLayout(preset_row)

        self.openset_preset_details_label = QLabel()
        self.openset_preset_details_label.setWordWrap(True)
        self.openset_preset_details_label.setStyleSheet(
            "background: #f4f6f8; border: 1px solid #d5d9de; border-radius: 6px; padding: 8px; color: #4b5563;"
        )
        preset_layout.addWidget(self.openset_preset_details_label)
        self._update_open_set_preset_details()

        video_box, video_layout = self._create_group_box("Видео")
        video_settings = QHBoxLayout()
        video_settings.addWidget(QLabel("Шаг кадров, сек:"))

        self.openset_sample_interval_spin = QDoubleSpinBox()
        self.openset_sample_interval_spin.setDecimals(2)
        self.openset_sample_interval_spin.setRange(0.04, 10.0)
        self.openset_sample_interval_spin.setSingleStep(0.1)
        self.openset_sample_interval_spin.setValue(self.openset_sample_interval_seconds)
        self.openset_sample_interval_spin.setSuffix(" сек")
        self.openset_sample_interval_spin.valueChanged.connect(self.on_open_set_sample_interval_changed)
        video_settings.addWidget(self.openset_sample_interval_spin)
        video_settings.addStretch(1)
        video_layout.addLayout(video_settings)

        video_hint = QLabel("Используется только для видео: чем больше интервал, тем быстрее анализ и меньше просмотренных кадров.")
        video_hint.setWordWrap(True)
        video_hint.setStyleSheet("color: gray; font-size: 10px;")
        video_layout.addWidget(video_hint)

        unknown_crop_layout = QHBoxLayout()
        unknown_crop_layout.addWidget(QLabel("Запас unknown crop:"))

        self.openset_unknown_crop_margin_spin = QDoubleSpinBox()
        self.openset_unknown_crop_margin_spin.setDecimals(2)
        self.openset_unknown_crop_margin_spin.setRange(0.0, 0.5)
        self.openset_unknown_crop_margin_spin.setSingleStep(0.02)
        self.openset_unknown_crop_margin_spin.setValue(self.openset_unknown_crop_margin_scale)
        self.openset_unknown_crop_margin_spin.valueChanged.connect(self.on_open_set_unknown_crop_margin_changed)
        unknown_crop_layout.addWidget(self.openset_unknown_crop_margin_spin)

        unknown_crop_percent = QLabel("доля кадра")
        unknown_crop_percent.setStyleSheet("color: gray;")
        unknown_crop_layout.addWidget(unknown_crop_percent)
        unknown_crop_layout.addStretch(1)
        video_layout.addLayout(unknown_crop_layout)

        unknown_crop_hint = QLabel("0.00 — строго по объекту, 0.12 — умеренный запас, 0.30+ — широкий контекст вокруг unknown-объекта.")
        unknown_crop_hint.setWordWrap(True)
        unknown_crop_hint.setStyleSheet("color: gray; font-size: 10px;")
        video_layout.addWidget(unknown_crop_hint)

        unknown_class_layout = QHBoxLayout()
        unknown_class_layout.addWidget(QLabel("Схожесть новых классов:"))

        self.openset_unknown_class_similarity_spin = QDoubleSpinBox()
        self.openset_unknown_class_similarity_spin.setDecimals(2)
        self.openset_unknown_class_similarity_spin.setRange(0.50, 0.99)
        self.openset_unknown_class_similarity_spin.setSingleStep(0.01)
        self.openset_unknown_class_similarity_spin.setValue(self.openset_unknown_class_similarity_threshold)
        self.openset_unknown_class_similarity_spin.valueChanged.connect(self.on_open_set_unknown_class_similarity_changed)
        unknown_class_layout.addWidget(self.openset_unknown_class_similarity_spin)
        unknown_class_layout.addStretch(1)
        video_layout.addLayout(unknown_class_layout)

        unknown_class_hint = QLabel("Меньше значение — объединяет больше unknown-скринов в один новый класс. Больше значение — дробит на более строгие группы.")
        unknown_class_hint.setWordWrap(True)
        unknown_class_hint.setStyleSheet("color: gray; font-size: 10px;")
        video_layout.addWidget(unknown_class_hint)

        self.openset_hdbscan_check = QCheckBox("Использовать HDBSCAN для кластеризации (рекомендуется для папки)")
        self.openset_hdbscan_check.setChecked(self.openset_use_hdbscan)
        self.openset_hdbscan_check.stateChanged.connect(self.on_open_set_hdbscan_changed)
        video_layout.addWidget(self.openset_hdbscan_check)

        hdbscan_hint = QLabel(
            "HDBSCAN + cosine distance автоматически определяет число классов и выявляет шумовые объекты (UnknownClass_Noise). "
            "Требует scikit-learn ≥ 1.3 или пакет hdbscan. При одиночном файле жадный алгоритм предпочтительнее."
        )
        hdbscan_hint.setWordWrap(True)
        hdbscan_hint.setStyleSheet("color: gray; font-size: 10px;")
        video_layout.addWidget(hdbscan_hint)

        self.btn_analyze_open_set = QPushButton("Запустить Open-set detection")
        self.btn_analyze_open_set.clicked.connect(self.run_open_set_analysis)
        self.btn_analyze_open_set.setEnabled(False)
        video_layout.addWidget(self.btn_analyze_open_set)

        buttons_row = QHBoxLayout()

        self.btn_open_openset_unknown_folder = QPushButton("Открыть папку Unknown")
        self.btn_open_openset_unknown_folder.clicked.connect(self.open_openset_unknown_folder)
        self.btn_open_openset_unknown_folder.setEnabled(False)
        buttons_row.addWidget(self.btn_open_openset_unknown_folder)

        self.btn_open_openset_unknown_classes_folder = QPushButton("Открыть классы базы UnknownClass")
        self.btn_open_openset_unknown_classes_folder.clicked.connect(self.open_openset_unknown_classes_folder)
        self.btn_open_openset_unknown_classes_folder.setEnabled(False)
        buttons_row.addWidget(self.btn_open_openset_unknown_classes_folder)
        video_layout.addLayout(buttons_row)

        layout.addWidget(source_box)
        layout.addWidget(preset_box)
        layout.addWidget(video_box)
        layout.addStretch(1)
        return tab

    def _build_unknown_merge_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        source_box, source_layout = self._create_group_box("Импорт в базу UnknownClass")
        self.btn_unknown_merge_folder = QPushButton("Выбрать папку для импорта")
        self.btn_unknown_merge_folder.clicked.connect(self.open_unknown_merge_folder)
        source_layout.addWidget(self.btn_unknown_merge_folder)

        self.unknown_merge_path_label = self._create_status_label(
            "Выберите папку, из которой нужно дополнить единую базу UnknownClass"
        )
        source_layout.addWidget(self.unknown_merge_path_label)

        source_hint = QLabel(
            "OpenSetAnalyzer теперь формирует постоянную базу UnknownClass. Эта вкладка рекурсивно обходит выбранную папку и добавляет изображения в ту же базу по визуальной схожести."
        )
        source_hint.setWordWrap(True)
        source_hint.setStyleSheet("color: gray; font-size: 10px;")
        source_layout.addWidget(source_hint)

        settings_box, settings_layout = self._create_group_box("Параметры объединения")
        similarity_layout = QHBoxLayout()
        similarity_layout.addWidget(QLabel("Порог схожести:"))

        self.unknown_merge_similarity_spin = QDoubleSpinBox()
        self.unknown_merge_similarity_spin.setDecimals(2)
        self.unknown_merge_similarity_spin.setRange(0.50, 0.99)
        self.unknown_merge_similarity_spin.setSingleStep(0.01)
        self.unknown_merge_similarity_spin.setValue(self.unknown_merge_similarity_threshold)
        self.unknown_merge_similarity_spin.valueChanged.connect(self.on_unknown_merge_similarity_changed)
        similarity_layout.addWidget(self.unknown_merge_similarity_spin)
        similarity_layout.addStretch(1)
        settings_layout.addLayout(similarity_layout)

        self.unknown_merge_similarity_hint = QLabel()
        self.unknown_merge_similarity_hint.setWordWrap(True)
        self.unknown_merge_similarity_hint.setStyleSheet(
            "background: #f4f6f8; border: 1px solid #d5d9de; border-radius: 6px; padding: 8px; color: #4b5563;"
        )
        settings_layout.addWidget(self.unknown_merge_similarity_hint)
        self._update_unknown_merge_similarity_hint()

        action_box, action_layout = self._create_group_box("Обновление базы")
        self.btn_run_unknown_merge = QPushButton("Дополнить базу UnknownClass")
        self.btn_run_unknown_merge.clicked.connect(self.run_unknown_merge_analysis)
        self.btn_run_unknown_merge.setEnabled(False)
        action_layout.addWidget(self.btn_run_unknown_merge)

        buttons_row = QHBoxLayout()

        self.btn_open_unknown_merge_output_folder = QPushButton("Открыть папку базы")
        self.btn_open_unknown_merge_output_folder.clicked.connect(self.open_unknown_merge_output_folder)
        self.btn_open_unknown_merge_output_folder.setEnabled(False)
        buttons_row.addWidget(self.btn_open_unknown_merge_output_folder)

        self.btn_open_unknown_merge_classes_folder = QPushButton("Открыть классы базы")
        self.btn_open_unknown_merge_classes_folder.clicked.connect(self.open_unknown_merge_classes_folder)
        self.btn_open_unknown_merge_classes_folder.setEnabled(False)
        buttons_row.addWidget(self.btn_open_unknown_merge_classes_folder)

        action_layout.addLayout(buttons_row)

        consolidate_box, consolidate_layout = self._create_group_box("Консолидация базы")

        consolidate_hint = QLabel(
            "Просматривает все классы базы и объединяет те, чьи центроиды похожи сильнее порога. "
            "Меньше значение — сливает больше классов; больше значение — сохраняет более тонкое разделение."
        )
        consolidate_hint.setWordWrap(True)
        consolidate_hint.setStyleSheet("color: gray; font-size: 10px;")
        consolidate_layout.addWidget(consolidate_hint)

        consolidate_threshold_layout = QHBoxLayout()
        consolidate_threshold_layout.addWidget(QLabel("Порог объединения:"))
        self.unknown_consolidate_threshold_spin = QDoubleSpinBox()
        self.unknown_consolidate_threshold_spin.setDecimals(2)
        self.unknown_consolidate_threshold_spin.setRange(0.50, 0.99)
        self.unknown_consolidate_threshold_spin.setSingleStep(0.01)
        self.unknown_consolidate_threshold_spin.setValue(self.unknown_consolidate_threshold)
        self.unknown_consolidate_threshold_spin.valueChanged.connect(self.on_unknown_consolidate_threshold_changed)
        consolidate_threshold_layout.addWidget(self.unknown_consolidate_threshold_spin)
        consolidate_threshold_layout.addStretch(1)
        consolidate_layout.addLayout(consolidate_threshold_layout)

        self.btn_run_consolidate = QPushButton("Объединить похожие классы")
        self.btn_run_consolidate.clicked.connect(self.run_consolidate_unknown_class)
        consolidate_layout.addWidget(self.btn_run_consolidate)

        preview_box, preview_layout = self._create_group_box("Предпросмотр базы UnknownClass")
        self.unknown_merge_preview_label = self._create_status_label("База UnknownClass ещё не сформирована")
        preview_layout.addWidget(self.unknown_merge_preview_label)

        self.unknown_merge_preview_list = QListWidget()
        self.unknown_merge_preview_list.setMinimumHeight(180)
        preview_layout.addWidget(self.unknown_merge_preview_list)

        layout.addWidget(source_box)
        layout.addWidget(settings_box)
        layout.addWidget(action_box)
        layout.addWidget(consolidate_box)
        layout.addWidget(preview_box)
        layout.addStretch(1)
        return tab

    # ─────────────────────────────────────────────────────────────────────────
    #  ТАБ: ТРЕНИРОВКА ДЕТЕКТОРА БЛОКИРАТОРОВ
    # ─────────────────────────────────────────────────────────────────────────

    def _build_train_blocker_tab(self):
        import os as _os
        from PyQt5.QtWidgets import QSpinBox as _QSpinBox, QScrollArea as _QScrollArea, QLineEdit as _QLineEdit
        from PyQt5.QtCore import Qt as _Qt

        _BASE = _os.path.abspath(_os.path.join(_os.path.dirname(__file__)))

        # Внутренняя обёртка со скроллом
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = _QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(_QScrollArea.NoFrame)
        outer_layout.addWidget(scroll)

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)
        scroll.setWidget(tab)

        # ── Источник + Параметры (два блока рядом) ────────────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(5)

        # Источник
        src_box, src_layout = self._create_group_box("Источник кропов")
        src_layout.setSpacing(3)
        self.blocker_source_label = QLabel(_os.path.join(_BASE, "LearnDataset"))
        self.blocker_source_label.setStyleSheet("color:#4b5563; font-size:10px;")
        self.blocker_source_label.setWordWrap(True)
        src_layout.addWidget(self.blocker_source_label)
        self.btn_blocker_source = QPushButton("Выбрать папку…")
        self.btn_blocker_source.clicked.connect(self._blocker_choose_source)
        src_layout.addWidget(self.btn_blocker_source)
        src_layout.addStretch(1)
        top_row.addWidget(src_box, 1)

        # Параметры (2 колонки через QGridLayout)
        params_box, params_layout = self._create_group_box("Параметры тренировки")
        params_layout.setSpacing(3)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        grid.addWidget(QLabel("Эпохи:"), 0, 0)
        self.blocker_epochs_spin = _QSpinBox()
        self.blocker_epochs_spin.setRange(5, 500)
        self.blocker_epochs_spin.setValue(80)
        grid.addWidget(self.blocker_epochs_spin, 0, 1)

        grid.addWidget(QLabel("Размер:"), 0, 2)
        self.blocker_imgsz_combo = QComboBox()
        self.blocker_imgsz_combo.addItems(["320", "416", "512", "640", "768", "1024", "1280", "1920"])
        self.blocker_imgsz_combo.setCurrentText("1280")
        grid.addWidget(self.blocker_imgsz_combo, 0, 3)

        grid.addWidget(QLabel("Батч:"), 1, 0)
        self.blocker_batch_spin = _QSpinBox()
        self.blocker_batch_spin.setRange(1, 128)
        self.blocker_batch_spin.setValue(2)
        grid.addWidget(self.blocker_batch_spin, 1, 1)

        grid.addWidget(QLabel("Device:"), 1, 2)
        self.blocker_device_combo = QComboBox()
        self.blocker_device_combo.addItems(["auto", "cpu", "0", "0,1"])
        grid.addWidget(self.blocker_device_combo, 1, 3)

        grid.addWidget(QLabel("Patience:"), 2, 0)
        self.blocker_patience_spin = _QSpinBox()
        self.blocker_patience_spin.setRange(0, 500)
        self.blocker_patience_spin.setValue(15)
        self.blocker_patience_spin.setToolTip(
            "Early stopping: остановить тренировку если за N эпох нет улучшения mAP.\n"
            "0 = отключить early stopping (дойти до конца)."
        )
        grid.addWidget(self.blocker_patience_spin, 2, 1)
        lbl_patience = QLabel("(0 = выкл)")
        lbl_patience.setStyleSheet("color:#6b7280; font-size:10px;")
        grid.addWidget(lbl_patience, 2, 2)

        params_layout.addLayout(grid)

        weights_row = QHBoxLayout()
        weights_row.setSpacing(4)
        weights_row.addWidget(QLabel("Веса:"))
        self.blocker_weights_label = QLabel("Автовыбор")
        self.blocker_weights_label.setStyleSheet("color:#4b5563; font-size:10px;")
        weights_row.addWidget(self.blocker_weights_label, 1)
        self.btn_blocker_weights = QPushButton("…")
        self.btn_blocker_weights.setFixedWidth(32)
        self.btn_blocker_weights.setToolTip("Выбрать .pt файл базовых весов")
        self.btn_blocker_weights.clicked.connect(self._blocker_choose_weights)
        weights_row.addWidget(self.btn_blocker_weights)
        params_layout.addLayout(weights_row)

        class_row = QHBoxLayout()
        class_row.setSpacing(4)
        class_row.addWidget(QLabel("Класс:"))
        self.blocker_class_name_edit = _QLineEdit("parking_blocker")
        self.blocker_class_name_edit.setPlaceholderText("например: wheel, car, cone")
        self.blocker_class_name_edit.setToolTip(
            "Название класса объекта в датасете.\n"
            "Используется в data.yaml и метках.\n"
            "Только латинские буквы, цифры и _"
        )
        class_row.addWidget(self.blocker_class_name_edit, 1)
        params_layout.addLayout(class_row)

        self.blocker_overwrite_check = QCheckBox("Пересоздать датасет")
        params_layout.addWidget(self.blocker_overwrite_check)
        params_layout.addStretch(1)
        top_row.addWidget(params_box, 2)
        layout.addLayout(top_row)

        # ── Готовый YOLO-датасет ──────────────────────────────────────────────
        yaml_box, yaml_layout = self._create_group_box("Готовый YOLO-датасет (data.yaml)")
        yaml_layout.setSpacing(3)
        yaml_hint = QLabel(
            "Если выбрать готовый data.yaml — шаги подготовки датасета (кропы/синтетика) будут пропущены."
        )
        yaml_hint.setStyleSheet("color:#6b7280; font-size:10px;")
        yaml_hint.setWordWrap(True)
        yaml_layout.addWidget(yaml_hint)
        yaml_row = QHBoxLayout()
        yaml_row.setSpacing(4)
        self._blocker_yaml_label = QLabel("не выбран")
        self._blocker_yaml_label.setStyleSheet("color:#4b5563; font-size:10px;")
        self._blocker_yaml_label.setWordWrap(True)
        yaml_row.addWidget(self._blocker_yaml_label, 1)
        self._btn_blocker_yaml = QPushButton("Выбрать data.yaml…")
        self._btn_blocker_yaml.clicked.connect(self._blocker_choose_yaml)
        yaml_row.addWidget(self._btn_blocker_yaml)
        self._btn_blocker_yaml_clear = QPushButton("Очистить")
        self._btn_blocker_yaml_clear.setEnabled(False)
        self._btn_blocker_yaml_clear.clicked.connect(self._blocker_clear_yaml)
        yaml_row.addWidget(self._btn_blocker_yaml_clear)
        yaml_layout.addLayout(yaml_row)
        layout.addWidget(yaml_box)

        # ── Синтетический датасет ─────────────────────────────────────────────
        synth_box, synth_layout = self._create_group_box("Синтетический датасет")
        synth_layout.setSpacing(3)
        synth_row = QHBoxLayout()
        synth_row.setSpacing(6)
        self.blocker_synthetic_check = QCheckBox("Включить (кропы на фонах, реальные bbox)")
        self.blocker_synthetic_check.setChecked(True)
        self.blocker_synthetic_check.setToolTip(
            "Рекомендуется: кропы блокираторов вставляются на фоновые кадры,\n"
            "аннотации содержат точные координаты bbox.\n\n"
            "Снять галочку — использовать обычный режим (bbox = весь кроп)."
        )
        self.blocker_synthetic_check.toggled.connect(self._blocker_on_synthetic_toggled)
        synth_row.addWidget(self.blocker_synthetic_check)
        synth_row.addSpacing(12)
        synth_row.addWidget(QLabel("Train:"))
        self._blocker_ntrain_spin = _QSpinBox()
        self._blocker_ntrain_spin.setRange(200, 20000)
        self._blocker_ntrain_spin.setSingleStep(200)
        self._blocker_ntrain_spin.setValue(2000)
        self._blocker_ntrain_spin.setFixedWidth(80)
        synth_row.addWidget(self._blocker_ntrain_spin)
        synth_row.addWidget(QLabel("Val:"))
        self._blocker_nval_spin = _QSpinBox()
        self._blocker_nval_spin.setRange(50, 5000)
        self._blocker_nval_spin.setSingleStep(50)
        self._blocker_nval_spin.setValue(300)
        self._blocker_nval_spin.setFixedWidth(80)
        synth_row.addWidget(self._blocker_nval_spin)
        synth_row.addStretch(1)
        synth_layout.addLayout(synth_row)
        layout.addWidget(synth_box)

        # ── Аугментация кропов ────────────────────────────────────────────────
        aug_box, aug_layout = self._create_group_box("Аугментация кропов")
        aug_layout.setSpacing(3)
        aug_ctrl_row = QHBoxLayout()
        aug_ctrl_row.setSpacing(6)
        aug_ctrl_row.addWidget(QLabel("Итого:"))
        self._aug_target_spin = _QSpinBox()
        self._aug_target_spin.setRange(100, 50000)
        self._aug_target_spin.setSingleStep(500)
        self._aug_target_spin.setValue(3000)
        self._aug_target_spin.setFixedWidth(90)
        self._aug_target_spin.setToolTip(
            "Желаемое итоговое число файлов в LearnDataset.\n"
            "Старые _aug_* файлы удаляются и генерируются заново."
        )
        aug_ctrl_row.addWidget(self._aug_target_spin)
        self._btn_aug_run = QPushButton("▶ Аугментировать")
        self._btn_aug_run.clicked.connect(self._blocker_run_augmentation)
        aug_ctrl_row.addWidget(self._btn_aug_run)
        self._btn_aug_stop = QPushButton("■ Стоп")
        self._btn_aug_stop.setEnabled(False)
        self._btn_aug_stop.clicked.connect(self._blocker_stop_augmentation)
        aug_ctrl_row.addWidget(self._btn_aug_stop)
        self._aug_status_label = self._create_status_label("Готово")
        aug_ctrl_row.addWidget(self._aug_status_label, 1)
        aug_layout.addLayout(aug_ctrl_row)
        self._aug_log = QPlainTextEdit()
        self._aug_log.setReadOnly(True)
        self._aug_log.setFixedHeight(60)
        self._aug_log.setStyleSheet(
            "background:#1e1e1e; color:#d4d4d4; font-family:Consolas,monospace; font-size:10px;"
        )
        aug_layout.addWidget(self._aug_log)
        layout.addWidget(aug_box)

        # ── Запуск ────────────────────────────────────────────────────────────
        action_box, action_layout = self._create_group_box("Запуск")
        action_layout.setSpacing(3)

        out_row = QHBoxLayout()
        out_row.setSpacing(4)
        out_row.addWidget(QLabel("Выходной файл:"))
        self.blocker_output_name_edit = _QLineEdit("parking_blocker.pt")
        self.blocker_output_name_edit.setPlaceholderText("например: wheel.pt, cone.pt")
        self.blocker_output_name_edit.setToolTip(
            "Имя файла модели в папке Models/.\n"
            "Расширение .pt добавляется автоматически если не указано."
        )
        out_row.addWidget(self.blocker_output_name_edit, 1)
        action_layout.addLayout(out_row)

        btn_row = QHBoxLayout()
        self.btn_blocker_train = QPushButton("▶  Начать тренировку")
        self.btn_blocker_train.clicked.connect(self._blocker_start_train)
        self.btn_blocker_stop = QPushButton("■  Остановить")
        self.btn_blocker_stop.setEnabled(False)
        self.btn_blocker_stop.clicked.connect(self._blocker_cancel_train)
        btn_row.addWidget(self.btn_blocker_train)
        btn_row.addWidget(self.btn_blocker_stop)
        self.blocker_status_label = self._create_status_label("Готов к тренировке")
        btn_row.addWidget(self.blocker_status_label, 1)
        action_layout.addLayout(btn_row)

        self.blocker_result_label = self._create_status_label("—")
        action_layout.addWidget(self.blocker_result_label)
        self.btn_blocker_open_model = QPushButton("Открыть папку Models/")
        self.btn_blocker_open_model.clicked.connect(self._blocker_open_models_folder)
        self.btn_blocker_open_model.setEnabled(False)
        action_layout.addWidget(self.btn_blocker_open_model)
        layout.addWidget(action_box)

        # ── Лог тренировки ────────────────────────────────────────────────────
        log_box, log_layout = self._create_group_box("Лог тренировки")
        self.blocker_log = QPlainTextEdit()
        self.blocker_log.setReadOnly(True)
        self.blocker_log.setMinimumHeight(140)
        self.blocker_log.setMaximumHeight(240)
        self.blocker_log.setStyleSheet(
            "background:#1e1e1e; color:#d4d4d4; font-family:Consolas,monospace; font-size:13px;"
        )
        log_layout.addWidget(self.blocker_log)
        layout.addWidget(log_box)

        # ── Графики обучения ─────────────────────────────────────────────────
        charts_box, charts_layout = self._create_group_box("Графики обучения")
        charts_layout.setSpacing(3)
        charts_ctrl_row = QHBoxLayout()
        charts_ctrl_row.addWidget(QLabel("График:"))
        self._blocker_charts_combo = QComboBox()
        self._blocker_charts_combo.setMinimumWidth(160)
        self._blocker_charts_combo.currentTextChanged.connect(self._blocker_show_chart)
        charts_ctrl_row.addWidget(self._blocker_charts_combo, 1)
        btn_charts_refresh = QPushButton("⟳ Обновить")
        btn_charts_refresh.clicked.connect(self._blocker_load_charts)
        charts_ctrl_row.addWidget(btn_charts_refresh)
        charts_layout.addLayout(charts_ctrl_row)

        self._blocker_chart_label = QLabel("Графики появятся после завершения тренировки")
        self._blocker_chart_label.setAlignment(_Qt.AlignCenter)
        self._blocker_chart_label.setMinimumHeight(220)
        self._blocker_chart_label.setStyleSheet("background:#111; color:#666;")
        chart_scroll = _QScrollArea()
        chart_scroll.setWidgetResizable(True)
        chart_scroll.setWidget(self._blocker_chart_label)
        chart_scroll.setMinimumHeight(240)
        chart_scroll.setMaximumHeight(340)
        self._blocker_chart_scroll = chart_scroll
        charts_layout.addWidget(chart_scroll)
        layout.addWidget(charts_box)
        layout.addStretch(1)

        # Внутренние поля
        self._blocker_source_dir   = _os.path.join(_BASE, "LearnDataset")
        self._blocker_weights_path = None
        self._blocker_custom_yaml  = None
        self._blocker_worker       = None
        self._blocker_run_dir      = _os.path.join(_BASE, "Models", "training_runs", "parking_blocker")
        self._aug_worker           = None

        # Таймер для автообновления графиков во время тренировки
        from PyQt5.QtCore import QTimer as _QTimer
        self._blocker_chart_timer = _QTimer(self)
        self._blocker_chart_timer.setInterval(15000)  # каждые 15 секунд
        self._blocker_chart_timer.timeout.connect(self._blocker_load_charts)

        return outer

    def _blocker_choose_source(self):
        folder = QFileDialog.getExistingDirectory(self, "Папка с кропами блокираторов")
        if folder:
            self._blocker_source_dir = folder
            self.blocker_source_label.setText(folder)

    def _blocker_choose_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Базовые веса YOLO", "", "Model files (*.pt);;All files (*)"
        )
        if path:
            self._blocker_weights_path = path
            self.blocker_weights_label.setText(os.path.basename(path))

    def _blocker_choose_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать data.yaml готового YOLO-датасета", "",
            "YAML files (*.yaml *.yml);;All files (*)"
        )
        if path:
            self._blocker_custom_yaml = path
            self._blocker_yaml_label.setText(path)
            self._blocker_yaml_label.setStyleSheet("color:#16a34a; font-size:10px;")
            self._btn_blocker_yaml_clear.setEnabled(True)

    def _blocker_clear_yaml(self):
        self._blocker_custom_yaml = None
        self._blocker_yaml_label.setText("не выбран")
        self._blocker_yaml_label.setStyleSheet("color:#4b5563; font-size:10px;")
        self._btn_blocker_yaml_clear.setEnabled(False)

    def _blocker_start_train(self):
        self.btn_blocker_train.setEnabled(False)
        self.btn_blocker_stop.setEnabled(True)
        self.blocker_log.clear()
        self.blocker_status_label.setText("Идёт тренировка...")
        self.btn_blocker_open_model.setEnabled(False)
        self.blocker_result_label.setText("—")

        from Plugins.ParkingBlockerTrainer.parking_blocker_trainer import OUTPUT_MODEL_PATH, MODELS_DIR

        raw_class = self.blocker_class_name_edit.text().strip()
        class_name = raw_class if raw_class else "parking_blocker"

        raw_out = self.blocker_output_name_edit.text().strip()
        if not raw_out:
            raw_out = "parking_blocker.pt"
        if not raw_out.endswith(".pt"):
            raw_out += ".pt"
        output_path = os.path.join(MODELS_DIR, raw_out)

        # Обновляем путь к папке с графиками под текущий класс
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self._blocker_run_dir = os.path.join(base_dir, "Models", "training_runs", class_name)

        self._blocker_worker = TrainBlockerWorker(
            source_dir        = self._blocker_source_dir,
            epochs            = self.blocker_epochs_spin.value(),
            imgsz             = int(self.blocker_imgsz_combo.currentText()),
            batch             = self.blocker_batch_spin.value(),
            weights           = self._blocker_weights_path,
            output_path       = output_path,
            overwrite_dataset = self.blocker_overwrite_check.isChecked(),
            device            = self.blocker_device_combo.currentText(),
            use_synthetic     = self.blocker_synthetic_check.isChecked(),
            n_train           = self._blocker_ntrain_spin.value(),
            n_val             = self._blocker_nval_spin.value(),
            patience          = self.blocker_patience_spin.value(),
            data_yaml         = self._blocker_custom_yaml,
            class_name        = class_name,
        )
        self._blocker_worker.log_message.connect(self._blocker_on_log)
        self._blocker_worker.train_done.connect(self._blocker_on_done)
        self._blocker_worker.train_error.connect(self._blocker_on_error)
        self._blocker_worker.finished.connect(self._blocker_on_finished)
        self._blocker_worker.start()

        # Запускаем автообновление графиков каждые 15 секунд
        self._blocker_chart_timer.start()

    def _blocker_cancel_train(self):
        if self._blocker_worker:
            self._blocker_worker.cancel()
        self._blocker_chart_timer.stop()
        self.blocker_status_label.setText("Отменяется...")

    def _blocker_on_synthetic_toggled(self, checked: bool):
        self._blocker_ntrain_spin.setEnabled(checked)
        self._blocker_nval_spin.setEnabled(checked)

    def _blocker_run_augmentation(self):
        self._btn_aug_run.setEnabled(False)
        self._btn_aug_stop.setEnabled(True)
        self._aug_log.clear()
        self._aug_status_label.setText("Аугментация...")

        self._aug_worker = AugmentCropsWorker(
            crops_dir    = self._blocker_source_dir,
            target_count = self._aug_target_spin.value(),
        )
        self._aug_worker.log_message.connect(lambda msg: self._aug_log.appendPlainText(msg))
        self._aug_worker.aug_done.connect(self._blocker_aug_done)
        self._aug_worker.aug_error.connect(self._blocker_aug_error)
        self._aug_worker.finished.connect(self._blocker_aug_finished)
        self._aug_worker.start()

    def _blocker_stop_augmentation(self):
        if self._aug_worker and self._aug_worker.isRunning():
            self._aug_worker.terminate()
            self._aug_status_label.setText("Остановлено")

    def _blocker_aug_done(self, total: int):
        self._aug_status_label.setText(f"Готово: {total} файлов в LearnDataset")

    def _blocker_aug_error(self, msg: str):
        self._aug_status_label.setText("Ошибка!")
        self._aug_log.appendPlainText(f"[ОШИБКА] {msg}")
        QMessageBox.critical(self, "Ошибка аугментации", msg)

    def _blocker_aug_finished(self):
        self._btn_aug_run.setEnabled(True)
        self._btn_aug_stop.setEnabled(False)
        self._aug_worker = None

    # ── Тест на видео ─────────────────────────────────────────────────────────

    def _vtest_choose_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать видеофайл", "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.wmv);;All files (*)"
        )
        if path:
            self._vtest_video_path = path
            self._vtest_video_label.setText(os.path.basename(path))
            self._btn_vtest_open.setEnabled(False)

    def _vtest_run(self):
        base = os.path.dirname(os.path.abspath(__file__))

        # Берём текущую модель из Инспектора, иначе — parking_blocker.pt
        model_path = getattr(self, "_mi_model_path", None)
        if not model_path:
            model_path = os.path.join(base, "Models", "parking_blocker.pt")
        if not os.path.isfile(model_path):
            QMessageBox.warning(self, "Тест на видео",
                                f"Файл модели не найден:\n{model_path}\n\n"
                                "Загрузите модель через кнопку «Выбрать .pt…» выше.")
            return

        video_path = self._vtest_video_path
        if not video_path or not os.path.isfile(video_path):
            QMessageBox.warning(self, "Тест на видео", "Выберите видеофайл.")
            return

        # Путь для выходного видео рядом с исходным
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir   = os.path.dirname(video_path)
        self._vtest_output_path = os.path.join(out_dir, f"{base_name}_detected.mp4")

        self._btn_vtest_run.setEnabled(False)
        self._btn_vtest_stop.setEnabled(True)
        self._btn_vtest_open.setEnabled(False)
        self._vtest_log.clear()
        self._vtest_status_label.setText("Обработка...")

        self._vtest_worker = VideoTestWorker(
            model_path   = model_path,
            video_path   = video_path,
            output_path  = self._vtest_output_path,
            conf         = self._vtest_conf_spin.value(),
            iou          = self._vtest_iou_spin.value(),
            imgsz        = int(self._vtest_imgsz_combo.currentText()),
            skip_frames  = self._vtest_skip_spin.value(),
        )
        self._vtest_worker.log_message.connect(
            lambda msg: self._vtest_log.appendPlainText(msg)
        )
        self._vtest_worker.progress.connect(
            lambda cur, tot: self._vtest_status_label.setText(f"Кадр {cur}/{tot}")
        )
        self._vtest_worker.stats_ready.connect(self._vtest_save_stats)
        self._vtest_worker.test_done.connect(self._vtest_done)
        self._vtest_worker.test_error.connect(self._vtest_error)
        self._vtest_worker.finished.connect(self._vtest_finished)
        self._vtest_worker.start()

    def _vtest_stop(self):
        if self._vtest_worker and self._vtest_worker.isRunning():
            self._vtest_worker.cancel()
        self._vtest_status_label.setText("Остановка...")

    def _vtest_save_stats(self, entries: list):
        """Сохраняет накопленную статистику видеотеста в JSON-файл."""
        try:
            save_detection_stats_batch(entries)
        except Exception as _exc:
            self._vtest_log.appendPlainText(f"[Статистика] Не удалось сохранить: {_exc}")

    def _vtest_done(self, output_path: str):
        self._vtest_status_label.setText("Готово!")
        self._btn_vtest_open.setEnabled(True)
        self._vtest_output_path = output_path
        self._vtest_log.appendPlainText(f"Файл сохранён: {output_path}")

    def _vtest_error(self, msg: str):
        self._vtest_status_label.setText("Ошибка!")
        self._vtest_log.appendPlainText(f"[ОШИБКА] {msg}")
        QMessageBox.critical(self, "Ошибка теста на видео", msg)

    def _vtest_finished(self):
        self._btn_vtest_run.setEnabled(True)
        self._btn_vtest_stop.setEnabled(False)
        self._vtest_worker = None

    def _vtest_open_result(self):
        if self._vtest_output_path and os.path.isfile(self._vtest_output_path):
            subprocess.Popen(f'explorer /select,"{self._vtest_output_path}"')

    def _blocker_on_log(self, msg):
        self.blocker_log.appendPlainText(msg)

    def _blocker_on_done(self, model_path):
        self._blocker_chart_timer.stop()
        self.blocker_status_label.setText("Тренировка завершена!")
        self.blocker_result_label.setText(f"Модель: {model_path}")
        self.btn_blocker_open_model.setEnabled(True)
        self._blocker_load_charts()
        QMessageBox.information(
            self, "Готово",
            f"Тренировка завершена!\n\nМодель сохранена:\n{model_path}"
        )

    def _blocker_on_error(self, error_msg):
        self._blocker_chart_timer.stop()
        self.blocker_status_label.setText("Ошибка тренировки!")
        self.blocker_log.appendPlainText(f"\n[ОШИБКА] {error_msg}")
        QMessageBox.critical(self, "Ошибка тренировки", error_msg)

    def _blocker_on_finished(self):
        self._blocker_chart_timer.stop()
        self.btn_blocker_train.setEnabled(True)
        self.btn_blocker_stop.setEnabled(False)
        self._blocker_worker = None

    def _blocker_open_models_folder(self):
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")
        if os.path.isdir(models_dir):
            subprocess.Popen(f'explorer "{models_dir}"')

    def _blocker_load_charts(self):
        """Сканирует папку прогона и заполняет комбобокс доступными графиками."""
        chart_names = [
            "results.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "F1_curve.png",
            "PR_curve.png",
            "P_curve.png",
            "R_curve.png",
        ]
        run_dir = getattr(self, "_blocker_run_dir", None)
        found = []
        if run_dir and os.path.isdir(run_dir):
            for name in chart_names:
                if os.path.isfile(os.path.join(run_dir, name)):
                    found.append(name)

        self._blocker_charts_combo.blockSignals(True)
        self._blocker_charts_combo.clear()
        if found:
            self._blocker_charts_combo.addItems(found)
        self._blocker_charts_combo.blockSignals(False)

        if found:
            self._blocker_show_chart(found[0])
        else:
            self._blocker_chart_label.clear()
            self._blocker_chart_label.setText(
                "Графики не найдены.\nЗапустите тренировку или нажмите \"Обновить\".\n\n"
                + (run_dir or "Папка не задана")
            )

    def _blocker_show_chart(self, name: str):
        """Загружает и отображает выбранный график."""
        from PyQt5.QtGui import QPixmap as _QPixmap
        from PyQt5.QtCore import Qt as _Qt
        run_dir = getattr(self, "_blocker_run_dir", None)
        if not name or not run_dir:
            return
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            self._blocker_chart_label.setText(f"Файл не найден:\n{path}")
            return
        pix = _QPixmap(path)
        if pix.isNull():
            self._blocker_chart_label.setText(f"Не удалось загрузить:\n{path}")
            return
        vp = self._blocker_chart_scroll.viewport()
        w = max(vp.width() - 4, 400)
        h = max(vp.height() - 4, 280)
        scaled = pix.scaled(w, h, _Qt.KeepAspectRatio, _Qt.SmoothTransformation)
        self._blocker_chart_label.setPixmap(scaled)
        self._blocker_chart_label.setMinimumSize(scaled.width(), scaled.height())

    # ═══════════════════════════════════════════════════════════════════════
    #  ВКЛ. ClassManager — управление классами для обучения
    # ═══════════════════════════════════════════════════════════════════════

    def _build_class_manager_tab(self):
        """Вкладка Инспектор датасетов — просмотр, слияние, переименование, удаление YOLO-датасетов."""
        from PyQt5.QtWidgets import (
            QSplitter as _QSplitter,
            QScrollArea as _QScrollArea,
            QListWidgetItem as _QListWidgetItem,
            QInputDialog as _QInputDialog,
            QLineEdit as _QLineEdit,
        )
        from PyQt5.QtCore import Qt as _Qt

        tab = QWidget()
        root_layout = QVBoxLayout(tab)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        # ── Выбор папки датасетов ─────────────────────────────────────────
        folder_box, folder_layout = self._create_group_box("Папка с датасетами")
        folder_row = QHBoxLayout()
        self._cm_folder_label = QLabel("Папка не выбрана")
        self._cm_folder_label.setStyleSheet("color: #4b5563; font-size: 10px;")
        self._cm_folder_label.setWordWrap(True)
        folder_row.addWidget(self._cm_folder_label, 1)

        btn_cm_choose = QPushButton("Выбрать папку…")
        btn_cm_choose.clicked.connect(self._cm_choose_folder)
        folder_row.addWidget(btn_cm_choose)

        self._btn_cm_from_openset = QPushButton("← Из Open-set")
        self._btn_cm_from_openset.setToolTip("Подставить папку датасетов из последнего Open-set анализа")
        self._btn_cm_from_openset.clicked.connect(self._cm_load_from_openset)
        folder_row.addWidget(self._btn_cm_from_openset)

        btn_cm_refresh = QPushButton("⟳")
        btn_cm_refresh.setFixedWidth(32)
        btn_cm_refresh.setToolTip("Обновить список")
        btn_cm_refresh.clicked.connect(self._cm_load_classes)
        folder_row.addWidget(btn_cm_refresh)
        folder_layout.addLayout(folder_row)
        root_layout.addWidget(folder_box)

        # ── Сплиттер: список датасетов + превью ───────────────────────────
        splitter = _QSplitter(_Qt.Horizontal)

        # ── Левая панель: список датасетов ────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(4)

        left_layout.addWidget(QLabel("Датасеты (Ctrl+клик — множественный выбор):"))

        self._cm_list = QListWidget()
        self._cm_list.setSelectionMode(QListWidget.ExtendedSelection)
        self._cm_list.itemSelectionChanged.connect(self._cm_on_selection_changed)
        self._cm_list.keyPressEvent = self._cm_list_key_press
        left_layout.addWidget(self._cm_list, 1)

        self._cm_info_label = QLabel("—")
        self._cm_info_label.setStyleSheet("color: gray; font-size: 10px;")
        left_layout.addWidget(self._cm_info_label)

        # Поле нового имени класса
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Имя класса:"))
        self._cm_new_name_edit = _QLineEdit()
        self._cm_new_name_edit.setPlaceholderText("Новое название...")
        name_row.addWidget(self._cm_new_name_edit, 1)
        left_layout.addLayout(name_row)

        # Кнопки действий — ряд 1
        act1 = QHBoxLayout()
        self._btn_cm_merge = QPushButton("⊕ Слить датасеты")
        self._btn_cm_merge.setToolTip(
            "Объединить ≥2 выбранных датасета в один.\n"
            "Укажите имя нового класса в поле выше или оставьте пустым для запроса."
        )
        self._btn_cm_merge.setEnabled(False)
        self._btn_cm_merge.clicked.connect(self._cm_merge)
        act1.addWidget(self._btn_cm_merge)

        self._btn_cm_rename = QPushButton("✏ Переименовать")
        self._btn_cm_rename.setToolTip(
            "Переименовать выбранный датасет.\n"
            "Укажите новое имя в поле выше или оставьте пустым для запроса."
        )
        self._btn_cm_rename.setEnabled(False)
        self._btn_cm_rename.clicked.connect(self._cm_rename)
        act1.addWidget(self._btn_cm_rename)
        left_layout.addLayout(act1)

        # Кнопки действий — ряд 2
        act2 = QHBoxLayout()
        self._btn_cm_delete = QPushButton("🗑 Удалить")
        self._btn_cm_delete.setToolTip("Удалить выбранные датасеты со всеми файлами (необратимо)")
        self._btn_cm_delete.setEnabled(False)
        self._btn_cm_delete.clicked.connect(self._cm_delete)
        act2.addWidget(self._btn_cm_delete)

        self._btn_cm_open_folder = QPushButton("📁 Открыть")
        self._btn_cm_open_folder.setToolTip("Открыть папку датасета в проводнике")
        self._btn_cm_open_folder.clicked.connect(self._cm_open_in_explorer)
        act2.addWidget(self._btn_cm_open_folder)
        left_layout.addLayout(act2)

        splitter.addWidget(left)

        # ── Правая панель: два превью train + val рядом ──────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(4)

        self._cm_preview_title = QLabel("Примеры кадров датасета")
        self._cm_preview_title.setStyleSheet("font-weight: bold; font-size: 11px;")
        right_layout.addWidget(self._cm_preview_title)

        # Слайдер масштаба миниатюр
        _scale_row = QWidget()
        _scale_hbox = QHBoxLayout(_scale_row)
        _scale_hbox.setContentsMargins(0, 0, 0, 0)
        _scale_hbox.setSpacing(6)
        _scale_lbl = QLabel("Размер:")
        _scale_lbl.setStyleSheet("font-size: 10px; color: #aaa;")
        _scale_hbox.addWidget(_scale_lbl)
        self._cm_thumb_slider = QSlider(_Qt.Horizontal)
        self._cm_thumb_slider.setMinimum(80)
        self._cm_thumb_slider.setMaximum(400)
        self._cm_thumb_slider.setValue(200)
        self._cm_thumb_slider.setFixedWidth(160)
        self._cm_thumb_slider.setTickInterval(40)
        self._cm_thumb_slider.setTickPosition(QSlider.TicksBelow)
        _scale_hbox.addWidget(self._cm_thumb_slider)
        self._cm_thumb_size_lbl = QLabel("200 px")
        self._cm_thumb_size_lbl.setStyleSheet("font-size: 10px; color: #aaa; min-width: 42px;")
        _scale_hbox.addWidget(self._cm_thumb_size_lbl)
        _scale_hbox.addStretch()
        right_layout.addWidget(_scale_row)

        # Горизонтальный сплиттер train | val
        preview_splitter = _QSplitter(_Qt.Horizontal)

        # -- Train --
        train_widget = QWidget()
        train_vbox = QVBoxLayout(train_widget)
        train_vbox.setContentsMargins(0, 0, 2, 0)
        train_vbox.setSpacing(2)
        self._cm_train_label = QLabel("Train")
        self._cm_train_label.setStyleSheet("color:#4ade80; font-weight:bold; font-size:10px;")
        train_vbox.addWidget(self._cm_train_label)
        self._cm_preview_scroll_train = _QScrollArea()
        self._cm_preview_scroll_train.setWidgetResizable(True)
        self._cm_preview_scroll_train.setFrameShape(_QScrollArea.StyledPanel)
        self._cm_preview_container_train = None
        self._cm_preview_grid_train = None
        train_vbox.addWidget(self._cm_preview_scroll_train, 1)
        preview_splitter.addWidget(train_widget)

        # -- Val --
        val_widget = QWidget()
        val_vbox = QVBoxLayout(val_widget)
        val_vbox.setContentsMargins(2, 0, 0, 0)
        val_vbox.setSpacing(2)
        self._cm_val_label = QLabel("Val")
        self._cm_val_label.setStyleSheet("color:#60a5fa; font-weight:bold; font-size:10px;")
        val_vbox.addWidget(self._cm_val_label)
        self._cm_preview_scroll_val = _QScrollArea()
        self._cm_preview_scroll_val.setWidgetResizable(True)
        self._cm_preview_scroll_val.setFrameShape(_QScrollArea.StyledPanel)
        self._cm_preview_container_val = None
        self._cm_preview_grid_val = None
        val_vbox.addWidget(self._cm_preview_scroll_val, 1)
        preview_splitter.addWidget(val_widget)

        right_layout.addWidget(preview_splitter, 1)

        splitter.addWidget(right)
        splitter.setSizes([300, 520])
        root_layout.addWidget(splitter, 1)

        self._cm_thumb_slider.valueChanged.connect(self._cm_on_thumb_size_changed)

        # Инициализация атрибутов ДО вызова _cm_clear_preview
        self._cm_preview_container_train = None
        self._cm_preview_grid_train = None
        self._cm_preview_container_val = None
        self._cm_preview_grid_val = None
        self._cm_clear_preview()

        # Состояние
        self._cm_classes_dir = None
        self._cm_classes_data = []

        return tab

    def _cm_choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Папка с YOLO-датасетами")
        if folder:
            self._cm_classes_dir = folder
            self._cm_folder_label.setText(folder)
            self._cm_load_classes()

    def _cm_load_from_openset(self):
        """Подставляет папку yolo_datasets из последнего Open-set анализа."""
        # Сначала пробуем combined_yolo_datasets (режим папки)
        yolo_dir = getattr(self, "_cm_last_yolo_datasets_dir", None)
        if not yolo_dir or not os.path.isdir(yolo_dir):
            # Fallback: поищем рядом с unknown_classes_dir
            classes_dir = getattr(self, "openset_unknown_classes_dir", None)
            if classes_dir and os.path.isdir(classes_dir):
                parent = os.path.dirname(classes_dir)
                candidate = os.path.join(parent, "combined_yolo_datasets")
                if os.path.isdir(candidate):
                    yolo_dir = candidate
        if not yolo_dir or not os.path.isdir(yolo_dir):
            QMessageBox.information(
                self, "Инспектор датасетов",
                "Нет данных из последнего Open-set анализа.\n"
                "Запустите анализ папки сначала."
            )
            return
        self._cm_classes_dir = yolo_dir
        self._cm_folder_label.setText(yolo_dir)
        self._cm_load_classes()

    def _cm_load_classes(self):
        from PyQt5.QtWidgets import QListWidgetItem as _QListWidgetItem
        self._cm_list.clear()
        self._cm_classes_data = []
        self._cm_clear_preview()
        self._cm_preview_title.setText("Примеры кадров датасета")
        self._cm_update_buttons()

        if not self._cm_classes_dir or not os.path.isdir(self._cm_classes_dir):
            self._cm_info_label.setText("Папка не выбрана или не существует")
            return

        IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
        entries = []
        for name in sorted(os.listdir(self._cm_classes_dir)):
            ds_path = os.path.join(self._cm_classes_dir, name)
            if not os.path.isdir(ds_path):
                continue
            # Считаем кадры в images/train и images/val
            train_dir = os.path.join(ds_path, "images", "train")
            val_dir   = os.path.join(ds_path, "images", "val")
            train_cnt = sum(
                1 for f in os.listdir(train_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXT
            ) if os.path.isdir(train_dir) else 0
            val_cnt = sum(
                1 for f in os.listdir(val_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXT
            ) if os.path.isdir(val_dir) else 0
            # Кропы лежат в корне папки датасета (не в images/)
            crop_cnt = sum(
                1 for f in os.listdir(ds_path)
                if os.path.isfile(os.path.join(ds_path, f))
                and os.path.splitext(f)[1].lower() in IMAGE_EXT
            )
            yaml_path = os.path.join(ds_path, "data.yaml")
            entries.append({
                "name":      name,
                "dir":       ds_path,
                "yaml":      yaml_path if os.path.isfile(yaml_path) else None,
                "train_cnt": train_cnt,
                "val_cnt":   val_cnt,
                "crop_cnt":  crop_cnt,
                "total":     train_cnt + val_cnt,
            })

        self._cm_classes_data = entries
        for e in entries:
            crop_info = f", кропов: {e['crop_cnt']}" if e["crop_cnt"] > 0 else ""
            item = _QListWidgetItem(
                f"{e['name']}  (train: {e['train_cnt']}, val: {e['val_cnt']}{crop_info})"
            )
            item.setData(Qt.UserRole, e)
            self._cm_list.addItem(item)

        total_frames = sum(e["total"] for e in entries)
        total_crops  = sum(e["crop_cnt"] for e in entries)
        crops_info = f",  кропов: {total_crops}" if total_crops > 0 else ""
        self._cm_info_label.setText(f"Датасетов: {len(entries)},  кадров всего: {total_frames}{crops_info}")

    def _cm_on_selection_changed(self):
        self._cm_update_buttons()
        items = self._cm_list.selectedItems()
        if not items:
            self._cm_clear_preview()
            self._cm_preview_title.setText("Примеры кадров датасета")
            return
        entry = items[0].data(Qt.UserRole)
        if len(items) == 1:
            crop_part = f", кропов: {entry['crop_cnt']}" if entry.get("crop_cnt", 0) > 0 else ""
            self._cm_preview_title.setText(
                f"{entry['name']}  —  train: {entry['train_cnt']}, val: {entry['val_cnt']}{crop_part}"
            )
            self._cm_new_name_edit.setText(entry["name"])
        else:
            self._cm_preview_title.setText(f"Выбрано датасетов: {len(items)}")
        self._cm_refresh_preview()

    def _cm_refresh_preview(self):
        items = self._cm_list.selectedItems()
        if not items:
            self._cm_clear_preview()
            return
        entry = items[0].data(Qt.UserRole)
        train_dir = os.path.join(entry["dir"], "images", "train")
        val_dir   = os.path.join(entry["dir"], "images", "val")
        self._cm_train_label.setText(f"Train  ({entry['train_cnt']} кадров)")
        self._cm_val_label.setText(f"Val  ({entry['val_cnt']} кадров)")
        # Читаем имена классов из data.yaml
        class_names = {}
        yaml_path = entry.get("yaml")
        if yaml_path and os.path.isfile(yaml_path):
            try:
                import yaml as _yaml
                with open(yaml_path, "r", encoding="utf-8") as _yf:
                    _cfg = _yaml.safe_load(_yf) or {}
                raw = _cfg.get("names", {})
                if isinstance(raw, dict):
                    class_names = {int(k): str(v) for k, v in raw.items()}
                elif isinstance(raw, list):
                    class_names = {i: str(v) for i, v in enumerate(raw)}
            except Exception:
                pass
        self._cm_show_thumbnails_in(train_dir, "train", class_names=class_names,
                                      thumb_size=self._cm_thumb_slider.value())
        self._cm_show_thumbnails_in(val_dir, "val", class_names=class_names,
                                    thumb_size=self._cm_thumb_slider.value())

    def _cm_on_thumb_size_changed(self, value):
        self._cm_thumb_size_lbl.setText(f"{value} px")
        self._cm_refresh_preview()

    def _cm_update_buttons(self):
        n = len(self._cm_list.selectedItems())
        self._btn_cm_merge.setEnabled(n >= 2)
        self._btn_cm_rename.setEnabled(n == 1)
        self._btn_cm_delete.setEnabled(n >= 1)

    def _cm_list_key_press(self, event):
        from PyQt5.QtCore import Qt as _Qt
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self._btn_cm_delete.isEnabled():
                self._cm_delete()
        else:
            QListWidget.keyPressEvent(self._cm_list, event)

    def _cm_show_thumbnails_in(self, image_dir, which, max_thumbs=32, thumb_size=200, class_names=None):
        """Заполняет превью-панель для 'train' или 'val'. Рисует bbox с именами классов."""
        from PyQt5.QtGui import QPixmap as _QPixmap, QPainter as _QPainter, QPen as _QPen, QColor as _QColor, QFont as _QFont
        from PyQt5.QtCore import Qt as _Qt
        if class_names is None:
            class_names = {}

        # Цвета для классов (циклически)
        _PALETTE = [
            (0,   255, 60),   # ярко-зелёный
            (80,  140, 255),  # синий
            (255, 200, 50),   # жёлтый
            (200, 80,  255),  # фиолетовый
            (50,  220, 220),  # циан
            (255, 80,  80),   # красный
        ]

        # Выбираем нужный scroll/grid
        if which == "train":
            scroll = self._cm_preview_scroll_train
            container_attr = "_cm_preview_container_train"
            grid_attr = "_cm_preview_grid_train"
        else:
            scroll = self._cm_preview_scroll_val
            container_attr = "_cm_preview_container_val"
            grid_attr = "_cm_preview_grid_val"

        # Создаём новый контейнер
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(3)
        grid.setContentsMargins(3, 3, 3, 3)
        grid.setAlignment(_Qt.AlignTop | _Qt.AlignLeft)
        old = getattr(self, container_attr, None)
        if old is not None:
            old.deleteLater()
        setattr(self, container_attr, container)
        setattr(self, grid_attr, grid)
        scroll.setWidget(container)

        IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
        files = []
        if os.path.isdir(image_dir):
            files = [
                os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))
                if os.path.splitext(f)[1].lower() in IMAGE_EXT
            ][:max_thumbs]

        labels_dir = image_dir.replace(
            os.path.join("images", which), os.path.join("labels", which)
        )

        font = _QFont()
        font.setPointSize(7)
        font.setBold(True)

        cols = 3
        for idx, fpath in enumerate(files):
            pix = _QPixmap(fpath)
            if pix.isNull():
                continue

            orig_w = pix.width()
            orig_h = pix.height()

            # Читаем метки
            stem = os.path.splitext(os.path.basename(fpath))[0]
            label_path = os.path.join(labels_dir, stem + ".txt")
            annotations = []  # список (class_id, x, y, w, h) в пикселях
            if os.path.isfile(label_path):
                try:
                    with open(label_path, "r") as _lf:
                        for line in _lf:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cid = int(parts[0])
                                cx, cy, bw, bh = map(float, parts[1:])
                                x1 = int((cx - bw / 2) * orig_w)
                                y1 = int((cy - bh / 2) * orig_h)
                                bw_px = int(bw * orig_w)
                                bh_px = int(bh * orig_h)
                                annotations.append((cid, x1, y1, bw_px, bh_px))
                except Exception:
                    pass

            # Рисуем bbox на оригинальном pixmap (до масштабирования — точнее)
            if annotations:
                pix = pix.copy()
                painter = _QPainter(pix)
                painter.setFont(font)
                for (cid, x, y, bw_px, bh_px) in annotations:
                    r, g, b = _PALETTE[cid % len(_PALETTE)]
                    color = _QColor(r, g, b, 230)
                    pen = _QPen(color)
                    pen.setWidth(max(3, orig_w // 300))
                    painter.setPen(pen)
                    painter.drawRect(x, y, bw_px, bh_px)
                    # Подпись класса
                    label_text = class_names.get(cid, str(cid))
                    bg_color = _QColor(r, g, b, 180)
                    text_rect = painter.fontMetrics().boundingRect(label_text)
                    ty = max(y - 2, text_rect.height())
                    painter.fillRect(x, ty - text_rect.height(), text_rect.width() + 4, text_rect.height() + 2, bg_color)
                    painter.setPen(_QColor(255, 255, 255, 255))
                    painter.drawText(x + 2, ty, label_text)
                painter.end()

            pix = pix.scaled(thumb_size, thumb_size, _Qt.KeepAspectRatio, _Qt.SmoothTransformation)

            n_obj = len(annotations)
            obj_names = ", ".join(class_names.get(a[0], str(a[0])) for a in annotations)
            tooltip = os.path.basename(fpath)
            if n_obj > 0:
                tooltip += f"\nОбъектов: {n_obj}  ({obj_names})"

            lbl = QLabel()
            lbl.setPixmap(pix)
            lbl.setFixedSize(thumb_size + 4, thumb_size + 4)
            lbl.setAlignment(_Qt.AlignCenter)
            lbl.setToolTip(tooltip)
            border_color = "#e05050" if n_obj > 0 else "#555"
            lbl.setStyleSheet(f"border: 1px solid {border_color}; background: #222;")
            row, col = divmod(idx, cols)
            grid.addWidget(lbl, row, col)

        if not files:
            no_img = QLabel("Нет кадров")
            no_img.setStyleSheet("color: gray; font-size: 10px;")
            grid.addWidget(no_img, 0, 0)

    def _cm_clear_preview(self):
        from PyQt5.QtCore import Qt as _Qt
        # Очищаем обе панели
        for which in ("train", "val"):
            scroll = getattr(self, f"_cm_preview_scroll_{which}", None)
            container_attr = f"_cm_preview_container_{which}"
            if scroll is None:
                continue
            container = QWidget()
            grid = QGridLayout(container)
            grid.setSpacing(3)
            grid.setContentsMargins(3, 3, 3, 3)
            grid.setAlignment(_Qt.AlignTop | _Qt.AlignLeft)
            old = getattr(self, container_attr, None)
            if old is not None:
                old.deleteLater()
            setattr(self, container_attr, container)
            scroll.setWidget(container)

    def _cm_merge(self):
        """Слияние ≥2 датасетов в один новый YOLO-датасет.

        Алгоритм:
          1. Копируем все кадры/метки источников во временную папку.
          2. Удаляем ВСЕ выбранные датасеты.
          3. Переименовываем временную папку в целевую.
        После этого в папке датасетов остаётся ровно один новый датасет.
        """
        import shutil as _shutil
        import tempfile as _tmp
        import yaml as _yaml
        from PyQt5.QtWidgets import QInputDialog as _QInputDialog

        items = self._cm_list.selectedItems()
        if len(items) < 2:
            return
        entries = [it.data(Qt.UserRole) for it in items]

        # Имя нового класса — из поля или через диалог
        name = self._cm_new_name_edit.text().strip()
        if not name:
            default = entries[0]["name"] + "_merged"
            name, ok = _QInputDialog.getText(
                self, "Слить датасеты",
                f"Объединение {len(entries)} датасетов.\nНазвание нового класса:",
                text=default,
            )
            if not ok or not name.strip():
                return
            name = name.strip()

        target_dir = os.path.join(self._cm_classes_dir, name)

        # Если целевая папка уже существует и НЕ является одним из источников —
        # спрашиваем подтверждение (перезаписать).
        source_dirs_abs = {os.path.abspath(e["dir"]) for e in entries}
        if os.path.exists(target_dir) and os.path.abspath(target_dir) not in source_dirs_abs:
            reply = QMessageBox.question(
                self, "Слить датасеты",
                f"Папка «{name}» уже существует и не входит в выборку.\n"
                f"Перезаписать её содержимое?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

        # Шаг 1: копируем всё во временную папку
        tmp_dir = _tmp.mkdtemp(dir=self._cm_classes_dir, prefix="_merge_tmp_")
        try:
            for split in ("train", "val"):
                os.makedirs(os.path.join(tmp_dir, "images", split), exist_ok=True)
                os.makedirs(os.path.join(tmp_dir, "labels", split), exist_ok=True)

            total_train = 0
            total_val = 0

            for entry in entries:
                src_ds = entry["dir"]
                for split in ("train", "val"):
                    img_src = os.path.join(src_ds, "images", split)
                    lbl_src = os.path.join(src_ds, "labels", split)
                    img_dst = os.path.join(tmp_dir, "images", split)
                    lbl_dst = os.path.join(tmp_dir, "labels", split)
                    if not os.path.isdir(img_src):
                        continue
                    for fname in os.listdir(img_src):
                        src = os.path.join(img_src, fname)
                        dst = os.path.join(img_dst, fname)
                        if os.path.exists(dst):
                            base, ext = os.path.splitext(fname)
                            c = 1
                            while os.path.exists(dst):
                                dst = os.path.join(img_dst, f"{base}_{c}{ext}")
                                c += 1
                        _shutil.copy2(src, dst)
                        stem = os.path.splitext(fname)[0]
                        dst_stem = os.path.splitext(os.path.basename(dst))[0]
                        lbl_file = os.path.join(lbl_src, stem + ".txt")
                        if os.path.isfile(lbl_file):
                            _shutil.copy2(lbl_file, os.path.join(lbl_dst, dst_stem + ".txt"))
                        if os.path.splitext(fname)[1].lower() in IMAGE_EXT:
                            if split == "train":
                                total_train += 1
                            else:
                                total_val += 1

            # Записываем data.yaml во временную папку
            data_cfg = {
                "path":  target_dir,
                "train": "images/train",
                "val":   "images/val",
                "nc":    1,
                "names": {0: name},
            }
            with open(os.path.join(tmp_dir, "data.yaml"), "w", encoding="utf-8") as _yf:
                _yaml.dump(data_cfg, _yf, allow_unicode=True, default_flow_style=False)

            # Шаг 2: удаляем ВСЕ выбранные источники
            for entry in entries:
                _shutil.rmtree(entry["dir"], ignore_errors=True)

            # Удаляем существующую target_dir, если осталась (не входила в sources)
            if os.path.exists(target_dir):
                _shutil.rmtree(target_dir, ignore_errors=True)

            # Шаг 3: переименовываем временную папку в целевую
            os.rename(tmp_dir, target_dir)
            tmp_dir = None  # больше не нужно чистить в finally

        except Exception as exc:
            # Что-то пошло не так — удаляем временную папку
            if tmp_dir and os.path.exists(tmp_dir):
                _shutil.rmtree(tmp_dir, ignore_errors=True)
            QMessageBox.critical(self, "Ошибка слияния", str(exc))
            self._cm_load_classes()
            return
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                _shutil.rmtree(tmp_dir, ignore_errors=True)

        QMessageBox.information(
            self, "Слияние завершено",
            f"Датасет «{name}» создан.\n"
            f"Train: {total_train} кадров, Val: {total_val} кадров.\n"
            f"Удалено источников: {len(entries)}.",
        )
        self._cm_new_name_edit.clear()
        self._cm_load_classes()

    def _cm_rename(self):
        """Переименовывает датасет и обновляет data.yaml."""
        import shutil as _shutil
        import yaml as _yaml
        from PyQt5.QtWidgets import QInputDialog as _QInputDialog

        items = self._cm_list.selectedItems()
        if len(items) != 1:
            return
        entry = items[0].data(Qt.UserRole)

        # Имя из поля или через диалог
        name = self._cm_new_name_edit.text().strip()
        if not name or name == entry["name"]:
            name, ok = _QInputDialog.getText(
                self, "Переименовать датасет",
                "Новое название класса:",
                text=entry["name"],
            )
            if not ok or not name.strip() or name.strip() == entry["name"]:
                return
            name = name.strip()

        new_dir = os.path.join(self._cm_classes_dir, name)
        if os.path.exists(new_dir):
            QMessageBox.warning(self, "Переименование", f"Датасет «{name}» уже существует.")
            return

        _shutil.move(entry["dir"], new_dir)

        # Обновляем data.yaml
        yaml_path = os.path.join(new_dir, "data.yaml")
        if os.path.isfile(yaml_path):
            try:
                with open(yaml_path, "r", encoding="utf-8") as _yf:
                    cfg = _yaml.safe_load(_yf) or {}
                cfg["path"] = new_dir
                cfg["names"] = {0: name}
                with open(yaml_path, "w", encoding="utf-8") as _yf:
                    _yaml.dump(cfg, _yf, allow_unicode=True, default_flow_style=False)
            except Exception:
                pass

        self._cm_new_name_edit.clear()
        self._cm_load_classes()

    def _cm_delete(self):
        import shutil as _shutil

        items = self._cm_list.selectedItems()
        if not items:
            return
        entries = [it.data(Qt.UserRole) for it in items]
        total_frames = sum(e["total"] for e in entries)

        # Если пользователь ранее выбрал "запомнить" — не спрашиваем
        if not getattr(self, "_cm_delete_skip_confirm", False):
            names_preview = "\n".join(f"  • {e['name']}" for e in entries[:12])
            if len(entries) > 12:
                names_preview += f"\n  ... и ещё {len(entries) - 12}"

            dlg = QDialog(self)
            dlg.setWindowTitle("Удаление датасетов")
            dlg.setMinimumWidth(380)
            _vbox = QVBoxLayout(dlg)

            msg = QLabel(
                f"Удалить {len(entries)} датасет(ов) с {total_frames} кадрами?\n\n"
                f"{names_preview}\n\nДействие необратимо!"
            )
            msg.setWordWrap(True)
            _vbox.addWidget(msg)

            remember_chk = QCheckBox("Запомнить выбор (не спрашивать снова)")
            _vbox.addWidget(remember_chk)

            _btns = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
            _btns.button(QDialogButtonBox.No).setDefault(True)
            _btns.accepted.connect(dlg.accept)
            _btns.rejected.connect(dlg.reject)
            _vbox.addWidget(_btns)

            if dlg.exec_() != QDialog.Accepted:
                return

            if remember_chk.isChecked():
                self._cm_delete_skip_confirm = True

        for entry in entries:
            _shutil.rmtree(entry["dir"], ignore_errors=True)
        self._cm_load_classes()
    def _cm_open_in_explorer(self):
        items = self._cm_list.selectedItems()
        if items:
            folder = items[0].data(Qt.UserRole)["dir"]
        elif self._cm_classes_dir and os.path.isdir(self._cm_classes_dir):
            folder = self._cm_classes_dir
        else:
            return
        subprocess.Popen(f'explorer "{folder}"')

    # ═══════════════════════════════════════════════════════════════════════
    #  ВКЛ. ClusterInspector — инспектор combined_unknown_classes
    # ═══════════════════════════════════════════════════════════════════════

    def _build_cluster_inspector_tab(self):
        from PyQt5.QtWidgets import QScrollArea as _QSA, QSplitter as _QSpl, QLineEdit as _QLE
        from PyQt5.QtCore import Qt as _Qt

        self._ci_folder = None          # путь к combined_unknown_classes
        self._ci_clusters = []          # [{"name": str, "dir": str, "count": int}, ...]

        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Верхняя строка: загрузка папки ───────────────────────────────
        top_row = QHBoxLayout()
        btn_open = QPushButton("📂 Открыть combined_unknown_classes...")
        btn_open.clicked.connect(self._ci_open_folder)
        top_row.addWidget(btn_open)
        self._ci_path_lbl = QLabel("Папка не выбрана")
        self._ci_path_lbl.setStyleSheet("color: gray; font-size: 10px;")
        top_row.addWidget(self._ci_path_lbl, 1)
        root.addLayout(top_row)

        # ── Поиск/фильтр ─────────────────────────────────────────────────
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Фильтр:"))
        self._ci_filter_edit = _QLE()
        self._ci_filter_edit.setPlaceholderText("Часть имени кластера...")
        self._ci_filter_edit.textChanged.connect(self._ci_apply_filter)
        filter_row.addWidget(self._ci_filter_edit, 1)
        self._ci_count_lbl = QLabel("")
        self._ci_count_lbl.setStyleSheet("font-size: 10px; color: #aaa;")
        filter_row.addWidget(self._ci_count_lbl)
        root.addLayout(filter_row)

        # ── Сплиттер: список кластеров | превью миниатюр ──────────────────
        splitter = _QSpl(_Qt.Horizontal)

        # Левая панель — список
        left = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(3)
        self._ci_list = QListWidget()
        self._ci_list.setSelectionMode(QListWidget.ExtendedSelection)
        self._ci_list.itemSelectionChanged.connect(self._ci_on_selection_changed)
        self._ci_list.keyPressEvent = self._ci_list_key_press
        left_v.addWidget(self._ci_list, 1)
        splitter.addWidget(left)

        # Правая панель — превью
        right = QWidget()
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(3)

        # Слайдер масштаба
        scale_row = QHBoxLayout()
        scale_lbl = QLabel("Размер:")
        scale_lbl.setStyleSheet("font-size: 10px; color: #aaa;")
        scale_row.addWidget(scale_lbl)
        self._ci_thumb_slider = QSlider(_Qt.Horizontal)
        self._ci_thumb_slider.setMinimum(60)
        self._ci_thumb_slider.setMaximum(300)
        self._ci_thumb_slider.setValue(120)
        self._ci_thumb_slider.setFixedWidth(140)
        self._ci_thumb_slider.setTickInterval(30)
        self._ci_thumb_slider.setTickPosition(QSlider.TicksBelow)
        self._ci_thumb_slider.valueChanged.connect(self._ci_refresh_preview)
        scale_row.addWidget(self._ci_thumb_slider)
        self._ci_thumb_lbl = QLabel("120 px")
        self._ci_thumb_lbl.setStyleSheet("font-size: 10px; color: #aaa; min-width: 42px;")
        self._ci_thumb_slider.valueChanged.connect(
            lambda v: self._ci_thumb_lbl.setText(f"{v} px"))
        scale_row.addWidget(self._ci_thumb_lbl)
        scale_row.addStretch()
        right_v.addLayout(scale_row)

        self._ci_preview_title = QLabel("Выберите кластер для просмотра")
        self._ci_preview_title.setStyleSheet("font-weight: bold; font-size: 10px;")
        right_v.addWidget(self._ci_preview_title)

        self._ci_preview_scroll = _QSA()
        self._ci_preview_scroll.setWidgetResizable(True)
        self._ci_preview_container = None
        right_v.addWidget(self._ci_preview_scroll, 1)
        splitter.addWidget(right)
        splitter.setSizes([260, 500])
        root.addWidget(splitter, 1)

        # ── Нижняя панель: действия ───────────────────────────────────────
        action_row = QHBoxLayout()
        self._ci_btn_merge = QPushButton("🔀 Объединить выбранные")
        self._ci_btn_merge.setEnabled(False)
        self._ci_btn_merge.clicked.connect(self._ci_merge_selected)
        action_row.addWidget(self._ci_btn_merge)

        action_row.addWidget(QLabel("Имя датасета:"))
        self._ci_ds_name_edit = _QLE()
        self._ci_ds_name_edit.setPlaceholderText("Например: My_Object")
        self._ci_ds_name_edit.setFixedWidth(160)
        action_row.addWidget(self._ci_ds_name_edit)

        self._ci_btn_create_ds = QPushButton("✅ Создать датасет")
        self._ci_btn_create_ds.setEnabled(False)
        self._ci_btn_create_ds.clicked.connect(self._ci_create_dataset)
        action_row.addWidget(self._ci_btn_create_ds)

        self._ci_btn_split = QPushButton("✂ Разделить кластер")
        self._ci_btn_split.setEnabled(False)
        self._ci_btn_split.setToolTip("Разделить один кластер на два по заданной пропорции")
        self._ci_btn_split.clicked.connect(self._ci_split_selected)
        action_row.addWidget(self._ci_btn_split)

        action_row.addStretch()
        self._ci_btn_del = QPushButton("🗑 Удалить кластер(ы)")
        self._ci_btn_del.setEnabled(False)
        self._ci_btn_del.clicked.connect(self._ci_delete_selected)
        action_row.addWidget(self._ci_btn_del)
        root.addLayout(action_row)

        self._ci_status_lbl = QLabel("")
        self._ci_status_lbl.setStyleSheet("font-size: 10px; color: #6cf;")
        root.addWidget(self._ci_status_lbl)

        return tab

    # ── Методы CI ────────────────────────────────────────────────────────

    def _ci_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите combined_unknown_classes")
        if not folder:
            return
        self._ci_folder = folder
        self._ci_path_lbl.setText(folder)
        self._ci_load_clusters()

    def _ci_load_clusters(self):
        self._ci_list.clear()
        self._ci_clusters = []
        if not self._ci_folder or not os.path.isdir(self._ci_folder):
            return
        IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
        names = sorted(
            d for d in os.listdir(self._ci_folder)
            if os.path.isdir(os.path.join(self._ci_folder, d))
        )
        for name in names:
            d = os.path.join(self._ci_folder, name)
            cnt = sum(
                1 for f in os.listdir(d)
                if os.path.splitext(f)[1].lower() in IMAGE_EXT
            )
            self._ci_clusters.append({"name": name, "dir": d, "count": cnt})
        self._ci_apply_filter()

    def _ci_apply_filter(self):
        text = self._ci_filter_edit.text().strip().lower() if hasattr(self, "_ci_filter_edit") else ""
        from PyQt5.QtWidgets import QListWidgetItem as _QLWI
        self._ci_list.clear()
        shown = 0
        for c in self._ci_clusters:
            if text and text not in c["name"].lower():
                continue
            item = _QLWI(f"{c['name']}  ({c['count']} кропов)")
            item.setData(Qt.UserRole, c)
            self._ci_list.addItem(item)
            shown += 1
        total = len(self._ci_clusters)
        self._ci_count_lbl.setText(f"{shown}/{total} кластеров")

    def _ci_on_selection_changed(self):
        items = self._ci_list.selectedItems()
        n = len(items)
        self._ci_btn_merge.setEnabled(n >= 2)
        self._ci_btn_split.setEnabled(n == 1)
        self._ci_btn_create_ds.setEnabled(n >= 1)
        self._ci_btn_del.setEnabled(n >= 1)
        if n == 1:
            c = items[0].data(Qt.UserRole)
            self._ci_preview_title.setText(f"{c['name']}  —  {c['count']} кропов")
        elif n > 1:
            total = sum(it.data(Qt.UserRole)["count"] for it in items)
            self._ci_preview_title.setText(f"Выбрано: {n} кластеров, {total} кропов")
        else:
            self._ci_preview_title.setText("Выберите кластер для просмотра")
        self._ci_refresh_preview()

    def _ci_refresh_preview(self, _=None):
        from PyQt5.QtWidgets import QScrollArea as _QSA
        from PyQt5.QtGui import QPixmap as _QPixmap
        from PyQt5.QtCore import Qt as _Qt

        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(3)
        grid.setContentsMargins(3, 3, 3, 3)
        grid.setAlignment(_Qt.AlignTop | _Qt.AlignLeft)
        old = self._ci_preview_container
        if old is not None:
            old.deleteLater()
        self._ci_preview_container = container
        self._ci_preview_scroll.setWidget(container)

        items = self._ci_list.selectedItems()
        if not items:
            no_lbl = QLabel("Нет выбранных кластеров")
            no_lbl.setStyleSheet("color: gray; font-size: 10px;")
            grid.addWidget(no_lbl, 0, 0)
            return

        thumb = self._ci_thumb_slider.value()
        IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
        MAX_TOTAL = 64

        all_files = []
        for it in items:
            c = it.data(Qt.UserRole)
            d = c["dir"]
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if os.path.splitext(f)[1].lower() in IMAGE_EXT:
                        all_files.append(os.path.join(d, f))
        all_files = all_files[:MAX_TOTAL]

        cols = max(1, 500 // (thumb + 6))
        for idx, fpath in enumerate(all_files):
            pix = _QPixmap(fpath)
            if pix.isNull():
                continue
            pix = pix.scaled(thumb, thumb, _Qt.KeepAspectRatio, _Qt.SmoothTransformation)
            lbl = QLabel()
            lbl.setPixmap(pix)
            lbl.setFixedSize(thumb + 4, thumb + 4)
            lbl.setAlignment(_Qt.AlignCenter)
            lbl.setToolTip(os.path.basename(fpath))
            lbl.setStyleSheet("border: 1px solid #555; background: #222;")
            row, col = divmod(idx, cols)
            grid.addWidget(lbl, row, col)

        if not all_files:
            no_lbl = QLabel("Нет изображений")
            no_lbl.setStyleSheet("color: gray; font-size: 10px;")
            grid.addWidget(no_lbl, 0, 0)

    def _ci_list_key_press(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self._ci_btn_del.isEnabled():
                self._ci_delete_selected()
        else:
            QListWidget.keyPressEvent(self._ci_list, event)

    def _ci_merge_selected(self):
        import shutil as _sh
        items = self._ci_list.selectedItems()
        if len(items) < 2:
            return
        clusters = [it.data(Qt.UserRole) for it in items]
        # Сливаем в первый (по алфавиту)
        clusters.sort(key=lambda c: c["name"])
        target = clusters[0]
        for src in clusters[1:]:
            if not os.path.isdir(src["dir"]):
                continue
            for fname in os.listdir(src["dir"]):
                src_file = os.path.join(src["dir"], fname)
                if not os.path.isfile(src_file):
                    continue
                dst_file = os.path.join(target["dir"], fname)
                if os.path.exists(dst_file):
                    base, ext = os.path.splitext(fname)
                    counter = 1
                    while os.path.exists(dst_file):
                        dst_file = os.path.join(target["dir"], f"{base}_m{counter}{ext}")
                        counter += 1
                _sh.move(src_file, dst_file)
            try:
                os.rmdir(src["dir"])
            except OSError:
                pass
        self._ci_status_lbl.setText(
            f"Объединено {len(clusters)} кластеров → {target['name']}"
        )
        self._ci_load_clusters()

    def _ci_create_dataset(self):
        import shutil as _sh, random as _rng, math as _math
        from PyQt5.QtWidgets import QInputDialog as _QID

        items = self._ci_list.selectedItems()
        if not items:
            return

        ds_name = self._ci_ds_name_edit.text().strip()
        if not ds_name:
            ds_name, ok = _QID.getText(self, "Имя датасета", "Введите имя класса/датасета:")
            if not ok or not ds_name.strip():
                return
            ds_name = ds_name.strip()
        self._ci_ds_name_edit.setText(ds_name)

        # Папка вывода — combined_yolo_datasets рядом с combined_unknown_classes
        parent = os.path.dirname(self._ci_folder)
        out_dir = os.path.join(parent, "combined_yolo_datasets", ds_name)
        for split in ("train", "val"):
            os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

        IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
        all_images = []
        for it in items:
            c = it.data(Qt.UserRole)
            d = c["dir"]
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if os.path.splitext(f)[1].lower() in IMAGE_EXT:
                        all_images.append(os.path.join(d, f))

        if not all_images:
            self._ci_status_lbl.setText("Нет изображений в выбранных кластерах")
            return

        _rng.seed(42)
        _rng.shuffle(all_images)
        val_count = max(1, int(len(all_images) * 0.15))
        train_imgs = all_images[val_count:]
        val_imgs   = all_images[:val_count]

        # Объект занимает весь кроп → bbox = 0 0.5 0.5 1.0 1.0
        yolo_line = "0 0.500000 0.500000 1.000000 1.000000\n"

        def _copy_split(imgs, split):
            for src in imgs:
                fname = os.path.basename(src)
                stem  = os.path.splitext(fname)[0]
                _sh.copy2(src, os.path.join(out_dir, "images", split, fname))
                with open(os.path.join(out_dir, "labels", split, stem + ".txt"), "w") as _f:
                    _f.write(yolo_line)

        _copy_split(train_imgs, "train")
        _copy_split(val_imgs,   "val")

        # data.yaml
        try:
            import yaml as _yaml
            cfg = {
                "path":  out_dir,
                "train": "images/train",
                "val":   "images/val",
                "nc":    1,
                "names": {0: ds_name},
            }
            with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as _yf:
                _yaml.dump(cfg, _yf, allow_unicode=True, default_flow_style=False)
        except ImportError:
            pass

        self._ci_status_lbl.setText(
            f"✅ Датасет «{ds_name}» создан: train={len(train_imgs)}, val={len(val_imgs)}  →  {out_dir}"
        )

    def _ci_split_selected(self):
        """Разделить один кластер на два по указанной пропорции."""
        import shutil as _sh, random as _rng
        from PyQt5.QtWidgets import QDialog as _QDlg, QSpinBox as _QSB, QDialogButtonBox as _QDBB
        from PyQt5.QtCore import Qt as _Qt

        items = self._ci_list.selectedItems()
        if len(items) != 1:
            return
        cluster = items[0].data(Qt.UserRole)

        IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
        all_files = sorted(
            f for f in os.listdir(cluster["dir"])
            if os.path.splitext(f)[1].lower() in IMAGE_EXT
        )
        total = len(all_files)
        if total < 2:
            self._ci_status_lbl.setText("Недостаточно изображений для разделения (нужно минимум 2)")
            return

        # ── Диалог выбора пропорции ─────────────────────────────────────
        dlg = _QDlg(self)
        dlg.setWindowTitle("Разделить кластер")
        dlg.setWindowFlags(dlg.windowFlags() & ~_Qt.WindowContextHelpButtonHint)
        dlg_layout = QVBoxLayout(dlg)

        info = QLabel(
            f"Кластер: <b>{cluster['name']}</b><br>"
            f"Изображений: <b>{total}</b><br><br>"
            f"Укажите долю первой части (%):"
        )
        info.setTextFormat(_Qt.RichText)
        dlg_layout.addWidget(info)

        ratio_row = QHBoxLayout()
        spin = _QSB()
        spin.setRange(1, 99)
        spin.setValue(50)
        spin.setSuffix(" %")
        spin.setFixedWidth(90)
        ratio_row.addWidget(spin)
        preview_lbl = QLabel()

        def _update_preview(v):
            n1 = max(1, round(total * v / 100))
            n2 = total - n1
            preview_lbl.setText(f"→ часть 1: {n1} шт.  /  часть 2: {n2} шт.")

        spin.valueChanged.connect(_update_preview)
        _update_preview(spin.value())
        ratio_row.addWidget(preview_lbl)
        ratio_row.addStretch()
        dlg_layout.addLayout(ratio_row)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Имя второй части:"))
        from PyQt5.QtWidgets import QLineEdit as _QLE2
        name_edit = _QLE2()
        name_edit.setPlaceholderText(f"{cluster['name']}_B")
        name_edit.setFixedWidth(220)
        name_row.addWidget(name_edit)
        name_row.addStretch()
        dlg_layout.addLayout(name_row)

        btns = _QDBB(_QDBB.Ok | _QDBB.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        dlg_layout.addWidget(btns)

        if dlg.exec_() != _QDlg.Accepted:
            return

        pct   = spin.value()
        n1    = max(1, round(total * pct / 100))
        n2    = total - n1
        if n2 < 1:
            self._ci_status_lbl.setText("Нельзя разделить: одна часть оказалась пустой")
            return

        new_name = name_edit.text().strip() or f"{cluster['name']}_B"
        new_dir  = os.path.join(os.path.dirname(cluster["dir"]), new_name)
        if os.path.exists(new_dir):
            self._ci_status_lbl.setText(f"Папка '{new_name}' уже существует — выберите другое имя")
            return
        os.makedirs(new_dir, exist_ok=False)

        _rng.seed(None)
        _rng.shuffle(all_files)
        second_part = all_files[n1:]      # файлы для второй части

        for fname in second_part:
            src = os.path.join(cluster["dir"], fname)
            _sh.move(src, os.path.join(new_dir, fname))

        self._ci_status_lbl.setText(
            f"✅ Кластер '{cluster['name']}' разделён: {n1} шт. оставлено, {n2} шт. → '{new_name}'"
        )
        self._ci_load_clusters()

    def _ci_delete_selected(self):
        import shutil as _sh
        items = self._ci_list.selectedItems()
        if not items:
            return
        clusters = [it.data(Qt.UserRole) for it in items]
        reply = QMessageBox.question(
            self, "Удаление кластеров",
            f"Удалить {len(clusters)} кластер(ов)?\n"
            + "\n".join(f"  • {c['name']}" for c in clusters[:10])
            + ("\n  ..." if len(clusters) > 10 else ""),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        for c in clusters:
            _sh.rmtree(c["dir"], ignore_errors=True)
        self._ci_load_clusters()
        self._ci_status_lbl.setText(f"Удалено кластеров: {len(clusters)}")

    # ═══════════════════════════════════════════════════════════════════════
    #  ВКЛ. ModelInspector — загрузка, свойства, тест детекции
    # ═══════════════════════════════════════════════════════════════════════

    def _build_model_inspector_tab(self):
        import os as _os
        from PyQt5.QtWidgets import (
            QScrollArea, QPlainTextEdit, QSpinBox, QSplitter
        )
        from PyQt5.QtGui import QPixmap, QImage
        from PyQt5.QtCore import Qt as _Qt

        tab = QWidget()
        root_layout = QVBoxLayout(tab)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(10)

        # ── Загрузка модели ───────────────────────────────────────────────
        model_box, model_layout = self._create_group_box("Модель")

        model_row = QHBoxLayout()
        self._mi_model_label = self._create_status_label("Модель не выбрана")
        model_row.addWidget(self._mi_model_label, 1)
        btn_pick = QPushButton("Выбрать .pt…")
        btn_pick.clicked.connect(self._mi_pick_model)
        model_row.addWidget(btn_pick)
        btn_unload = QPushButton("Выгрузить")
        btn_unload.clicked.connect(self._mi_unload_model)
        model_row.addWidget(btn_unload)
        model_layout.addLayout(model_row)

        root_layout.addWidget(model_box)

        # ── Свойства ──────────────────────────────────────────────────────
        info_box, info_layout = self._create_group_box("Свойства модели")
        self._mi_info_text = QPlainTextEdit()
        self._mi_info_text.setReadOnly(True)
        self._mi_info_text.setFixedHeight(160)
        self._mi_info_text.setStyleSheet(
            "background:#1e1e1e; color:#d4d4d4; font-family:Consolas,monospace; font-size:11px;"
        )
        self._mi_info_text.setPlaceholderText("Загрузите .pt файл для просмотра свойств…")
        info_layout.addWidget(self._mi_info_text)
        root_layout.addWidget(info_box)

        # ── Тест детекции ─────────────────────────────────────────────────
        detect_box, detect_layout = self._create_group_box("Тест детекции")

        # параметры
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Conf:"))
        self._mi_conf_spin = QDoubleSpinBox()
        self._mi_conf_spin.setRange(0.01, 1.0)
        self._mi_conf_spin.setSingleStep(0.05)
        self._mi_conf_spin.setValue(0.25)
        self._mi_conf_spin.setFixedWidth(70)
        param_row.addWidget(self._mi_conf_spin)

        param_row.addWidget(QLabel("IoU:"))
        self._mi_iou_spin = QDoubleSpinBox()
        self._mi_iou_spin.setRange(0.01, 1.0)
        self._mi_iou_spin.setSingleStep(0.05)
        self._mi_iou_spin.setValue(0.45)
        self._mi_iou_spin.setFixedWidth(70)
        param_row.addWidget(self._mi_iou_spin)

        from PyQt5.QtWidgets import QSpinBox as _QSpinBox2
        param_row.addWidget(QLabel("imgsz:"))
        self._mi_imgsz_combo = QComboBox()
        self._mi_imgsz_combo.addItems(["320", "416", "512", "640", "768", "1024"])
        self._mi_imgsz_combo.setCurrentText("640")
        param_row.addWidget(self._mi_imgsz_combo)

        from PyQt5.QtWidgets import QCheckBox as _QCheckBox
        self._mi_tile_check = _QCheckBox("Tile mode")
        self._mi_tile_check.setToolTip(
            "Скользящее окно: изображение разбивается на перекрывающиеся патчи.\n"
            "Используйте для больших изображений, где объект плохо находится."
        )
        param_row.addWidget(self._mi_tile_check)
        param_row.addStretch(1)
        detect_layout.addLayout(param_row)

        # выбор изображений
        img_row = QHBoxLayout()
        self._mi_images_label = self._create_status_label("Изображения не выбраны")
        img_row.addWidget(self._mi_images_label, 1)
        btn_pick_imgs = QPushButton("Выбрать изображение(я)…")
        btn_pick_imgs.clicked.connect(self._mi_pick_images)
        img_row.addWidget(btn_pick_imgs)
        detect_layout.addLayout(img_row)

        # кнопка запуска
        run_row = QHBoxLayout()
        self._mi_btn_run = QPushButton("▶  Запустить детекцию")
        self._mi_btn_run.clicked.connect(self._mi_run_detection)
        run_row.addWidget(self._mi_btn_run)
        self._mi_btn_stats = QPushButton("📊  Показать графики")
        self._mi_btn_stats.setToolTip("Открыть графики накопленной статистики детекций")
        self._mi_btn_stats.clicked.connect(self._mi_show_stats_charts)
        run_row.addWidget(self._mi_btn_stats)
        self._mi_status_label = self._create_status_label("")
        run_row.addWidget(self._mi_status_label, 1)
        detect_layout.addLayout(run_row)

        # навигация по результатам
        nav_row = QHBoxLayout()
        self._mi_btn_prev = QPushButton("◀")
        self._mi_btn_prev.setFixedWidth(40)
        self._mi_btn_prev.clicked.connect(lambda: self._mi_navigate(-1))
        self._mi_page_label = QLabel("0 / 0")
        self._mi_page_label.setAlignment(_Qt.AlignCenter)
        self._mi_page_label.setFixedWidth(70)
        self._mi_btn_next = QPushButton("▶")
        self._mi_btn_next.setFixedWidth(40)
        self._mi_btn_next.clicked.connect(lambda: self._mi_navigate(+1))
        self._mi_btn_open_full = QPushButton("⤢ На весь экран")
        self._mi_btn_open_full.setFixedWidth(130)
        self._mi_btn_open_full.setToolTip("Открыть текущее изображение в отдельном окне")
        self._mi_btn_open_full.clicked.connect(self._mi_open_fullsize)
        nav_row.addStretch(1)
        nav_row.addWidget(self._mi_btn_prev)
        nav_row.addWidget(self._mi_page_label)
        nav_row.addWidget(self._mi_btn_next)
        nav_row.addWidget(self._mi_btn_open_full)
        nav_row.addStretch(1)
        detect_layout.addLayout(nav_row)

        # превью + лог
        splitter = QSplitter(_Qt.Horizontal)

        self._mi_preview = QLabel()
        self._mi_preview.setAlignment(_Qt.AlignCenter)
        self._mi_preview.setMinimumSize(300, 240)
        self._mi_preview.setStyleSheet("background:#111; border:1px solid #333;")
        self._mi_preview.setText("Превью")
        splitter.addWidget(self._mi_preview)

        self._mi_log = QPlainTextEdit()
        self._mi_log.setReadOnly(True)
        self._mi_log.setStyleSheet(
            "background:#1e1e1e; color:#d4d4d4; font-family:Consolas,monospace; font-size:11px;"
        )
        self._mi_log.setPlaceholderText("Результаты детекции…")
        splitter.addWidget(self._mi_log)
        splitter.setSizes([380, 280])

        detect_layout.addWidget(splitter)
        root_layout.addWidget(detect_box)

        # ── Тест на видео ─────────────────────────────────────────────────────
        vtest_box, vtest_layout = self._create_group_box("Тест на видео")
        vtest_layout.setSpacing(4)

        vtest_files_row = QHBoxLayout()
        vtest_files_row.setSpacing(4)
        vtest_files_row.addWidget(QLabel("Видео:"))
        self._vtest_video_label = QLabel("не выбрано")
        self._vtest_video_label.setStyleSheet("color:#4b5563; font-size:10px;")
        vtest_files_row.addWidget(self._vtest_video_label, 1)
        btn_vtest_video = QPushButton("…")
        btn_vtest_video.setFixedWidth(32)
        btn_vtest_video.setToolTip("Выбрать входной видеофайл")
        btn_vtest_video.clicked.connect(self._vtest_choose_video)
        vtest_files_row.addWidget(btn_vtest_video)
        vtest_layout.addLayout(vtest_files_row)

        vtest_params_row = QHBoxLayout()
        vtest_params_row.setSpacing(6)
        vtest_params_row.addWidget(QLabel("Conf:"))
        self._vtest_conf_spin = QDoubleSpinBox()
        self._vtest_conf_spin.setRange(0.01, 0.99)
        self._vtest_conf_spin.setSingleStep(0.05)
        self._vtest_conf_spin.setValue(0.25)
        self._vtest_conf_spin.setFixedWidth(70)
        vtest_params_row.addWidget(self._vtest_conf_spin)
        vtest_params_row.addWidget(QLabel("IoU:"))
        self._vtest_iou_spin = QDoubleSpinBox()
        self._vtest_iou_spin.setRange(0.01, 0.99)
        self._vtest_iou_spin.setSingleStep(0.05)
        self._vtest_iou_spin.setValue(0.45)
        self._vtest_iou_spin.setFixedWidth(70)
        vtest_params_row.addWidget(self._vtest_iou_spin)
        vtest_params_row.addWidget(QLabel("ImgSz:"))
        self._vtest_imgsz_combo = QComboBox()
        self._vtest_imgsz_combo.addItems(["320", "416", "512", "640", "1280"])
        self._vtest_imgsz_combo.setCurrentText("640")
        self._vtest_imgsz_combo.setFixedWidth(70)
        vtest_params_row.addWidget(self._vtest_imgsz_combo)
        vtest_params_row.addWidget(QLabel("Пропуск кадров:"))
        from PyQt5.QtWidgets import QSpinBox as _QSpinBoxVT
        self._vtest_skip_spin = _QSpinBoxVT()
        self._vtest_skip_spin.setRange(0, 10)
        self._vtest_skip_spin.setValue(0)
        self._vtest_skip_spin.setFixedWidth(55)
        self._vtest_skip_spin.setToolTip(
            "0 = каждый кадр\n1 = каждый второй\n2 = каждый третий…"
        )
        vtest_params_row.addWidget(self._vtest_skip_spin)
        vtest_params_row.addStretch(1)
        vtest_layout.addLayout(vtest_params_row)

        vtest_ctrl_row = QHBoxLayout()
        vtest_ctrl_row.setSpacing(6)
        self._btn_vtest_run = QPushButton("▶  Тест")
        self._btn_vtest_run.clicked.connect(self._vtest_run)
        vtest_ctrl_row.addWidget(self._btn_vtest_run)
        self._btn_vtest_stop = QPushButton("■  Стоп")
        self._btn_vtest_stop.setEnabled(False)
        self._btn_vtest_stop.clicked.connect(self._vtest_stop)
        vtest_ctrl_row.addWidget(self._btn_vtest_stop)
        self._btn_vtest_open = QPushButton("▶  Открыть результат")
        self._btn_vtest_open.setEnabled(False)
        self._btn_vtest_open.clicked.connect(self._vtest_open_result)
        vtest_ctrl_row.addWidget(self._btn_vtest_open)
        self._vtest_status_label = self._create_status_label("Готов")
        vtest_ctrl_row.addWidget(self._vtest_status_label, 1)
        vtest_layout.addLayout(vtest_ctrl_row)

        self._vtest_log = QPlainTextEdit()
        self._vtest_log.setReadOnly(True)
        self._vtest_log.setFixedHeight(80)
        self._vtest_log.setStyleSheet(
            "background:#1e1e1e; color:#d4d4d4; font-family:Consolas,monospace; font-size:10px;"
        )
        vtest_layout.addWidget(self._vtest_log)
        root_layout.addWidget(vtest_box)

        # ── внутреннее состояние ──────────────────────────────────────────
        self._mi_model_path: str | None = None
        self._mi_image_paths: list = []
        self._mi_results: list = []   # список dict из detect_image()
        self._mi_current_idx: int = 0
        self._vtest_worker      = None
        self._vtest_video_path  = None
        self._vtest_output_path = None

        return tab

    def _mi_pick_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать модель YOLO", "", "YOLO weights (*.pt);;All files (*)"
        )
        if not path:
            return
        self._mi_model_path = path
        self._mi_model_label.setText(os.path.basename(path))
        self._mi_info_text.setPlainText("Загрузка…")
        try:
            info = get_model_info(path)
            self._mi_info_text.setPlainText(format_model_info(info))
        except Exception as e:
            self._mi_info_text.setPlainText(f"Ошибка загрузки:\n{e}")

    def _mi_unload_model(self):
        unload_model()
        self._mi_model_path = None
        self._mi_model_label.setText("Модель не выбрана")
        self._mi_info_text.clear()
        self._mi_log.clear()
        self._mi_preview.setText("Превью")
        self._mi_results.clear()
        self._mi_page_label.setText("0 / 0")

    def _mi_pick_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Выбрать изображения", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*)"
        )
        if paths:
            self._mi_image_paths = paths
            self._mi_images_label.setText(f"Выбрано: {len(paths)} шт.")

    def _mi_run_detection(self):
        if not self._mi_model_path:
            QMessageBox.warning(self, "Нет модели", "Сначала выберите .pt файл модели.")
            return
        if not self._mi_image_paths:
            QMessageBox.warning(self, "Нет изображений", "Выберите одно или несколько изображений.")
            return

        self._mi_btn_run.setEnabled(False)
        self._mi_status_label.setText("Обработка…")
        self._mi_results.clear()
        self._mi_log.clear()

        conf      = self._mi_conf_spin.value()
        iou       = self._mi_iou_spin.value()
        imgsz     = int(self._mi_imgsz_combo.currentText())
        use_tiles = self._mi_tile_check.isChecked()

        errors = 0
        for path in self._mi_image_paths:
            try:
                if use_tiles:
                    result = detect_image_tiled(
                        self._mi_model_path, path, conf=conf, iou=iou, imgsz=imgsz
                    )
                else:
                    result = detect_image(
                        self._mi_model_path, path, conf=conf, iou=iou, imgsz=imgsz
                    )
                result["_path"] = path
                self._mi_results.append(result)
                # сохраняем статистику
                ann = result.get("annotated_image")
                if ann is not None and ann.ndim == 3:
                    fh, fw = ann.shape[:2]
                else:
                    import cv2 as _cv2_stat
                    _img_stat = _cv2_stat.imread(path)
                    fh, fw = (_img_stat.shape[:2] if _img_stat is not None else (0, 0))
                try:
                    save_detection_stats(
                        self._mi_model_path, path, fw, fh, result["detections"]
                    )
                except Exception:
                    pass
            except Exception as e:
                self._mi_log.appendPlainText(f"[ОШИБКА] {os.path.basename(path)}: {e}")
                errors += 1

        total = len(self._mi_results)
        self._mi_status_label.setText(
            f"Готово: {total} успешно" + (f", {errors} ошибок" if errors else "")
        )
        self._mi_btn_run.setEnabled(True)

        if total > 0:
            self._mi_current_idx = 0
            self._mi_show_result(0)

    def _mi_navigate(self, delta: int):
        if not self._mi_results:
            return
        new_idx = self._mi_current_idx + delta
        new_idx = max(0, min(len(self._mi_results) - 1, new_idx))
        if new_idx != self._mi_current_idx:
            self._mi_current_idx = new_idx
            self._mi_show_result(new_idx)

    def _mi_show_result(self, idx: int):
        from PyQt5.QtGui import QPixmap, QImage
        from PyQt5.QtCore import Qt as _Qt

        self._mi_current_idx = idx
        total = len(self._mi_results)
        self._mi_page_label.setText(f"{idx + 1} / {total}")

        result = self._mi_results[idx]
        path   = result.get("_path", "")

        # ── лог ──────────────────────────────────────────────────────────
        header = f"[ {os.path.basename(path)} ]"
        self._mi_log.setPlainText(
            header + "\n" + format_detections(result["detections"], result["inference_ms"])
        )

        # ── превью ───────────────────────────────────────────────────────
        ann = result.get("annotated_image")
        if ann is not None and ann.size > 0:
            import cv2
            rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            pw = self._mi_preview.width()
            ph = self._mi_preview.height()
            self._mi_preview.setPixmap(
                pix.scaled(pw, ph, _Qt.KeepAspectRatio, _Qt.SmoothTransformation)
            )
        else:
            self._mi_preview.setText("Нет изображения")

    def _mi_open_fullsize(self):
        """Открывает текущий результат детекции в отдельном окне в полный размер."""
        from PyQt5.QtGui import QPixmap, QImage
        from PyQt5.QtCore import Qt as _Qt
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QScrollArea, QLabel as _QLabel

        if not self._mi_results or self._mi_current_idx >= len(self._mi_results):
            return
        result = self._mi_results[self._mi_current_idx]
        ann = result.get("annotated_image")
        if ann is None or ann.size == 0:
            return

        import cv2
        rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        dlg = QDialog(self)
        dlg.setWindowTitle(os.path.basename(result.get("_path", "Результат детекции")))
        dlg.setWindowFlags(dlg.windowFlags() | _Qt.Window)
        dlg.resize(min(w + 40, 1600), min(h + 60, 1000))

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setAlignment(_Qt.AlignCenter)

        img_label = _QLabel()
        img_label.setPixmap(pix)
        img_label.setFixedSize(w, h)
        img_label.setAlignment(_Qt.AlignCenter)

        scroll.setWidget(img_label)
        layout.addWidget(scroll)

        dlg.show()

    def _mi_show_stats_charts(self):
        """Открывает окно с графиками накопленной статистики детекций."""
        import io as _io
        import json as _json
        from collections import defaultdict
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QTabWidget, QWidget, QScrollArea,
            QLabel as _QLabel2,
        )
        from PyQt5.QtGui import QPixmap as _QPixmap
        from PyQt5.QtCore import Qt as _Qt2

        stats_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "Stats", "model_inspector_stats.json",
        )

        if not os.path.isfile(stats_file):
            QMessageBox.information(
                self, "Статистика",
                "Статистика пока не накоплена.\nЗапустите детекцию хотя бы один раз.",
            )
            return

        try:
            with open(stats_file, "r", encoding="utf-8") as _f:
                data = _json.load(_f)
        except Exception as _e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить статистику:\n{_e}")
            return

        if not data:
            QMessageBox.information(self, "Статистика", "Файл статистики пуст.")
            return

        try:
            import matplotlib as _mpl
            # Принудительно используем не-GUI бэкенд чтобы не конфликтовать с Qt
            _mpl.use("Agg")
            from matplotlib.figure import Figure as _Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg as _FigCanvasAgg
            import matplotlib.ticker as _ticker
        except ImportError:
            QMessageBox.warning(
                self, "Нет matplotlib",
                "Установите matplotlib для просмотра графиков:\npip install matplotlib",
            )
            return

        # группируем по разрешению
        by_res: dict = defaultdict(list)
        for entry in data:
            key = f"{entry.get('frame_w', 0)}x{entry.get('frame_h', 0)}"
            by_res[key].append(entry)

        dlg = QDialog(self)
        dlg.setWindowTitle("Статистика детекций — графики")
        dlg.setWindowFlags(dlg.windowFlags() | _Qt2.Window)
        dlg.resize(980, 540)

        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.setContentsMargins(6, 6, 6, 6)

        tabs = QTabWidget()

        for res_key in sorted(by_res.keys()):
            entries = by_res[res_key]

            fig = _Figure(figsize=(11, 4.2), tight_layout=True)
            _FigCanvasAgg(fig)  # подключаем Agg-рендерер к фигуре

            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            # ── Левый график: количество детекций по запускам ──────────────
            n_dets = [e.get("n_detections", 0) for e in entries]
            run_indices = list(range(1, len(entries) + 1))
            ax1.bar(run_indices, n_dets, color="#4c72b0", width=0.6)
            ax1.set_title(f"Количество детекций  ({res_key})")
            ax1.set_xlabel("Запуск №")
            ax1.set_ylabel("Объектов найдено")
            ax1.xaxis.set_major_locator(_ticker.MaxNLocator(integer=True))
            ax1.yaxis.set_major_locator(_ticker.MaxNLocator(integer=True))
            ax1.grid(axis="y", linestyle="--", alpha=0.5)

            # ── Правый график: распределение confidence ────────────────────
            all_confs = []
            for e in entries:
                all_confs.extend(e.get("confs", []))

            if all_confs:
                ax2.hist(
                    all_confs, bins=20, range=(0.0, 1.0),
                    color="#dd8452", edgecolor="white", linewidth=0.4,
                )
                ax2.set_title(f"Распределение confidence  ({res_key})")
                ax2.set_xlabel("Confidence")
                ax2.set_ylabel("Количество")
                ax2.grid(axis="y", linestyle="--", alpha=0.5)
            else:
                ax2.text(0.5, 0.5, "Нет детекций",
                         ha="center", va="center", transform=ax2.transAxes, fontsize=13)
                ax2.set_title(f"Распределение confidence  ({res_key})")

            # Рендерим фигуру в PNG-байты (полностью избегаем Qt-бэкенд matplotlib)
            buf = _io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            buf.seek(0)
            png_bytes = buf.read()
            buf.close()

            pix = _QPixmap()
            pix.loadFromData(png_bytes)

            img_label = _QLabel2()
            img_label.setPixmap(pix)
            img_label.setAlignment(_Qt2.AlignCenter)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(img_label)

            tab_w = QWidget()
            tab_layout = QVBoxLayout(tab_w)
            tab_layout.setContentsMargins(4, 4, 4, 4)
            tab_layout.addWidget(scroll)

            tabs.addTab(tab_w, res_key)

        dlg_layout.addWidget(tabs)
        dlg.exec_()

    def init_ui(self):
        self.setStyleSheet("""
            QWidget {
                font-size: 12px;
            }
            QPushButton {
                min-height: 32px;
                padding: 6px 10px;
            }
            QGroupBox {
                font-weight: 600;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QTabBar::tab {
                min-height: 28px;
                padding: 6px 12px;
            }
        """)

        self.resize(560, 700)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_detection_tab(), "Видео/фото")
        tabs.addTab(self._build_duplicates_tab(), "Дубликаты")
        tabs.addTab(self._build_vehicle_tab(), "Транспорт")
        tabs.addTab(self._build_people_phone_tab(), "Люди/телефоны")
        tabs.addTab(self._build_open_set_tab(), "Open-set")
        tabs.addTab(self._build_class_manager_tab(), "Инспектор датасетов")
        tabs.addTab(self._build_cluster_inspector_tab(), "Инспектор кластеров")
        tabs.addTab(self._build_unknown_merge_tab(), "UnknownClass DB")
        tabs.addTab(self._build_train_blocker_tab(), "Тренировка")
        tabs.addTab(self._build_model_inspector_tab(), "Инспектор модели")

        layout.addWidget(tabs)

        self.setLayout(layout)

        # Создаем папку для сохранения кадров, если она не существует
        if not os.path.exists("FRAMES"):
            os.makedirs("FRAMES")

    def open_folder(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Выберите папку с видео/изображениями", options=options)

        if folder_path:
            self.folder_path = folder_path
            self.file_path = None
            self.file_type = None
            self.btn_start.setEnabled(True)
            self.btn_start_file.setEnabled(False)

    def open_video_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите видеофайл", 
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
            options=options
        )

        if file_path:
            self.file_path = file_path
            self.folder_path = None
            self.file_type = 'video'
            self.btn_start_file.setEnabled(True)
            self.btn_start.setEnabled(False)

    def open_image_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите изображение", 
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.gif);;All Files (*)",
            options=options
        )

        if file_path:
            self.file_path = file_path
            self.folder_path = None
            self.file_type = 'image'
            self.btn_start_file.setEnabled(True)
            self.btn_start.setEnabled(False)

    def start_analysis(self):
        if self.folder_path:
            self.timer.start(30)  # Обработка кадров каждые 30 мс
            self.btn_stop.setEnabled(True)
            self.btn_start.setEnabled(False)

    def start_analysis_file(self):
        if self.file_path:
            self.timer.start(30)  # Обработка кадров каждые 30 мс
            self.btn_stop.setEnabled(True)
            self.btn_start_file.setEnabled(False)

    def stop_analysis(self):
        self.timer.stop()
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True if self.folder_path else False)
        self.btn_start_file.setEnabled(True if self.file_path else False)

    def process_files(self):
        print(f"[UI] process_files вызвана")
        print(f"[UI] file_path={self.file_path}, file_type={self.file_type}, folder_path={self.folder_path}")
        
        if self.file_path:
            if self.file_type == 'video':
                print(f"[UI] Обрабатываю видеофайл: {self.file_path}")
                process_single_video(self.file_path, self.tracking_objects, self.trail_points, self.show_video_checkbox, self.label, self.confidence_threshold)
            elif self.file_type == 'image':
                print(f"[UI] Обрабатываю изображение: {self.file_path}")
                process_single_image(self.file_path, self.tracking_objects, self.trail_points, self.show_video_checkbox, self.label, self.confidence_threshold)
            else:
                print(f"[UI] Неизвестный тип файла: {self.file_type}")
        elif self.folder_path:
            # Определяем тип файлов в папке (видео или изображения)
            video_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            image_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'))]
            
            print(f"[UI] В папке найдено видео: {len(video_files)}, изображений: {len(image_files)}")
            
            if video_files:
                print(f"[UI] Обрабатываю видео из папки")
                process_videos_in_folder(self.folder_path, self.tracking_objects, self.trail_points, self.show_video_checkbox, self.label, self.confidence_threshold)
            elif image_files:
                print(f"[UI] Обрабатываю изображения из папки")
                process_images_in_folder(self.folder_path, self.tracking_objects, self.trail_points, self.show_video_checkbox, self.label, self.confidence_threshold)
            else:
                print(f"[UI] В папке не найдено видео или изображений")
        
        print(f"[UI] Завершение process_files")
        self.timer.stop()
        self.btn_start.setEnabled(True if self.folder_path else False)
        self.btn_start_file.setEnabled(True if self.file_path else False)
        self.btn_stop.setEnabled(False)

    def open_duplicates_folder(self):
        """Открывает диалог выбора папки для поиска дубликатов"""
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(
            self, 
            "Выберите папку для поиска дубликатов", 
            options=options
        )

        if folder_path:
            self.duplicates_folder = folder_path
            self.duplicates_folder_label.setText(f"Выбрана папка:\n{folder_path}")
            self.btn_remove_duplicates.setEnabled(True)

    def on_hamming_changed(self, value):
        """Обновляет значение порога Хэмминга"""
        self.hamming_threshold = value
        self.hamming_value_label.setText(str(value))

    def on_ssim_changed(self, value):
        """Обновляет значение порога SSIM"""
        self.ssim_threshold = value / 100.0
        self.ssim_value_label.setText(f"{self.ssim_threshold:.2f}")

    def on_confidence_changed(self, value):
        """Обновляет значение порога уверенности детекции"""
        self.confidence_threshold = value / 100.0
        self.confidence_value_label.setText(f"{self.confidence_threshold:.2f}")

    def remove_duplicates(self):
        """Запускает процесс удаления дубликатов с установленными параметрами"""
        if not self.duplicates_folder:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, выберите папку для поиска дубликатов")
            return

        # Отключаем кнопку во время обработки
        self.btn_remove_duplicates.setEnabled(False)
        self.btn_select_duplicates_folder.setEnabled(False)

        try:
            # Запускаем процесс удаления дубликатов с установленными параметрами
            result = detect_and_remove_duplicates(
                self.duplicates_folder,
                hamming_threshold=self.hamming_threshold,
                ssim_threshold=self.ssim_threshold
            )
            
            if result:
                message = f"""Обработка завершена!

Параметры:
• Порог Хэмминга: {self.hamming_threshold}
• Порог SSIM: {self.ssim_threshold:.2f}

Результаты:
• Удалено дубликатов: {result['duplicates_deleted']}
• Обработано групп: {result['duplicate_groups_processed']}
• Перемещено уникальных изображений: {result['unique_images_moved']}

Уникальные изображения находятся в папке:
{result['output_folder']}"""
                
                QMessageBox.information(self, "Успешно", message)
                self.duplicates_folder_label.setText(f"✓ Обработка завершена!\n{result['output_folder']}")
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось обработать папку")
        
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при обработке:\n{str(e)}")
        
        finally:
            # Включаем кнопку обратно
            self.btn_remove_duplicates.setEnabled(True)
            self.btn_select_duplicates_folder.setEnabled(True)

    def open_grouping_folder(self):
        """Открывает диалог для выбора папки с изображениями для группировки"""
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(
            self, 
            "Выберите папку с изображениями для группировки", 
            options=options
        )

        if folder_path:
            self.grouping_folder = folder_path
            self.btn_group_images.setEnabled(True)
            folder_name = os.path.basename(folder_path)
            self.grouping_folder_label.setText(f"✓ Выбрана: {folder_name}")

    def group_images(self):
        """Запускает процесс группировки изображений"""
        if not self.grouping_folder:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, выберите папку для группировки")
            return

        # Отключаем кнопки во время обработки
        self.btn_group_images.setEnabled(False)
        self.btn_select_grouping_folder.setEnabled(False)

        try:
            # Запускаем процесс группировки с установленными параметрами
            result = group_images_by_similarity(
                self.grouping_folder,
                hamming_threshold=self.hamming_threshold,
                ssim_threshold=self.ssim_threshold
            )
            
            if result and result['success']:
                message = f"""✓ Группировка завершена!

Параметры:
• Порог Хэмминга: {self.hamming_threshold}
• Порог SSIM: {self.ssim_threshold:.2f}

Результаты:
• Создано групп: {result['total_groups']}
• Всего изображений: {result['total_images']}
• Скопировано: {result['copied_images']}
• Не сгруппировано: {result['ungrouped_images']}

Группы находятся в папке:
{result['output_dir']}

Подробный отчёт:
{result['report_file']}"""
                
                QMessageBox.information(self, "Успешно", message)
                self.grouping_folder_label.setText(f"✓ Группировка завершена!\n{result['output_dir']}")
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось сгруппировать изображения")
        
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при группировке:\n{str(e)}")
        
        finally:
            # Включаем кнопки обратно
            self.btn_group_images.setEnabled(True)
            self.btn_select_grouping_folder.setEnabled(True)

    # ─── Методы анализа транспорта ───

    def open_vehicle_file(self):
        """Выбор одного изображения транспорта для анализа."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение транспорта",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.gif);;All Files (*)",
            options=options
        )
        if file_path:
            self.vehicle_file = file_path
            self.vehicle_folder = None
            self.vehicle_path_label.setText(f"Файл: {os.path.basename(file_path)}")
            self.btn_analyze_vehicle.setEnabled(True)
            self.btn_analyze_vehicle_qwen.setEnabled(True)
            self.btn_analyze_vehicle_qwen2vl.setEnabled(True)
            self.btn_analyze_vehicle_openalpr.setEnabled(True)

    def open_vehicle_folder(self):
        """Выбор папки с изображениями транспорта."""
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку с изображениями транспорта",
            options=options
        )
        if folder_path:
            self.vehicle_folder = folder_path
            self.vehicle_file = None
            self.vehicle_path_label.setText(f"Папка: {os.path.basename(folder_path)}")
            self.btn_analyze_vehicle.setEnabled(True)
            self.btn_analyze_vehicle_qwen.setEnabled(True)
            self.btn_analyze_vehicle_qwen2vl.setEnabled(True)
            self.btn_analyze_vehicle_openalpr.setEnabled(True)

    def _create_progress(self, title, maximum):
        """Создаёт диалог прогресса."""
        progress = QProgressDialog(title, "Отмена", 0, maximum, self)
        progress.setWindowTitle("Анализ транспорта")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)
        return progress

    def run_vehicle_analysis(self):
        """Запускает анализ транспорта в фоновом потоке."""
        self.btn_analyze_vehicle.setEnabled(False)
        self.btn_vehicle_file.setEnabled(False)
        self.btn_vehicle_folder.setEnabled(False)

        if not self.vehicle_file and not self.vehicle_folder:
            QMessageBox.warning(self, "Ошибка", "Выберите файл или папку")
            self.btn_analyze_vehicle.setEnabled(True)
            self.btn_vehicle_file.setEnabled(True)
            self.btn_vehicle_folder.setEnabled(True)
            return

        # Создаём прогресс-диалог
        self._vehicle_progress = self._create_progress("Загрузка модели...", 4)
        self._vehicle_progress.canceled.connect(self._cancel_vehicle_analysis)

        # Запускаем воркер в QThread
        self._vehicle_worker = VehicleAnalysisWorker(
            file_path=self.vehicle_file,
            folder_path=self.vehicle_folder,
            parent=None
        )
        self._vehicle_worker.progress_update.connect(self._on_vehicle_progress)
        self._vehicle_worker.analysis_done.connect(self._on_vehicle_done)
        self._vehicle_worker.analysis_error.connect(self._on_vehicle_error)
        self._vehicle_worker.finished.connect(self._on_vehicle_finished)
        self._vehicle_worker.start()

    def _cancel_vehicle_analysis(self):
        """Отмена анализа."""
        if hasattr(self, '_vehicle_worker') and self._vehicle_worker is not None:
            self._vehicle_worker.cancel()

    def _on_vehicle_progress(self, label, value, maximum):
        """Обновление прогресса из фонового потока."""
        if hasattr(self, '_vehicle_progress') and self._vehicle_progress is not None:
            self._vehicle_progress.setMaximum(maximum)
            self._vehicle_progress.setValue(value)
            self._vehicle_progress.setLabelText(label)

    def _on_vehicle_done(self, results):
        """Обработка результатов анализа."""
        if hasattr(self, '_vehicle_progress') and self._vehicle_progress is not None:
            self._vehicle_progress.close()
            self._vehicle_progress = None

        if not results:
            QMessageBox.warning(self, "Ошибка", "Изображения не найдены")
            return

        # Сохраняем отчёт
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%d-%m-%Y_%H-%M-%S")
        report_dir = os.path.join("Results", f"VehicleAnalysis_{ts}")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "VEHICLE_REPORT.txt")
        elapsed = getattr(self._vehicle_worker, 'elapsed_seconds', 0)
        generate_report(results, report_path, elapsed_seconds=elapsed)

        # Формируем краткое сообщение
        if len(results) == 1:
            r = results[0]
            if "error" in r:
                msg = f"Ошибка: {r['error']}"
            else:
                msg = f"""Анализ завершён!\n
Файл: {r['file']}\n
Цвет: {r['color']['color']} ({r['color']['confidence']:.0%})
Модель: {r['model']['model']} ({r['model']['confidence']:.1%})
Номер: {r['plate']['message']}\n
Отчёт: {report_path}"""
        else:
            msg = f"""Анализ завершён!\n
Обработано изображений: {len(results)}\n
Подробный отчёт:\n{report_path}"""

        QMessageBox.information(self, "Анализ транспорта", msg)
        self.vehicle_path_label.setText(f"✓ Отчёт: {report_path}")

    def _on_vehicle_error(self, error_msg):
        """Обработка ошибки анализа."""
        if hasattr(self, '_vehicle_progress') and self._vehicle_progress is not None:
            self._vehicle_progress.close()
            self._vehicle_progress = None
        QMessageBox.critical(self, "Ошибка", f"Ошибка анализа транспорта:\n{error_msg}")

    def _on_vehicle_finished(self):
        """Разблокировка кнопок после завершения потока."""
        self.btn_analyze_vehicle.setEnabled(True)
        self.btn_analyze_vehicle_qwen.setEnabled(True)
        self.btn_analyze_vehicle_qwen2vl.setEnabled(True)
        self.btn_analyze_vehicle_openalpr.setEnabled(True)
        self.btn_vehicle_file.setEnabled(True)
        self.btn_vehicle_folder.setEnabled(True)
        self._vehicle_worker = None

    # ─── Qwen2.5-VL анализ транспорта ───

    def run_qwen_vehicle_analysis(self):
        """Запускает анализ транспорта через Qwen2.5-VL в фоновом потоке."""
        self.btn_analyze_vehicle.setEnabled(False)
        self.btn_analyze_vehicle_qwen.setEnabled(False)
        self.btn_analyze_vehicle_qwen2vl.setEnabled(False)
        self.btn_analyze_vehicle_openalpr.setEnabled(False)
        self.btn_vehicle_file.setEnabled(False)
        self.btn_vehicle_folder.setEnabled(False)

        if not self.vehicle_file and not self.vehicle_folder:
            QMessageBox.warning(self, "Ошибка", "Выберите файл или папку")
            self._on_qwen_vehicle_finished()
            return

        self._qwen_vehicle_progress = self._create_progress("Загрузка Qwen2.5-VL...", 4)
        self._qwen_vehicle_progress.setWindowTitle("Анализ транспорта (Qwen2.5-VL)")
        self._qwen_vehicle_progress.canceled.connect(self._cancel_qwen_vehicle)

        self._qwen_vehicle_worker = QwenVehicleAnalysisWorker(
            file_path=self.vehicle_file,
            folder_path=self.vehicle_folder,
            parent=None
        )
        self._qwen_vehicle_worker.progress_update.connect(self._on_qwen_vehicle_progress)
        self._qwen_vehicle_worker.analysis_done.connect(self._on_qwen_vehicle_done)
        self._qwen_vehicle_worker.analysis_error.connect(self._on_qwen_vehicle_error)
        self._qwen_vehicle_worker.finished.connect(self._on_qwen_vehicle_finished)
        self._qwen_vehicle_worker.start()

    def _cancel_qwen_vehicle(self):
        if hasattr(self, '_qwen_vehicle_worker') and self._qwen_vehicle_worker is not None:
            self._qwen_vehicle_worker.cancel()

    def _on_qwen_vehicle_progress(self, label, value, maximum):
        if hasattr(self, '_qwen_vehicle_progress') and self._qwen_vehicle_progress is not None:
            self._qwen_vehicle_progress.setMaximum(maximum)
            self._qwen_vehicle_progress.setValue(value)
            self._qwen_vehicle_progress.setLabelText(label)

    def _on_qwen_vehicle_done(self, results):
        if hasattr(self, '_qwen_vehicle_progress') and self._qwen_vehicle_progress is not None:
            self._qwen_vehicle_progress.close()
            self._qwen_vehicle_progress = None

        if not results:
            QMessageBox.warning(self, "Ошибка", "Изображения не найдены")
            return

        from datetime import datetime as _dt
        ts = _dt.now().strftime("%d-%m-%Y_%H-%M-%S")
        report_dir = os.path.join("Results", f"QwenVehicleAnalysis_{ts}")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "QWEN_VEHICLE_REPORT.txt")
        elapsed = getattr(self._qwen_vehicle_worker, 'elapsed_seconds', 0)
        generate_report_qwen(results, report_path, elapsed_seconds=elapsed)

        try:
            word_doc_path = generate_report_word_qwen(results, report_dir, elapsed_seconds=elapsed)
            word_info = f"\nWord отчёт: {word_doc_path}"
        except Exception as e:
            word_info = f"\nОшибка создания Word отчёта: {str(e)}"

        if len(results) == 1:
            r = results[0]
            if "error" in r:
                msg = f"Ошибка: {r['error']}"
            else:
                msg = (f"Анализ завершён (Qwen2.5-VL)!\n\n"
                       f"Файл: {r['file']}\n\n"
                       f"Цвет: {r['color']['color']} ({r['color']['confidence']:.0%})\n"
                       f"Модель: {r['model']['model']} ({r['model']['confidence']:.0%})\n"
                       f"Номер: {r['plate']['message']}\n\n"
                       f"Текстовый отчёт: {report_path}{word_info}")
        else:
            msg = (f"Анализ завершён (Qwen2.5-VL)!\n\n"
                   f"Обработано изображений: {len(results)}\n\n"
                   f"Текстовый отчёт: {report_path}{word_info}")

        QMessageBox.information(self, "Анализ транспорта (Qwen2.5-VL)", msg)
        self.vehicle_path_label.setText(f"✓ Отчёт: {report_path}")

    def _on_qwen_vehicle_error(self, error_msg):
        if hasattr(self, '_qwen_vehicle_progress') and self._qwen_vehicle_progress is not None:
            self._qwen_vehicle_progress.close()
            self._qwen_vehicle_progress = None
        QMessageBox.critical(self, "Ошибка", f"Ошибка анализа Qwen2.5-VL:\n{error_msg}")

    def _on_qwen_vehicle_finished(self):
        self.btn_analyze_vehicle.setEnabled(True)
        self.btn_analyze_vehicle_qwen.setEnabled(True)
        self.btn_analyze_vehicle_qwen2vl.setEnabled(True)
        self.btn_analyze_vehicle_openalpr.setEnabled(True)
        self.btn_vehicle_file.setEnabled(True)
        self.btn_vehicle_folder.setEnabled(True)
        self._qwen_vehicle_worker = None

    # ─── Qwen2-VL-2B анализ транспорта ───

    def run_qwen2vl_vehicle_analysis(self):
        """Запускает анализ транспорта через Qwen2-VL-2B в фоновом потоке."""
        self.btn_analyze_vehicle.setEnabled(False)
        self.btn_analyze_vehicle_qwen.setEnabled(False)
        self.btn_analyze_vehicle_qwen2vl.setEnabled(False)
        self.btn_analyze_vehicle_openalpr.setEnabled(False)
        self.btn_vehicle_file.setEnabled(False)
        self.btn_vehicle_folder.setEnabled(False)

        if not self.vehicle_file and not self.vehicle_folder:
            QMessageBox.warning(self, "Ошибка", "Выберите файл или папку")
            self._on_qwen2vl_vehicle_finished()
            return

        self._qwen2vl_vehicle_progress = self._create_progress("Загрузка Qwen2-VL-2B...", 4)
        self._qwen2vl_vehicle_progress.setWindowTitle("Анализ транспорта (Qwen2-VL-2B)")
        self._qwen2vl_vehicle_progress.canceled.connect(self._cancel_qwen2vl_vehicle)

        self._qwen2vl_vehicle_worker = Qwen2VLAnalysisWorker(
            file_path=self.vehicle_file,
            folder_path=self.vehicle_folder,
            parent=None
        )
        self._qwen2vl_vehicle_worker.progress_update.connect(self._on_qwen2vl_vehicle_progress)
        self._qwen2vl_vehicle_worker.analysis_done.connect(self._on_qwen2vl_vehicle_done)
        self._qwen2vl_vehicle_worker.analysis_error.connect(self._on_qwen2vl_vehicle_error)
        self._qwen2vl_vehicle_worker.finished.connect(self._on_qwen2vl_vehicle_finished)
        self._qwen2vl_vehicle_worker.start()

    def _cancel_qwen2vl_vehicle(self):
        if hasattr(self, '_qwen2vl_vehicle_worker') and self._qwen2vl_vehicle_worker is not None:
            self._qwen2vl_vehicle_worker.cancel()

    def _on_qwen2vl_vehicle_progress(self, label, value, maximum):
        if hasattr(self, '_qwen2vl_vehicle_progress') and self._qwen2vl_vehicle_progress is not None:
            self._qwen2vl_vehicle_progress.setMaximum(maximum)
            self._qwen2vl_vehicle_progress.setValue(value)
            self._qwen2vl_vehicle_progress.setLabelText(label)

    def _on_qwen2vl_vehicle_done(self, results):
        if hasattr(self, '_qwen2vl_vehicle_progress') and self._qwen2vl_vehicle_progress is not None:
            self._qwen2vl_vehicle_progress.close()
            self._qwen2vl_vehicle_progress = None

        if not results:
            QMessageBox.warning(self, "Ошибка", "Изображения не найдены")
            return

        from datetime import datetime as _dt
        ts = _dt.now().strftime("%d-%m-%Y_%H-%M-%S")
        report_dir = os.path.join("Results", f"Qwen2VLAnalysis_{ts}")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "QWEN2VL_VEHICLE_REPORT.txt")
        elapsed = getattr(self._qwen2vl_vehicle_worker, 'elapsed_seconds', 0)
        generate_report_qwen2vl(results, report_path, elapsed_seconds=elapsed)
        
        # Создаём Word документ с изображениями и результатами
        word_doc_path = None
        try:
            word_doc_path = generate_report_word_qwen2vl(results, report_dir, elapsed_seconds=elapsed)
            word_info = f"\nWord отчёт: {word_doc_path}"
        except Exception as e:
            word_info = f"\n⚠ Ошибка создания Word отчёта: {str(e)}"

        if len(results) == 1:
            r = results[0]
            if "error" in r:
                msg = f"Ошибка: {r['error']}"
            else:
                msg = (f"Анализ завершён (Qwen2-VL-2B)!\n\n"
                       f"Файл: {r['file']}\n\n"
                       f"Цвет: {r['color']['color']} ({r['color']['confidence']:.0%})\n"
                       f"Модель: {r['model']['model']} ({r['model']['confidence']:.0%})\n"
                       f"Номер: {r['plate']['message']}\n\n"
                       f"Текстовый отчёт: {report_path}{word_info}")
        else:
            msg = (f"Анализ завершён (Qwen2-VL-2B)!\n\n"
                   f"Обработано изображений: {len(results)}\n\n"
                   f"Текстовый отчёт: {report_path}{word_info}")

        QMessageBox.information(self, "Анализ транспорта (Qwen2-VL-2B)", msg)
        self.vehicle_path_label.setText(f"✓ Отчёт: {report_path}")

    def _on_qwen2vl_vehicle_error(self, error_msg):
        if hasattr(self, '_qwen2vl_vehicle_progress') and self._qwen2vl_vehicle_progress is not None:
            self._qwen2vl_vehicle_progress.close()
            self._qwen2vl_vehicle_progress = None
        QMessageBox.critical(self, "Ошибка", f"Ошибка анализа Qwen2-VL-2B:\n{error_msg}")

    def _on_qwen2vl_vehicle_finished(self):
        self.btn_analyze_vehicle.setEnabled(True)
        self.btn_analyze_vehicle_qwen.setEnabled(True)
        self.btn_analyze_vehicle_qwen2vl.setEnabled(True)
        self.btn_analyze_vehicle_openalpr.setEnabled(True)
        self.btn_vehicle_file.setEnabled(True)
        self.btn_vehicle_folder.setEnabled(True)
        self._qwen2vl_vehicle_worker = None

    # ─── OpenALPR анализ транспорта (полностью локально, без интернета) ───

    def run_openalpr_vehicle_analysis(self):
        """Запускает анализ транспорта через OpenALPR в фоновом потоке."""
        self.btn_analyze_vehicle.setEnabled(False)
        self.btn_analyze_vehicle_qwen.setEnabled(False)
        self.btn_analyze_vehicle_qwen2vl.setEnabled(False)
        self.btn_analyze_vehicle_openalpr.setEnabled(False)
        self.btn_vehicle_file.setEnabled(False)
        self.btn_vehicle_folder.setEnabled(False)

        if not self.vehicle_file and not self.vehicle_folder:
            QMessageBox.warning(self, "Ошибка", "Выберите файл или папку")
            self._on_openalpr_vehicle_finished()
            return

        self._openalpr_vehicle_progress = self._create_progress("Подготовка OpenALPR...", 4)
        self._openalpr_vehicle_progress.setWindowTitle("Анализ транспорта (OpenALPR)")
        self._openalpr_vehicle_progress.canceled.connect(self._cancel_openalpr_vehicle)

        self._openalpr_vehicle_worker = OpenALPRAnalysisWorker(
            file_path=self.vehicle_file,
            folder_path=self.vehicle_folder,
            parent=None
        )
        self._openalpr_vehicle_worker.progress_update.connect(self._on_openalpr_vehicle_progress)
        self._openalpr_vehicle_worker.analysis_done.connect(self._on_openalpr_vehicle_done)
        self._openalpr_vehicle_worker.analysis_error.connect(self._on_openalpr_vehicle_error)
        self._openalpr_vehicle_worker.finished.connect(self._on_openalpr_vehicle_finished)
        self._openalpr_vehicle_worker.start()

    def _cancel_openalpr_vehicle(self):
        if hasattr(self, '_openalpr_vehicle_worker') and self._openalpr_vehicle_worker is not None:
            self._openalpr_vehicle_worker.cancel()

    def _on_openalpr_vehicle_progress(self, label, value, maximum):
        if hasattr(self, '_openalpr_vehicle_progress') and self._openalpr_vehicle_progress is not None:
            self._openalpr_vehicle_progress.setMaximum(maximum)
            self._openalpr_vehicle_progress.setValue(value)
            self._openalpr_vehicle_progress.setLabelText(label)

    def _on_openalpr_vehicle_done(self, results):
        if hasattr(self, '_openalpr_vehicle_progress') and self._openalpr_vehicle_progress is not None:
            self._openalpr_vehicle_progress.close()
            self._openalpr_vehicle_progress = None

        if not results:
            QMessageBox.warning(self, "Ошибка", "Изображения не найдены")
            return

        from datetime import datetime as _dt
        ts = _dt.now().strftime("%d-%m-%Y_%H-%M-%S")
        report_dir = os.path.join("Results", f"OpenALPRAnalysis_{ts}")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "OPENALPR_REPORT.txt")
        elapsed = getattr(self._openalpr_vehicle_worker, 'elapsed_seconds', 0)
        generate_report_openalpr(results, report_path, elapsed_seconds=elapsed)

        if len(results) == 1:
            r = results[0]
            if "error" in r:
                msg = f"Ошибка: {r['error']}"
            else:
                msg = (f"Анализ завершён (OpenALPR - локально)!\n\n"
                       f"Файл: {r['file']}\n\n"
                       f"Цвет: {r['color']['color']} ({r['color']['confidence']:.0%})\n"
                       f"Модель: {r['model']['model']} ({r['model']['confidence']:.0%})\n"
                       f"Номер: {r['plate']['message']}\n\n"
                       f"Отчёт: {report_path}")
        else:
            msg = (f"Анализ завершён (OpenALPR - локально)!\n\n"
                   f"Обработано изображений: {len(results)}\n\n"
                   f"Подробный отчёт:\n{report_path}")

        QMessageBox.information(self, "Анализ транспорта (OpenALPR)", msg)
        self.vehicle_path_label.setText(f"✓ Отчёт: {report_path}")

    def _on_openalpr_vehicle_error(self, error_msg):
        if hasattr(self, '_openalpr_vehicle_progress') and self._openalpr_vehicle_progress is not None:
            self._openalpr_vehicle_progress.close()
            self._openalpr_vehicle_progress = None
        QMessageBox.critical(self, "Ошибка", f"Ошибка анализа OpenALPR:\n{error_msg}")

    def _on_openalpr_vehicle_finished(self):
        self.btn_analyze_vehicle.setEnabled(True)
        self.btn_analyze_vehicle_qwen.setEnabled(True)
        self.btn_analyze_vehicle_qwen2vl.setEnabled(True)
        self.btn_analyze_vehicle_openalpr.setEnabled(True)
        self.btn_vehicle_file.setEnabled(True)
        self.btn_vehicle_folder.setEnabled(True)
        self._openalpr_vehicle_worker = None

    # ─── Анализ людей (PersonAnalyzer) ───

    def open_person_file(self):
        """Выбор одного изображения человека для анализа."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение человека",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.gif);;All Files (*)",
            options=options
        )
        if file_path:
            self.person_file = file_path
            self.person_folder = None
            self.person_path_label.setText(f"Файл: {os.path.basename(file_path)}")
            self.btn_analyze_person.setEnabled(True)

    def open_person_folder(self):
        """Выбор папки с изображениями людей."""
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку с изображениями людей",
            options=options
        )
        if folder_path:
            self.person_folder = folder_path
            self.person_file = None
            self.person_path_label.setText(f"Папка: {os.path.basename(folder_path)}")
            self.btn_analyze_person.setEnabled(True)

    def run_person_analysis(self):
        """Запускает анализ людей в фоновом потоке."""
        self.btn_analyze_person.setEnabled(False)
        self.btn_person_file.setEnabled(False)
        self.btn_person_folder.setEnabled(False)

        if not self.person_file and not self.person_folder:
            QMessageBox.warning(self, "Ошибка", "Выберите файл или папку")
            self._on_person_finished()
            return

        # Перезагружаем базу лиц перед каждым анализом
        reload_face_database()

        self._person_progress = self._create_progress("Загрузка моделей...", 4)
        self._person_progress.setWindowTitle("Анализ людей")
        self._person_progress.canceled.connect(self._cancel_person)

        self._person_worker = PersonAnalysisWorker(
            file_path=self.person_file,
            folder_path=self.person_folder,
            parent=None
        )
        self._person_worker.progress_update.connect(self._on_person_progress)
        self._person_worker.analysis_done.connect(self._on_person_done)
        self._person_worker.analysis_error.connect(self._on_person_error)
        self._person_worker.finished.connect(self._on_person_finished)
        self._person_worker.start()

    def _cancel_person(self):
        if hasattr(self, '_person_worker') and self._person_worker is not None:
            self._person_worker.cancel()

    def _on_person_progress(self, label, value, maximum):
        if hasattr(self, '_person_progress') and self._person_progress is not None:
            self._person_progress.setMaximum(maximum)
            self._person_progress.setValue(value)
            self._person_progress.setLabelText(label)

    def _on_person_done(self, results):
        if hasattr(self, '_person_progress') and self._person_progress is not None:
            self._person_progress.close()
            self._person_progress = None

        if not results:
            QMessageBox.warning(self, "Ошибка", "Изображения не найдены")
            return

        from datetime import datetime as _dt
        ts = _dt.now().strftime("%d-%m-%Y_%H-%M-%S")
        report_dir = os.path.join("Results", f"PersonAnalysis_{ts}")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "PERSON_REPORT.txt")
        elapsed = getattr(self._person_worker, 'elapsed_seconds', 0)
        generate_report_person(results, report_path, elapsed_seconds=elapsed)

        # Word отчёт
        word_info = ""
        try:
            word_doc_path = generate_report_word_person(results, report_dir, elapsed_seconds=elapsed)
            word_info = f"\nWord отчёт: {word_doc_path}"
        except Exception as e:
            word_info = f"\n⚠ Ошибка создания Word отчёта: {str(e)}"

        if len(results) == 1:
            r = results[0]
            if "error" in r:
                msg = f"Ошибка: {r['error']}"
            else:
                g = r.get("gender", {})
                age = r.get("age")
                clothing = r.get("clothing", {})
                fm = r.get("face_match", {})

                upper = clothing.get("upper")
                lower = clothing.get("lower")
                cloth_str = ""
                if upper:
                    cloth_str += f"  Верх: {upper['color']} {upper['type']}\n"
                if lower:
                    cloth_str += f"  Низ: {lower['color']} {lower['type']}\n"

                face_str = (f"Совпадение — {fm['name']} ({fm['confidence']:.0%})"
                            if fm.get("matched")
                            else f"{fm.get('name', '?')} ({fm.get('confidence', 0):.0%})")

                msg = (f"Анализ завершён!\n\n"
                       f"Файл: {r['file']}\n\n"
                       f"Пол: {g.get('gender', '?')}\n"
                       f"Возраст: ~{age} лет\n\n"
                       f"Одежда:\n{cloth_str}\n"
                       f"Лицо: {face_str}\n\n"
                       f"Текстовый отчёт: {report_path}{word_info}")
        else:
            msg = (f"Анализ завершён!\n\n"
                   f"Обработано изображений: {len(results)}\n\n"
                   f"Текстовый отчёт: {report_path}{word_info}")

        QMessageBox.information(self, "Анализ людей", msg)
        self.person_path_label.setText(f"✓ Отчёт: {report_path}")

    def _on_person_error(self, error_msg):
        if hasattr(self, '_person_progress') and self._person_progress is not None:
            self._person_progress.close()
            self._person_progress = None
        QMessageBox.critical(self, "Ошибка", f"Ошибка анализа людей:\n{error_msg}")

    def _on_person_finished(self):
        self.btn_analyze_person.setEnabled(True)
        self.btn_person_file.setEnabled(True)
        self.btn_person_folder.setEnabled(True)
        self._person_worker = None

    # ─── Анализ мобильных телефонов (PhoneAnalyzer) ───

    def open_phone_file(self):
        """Выбор одного изображения телефона для анализа."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение телефона",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.gif);;All Files (*)",
            options=options
        )
        if file_path:
            self.phone_file = file_path
            self.phone_folder = None
            self.phone_path_label.setText(f"Файл: {os.path.basename(file_path)}")
            self.btn_analyze_phone.setEnabled(True)

    def open_phone_folder(self):
        """Выбор папки с изображениями телефонов."""
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку с изображениями телефонов",
            options=options
        )
        if folder_path:
            self.phone_folder = folder_path
            self.phone_file = None
            self.phone_path_label.setText(f"Папка: {os.path.basename(folder_path)}")
            self.btn_analyze_phone.setEnabled(True)

    def run_phone_analysis(self):
        """Запускает анализ телефонов в фоновом потоке."""
        self.btn_analyze_phone.setEnabled(False)
        self.btn_phone_file.setEnabled(False)
        self.btn_phone_folder.setEnabled(False)

        if not self.phone_file and not self.phone_folder:
            QMessageBox.warning(self, "Ошибка", "Выберите файл или папку")
            self._on_phone_finished()
            return

        self._phone_progress = self._create_progress("Загрузка моделей...", 4)
        self._phone_progress.setWindowTitle("Анализ мобильного телефона")
        self._phone_progress.canceled.connect(self._cancel_phone)

        self._phone_worker = PhoneAnalysisWorker(
            file_path=self.phone_file,
            folder_path=self.phone_folder,
            parent=None
        )
        self._phone_worker.progress_update.connect(self._on_phone_progress)
        self._phone_worker.analysis_done.connect(self._on_phone_done)
        self._phone_worker.analysis_error.connect(self._on_phone_error)
        self._phone_worker.finished.connect(self._on_phone_finished)
        self._phone_worker.start()

    def _cancel_phone(self):
        if hasattr(self, '_phone_worker') and self._phone_worker is not None:
            self._phone_worker.cancel()

    def _on_phone_progress(self, label, value, maximum):
        if hasattr(self, '_phone_progress') and self._phone_progress is not None:
            self._phone_progress.setMaximum(maximum)
            self._phone_progress.setValue(value)
            self._phone_progress.setLabelText(label)

    def _on_phone_done(self, results):
        if hasattr(self, '_phone_progress') and self._phone_progress is not None:
            self._phone_progress.close()
            self._phone_progress = None

        if not results:
            QMessageBox.warning(self, "Ошибка", "Изображения не найдены")
            return

        from datetime import datetime as _dt
        ts = _dt.now().strftime("%d-%m-%Y_%H-%M-%S")
        report_dir = os.path.join("Results", f"PhoneAnalysis_{ts}")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "PHONE_REPORT.txt")
        elapsed = getattr(self._phone_worker, 'elapsed_seconds', 0)
        generate_report_phone(results, report_path, elapsed_seconds=elapsed)

        # Word отчёт
        word_info = ""
        try:
            word_doc_path = generate_report_word_phone(results, report_dir, elapsed_seconds=elapsed)
            word_info = f"\nWord отчёт: {word_doc_path}"
        except Exception as e:
            word_info = f"\n⚠ Ошибка создания Word отчёта: {str(e)}"

        if len(results) == 1:
            r = results[0]
            if "error" in r:
                msg = f"Ошибка: {r['error']}"
            else:
                phone = r.get("phone", {})
                imei = r.get("imei", {})

                if phone.get("is_phone"):
                    phone_str = (f"Тип: {phone.get('category', '?')}\n"
                                 f"Модель: {phone.get('model', '?')} "
                                 f"({phone.get('model_confidence', 0):.0%})")
                else:
                    phone_str = f"Тип: {phone.get('category', 'Не определён')}"

                imei_str = imei.get("message", "Не найден")

                msg = (f"Анализ завершён!\n\n"
                       f"Файл: {r['file']}\n\n"
                       f"{phone_str}\n\n"
                       f"IMEI: {imei_str}\n\n"
                       f"Текстовый отчёт: {report_path}{word_info}")
        else:
            msg = (f"Анализ завершён!\n\n"
                   f"Обработано изображений: {len(results)}\n\n"
                   f"Текстовый отчёт: {report_path}{word_info}")

        QMessageBox.information(self, "Анализ мобильного телефона", msg)
        self.phone_path_label.setText(f"✓ Отчёт: {report_path}")

    def _on_phone_error(self, error_msg):
        if hasattr(self, '_phone_progress') and self._phone_progress is not None:
            self._phone_progress.close()
            self._phone_progress = None
        QMessageBox.critical(self, "Ошибка", f"Ошибка анализа телефона:\n{error_msg}")

    def _on_phone_finished(self):
        self.btn_analyze_phone.setEnabled(True)
        self.btn_phone_file.setEnabled(True)
        self.btn_phone_folder.setEnabled(True)
        self._phone_worker = None

    # ─── Open-set detection (OpenSetAnalyzer) ───

    def open_openset_file(self):
        """Выбор одного изображения для open-set анализа."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение для open-set анализа",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.gif);;All Files (*)",
            options=options
        )
        if file_path:
            self.openset_file = file_path
            self.openset_folder = None
            self._set_open_set_unknown_dir(None)
            self._set_open_set_unknown_classes_dir(None)
            self.openset_path_label.setText(f"Файл: {os.path.basename(file_path)}")
            self.btn_analyze_open_set.setEnabled(True)

    def open_openset_video(self):
        """Выбор одного видео для open-set анализа."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео для open-set анализа",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v);;All Files (*)",
            options=options
        )
        if file_path:
            self.openset_file = file_path
            self.openset_folder = None
            self._set_open_set_unknown_dir(None)
            self._set_open_set_unknown_classes_dir(None)
            self.openset_path_label.setText(f"Видео: {os.path.basename(file_path)}")
            self.btn_analyze_open_set.setEnabled(True)

    def open_openset_multi_video(self):
        pass  # удалена — папка теперь использует единый датасет

    def open_openset_folder(self):
        """Выбор папки с изображениями и видео для open-set анализа."""
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку для open-set анализа",
            options=options
        )
        if folder_path:
            self.openset_folder = folder_path
            self.openset_file = None
            self._set_open_set_unknown_dir(None)
            self._set_open_set_unknown_classes_dir(None)
            self.openset_path_label.setText(f"Папка: {os.path.basename(folder_path)}")
            self.btn_analyze_open_set.setEnabled(True)

    def on_open_set_sample_interval_changed(self, value):
        self.openset_sample_interval_seconds = float(value)

    def on_open_set_unknown_crop_margin_changed(self, value):
        self.openset_unknown_crop_margin_scale = float(value)
        self.openset_analysis_options = self._get_open_set_analysis_options(self.openset_preset_name)
        self._update_open_set_preset_details()

    def on_open_set_hdbscan_changed(self, state):
        self.openset_use_hdbscan = bool(state)
        self.openset_analysis_options = self._get_open_set_analysis_options(self.openset_preset_name)

    def on_open_set_unknown_class_similarity_changed(self, value):
        self.openset_unknown_class_similarity_threshold = float(value)
        self.openset_analysis_options = self._get_open_set_analysis_options(self.openset_preset_name)
        self._update_open_set_preset_details()

    def on_open_set_preset_changed(self, preset_name):
        self.openset_preset_name = preset_name
        self.openset_analysis_options = self._get_open_set_analysis_options(preset_name)
        self._update_open_set_preset_details()

    def _update_unknown_merge_similarity_hint(self):
        threshold = self.unknown_merge_similarity_threshold
        hint = (
            f"Текущий порог: {threshold:.2f}\n"
            "0.50-0.70: агрессивное укрупнение, разные объекты чаще сольются\n"
            "0.70-0.88: рабочий баланс для первичной сборки классов\n"
            "0.88-0.99: строгая группировка, больше мелких классов"
        )
        if hasattr(self, 'unknown_merge_similarity_hint'):
            self.unknown_merge_similarity_hint.setText(hint)

    def on_unknown_merge_similarity_changed(self, value):
        self.unknown_merge_similarity_threshold = float(value)
        self._update_unknown_merge_similarity_hint()

    def on_unknown_consolidate_threshold_changed(self, value):
        self.unknown_consolidate_threshold = float(value)

    def run_consolidate_unknown_class(self):
        self._clear_unknown_merge_preview("Идёт консолидация базы UnknownClass...")
        self.btn_run_consolidate.setEnabled(False)

        self._consolidate_progress = self._create_progress("Загрузка базы UnknownClass...", 100)
        self._consolidate_progress.setWindowTitle("Консолидация базы")
        self._consolidate_progress.canceled.connect(self._cancel_consolidate)

        self._consolidate_worker = ConsolidateUnknownClassWorker(
            similarity_threshold=self.unknown_consolidate_threshold,
            parent=None,
        )
        self._consolidate_worker.progress_update.connect(self._on_consolidate_progress)
        self._consolidate_worker.analysis_done.connect(self._on_consolidate_done)
        self._consolidate_worker.analysis_error.connect(self._on_consolidate_error)
        self._consolidate_worker.finished.connect(self._on_consolidate_finished)
        self._consolidate_worker.start()

    def _cancel_consolidate(self):
        if hasattr(self, '_consolidate_worker') and self._consolidate_worker is not None:
            self._consolidate_worker.cancel()

    def _on_consolidate_progress(self, label, value, maximum):
        if hasattr(self, '_consolidate_progress') and self._consolidate_progress is not None:
            self._consolidate_progress.setMaximum(maximum)
            self._consolidate_progress.setValue(value)
            self._consolidate_progress.setLabelText(label)

    def _on_consolidate_done(self, result):
        if hasattr(self, '_consolidate_progress') and self._consolidate_progress is not None:
            self._consolidate_progress.close()
            self._consolidate_progress = None

        if result.get("error"):
            QMessageBox.warning(self, "Ошибка", result["error"])
            return

        self._update_unknown_merge_preview(result)
        msg = (
            f"Консолидация завершена!\n\n"
            f"Порог объединения: {result.get('similarity_threshold', self.unknown_consolidate_threshold):.2f}\n"
            f"Классов до: {result.get('class_count_before', '—')}\n"
            f"Классов после: {result.get('class_count', 0)}\n"
            f"Объединений выполнено: {result.get('merged_count', 0)}\n"
            f"Изображений в базе: {result.get('total_image_count', 0)}\n\n"
            f"TXT отчёт: {result.get('report_path', '—')}\n"
            f"JSON отчёт: {result.get('json_report_path', '—')}"
        )
        QMessageBox.information(self, "UnknownClass DB — Консолидация", msg)

    def _on_consolidate_error(self, error_msg):
        if hasattr(self, '_consolidate_progress') and self._consolidate_progress is not None:
            self._consolidate_progress.close()
            self._consolidate_progress = None
        QMessageBox.critical(self, "Ошибка", f"Ошибка консолидации базы:\n{error_msg}")

    def _on_consolidate_finished(self):
        self.btn_run_consolidate.setEnabled(True)
        self._consolidate_worker = None

    def open_openset_unknown_folder(self):
        if not self._open_folder_in_explorer(self.openset_unknown_dir, "папку Unknown"):
            self._set_open_set_unknown_dir(None)
    
    def open_openset_unknown_classes_folder(self):
        if not self._open_folder_in_explorer(self.openset_unknown_classes_dir, "классы базы UnknownClass"):
            self._set_open_set_unknown_classes_dir(None)

    def open_unknown_merge_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку для импорта в базу UnknownClass"
        )
        if folder_path:
            self.unknown_merge_folder = folder_path
            self.unknown_merge_path_label.setText(f"Папка: {os.path.basename(folder_path)}")
            self.btn_run_unknown_merge.setEnabled(True)
            self._clear_unknown_merge_preview("Выбрана папка. Запустите импорт для пополнения базы UnknownClass")

    def open_unknown_merge_output_folder(self):
        if not self._open_folder_in_explorer(self.unknown_merge_output_dir, "папку базы UnknownClass"):
            self._set_unknown_merge_output_dir(None)

    def open_unknown_merge_classes_folder(self):
        if not self._open_folder_in_explorer(self.unknown_merge_classes_dir, "классы базы UnknownClass"):
            self._set_unknown_merge_classes_dir(None)

    def run_open_set_analysis(self):
        """Запускает open-set анализ в фоновом потоке."""
        self._set_open_set_unknown_dir(None)
        self._set_open_set_unknown_classes_dir(None)
        self.btn_analyze_open_set.setEnabled(False)
        self.btn_openset_file.setEnabled(False)
        self.btn_openset_video.setEnabled(False)
        self.btn_openset_folder.setEnabled(False)

        if not self.openset_file and not self.openset_folder:
            QMessageBox.warning(self, "Ошибка", "Выберите файл или папку")
            self._on_open_set_finished()
            return

        from datetime import datetime as _dt
        multi_video_output_dir = None
        if self.openset_folder:
            ts = _dt.now().strftime("%d-%m-%Y_%H-%M-%S")
            multi_video_output_dir = os.path.join("Results", f"OpenSetFolder_{ts}")

        self._open_set_progress = self._create_progress("Загрузка моделей...", 4)
        self._open_set_progress.setWindowTitle("Open-set detection")
        self._open_set_progress.canceled.connect(self._cancel_open_set)

        self._open_set_worker = OpenSetAnalysisWorker(
            file_path=self.openset_file,
            folder_path=self.openset_folder,
            multi_video_output_dir=multi_video_output_dir,
            sample_interval_seconds=self.openset_sample_interval_seconds,
            analysis_options=self.openset_analysis_options,
            parent=None
        )
        self._open_set_worker.progress_update.connect(self._on_open_set_progress)
        self._open_set_worker.analysis_done.connect(self._on_open_set_done)
        self._open_set_worker.analysis_error.connect(self._on_open_set_error)
        self._open_set_worker.finished.connect(self._on_open_set_finished)
        self._open_set_worker.start()

    def _cancel_open_set(self):
        if hasattr(self, '_open_set_worker') and self._open_set_worker is not None:
            self._open_set_worker.cancel()

    def _on_open_set_progress(self, label, value, maximum):
        if hasattr(self, '_open_set_progress') and self._open_set_progress is not None:
            self._open_set_progress.setMaximum(maximum)
            self._open_set_progress.setValue(value)
            self._open_set_progress.setLabelText(label)

    def _on_open_set_done(self, results):
        if hasattr(self, '_open_set_progress') and self._open_set_progress is not None:
            self._open_set_progress.close()
            self._open_set_progress = None

        if not results:
            QMessageBox.warning(self, "Ошибка", "Изображения не найдены")
            return

        from datetime import datetime as _dt
        ts = _dt.now().strftime("%d-%m-%Y_%H-%M-%S")
        report_dir = os.path.join("Results", f"OpenSetAnalysis_{ts}")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "OPEN_SET_REPORT.txt")
        generate_report_open_set(results, report_path)

        word_info = ""
        try:
            word_doc_path = generate_report_word_open_set(results, report_dir)
            word_info = f"\nWord отчёт: {word_doc_path}"
        except Exception as e:
            word_info = f"\n⚠ Ошибка создания Word отчёта: {str(e)}"

        if len(results) == 1:
            r = results[0]
            if "error" in r:
                msg = f"Ошибка: {r['error']}"
            elif r.get("type") == "multi_video":
                ds_count = len(r.get("yolo_datasets", []))
                file_count = r.get("file_count", r.get("video_count", 0))
                msg = (
                    f"Open-set анализ папки завершён!\n\n"
                    f"Файлов обработано: {file_count}\n"
                    f"Проанализировано кадров/изображений: {r.get('total_sampled_frames', 0)}\n"
                    f"Интервал выборки: {self.openset_sample_interval_seconds} сек\n"
                    f"Режим: {self.openset_preset_name}\n\n"
                    f"Known objects (всего): {r.get('total_known_count', 0)}\n"
                    f"Unknown objects (всего): {r.get('total_unknown_count', 0)}\n"
                    f"Единых классов: {r.get('combined_class_count', 0)}\n\n"
                    f"Классы: {r.get('combined_classes_dir') or '—'}\n"
                    f"YOLO датасеты ({ds_count} кл.): {r.get('yolo_datasets_dir') or '—'}\n\n"
                    f"Текстовый отчёт: {report_path}{word_info}"
                )
            elif r.get("type") == "video":
                msg = (
                    f"Open-set анализ видео завершён!\n\n"
                    f"Файл: {r['file']}\n\n"
                    f"Режим: {self.openset_preset_name}\n"
                    f"Всего кадров: {r.get('frame_count', 0)}\n"
                    f"Проанализировано кадров: {r.get('sampled_frames', 0)}\n"
                    f"Интервал выборки: {r.get('sample_interval_seconds', 0)} сек\n"
                    f"Known objects: {r.get('known_count', 0)}\n"
                    f"Unknown objects: {r.get('unknown_count', 0)}\n"
                    f"Кадров с Unknown: {r.get('frames_with_unknown', 0)}\n"
                    f"Запас unknown crop: {r.get('unknown_crop_margin_scale', self.openset_unknown_crop_margin_scale):.2f}\n"
                    f"Классов в базе UnknownClass: {r.get('unknown_class_count', 0)}\n"
                    f"Аннотированное видео: {r.get('annotated_video_path') or '—'}\n"
                    f"Аннотированные кадры ({r.get('saved_annotated_frames', 0)} шт.): {r.get('annotated_frames_dir') or '—'}\n"
                    f"Кадры: {r.get('saved_frames_dir') or '—'}\n\n"
                    f"Скрины Unknown: {r.get('unknown_objects_dir') or '—'}\n"
                    f"Классы базы UnknownClass: {r.get('unknown_classes_dir') or '—'}\n"
                    f"Таймкоды Unknown: {r.get('unknown_timestamps_path') or '—'}\n\n"
                    f"Текстовый отчёт: {report_path}{word_info}"
                )
            else:
                msg = (
                    f"Open-set анализ завершён!\n\n"
                    f"Файл: {r['file']}\n\n"
                    f"Режим: {self.openset_preset_name}\n"
                    f"Proposal boxes: {r.get('proposal_count', 0)}\n"
                    f"Known objects: {r.get('known_count', 0)}\n"
                    f"Unknown objects: {r.get('unknown_count', 0)}\n"
                    f"Запас unknown crop: {r.get('unknown_crop_margin_scale', self.openset_unknown_crop_margin_scale):.2f}\n"
                    f"Классов в базе UnknownClass: {r.get('unknown_class_count', 0)}\n"
                    f"Аннотированное изображение: {r.get('annotated_path', '—')}\n\n"
                    f"Скрины Unknown: {r.get('unknown_objects_dir') or '—'}\n\n"
                    f"Классы базы UnknownClass: {r.get('unknown_classes_dir') or '—'}\n\n"
                    f"Текстовый отчёт: {report_path}{word_info}"
                )
        else:
            msg = (
                f"Open-set анализ завершён!\n\n"
                f"Обработано элементов: {len(results)}\n\n"
                f"Текстовый отчёт: {report_path}{word_info}"
            )

        if len(results) == 1 and results[0].get("type") == "multi_video":
            r = results[0]
            self._set_open_set_unknown_dir(None)
            self._set_open_set_unknown_classes_dir(r.get("combined_classes_dir"))
            self._cm_last_yolo_datasets_dir = r.get("yolo_datasets_dir")
        elif len(results) == 1:
            self._set_open_set_unknown_dir(results[0].get("unknown_objects_dir"))
            self._set_open_set_unknown_classes_dir(results[0].get("unknown_classes_dir"))
            self._cm_last_yolo_datasets_dir = results[0].get("yolo_datasets_dir")
        else:
            self._set_open_set_unknown_dir(None)
            self._set_open_set_unknown_classes_dir(None)
            self._cm_last_yolo_datasets_dir = None

        QMessageBox.information(self, "Open-set detection", msg)
        self.openset_path_label.setText(f"✓ Отчёт: {report_path}")

    def _on_open_set_error(self, error_msg):
        if hasattr(self, '_open_set_progress') and self._open_set_progress is not None:
            self._open_set_progress.close()
            self._open_set_progress = None
        QMessageBox.critical(self, "Ошибка", f"Ошибка open-set анализа:\n{error_msg}")

    def _on_open_set_finished(self):
        self.btn_analyze_open_set.setEnabled(True)
        self.btn_openset_file.setEnabled(True)
        self.btn_openset_video.setEnabled(True)
        self.btn_openset_folder.setEnabled(True)
        self._open_set_worker = None

    def run_unknown_merge_analysis(self):
        self._set_unknown_merge_output_dir(None)
        self._set_unknown_merge_classes_dir(None)
        self.unknown_merge_report_path = None
        self.unknown_merge_json_report_path = None
        self.unknown_merge_word_report_path = None
        self._clear_unknown_merge_preview("Идёт пополнение базы UnknownClass...")
        self.btn_run_unknown_merge.setEnabled(False)
        self.btn_unknown_merge_folder.setEnabled(False)

        if not self.unknown_merge_folder:
            QMessageBox.warning(self, "Ошибка", "Выберите папку для импорта в базу UnknownClass")
            self._on_unknown_merge_finished()
            return

        self._unknown_merge_progress = self._create_progress("Поиск unknown-изображений...", 100)
        self._unknown_merge_progress.setWindowTitle("UnknownClass DB")
        self._unknown_merge_progress.canceled.connect(self._cancel_unknown_merge)

        self._unknown_merge_worker = UnknownClassMergeWorker(
            folder_path=self.unknown_merge_folder,
            similarity_threshold=self.unknown_merge_similarity_threshold,
            parent=None,
        )
        self._unknown_merge_worker.progress_update.connect(self._on_unknown_merge_progress)
        self._unknown_merge_worker.analysis_done.connect(self._on_unknown_merge_done)
        self._unknown_merge_worker.analysis_error.connect(self._on_unknown_merge_error)
        self._unknown_merge_worker.finished.connect(self._on_unknown_merge_finished)
        self._unknown_merge_worker.start()

    def _cancel_unknown_merge(self):
        if hasattr(self, '_unknown_merge_worker') and self._unknown_merge_worker is not None:
            self._unknown_merge_worker.cancel()

    def _on_unknown_merge_progress(self, label, value, maximum):
        if hasattr(self, '_unknown_merge_progress') and self._unknown_merge_progress is not None:
            self._unknown_merge_progress.setMaximum(maximum)
            self._unknown_merge_progress.setValue(value)
            self._unknown_merge_progress.setLabelText(label)

    def _on_unknown_merge_done(self, result):
        if hasattr(self, '_unknown_merge_progress') and self._unknown_merge_progress is not None:
            self._unknown_merge_progress.close()
            self._unknown_merge_progress = None

        if not result:
            QMessageBox.warning(self, "Ошибка", "Не удалось обновить базу UnknownClass")
            return

        if result.get("error"):
            QMessageBox.warning(self, "Ошибка", result["error"])
            return

        self.unknown_merge_report_path = result.get("report_path")
        self.unknown_merge_json_report_path = result.get("json_report_path")
        self.unknown_merge_word_report_path = result.get("word_report_path")
        self._set_unknown_merge_output_dir(result.get("output_dir"))
        self._set_unknown_merge_classes_dir(result.get("merged_classes_dir"))
        self._update_unknown_merge_preview(result)

        skipped_count = len(result.get("skipped_files", []))
        word_info = result.get("word_report_path") or "не создан"
        json_info = result.get("json_report_path") or "не создан"
        msg = (
            f"База UnknownClass обновлена!\n\n"
            f"Исходная папка: {result.get('input_root', '—')}\n"
            f"Изображений импортировано: {result.get('image_count', 0)}\n"
            f"Подпапок-источников: {result.get('source_dir_count', 0)}\n"
            f"Классов в базе: {result.get('class_count', 0)}\n"
            f"Всего изображений в базе: {result.get('total_image_count', 0)}\n"
            f"Порог схожести: {result.get('similarity_threshold', self.unknown_merge_similarity_threshold):.2f}\n"
            f"Пропущено файлов: {skipped_count}\n\n"
            f"Папка базы: {result.get('output_dir', '—')}\n"
            f"Папка классов базы: {result.get('merged_classes_dir', '—')}\n"
            f"TXT отчёт: {result.get('report_path', '—')}\n"
            f"JSON отчёт: {json_info}\n"
            f"Word отчёт: {word_info}"
        )
        QMessageBox.information(self, "UnknownClass DB", msg)
        self.unknown_merge_path_label.setText(f"✓ База обновлена: {result.get('report_path', '—')}")

    def _on_unknown_merge_error(self, error_msg):
        if hasattr(self, '_unknown_merge_progress') and self._unknown_merge_progress is not None:
            self._unknown_merge_progress.close()
            self._unknown_merge_progress = None
        QMessageBox.critical(self, "Ошибка", f"Ошибка обновления базы UnknownClass:\n{error_msg}")

    def _on_unknown_merge_finished(self):
        self.btn_run_unknown_merge.setEnabled(bool(self.unknown_merge_folder))
        self.btn_unknown_merge_folder.setEnabled(True)
        self._unknown_merge_worker = None