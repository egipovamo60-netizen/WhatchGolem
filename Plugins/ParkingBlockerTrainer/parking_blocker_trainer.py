"""
Плагин тренировки нового YOLO-класса: парковочный блокиратор.

Принцип:
  - Изображения из LearnDataset/ — кропы сцен с блокираторами.
  - Синтетический датасет: кропы вставляются на фоновые кадры с реальными bbox.
  - Fine-tuning выполняется на предобученных весах (yolo11n.pt или yolo11x.pt).
  - Готовая модель сохраняется в Models/parking_blocker.pt

Зависимости:
  pip install ultralytics opencv-python
"""

import os
import sys
import shutil
import random
import yaml
import logging

import cv2
import numpy as np

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _BASE_DIR)

try:
    from logger import get_logger
    _log = get_logger("ParkingBlockerTrainer")
except Exception:
    _log = logging.getLogger("ParkingBlockerTrainer")
    logging.basicConfig(level=logging.INFO)

# ─── Пути ───────────────────────────────────────────────────────────────────
LEARN_DATASET_DIR  = os.path.join(_BASE_DIR, "LearnDataset")
YOLO_DATASET_DIR   = os.path.join(_BASE_DIR, "LearnDataset_yolo")
DATA_YAML_PATH     = os.path.join(YOLO_DATASET_DIR, "data.yaml")
MODELS_DIR         = os.path.join(_BASE_DIR, "Models")
OUTPUT_MODEL_PATH  = os.path.join(MODELS_DIR, "parking_blocker.pt")

# Путь к основному видео (фоны для синтетического датасета)
DEFAULT_VIDEO_PATH = os.path.join(_BASE_DIR, "Test video", "Cam1.avi")

# Претренированные веса для старта (выбирается автоматически)
_PRETRAINED_CANDIDATES = [
    os.path.join(_BASE_DIR, "yolo11n.pt"),
    os.path.join(_BASE_DIR, "yolo11x.pt"),
    os.path.join(_BASE_DIR, "yolov8n.pt"),
]

CLASS_NAME  = "parking_blocker"
CLASS_ID    = 0
VAL_SPLIT   = 0.15          # 15 % — валидация
RANDOM_SEED = 42

# ─── Кэш загруженной модели ─────────────────────────────────────────────────
_loaded_model = None
_loaded_model_path = None


# ═════════════════════════════════════════════════════════════════════════════
#  1. ПОДГОТОВКА ДАТАСЕТА
# ═════════════════════════════════════════════════════════════════════════════

def prepare_synthetic_dataset(
    source_dir: str = LEARN_DATASET_DIR,
    dest_dir: str = YOLO_DATASET_DIR,
    video_path: str | None = None,
    n_train: int = 2000,
    n_val: int = 300,
    overwrite: bool = False,
    log_callback=None,
) -> str:
    """
    Генерирует синтетический YOLO-датасет:
    кропы блокираторов вставляются на фоновые кадры с реальными bbox.

    Параметры
    ----------
    source_dir  : папка с кропами блокираторов (LearnDataset)
    dest_dir    : выходная папка (LearnDataset_yolo)
    video_path  : путь к видео для извлечения фонов (необязательно)
    n_train     : количество синтетических тренировочных сцен
    n_val       : количество синтетических сцен для валидации
    overwrite   : пересоздать датасет если уже существует
    log_callback: callable(str) — для вывода прогресса в UI

    Возвращает путь к data.yaml.
    """
    from Plugins.ParkingBlockerTrainer.synthetic_dataset_generator import generate

    # Автоопределение пути к видео
    if video_path is None:
        if os.path.isfile(DEFAULT_VIDEO_PATH):
            video_path = DEFAULT_VIDEO_PATH
            if log_callback:
                log_callback(f"Видео для фонов: {os.path.basename(video_path)}")
        else:
            if log_callback:
                log_callback("Видео не найдено — фоны будут из кропов и синтетики")

    return generate(
        crops_dir      = source_dir,
        dest_dir       = dest_dir,
        data_yaml_path = DATA_YAML_PATH,
        video_path     = video_path,
        n_train        = n_train,
        n_val          = n_val,
        overwrite      = overwrite,
        log_callback   = log_callback,
    )


def prepare_dataset(
    source_dir: str = LEARN_DATASET_DIR,
    dest_dir: str = YOLO_DATASET_DIR,
    val_split: float = VAL_SPLIT,
    overwrite: bool = False,
    class_name: str = CLASS_NAME,
) -> str:
    """
    Готовит YOLO-датасет из папки с кропами.

    Каждый кроп вставляется на случайный фон (серый шум) в случайную позицию.
    YOLO-метка содержит реальные нормализованные координаты bbox кропа на холсте.

    Структура на выходе:
        dest_dir/
          images/train/   *.jpg
          images/val/     *.jpg
          labels/train/   *.txt
          labels/val/     *.txt
          data.yaml

    Возвращает путь к data.yaml.
    """
    if os.path.exists(dest_dir) and not overwrite:
        if os.path.exists(DATA_YAML_PATH):
            _log.info(f"Датасет уже существует: {dest_dir}  (overwrite=False)")
            return DATA_YAML_PATH

    # Собираем все .jpg / .png
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [
        f for f in os.listdir(source_dir)
        if os.path.splitext(f.lower())[1] in supported
    ]
    if not images:
        raise FileNotFoundError(f"В папке {source_dir} не найдено изображений.")

    _log.info(f"Найдено изображений: {len(images)}")

    random.seed(RANDOM_SEED)
    random.shuffle(images)

    split_idx = max(1, int(len(images) * (1 - val_split)))
    train_images = images[:split_idx]
    val_images   = images[split_idx:]

    _log.info(f"Train: {len(train_images)}  |  Val: {len(val_images)}")

    # Создаём папки
    for split in ("train", "val"):
        os.makedirs(os.path.join(dest_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "labels", split), exist_ok=True)

    # Параметры холста — стандартное HD-разрешение камеры
    CANVAS_W, CANVAS_H = 1280, 720
    # Мин./макс. доля высоты холста для масштабирования кропа
    SCALE_MIN, SCALE_MAX = 0.20, 0.55

    def _write_split(file_list: list, split: str) -> None:
        np_rng = np.random.default_rng(RANDOM_SEED)

        for fname in file_list:
            src = os.path.join(source_dir, fname)
            base = os.path.splitext(fname)[0]

            # Загружаем кроп
            crop = cv2.imread(src)
            if crop is None:
                _log.warning(f"Не удалось загрузить: {src}  — пропуск")
                continue

            crop_h, crop_w = crop.shape[:2]

            # Масштабируем кроп до случайной высоты (SCALE_MIN–SCALE_MAX от холста)
            target_h = int(CANVAS_H * random.uniform(SCALE_MIN, SCALE_MAX))
            target_h = max(32, target_h)
            target_w = int(crop_w * target_h / crop_h)

            # Если кроп шире холста — уменьшаем по ширине
            if target_w >= CANVAS_W:
                target_w = CANVAS_W - 20
                target_h = int(crop_h * target_w / crop_w)

            crop_scaled = cv2.resize(crop, (target_w, target_h),
                                     interpolation=cv2.INTER_AREA)

            # Генерируем фон — серый шум (имитация асфальта)
            base_gray = int(np_rng.integers(85, 165))
            noise = np_rng.integers(-18, 18,
                                    (CANVAS_H, CANVAS_W, 3), dtype=np.int16)
            canvas = np.clip(base_gray + noise, 0, 255).astype(np.uint8)

            # Случайное размещение кропа на холсте
            max_x = max(0, CANVAS_W - target_w)
            max_y = max(0, CANVAS_H - target_h)
            x0 = random.randint(0, max_x)
            y0 = random.randint(0, max_y)

            canvas[y0:y0 + target_h, x0:x0 + target_w] = crop_scaled

            # Вычисляем нормализованные YOLO-координаты
            cx = (x0 + target_w / 2.0) / CANVAS_W
            cy = (y0 + target_h / 2.0) / CANVAS_H
            bw = target_w / CANVAS_W
            bh = target_h / CANVAS_H

            # Сохраняем сцену (jpg) и метку
            dst_img = os.path.join(dest_dir, "images", split, base + ".jpg")
            cv2.imwrite(dst_img, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

            dst_lbl = os.path.join(dest_dir, "labels", split, base + ".txt")
            with open(dst_lbl, "w", encoding="utf-8") as lf:
                lf.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    _write_split(train_images, "train")
    _write_split(val_images,   "val")

    # Пишем data.yaml в папку датасета
    yaml_path = os.path.join(dest_dir, "data.yaml")
    data_cfg = {
        "path": dest_dir,
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": {0: class_name},
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_cfg, f, allow_unicode=True, default_flow_style=False)

    _log.info(f"data.yaml создан: {yaml_path}")
    return yaml_path


# ═════════════════════════════════════════════════════════════════════════════
#  2. ТРЕНИРОВКА
# ═════════════════════════════════════════════════════════════════════════════

def train(
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    pretrained_weights: str | None = None,
    output_path: str = OUTPUT_MODEL_PATH,
    overwrite_dataset: bool = False,
    device: str = "auto",
    patience: int = 50,
    log_callback=None,
    source_dir: str | None = None,
    data_yaml: str | None = None,
    class_name: str = CLASS_NAME,
) -> str:
    """
    Запускает fine-tuning YOLO.

    Параметры
    ----------
    epochs           : количество эпох
    imgsz            : размер изображения (640 стандарт)
    batch            : размер батча (-1 = авто)
    pretrained_weights: путь к .pt файлу базовой модели (если None — автовыбор)
    output_path      : куда сохранить итоговую модель
    overwrite_dataset: пересоздать датасет, даже если он уже есть
    device           : "auto", "cpu", "0", "0,1", ...
    log_callback     : callable(str) — вызывается после каждой эпохи
    source_dir       : папка с кропами (None = LearnDataset/)
    data_yaml        : путь к готовому data.yaml; если задан — пропускает prepare_dataset()

    Возвращает путь к итоговой модели.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics не установлен. Выполните: pip install ultralytics"
        )

    # 1. Подготовка датасета
    if data_yaml is not None:
        # Готовый датасет — пропускаем генерацию
        data_yaml = os.path.abspath(data_yaml)
        if not os.path.isfile(data_yaml):
            raise FileNotFoundError(f"data.yaml не найден: {data_yaml}")
        _log.info(f"Используется готовый датасет: {data_yaml}")
    else:
        _src = source_dir if source_dir is not None else LEARN_DATASET_DIR
        _src = os.path.abspath(_src)
        if _src == os.path.abspath(LEARN_DATASET_DIR):
            _dest = YOLO_DATASET_DIR
        else:
            _dest = _src + "_yolo"
        data_yaml = prepare_dataset(
            source_dir=_src,
            dest_dir=_dest,
            overwrite=overwrite_dataset,
            class_name=class_name,
        )

    # 2. Выбор базовых весов
    if pretrained_weights is None:
        pretrained_weights = _find_pretrained()
    if not os.path.exists(pretrained_weights):
        raise FileNotFoundError(
            f"Файл весов не найден: {pretrained_weights}\n"
            "Запустите download_yolo11.py или укажите путь вручную."
        )
    _log.info(f"Базовая модель: {pretrained_weights}")

    # 3. Определение device
    if device == "auto":
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # Проверяем, что запрошенный GPU действительно доступен
    if str(device).lower() not in ("cpu", ""):
        try:
            import torch
            if not torch.cuda.is_available():
                _log.warning(
                    f"Запрошен device={device}, но torch.cuda.is_available()=False. "
                    "Переключаюсь на CPU. "
                    "Для GPU установите PyTorch с CUDA: "
                    "pip install torch --index-url https://download.pytorch.org/whl/cu126"
                )
                device = "cpu"
        except ImportError:
            device = "cpu"
    _log.info(f"Device: {device}")

    # 3.1. Стабилизация для Windows + CPU.
    # На некоторых связках torch/ultralytics это предотвращает native crash 0xC0000005
    # на старте эпох (сразу после инициализации optimizer).
    safe_cpu_windows = (os.name == "nt" and str(device).lower() == "cpu")
    if safe_cpu_windows:
        try:
            import torch
            torch.set_num_threads(1)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(1)
            if hasattr(torch.backends, "mkldnn"):
                torch.backends.mkldnn.enabled = False
            _log.info("Включен safe-профиль для Windows+CPU: threads=1, mkldnn=False")
        except Exception as exc:
            _log.warning(f"Не удалось полностью применить safe-профиль torch: {exc}")

    # 4. Тренировка
    model = YOLO(pretrained_weights)
    _log.info(f"Запуск тренировки: epochs={epochs}, imgsz={imgsz}, batch={batch}")

    # Callback для отображения прогресса по эпохам
    if log_callback is not None:
        def _metrics_to_dict(metrics_obj):
            """Приводит trainer.metrics к обычному dict независимо от типа."""
            if metrics_obj is None:
                return {}
            if isinstance(metrics_obj, dict):
                return metrics_obj
            # DetMetrics и аналоги — используем results_dict
            if hasattr(metrics_obj, "results_dict"):
                try:
                    return dict(metrics_obj.results_dict)
                except Exception:
                    pass
            return {}

        def _on_train_epoch_end(trainer):
            ep     = trainer.epoch + 1
            total  = trainer.args.epochs
            metrics = _metrics_to_dict(trainer.metrics)
            box_loss = metrics.get("train/box_loss", metrics.get("box_loss"))
            cls_loss = metrics.get("train/cls_loss", metrics.get("cls_loss"))
            mAP50    = metrics.get("metrics/mAP50(B)", metrics.get("mAP50"))
            parts = [f"Эпоха {ep}/{total}"]
            if box_loss is not None:
                parts.append(f"box_loss={box_loss:.4f}")
            if cls_loss is not None:
                parts.append(f"cls_loss={cls_loss:.4f}")
            if mAP50 is not None:
                parts.append(f"mAP50={mAP50:.4f}")
            log_callback("  ".join(parts))

        def _on_val_end(trainer):
            metrics = _metrics_to_dict(trainer.metrics)
            mAP50   = metrics.get("metrics/mAP50(B)", metrics.get("mAP50"))
            mAP5095 = metrics.get("metrics/mAP50-95(B)", metrics.get("mAP50-95"))
            parts = ["  [val]"]
            if mAP50 is not None:
                parts.append(f"mAP50={mAP50:.4f}")
            if mAP5095 is not None:
                parts.append(f"mAP50-95={mAP5095:.4f}")
            if len(parts) > 1:
                log_callback("  ".join(parts))

        model.add_callback("on_train_epoch_end", _on_train_epoch_end)
        model.add_callback("on_val_end", _on_val_end)

    on_gpu = str(device).lower() not in ("cpu",)

    train_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name=class_name,
        project=os.path.join(_BASE_DIR, "Models", "training_runs"),
        exist_ok=True,
        verbose=True,
        plots=False,        # matplotlib крашит в subprocess на Windows (CPU и GPU)
        # ── GPU-оптимизации ───────────────────────────────────────────────────
        workers=0,                        # 0 = без multiprocessing (WinError 1455 при workers>0 + CUDA)
        amp=on_gpu,                       # mixed precision только на GPU
        cache=on_gpu,                     # кэш в RAM: безопасен на GPU, краш на CPU
        # Аугментации
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.1,
        fliplr=0.5,
        # mosaic/mixup: на GPU работают, на CPU/Windows в subprocess вызывают 0xC0000005
        mosaic=1.0 if on_gpu else 0.0,
        mixup=0.1 if on_gpu else 0.0,
        copy_paste=0.0,
        close_mosaic=10 if on_gpu else 0,
        # Гиперпараметры
        lr0=0.005 if on_gpu else 0.001,
        lrf=0.01,
        warmup_epochs=3,
        patience=patience,
    )

    # safe_cpu_windows: максимальная стабилизация только на CPU
    if safe_cpu_windows:
        train_kwargs.update({
            "deterministic": False,
            "optimizer": "SGD",
            "auto_augment": None,
            "erasing": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
        })

    results = model.train(**train_kwargs)

    # 5. Копируем best.pt в Models/parking_blocker.pt
    run_dir = results.save_dir
    best_src = os.path.join(str(run_dir), "weights", "best.pt")
    if not os.path.exists(best_src):
        best_src = os.path.join(str(run_dir), "weights", "last.pt")

    os.makedirs(MODELS_DIR, exist_ok=True)
    shutil.copy2(best_src, output_path)
    _log.info(f"Модель сохранена: {output_path}")

    return output_path


# ═════════════════════════════════════════════════════════════════════════════
#  3. ИНФЕРЕНС
# ═════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str = OUTPUT_MODEL_PATH):
    """Загружает модель (ленивая загрузка, кэш)."""
    global _loaded_model, _loaded_model_path
    if _loaded_model is not None and _loaded_model_path == model_path:
        return _loaded_model
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}\n"
            "Сначала запустите train() или train_parking_blocker.py."
        )
    _loaded_model = YOLO(model_path)
    _loaded_model_path = model_path
    _log.info(f"Модель загружена: {model_path}")
    return _loaded_model


def detect(
    image,
    model_path: str = OUTPUT_MODEL_PATH,
    conf: float = 0.4,
    iou: float = 0.45,
):
    """
    Запускает детекцию парковочных блокираторов на изображении.

    Параметры
    ----------
    image      : np.ndarray (BGR) или путь к файлу
    model_path : путь к .pt файлу модели
    conf       : минимальная уверенность
    iou        : порог IoU для NMS

    Возвращает список словарей:
        [{"bbox": [x1,y1,x2,y2], "conf": float, "class": "parking_blocker"}, ...]
    """
    model = load_model(model_path)
    results = model.predict(image, conf=conf, iou=iou, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "bbox":  [x1, y1, x2, y2],
                "conf":  round(float(box.conf[0]), 3),
                "class": CLASS_NAME,
            })
    return detections


# ═════════════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═════════════════════════════════════════════════════════════════════════════

def _find_pretrained() -> str:
    """Автовыбор предобученных весов из корня проекта."""
    for path in _PRETRAINED_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Не найдено ни одного .pt файла для fine-tuning. "
        "Проверьте наличие yolo11n.pt / yolo11x.pt / yolov8n.pt в корне проекта."
    )
