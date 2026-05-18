"""
Плагин ModelInspector — загрузка, просмотр свойств и тест детекции
для произвольной YOLO-модели (.pt).

Зависимости:
    pip install ultralytics
"""

import os
import sys
import logging
import time
import json
import datetime
from pathlib import Path

import numpy as np

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _BASE_DIR)

_STATS_FILE = os.path.join(_BASE_DIR, "Stats", "model_inspector_stats.json")

try:
    from logger import get_logger
    _log = get_logger("ModelInspector")
except Exception:
    _log = logging.getLogger("ModelInspector")
    logging.basicConfig(level=logging.INFO)

# ─── кэш загруженной модели ──────────────────────────────────────────────────
_cached_model = None
_cached_path: str | None = None


# ═════════════════════════════════════════════════════════════════════════════
#  ЗАГРУЗКА
# ═════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str):
    """Загружает YOLO-модель (с кэшированием). Возвращает объект YOLO."""
    global _cached_model, _cached_path
    if _cached_model is not None and _cached_path == model_path:
        return _cached_model
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Установите: pip install ultralytics")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    _log.info(f"Загрузка модели: {model_path}")
    _cached_model = YOLO(model_path)
    _cached_path = model_path
    return _cached_model


def unload_model():
    """Сбрасывает кэш модели."""
    global _cached_model, _cached_path
    _cached_model = None
    _cached_path = None


# ═════════════════════════════════════════════════════════════════════════════
#  СВОЙСТВА МОДЕЛИ
# ═════════════════════════════════════════════════════════════════════════════

def get_model_info(model_path: str) -> dict:
    """
    Возвращает словарь со свойствами модели:
      file_name, file_size_mb, task, imgsz, num_classes, class_names,
      num_layers, num_params_m, gflops, arch
    """
    model = load_model(model_path)

    # ── базовые атрибуты ──────────────────────────────────────────────────
    info = {}
    info["file_name"]    = os.path.basename(model_path)
    info["file_size_mb"] = round(os.path.getsize(model_path) / 1024 / 1024, 2)

    # task
    try:
        info["task"] = model.task or "detect"
    except Exception:
        info["task"] = "detect"

    # классы
    try:
        names = model.names  # dict {id: name}
        info["num_classes"] = len(names)
        info["class_names"] = list(names.values())
    except Exception:
        info["num_classes"] = 0
        info["class_names"] = []

    # arch / слои / параметры / GFLOPs — из model.info()
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summary = model.info(verbose=True)

        # model.info() возвращает tuple (layers, params, gradients, flops) в новых версиях
        if isinstance(summary, (list, tuple)) and len(summary) >= 2:
            info["num_layers"]   = int(summary[0]) if summary[0] else "—"
            info["num_params_m"] = round(int(summary[1]) / 1e6, 3) if summary[1] else "—"
            info["gflops"]       = round(float(summary[3]), 2) if len(summary) > 3 and summary[3] else "—"
        else:
            info["num_layers"]   = "—"
            info["num_params_m"] = "—"
            info["gflops"]       = "—"
    except Exception:
        info["num_layers"]   = "—"
        info["num_params_m"] = "—"
        info["gflops"]       = "—"

    # arch — из имени файла или model.yaml
    try:
        arch = Path(model_path).stem
        info["arch"] = arch
    except Exception:
        info["arch"] = "—"

    return info


def format_model_info(info: dict) -> str:
    """Форматирует dict из get_model_info() в читабельный текст."""
    classes_str = ", ".join(info["class_names"][:30])
    if len(info["class_names"]) > 30:
        classes_str += f"  … (+{len(info['class_names']) - 30})"
    if not classes_str:
        classes_str = "—"

    lines = [
        f"Файл          : {info['file_name']}",
        f"Размер        : {info['file_size_mb']} МБ",
        f"Задача        : {info['task']}",
        f"Архитектура   : {info['arch']}",
        f"Слои          : {info['num_layers']}",
        f"Параметры     : {info['num_params_m']} M",
        f"GFLOPs        : {info['gflops']}",
        f"Классов       : {info['num_classes']}",
        f"Имена классов : {classes_str}",
    ]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  ДЕТЕКЦИЯ
# ═════════════════════════════════════════════════════════════════════════════

def detect_image(
    model_path: str,
    image_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> dict:
    """
    Запускает детекцию на одном изображении.

    Возвращает:
        {
          "detections": [{"bbox": [x1,y1,x2,y2], "conf": float, "class_id": int,
                          "class_name": str}, ...],
          "inference_ms": float,
          "annotated_image": np.ndarray (BGR),
        }
    """
    import cv2

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    model = load_model(model_path)

    t0 = time.perf_counter()
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
        save=False,
    )
    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    detections = []
    annotated = None

    for r in results:
        # аннотированный кадр
        try:
            annotated = r.plot()  # BGR np.ndarray
        except Exception:
            annotated = cv2.imread(image_path)

        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, str(cls_id))
            detections.append({
                "bbox":       [x1, y1, x2, y2],
                "conf":       round(float(box.conf[0]), 3),
                "class_id":   cls_id,
                "class_name": cls_name,
            })

    if annotated is None:
        annotated = cv2.imread(image_path) or np.zeros((480, 640, 3), dtype=np.uint8)

    return {
        "detections":    detections,
        "inference_ms":  inference_ms,
        "annotated_image": annotated,
    }


def _nms_detections(detections: list, iou_threshold: float = 0.45) -> list:
    """NMS поверх объединённых детекций из разных тайлов."""
    if not detections:
        return []
    from collections import defaultdict
    by_class: dict = defaultdict(list)
    for d in detections:
        by_class[d["class_id"]].append(d)

    result = []
    for dets in by_class.values():
        boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)
        scores = np.array([d["conf"] for d in dets], dtype=np.float32)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        while order.size > 0:
            i = order[0]
            result.append(dets[i])
            if order.size == 1:
                break
            rest = order[1:]
            ix1 = np.maximum(x1[i], x1[rest])
            iy1 = np.maximum(y1[i], y1[rest])
            ix2 = np.minimum(x2[i], x2[rest])
            iy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            union = areas[i] + areas[rest] - inter
            iou_vals = inter / np.maximum(union, 1e-6)
            order = rest[iou_vals <= iou_threshold]
    return result


def detect_image_tiled(
    model_path: str,
    image_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    tile_overlap: float = 0.3,
) -> dict:
    """
    Детекция на большом изображении методом скользящего окна.
    Изображение разбивается на перекрывающиеся патчи imgsz×imgsz,
    результаты объединяются через NMS.

    Краевые патчи дополняются нулями до tile_size×tile_size, чтобы
    YOLO не применял внутренний letterboxing — иначе координаты
    возвращались бы в пространстве дополненного изображения и
    боксы оказывались бы шире реальных объектов.
    """
    import cv2

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    model = load_model(model_path)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    H, W = img.shape[:2]
    tile_size = imgsz

    # Если изображение меньше тайла — обычная детекция
    if H <= tile_size and W <= tile_size:
        return detect_image(model_path, image_path, conf=conf, iou=iou, imgsz=imgsz)

    step = max(1, int(tile_size * (1 - tile_overlap)))

    def _tile_starts(length: int) -> list:
        starts = list(range(0, length - tile_size, step))
        starts.append(max(0, length - tile_size))
        return sorted(set(starts))

    all_raw: list = []
    t0 = time.perf_counter()

    # Буфер tile_size×tile_size: используется повторно, сбрасывается перед каждым тайлом
    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

    for y0 in _tile_starts(H):
        for x0 in _tile_starts(W):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            ph = y1 - y0  # реальные размеры патча (≤ tile_size)
            pw = x1 - x0

            # Сброс + копирование патча в верхний левый угол буфера
            padded[:] = 0
            padded[:ph, :pw] = img[y0:y1, x0:x1]

            results = model.predict(
                source=padded,
                conf=conf,
                iou=iou,
                imgsz=tile_size,   # патч уже tile_size×tile_size — resize не нужен
                verbose=False,
                save=False,
            )
            for r in results:
                for box in r.boxes:
                    px1, py1, px2, py2 = [int(v) for v in box.xyxy[0].tolist()]
                    # Центр бокса не должен попадать в область дополнения (чёрные пиксели)
                    if (px1 + px2) / 2 >= pw or (py1 + py2) / 2 >= ph:
                        continue
                    # Обрезаем до реального размера патча
                    px1 = min(px1, pw - 1)
                    py1 = min(py1, ph - 1)
                    px2 = min(px2, pw)
                    py2 = min(py2, ph)
                    cls_id = int(box.cls[0])
                    all_raw.append({
                        "bbox":       [x0 + px1, y0 + py1, x0 + px2, y0 + py2],
                        "conf":       round(float(box.conf[0]), 3),
                        "class_id":   cls_id,
                        "class_name": model.names.get(cls_id, str(cls_id)),
                    })

    inference_ms = round((time.perf_counter() - t0) * 1000, 1)
    detections = _nms_detections(all_raw, iou_threshold=0.30)

    # Рисуем боксы на оригинальном изображении
    annotated = img.copy()
    for d in detections:
        bx1, by1, bx2, by2 = d["bbox"]
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        label = f"{d['class_name']} {d['conf']:.2f}"
        cv2.putText(annotated, label, (bx1, max(by1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return {
        "detections":     detections,
        "inference_ms":   inference_ms,
        "annotated_image": annotated,
    }


def format_detections(detections: list, inference_ms: float) -> str:
    """Форматирует список детекций в текст для лога."""
    lines = [f"Время инференса: {inference_ms} мс", f"Найдено объектов: {len(detections)}"]
    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = d["bbox"]
        lines.append(
            f"  [{i}] {d['class_name']}  conf={d['conf']:.3f}  "
            f"bbox=[{x1},{y1},{x2},{y2}]"
        )
    if not detections:
        lines.append("  (ничего не найдено)")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  СТАТИСТИКА ДЕТЕКЦИЙ
# ═════════════════════════════════════════════════════════════════════════════

def save_detection_stats(
    model_path: str,
    image_path: str,
    frame_w: int,
    frame_h: int,
    detections: list,
) -> None:
    """
    Добавляет запись о результатах детекции в накопительный JSON-файл.

    Каждая запись:
        {
          "ts":           "<ISO-timestamp>",
          "model":        "<имя .pt файла>",
          "image":        "<имя изображения>",
          "frame_w":      <int>,
          "frame_h":      <int>,
          "n_detections": <int>,
          "confs":        [<float>, ...]
        }
    """
    stats_dir = os.path.dirname(_STATS_FILE)
    os.makedirs(stats_dir, exist_ok=True)

    existing: list = []
    if os.path.isfile(_STATS_FILE):
        try:
            with open(_STATS_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

    entry = {
        "ts":           datetime.datetime.now().isoformat(timespec="seconds"),
        "model":        os.path.basename(model_path),
        "image":        os.path.basename(image_path),
        "frame_w":      int(frame_w),
        "frame_h":      int(frame_h),
        "n_detections": len(detections),
        "confs":        [d["conf"] for d in detections],
    }
    existing.append(entry)

    try:
        with open(_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        _log.warning(f"Не удалось сохранить статистику детекции: {exc}")


def save_detection_stats_batch(entries: list) -> None:
    """
    Добавляет сразу несколько записей статистики в накопительный JSON-файл.

    Каждый элемент entries — dict с полями:
        ts, model, image, frame_w, frame_h, n_detections, confs
    """
    if not entries:
        return

    stats_dir = os.path.dirname(_STATS_FILE)
    os.makedirs(stats_dir, exist_ok=True)

    existing: list = []
    if os.path.isfile(_STATS_FILE):
        try:
            with open(_STATS_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

    existing.extend(entries)

    try:
        with open(_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        _log.warning(f"Не удалось сохранить пакетную статистику детекции: {exc}")
