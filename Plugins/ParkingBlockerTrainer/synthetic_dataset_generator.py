"""
Генератор синтетического датасета для детектора парковочных блокираторов.

Принцип:
  - Берёт кропы блокираторов из LearnDataset (каждый кроп = изображение блокиратора)
  - Вставляет их (уменьшенными до реалистичного масштаба) на фоновые кадры
  - Фон: кадры из видео Cam1.avi  →  сами кропы как фон  →  синтетические текстуры
  - Записывает аннотации в YOLO-формате с реальными координатами bbox

Выходная структура:
    dest_dir/
      images/train/  images/val/
      labels/train/  labels/val/
      data.yaml
"""

import os
import sys
import random
import shutil
import logging
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import yaml

_log = logging.getLogger("SyntheticDatasetGenerator")

# ─── константы ───────────────────────────────────────────────────────────────
CANVAS_W, CANVAS_H = 1280, 720
BLOCKER_SCALE_MIN  = 0.03   # ширина блокиратора как доля CANVAS_W
BLOCKER_SCALE_MAX  = 0.22
MIN_BLOCKERS       = 1
MAX_BLOCKERS       = 3
RANDOM_SEED        = 42
VAL_SPLIT          = 0.15
CLASS_ID           = 0
CLASS_NAME         = "parking_blocker"

# ─── вспомогательные функции ─────────────────────────────────────────────────

def _list_images(folder: str) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f.lower())[1] in exts
    ]


def _extract_video_backgrounds(
    video_path: str,
    n_frames: int = 200,
    out_dir: str | None = None,
    log_cb: Callable | None = None,
) -> list:
    """
    Извлекает n_frames равномерно распределённых кадров из видео.
    Если out_dir задан — сохраняет JPG, иначе возвращает список np.ndarray.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if log_cb:
            log_cb(f"[WARN] Не удалось открыть видео: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sorted(random.sample(range(max(1, total)), min(n_frames, total)))
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        if out_dir:
            path = os.path.join(out_dir, f"bg_video_{idx:06d}.jpg")
            cv2.imwrite(path, frame)
            frames.append(path)
        else:
            frames.append(frame)

    cap.release()
    if log_cb:
        log_cb(f"  Извлечено {len(frames)} кадров из {os.path.basename(video_path)}")
    return frames


def _make_asphalt_bg(h: int = CANVAS_H, w: int = CANVAS_W) -> np.ndarray:
    """Создаёт синтетическую текстуру асфальта."""
    # Базовый серый с шумом
    base = random.randint(50, 100)
    bg = np.full((h, w, 3), base, dtype=np.uint8)
    noise = np.random.randint(-25, 25, (h, w, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Небольшие случайные пятна (следы масла, разметка и т.д.)
    for _ in range(random.randint(3, 12)):
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        rr = random.randint(20, 120)
        color = [random.randint(30, 180)] * 3
        cv2.circle(bg, (cx, cy), rr, color, -1)
        # Размываем каждое пятно отдельно для мягкости
    bg = cv2.GaussianBlur(bg, (31, 31), 8)
    return bg


def _load_background(path_or_array, log_cb=None) -> Optional[np.ndarray]:
    """Загружает фон и масштабирует до CANVAS_W×CANVAS_H."""
    if isinstance(path_or_array, np.ndarray):
        img = path_or_array
    else:
        img = cv2.imread(path_or_array)
        if img is None:
            return None
    return cv2.resize(img, (CANVAS_W, CANVAS_H))


def _paste_blocker(
    canvas: np.ndarray,
    crop: np.ndarray,
    scale: float,
) -> Optional[tuple]:
    """
    Вставляет уменьшенный кроп на canvas в случайной позиции.
    Возвращает (x1, y1, x2, y2) в пикселях или None если не поместился.
    """
    H, W = canvas.shape[:2]
    ch, cw = crop.shape[:2]

    # Целевая ширина
    target_w = max(20, int(CANVAS_W * scale))
    target_h = max(10, int(ch * target_w / max(cw, 1)))

    if target_w >= W or target_h >= H:
        return None

    resized = cv2.resize(crop, (target_w, target_h))

    # Случайная позиция (целиком в пределах canvas)
    margin = 5
    x1 = random.randint(margin, W - target_w - margin)
    y1 = random.randint(margin, H - target_h - margin)
    x2 = x1 + target_w
    y2 = y1 + target_h

    # Лёгкое смешение краёв через маску
    mask = np.ones((target_h, target_w), dtype=np.float32)
    fade = max(2, min(target_h, target_w) // 8)
    for i in range(fade):
        alpha = i / fade
        mask[i, :]  = np.minimum(mask[i, :],  alpha)
        mask[-i-1,:] = np.minimum(mask[-i-1,:], alpha)
        mask[:, i]  = np.minimum(mask[:, i],  alpha)
        mask[:,-i-1] = np.minimum(mask[:,-i-1], alpha)
    mask = mask[:, :, np.newaxis]

    roi = canvas[y1:y2, x1:x2].astype(np.float32)
    blended = roi * (1 - mask) + resized.astype(np.float32) * mask
    canvas[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return (x1, y1, x2, y2)


def _augment(canvas: np.ndarray) -> np.ndarray:
    """Случайные цветовые и лёгкие геометрические аугментации."""
    # Яркость/контраст
    alpha = random.uniform(0.75, 1.25)
    beta  = random.randint(-30, 30)
    canvas = cv2.convertScaleAbs(canvas, alpha=alpha, beta=beta)

    # Горизонтальный флип
    if random.random() < 0.5:
        canvas = cv2.flip(canvas, 1)

    # Лёгкое размытие (имитация расфокуса камеры)
    if random.random() < 0.2:
        k = random.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)

    return canvas


def _augment_crop(img: np.ndarray) -> np.ndarray:
    """
    Полный набор аугментаций для одного кропа блокиратора.
    Применяет случайную комбинацию преобразований при каждом вызове.
    """
    out = img.copy()

    # 1. Яркость / контраст
    alpha = random.uniform(0.6, 1.4)
    beta  = random.randint(-40, 40)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # 2. HSV — оттенок и насыщенность (ржавчина бывает разных оттенков)
    if random.random() < 0.7:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.randint(-15, 15), 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-40, 40), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3. Горизонтальный флип
    if random.random() < 0.5:
        out = cv2.flip(out, 1)

    # 4. Вертикальный флип (редко — блокиратор иногда перевёрнут)
    if random.random() < 0.15:
        out = cv2.flip(out, 0)

    # 5. Поворот ±20°
    if random.random() < 0.6:
        angle = random.uniform(-20, 20)
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # 6. Масштаб (zoom in/out с сохранением размера)
    if random.random() < 0.5:
        scale = random.uniform(0.75, 1.25)
        h, w = out.shape[:2]
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(out, (nw, nh))
        # Обрезаем или добавляем поля до исходного размера
        if scale > 1:
            y0 = (nh - h) // 2
            x0 = (nw - w) // 2
            out = resized[y0:y0 + h, x0:x0 + w]
        else:
            pad_h, pad_w = h - nh, w - nw
            out = cv2.copyMakeBorder(
                resized,
                pad_h // 2, pad_h - pad_h // 2,
                pad_w // 2, pad_w - pad_w // 2,
                cv2.BORDER_REFLECT_101,
            )

    # 7. Гауссовский шум
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(3, 12), out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 8. Размытие (расфокус / дождь)
    if random.random() < 0.3:
        k = random.choice([3, 5, 7])
        out = cv2.GaussianBlur(out, (k, k), 0)

    # 9. Случайные прямоугольные «заслонки» (имитация частичного перекрытия)
    if random.random() < 0.25:
        h, w = out.shape[:2]
        bh = random.randint(h // 8, h // 3)
        bw = random.randint(w // 8, w // 3)
        bx = random.randint(0, w - bw)
        by = random.randint(0, h - bh)
        color = [random.randint(20, 120)] * 3
        out[by:by + bh, bx:bx + bw] = color

    # 10. JPEG-компрессия (имитация артефактов видеокамеры)
    if random.random() < 0.3:
        q = random.randint(40, 85)
        _, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return out


# ═════════════════════════════════════════════════════════════════════════════
#  АУГМЕНТАЦИЯ КРОПОВ
# ═════════════════════════════════════════════════════════════════════════════

def augment_crops(
    crops_dir: str,
    target_count: int = 2000,
    log_callback: Callable | None = None,
) -> int:
    """
    Аугментирует кропы блокираторов из crops_dir до target_count экземпляров.
    Новые файлы сохраняются в ту же папку с суффиксом _aug_XXXXX.jpg.
    Уже существующие аугментированные файлы удаляются перед началом.

    Параметры
    ----------
    crops_dir    : папка с исходными кропами (LearnDataset)
    target_count : желаемое итоговое число файлов в папке
    log_callback : callable(str)

    Возвращает итоговое число файлов в папке.
    """
    def _log(msg: str):
        _log_obj.info(msg)
        if log_callback:
            log_callback(msg)

    _log_obj = logging.getLogger("SyntheticDatasetGenerator.augment_crops")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Разделяем оригинальные и уже аугментированные файлы
    all_files = [
        f for f in os.listdir(crops_dir)
        if os.path.splitext(f.lower())[1] in exts
    ]
    originals = [f for f in all_files if "_aug_" not in f]
    aug_old   = [f for f in all_files if "_aug_" in f]

    _log(f"Оригинальных кропов: {len(originals)}")
    _log(f"Старых аугментированных: {len(aug_old)} — удаляем...")
    for f in aug_old:
        try:
            os.remove(os.path.join(crops_dir, f))
        except OSError:
            pass

    if not originals:
        raise FileNotFoundError(f"Нет исходных кропов в {crops_dir}")

    need = max(0, target_count - len(originals))
    if need == 0:
        _log(f"Уже достаточно кропов ({len(originals)} >= {target_count}), аугментация не нужна.")
        return len(originals)

    _log(f"Генерируем {need} аугментированных кропов (цель: {target_count})...")

    # Загружаем оригиналы
    loaded = []
    for f in originals:
        img = cv2.imread(os.path.join(crops_dir, f))
        if img is not None:
            loaded.append(img)

    if not loaded:
        raise RuntimeError("Не удалось загрузить ни одного оригинального кропа")

    random.seed(None)   # случайный seed для каждого запуска
    np.random.seed(None)

    written = 0
    for i in range(need):
        if i % 200 == 0 and i > 0:
            _log(f"  {i}/{need} ...")
        src = random.choice(loaded)
        aug = _augment_crop(src)
        fname = f"_aug_{i:05d}.jpg"
        cv2.imwrite(
            os.path.join(crops_dir, fname),
            aug,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        written += 1

    total = len(originals) + written
    _log(f"Готово: {len(originals)} оригинал + {written} аугментировано = {total} файлов")
    return total


# ═════════════════════════════════════════════════════════════════════════════
#  ОСНОВНАЯ ФУНКЦИЯ
# ═════════════════════════════════════════════════════════════════════════════

def generate(
    crops_dir: str,
    dest_dir: str,
    data_yaml_path: str,
    video_path: str | None = None,
    n_train: int = 2000,
    n_val: int = 300,
    overwrite: bool = False,
    log_callback: Callable | None = None,
) -> str:
    """
    Генерирует синтетический YOLO-датасет.

    Параметры
    ----------
    crops_dir      : папка с кропами блокираторов (LearnDataset)
    dest_dir       : куда сохранять датасет (LearnDataset_yolo)
    data_yaml_path : путь к data.yaml
    video_path     : опционально — путь к видео для извлечения фонов
    n_train/n_val  : количество синтетических сцен
    overwrite      : пересоздать датасет если уже существует
    log_callback   : callable(str) для вывода прогресса

    Возвращает путь к data.yaml.
    """
    def _log_msg(msg: str):
        _log.info(msg)
        if log_callback:
            log_callback(msg)

    if os.path.exists(dest_dir) and not overwrite:
        if os.path.exists(data_yaml_path):
            _log_msg(f"Датасет уже существует: {dest_dir}  (overwrite=False)")
            return data_yaml_path

    # 1. Загружаем кропы блокираторов
    crop_paths = _list_images(crops_dir)
    if not crop_paths:
        raise FileNotFoundError(f"Не найдено кропов в {crops_dir}")
    _log_msg(f"Кропов блокираторов: {len(crop_paths)}")

    crops_loaded = []
    for p in crop_paths:
        img = cv2.imread(p)
        if img is not None:
            crops_loaded.append(img)
    if not crops_loaded:
        raise RuntimeError("Не удалось загрузить ни одного кропа")
    _log_msg(f"Загружено кропов: {len(crops_loaded)}")

    # 2. Собираем фоны
    backgrounds: list = []

    # 2а. Из видео
    if video_path and os.path.isfile(video_path):
        _log_msg(f"Извлекаем кадры из видео: {os.path.basename(video_path)} ...")
        frames = _extract_video_backgrounds(video_path, n_frames=250, log_cb=_log_msg)
        backgrounds.extend(frames)

    # 2б. Сами кропы как фон (они уже содержат сцены парковки)
    bg_from_crops = [cv2.resize(c, (CANVAS_W, CANVAS_H)) for c in crops_loaded]
    backgrounds.extend(bg_from_crops)
    _log_msg(f"Фонов из кропов: {len(bg_from_crops)}")

    _log_msg(f"Итого фонов: {len(backgrounds)}")

    # 3. Создаём папки
    for split in ("train", "val"):
        os.makedirs(os.path.join(dest_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "labels", split), exist_ok=True)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 4. Генерируем сцены
    def _generate_split(n: int, split: str):
        for i in range(n):
            if i % 200 == 0:
                _log_msg(f"  [{split}] {i}/{n} ...")

            # Выбираем фон
            if backgrounds and random.random() < 0.85:
                bg_src = random.choice(backgrounds)
                canvas = _load_background(bg_src)
                if canvas is None:
                    canvas = _make_asphalt_bg()
            else:
                canvas = _make_asphalt_bg()

            # Горизонтальный флип фона (расширяет разнообразие)
            if random.random() < 0.5:
                canvas = cv2.flip(canvas, 1)

            annotations = []
            n_obj = random.randint(MIN_BLOCKERS, MAX_BLOCKERS)

            for _ in range(n_obj):
                crop = random.choice(crops_loaded)
                scale = random.uniform(BLOCKER_SCALE_MIN, BLOCKER_SCALE_MAX)
                bbox = _paste_blocker(canvas, crop, scale)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                H, W = canvas.shape[:2]
                cx = (x1 + x2) / 2 / W
                cy = (y1 + y2) / 2 / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                annotations.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # Если не поместился ни один блокиратор — пропускаем сцену
            if not annotations:
                continue

            canvas = _augment(canvas)

            fname = f"syn_{split}_{i:06d}.jpg"
            cv2.imwrite(
                os.path.join(dest_dir, "images", split, fname),
                canvas,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            with open(
                os.path.join(dest_dir, "labels", split,
                             os.path.splitext(fname)[0] + ".txt"),
                "w", encoding="utf-8",
            ) as f:
                f.write("\n".join(annotations) + "\n")

    _generate_split(n_train, "train")
    _log_msg(f"Train: {n_train} сцен сгенерировано")
    _generate_split(n_val, "val")
    _log_msg(f"Val: {n_val} сцен сгенерировано")

    # 5. Записываем data.yaml
    data_cfg = {
        "path":  dest_dir,
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": {0: CLASS_NAME},
    }
    os.makedirs(os.path.dirname(data_yaml_path), exist_ok=True)
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_cfg, f, allow_unicode=True, default_flow_style=False)

    _log_msg(f"data.yaml сохранён: {data_yaml_path}")
    _log_msg("Синтетический датасет готов!")
    return data_yaml_path
