"""
Плагин анализа транспортных средств для Watch Golem.

Функционал:
  - Определение цвета транспорта (HSV-анализ)
  - Определение модели транспорта (CLIP zero-shot классификация)
  - Чтение государственного номера (EasyOCR)

Зависимости:
  pip install easyocr open-clip-torch
"""

import os
# Предотвращаем краш от дублирования OpenMP (libiomp5md.dll)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import re
import importlib.util
from datetime import datetime

# ──────────────────────────────────────────────
# Опциональные зависимости с мягким fallback
# ──────────────────────────────────────────────

_easyocr_available = False
_clip_available = importlib.util.find_spec("open_clip") is not None
open_clip = None
torch = None
PILImage = None

try:
    import easyocr
    _easyocr_available = True
except ImportError:
    pass

# open_clip/torch/PIL подгружаем лениво внутри _load_clip(),
# чтобы исключить нативные падения при импорте модуля.


# ═══════════════════════════════════════════════
#  0. ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЯ
# ═══════════════════════════════════════════════

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Общая предобработка изображения перед анализом:
    CLAHE + лёгкое шумоподавление + усиление резкости.
    """
    if image is None or image.size == 0:
        return image

    # Шумоподавление (мягкий bilateral — сохраняет границы)
    denoised = cv2.bilateralFilter(image, 7, 50, 50)

    # CLAHE — адаптивное выравнивание контраста в LAB (только по яркости)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    enhanced = cv2.merge([l_ch, a_ch, b_ch])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Unsharp mask — усиление резкости
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    return sharpened


def preprocess_plate_region(plate_bgr: np.ndarray) -> list:
    """
    Расширенная предобработка кропа номерной пластины.
    Возвращает список бинаризованных вариантов для OCR.
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return []

    # Увеличение в 3× (бикубическая интерполяция)
    scaled = cv2.resize(plate_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

    # CLAHE для выравнивания контраста
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Шумоподавление
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    variants = []

    # Вариант 1: адаптивный порог (Gaussian)
    th1 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )
    variants.append(th1)

    # Вариант 2: адаптивный порог (Mean)
    th2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 15, 4
    )
    variants.append(th2)

    # Вариант 3: Otsu
    _, th3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(th3)

    # Вариант 4: инвертированный Otsu (белый текст на тёмном фоне)
    variants.append(cv2.bitwise_not(th3))

    # Морфология для очистки каждого варианта
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = []
    for v in variants:
        v = cv2.morphologyEx(v, cv2.MORPH_OPEN, kernel, iterations=1)
        v = cv2.morphologyEx(v, cv2.MORPH_CLOSE, kernel, iterations=1)
        cleaned.append(v)

    return cleaned


# ═══════════════════════════════════════════════
#  1. ОПРЕДЕЛЕНИЕ ЦВЕТА ТРАНСПОРТА (K-Means LAB + CLIP)
# ═══════════════════════════════════════════════

# Эталонные цвета в LAB для Delta E (CIE76) сопоставления
_REFERENCE_COLORS_LAB = [
    # (L, A, B, название)
    (100.0,  0.0,    0.0,   "Белый"),
    (5.0,    0.0,    0.0,   "Чёрный"),
    (22.0,   0.0,   -2.0,   "Антрацитовый"),
    (32.0,   0.0,   -3.0,   "Графитовый"),
    (42.0,   0.0,   -1.0,   "Тёмно-серый"),
    (53.6,   0.0,    0.0,   "Серый"),
    (65.0,   0.0,    0.0,   "Светло-серый"),
    (75.0,   0.0,    2.0,   "Серебристый"),
    (88.0,   0.0,    3.0,   "Жемчужный"),
    (53.2,   80.1,   67.2,  "Красный"),
    (24.8,   60.1,   38.2,  "Бордовый"),
    (87.7,  -86.2,   83.2,  "Зелёный"),
    (60.3,  -31.1,   35.0,  "Тёмно-зелёный"),
    (32.3,   79.2,  -107.9, "Синий"),
    (61.2,  -3.4,   -43.1,  "Голубой"),
    (29.0,   29.0,  -51.0,  "Тёмно-синий"),
    (97.1,  -21.6,   94.5,  "Жёлтый"),
    (66.6,   48.8,   73.0,  "Оранжевый"),
    (60.3,   98.2,  -60.8,  "Фиолетовый"),
    (44.0,   36.0,  -14.0,  "Коричневый"),
    (51.6,   74.9,  -26.5,  "Розовый"),
    (49.2,   4.7,   -28.1,  "Бежевый"),
    (35.0,   15.0,   25.0,  "Тёмно-коричневый"),
    (55.0,   25.0,   35.0,  "Бронзовый"),
    (38.0,  -10.0,  -15.0,  "Тёмно-синий (стальной)"),
]

# Цвета для CLIP zero-shot (английские промпты)
_COLOR_CLIP_LABELS = [
    ("white",           "Белый"),
    ("pearl white",     "Жемчужный"),
    ("black",           "Чёрный"),
    ("anthracite",      "Антрацитовый"),
    ("graphite",        "Графитовый"),
    ("dark gray",       "Тёмно-серый"),
    ("gray",            "Серый"),
    ("light gray",      "Светло-серый"),
    ("silver",          "Серебристый"),
    ("red",             "Красный"),
    ("dark red",        "Бордовый"),
    ("green",           "Зелёный"),
    ("dark green",      "Тёмно-зелёный"),
    ("blue",            "Синий"),
    ("light blue",      "Голубой"),
    ("dark blue",       "Тёмно-синий"),
    ("navy blue",       "Тёмно-синий (стальной)"),
    ("yellow",          "Жёлтый"),
    ("orange",          "Оранжевый"),
    ("purple",          "Фиолетовый"),
    ("brown",           "Коричневый"),
    ("dark brown",      "Тёмно-коричневый"),
    ("bronze",          "Бронзовый"),
    ("pink",            "Розовый"),
    ("beige",           "Бежевый"),
]


def _delta_e(lab1, lab2):
    """Вычисляет CIEDE2000 — перцептуально точная метрика цветового различия."""
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2.0
    C_avg7 = C_avg**7
    G = 0.5 * (1.0 - np.sqrt(C_avg7 / (C_avg7 + 25.0**7)))
    a1p, a2p = a1 * (1.0 + G), a2 * (1.0 + G)
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    h1p = float(np.degrees(np.arctan2(b1, a1p)) % 360.0)
    h2p = float(np.degrees(np.arctan2(b2, a2p)) % 360.0)
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0.0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180.0:
        dhp = h2p - h1p
    elif h2p - h1p > 180.0:
        dhp = h2p - h1p - 360.0
    else:
        dhp = h2p - h1p + 360.0
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))
    Lp_avg = (L1 + L2) / 2.0
    Cp_avg = (C1p + C2p) / 2.0
    if C1p * C2p == 0.0:
        hp_avg = h1p + h2p
    elif abs(h1p - h2p) <= 180.0:
        hp_avg = (h1p + h2p) / 2.0
    elif h1p + h2p < 360.0:
        hp_avg = (h1p + h2p + 360.0) / 2.0
    else:
        hp_avg = (h1p + h2p - 360.0) / 2.0
    T = (1.0
         - 0.17 * np.cos(np.radians(hp_avg - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hp_avg))
         + 0.32 * np.cos(np.radians(3.0 * hp_avg + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hp_avg - 63.0)))
    SL = 1.0 + 0.015 * (Lp_avg - 50.0)**2 / np.sqrt(20.0 + (Lp_avg - 50.0)**2)
    SC = 1.0 + 0.045 * Cp_avg
    SH = 1.0 + 0.015 * Cp_avg * T
    Cp_avg7 = Cp_avg**7
    RC = 2.0 * np.sqrt(Cp_avg7 / (Cp_avg7 + 25.0**7))
    d_theta = 30.0 * np.exp(-((hp_avg - 275.0) / 25.0)**2)
    RT = -np.sin(np.radians(2.0 * d_theta)) * RC
    return float(np.sqrt(
        (dLp / SL)**2 + (dCp / SC)**2 + (dHp / SH)**2
        + RT * (dCp / SC) * (dHp / SH)
    ))


def _kmeans_dominant_colors(image_bgr: np.ndarray, k: int = 7):
    """
    Находит k доминирующих цветов через K-Means в LAB
    с предварительной фильтрацией фона (тени, блики, асфальт).

    Returns:
        Список (lab_color, доля_пикселей), отсортированный по доле.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    flat = lab.reshape(-1, 3)

    # Переводим в стандартный LAB для фильтрации
    L_norm = flat[:, 0] * 100.0 / 255.0
    A_norm = flat[:, 1] - 128.0
    B_norm = flat[:, 2] - 128.0
    saturation = np.sqrt(A_norm**2 + B_norm**2)

    # Убираем: глубокие тени (L<4), сильные блики/стёкла (L>93), нейтральный фон
    # Для тёмных металлических цветов (L≈20-45) снижаем порог до 4
    mask = (
        (L_norm > 4.0) & (L_norm < 93.0)
        & ~(
            # Небо: высокая яркость + очень низкая насыщенность
            (L_norm > 70.0) & (saturation < 5.0)
        )
        & ~(
            # Асфальт/дорога: средняя яркость + нейтральный серый
            (L_norm > 35.0) & (L_norm < 60.0) & (saturation < 4.0)
        )
    )
    pixels = flat[mask]

    # Если после фильтрации осталось мало пикселей — берём всё
    if len(pixels) < k * 10:
        pixels = flat

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )

    total = len(labels)
    cluster_sizes = []
    for i in range(k):
        count = int(np.sum(labels == i))
        L = centers[i][0] * 100.0 / 255.0
        A = centers[i][1] - 128.0
        B = centers[i][2] - 128.0
        cluster_sizes.append(((L, A, B), count / total))

    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    return cluster_sizes


def _lab_to_color_name(lab):
    """Сопоставляет LAB-цвет с ближайшим названием по Delta E."""
    best_name = "Не определён"
    best_dist = float("inf")
    for L, A, B, name in _REFERENCE_COLORS_LAB:
        dist = _delta_e(lab, (L, A, B))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name, best_dist


def _detect_color_clip(image: np.ndarray) -> tuple:
    """
    Определяет цвет транспорта через CLIP zero-shot.

    Returns:
        (название_цвета_рус, уверенность) или (None, 0) если CLIP недоступен.
    """
    if not _load_clip():
        return None, 0.0

    try:
        device = _get_clip_device()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)
        img_tensor = _clip_preprocess(pil_image).unsqueeze(0).to(device)

        # Ensemble: 4 промпта на цвет — усредняем для устойчивости
        templates = [
            "a car with {} paint",
            "a {} colored car body",
            "a {} vehicle exterior",
            "a {} metallic car",
        ]
        all_scores = np.zeros(len(_COLOR_CLIP_LABELS))
        for tmpl in templates:
            prompts = [tmpl.format(en) for en, _ in _COLOR_CLIP_LABELS]
            text_tokens = _clip_tokenizer(prompts).to(device)
            with torch.no_grad():
                img_features = _clip_model.encode_image(img_tensor)
                txt_features = _clip_model.encode_text(text_tokens)
                img_features /= img_features.norm(dim=-1, keepdim=True)
                txt_features /= txt_features.norm(dim=-1, keepdim=True)
                sim = (img_features @ txt_features.T).squeeze(0)
                all_scores += sim.cpu().numpy()
        scores = all_scores / len(templates)

        best_idx = int(scores.argmax())
        # Используем softmax для более точной оценки уверенности
        exp_scores = np.exp((scores - scores.max()) * 10)
        softmax_probs = exp_scores / exp_scores.sum()
        confidence = float(np.clip(softmax_probs[best_idx] * 2.5, 0.0, 1.0))
        return _COLOR_CLIP_LABELS[best_idx][1], confidence

    except Exception:
        return None, 0.0


def detect_vehicle_color(image: np.ndarray) -> dict:
    """
    Определяет доминирующий цвет транспортного средства.
    Метод: K-Means кластеризация в LAB + Delta E + CLIP-валидация.

    Args:
        image: BGR-изображение (кроп транспорта).

    Returns:
        dict с ключами:
            color      — название цвета (str)
            confidence — уверенность (0-1)
            palette    — список (название, доля) — все обнаруженные цвета
            method     — использованный метод
    """
    if image is None or image.size == 0:
        return {"color": "Не определён", "confidence": 0.0, "palette": [], "method": "none"}

    # Убираем крайние 15 % по краям, чтобы исключить фон
    h, w = image.shape[:2]
    margin_y, margin_x = max(1, h * 15 // 100), max(1, w * 15 // 100)
    cropped = image[margin_y:h - margin_y, margin_x:w - margin_x]

    if cropped.size == 0:
        return {"color": "Не определён", "confidence": 0.0, "palette": [], "method": "none"}

    # ── K-Means LAB ──
    clusters = _kmeans_dominant_colors(cropped)

    # lab[0] из _kmeans_dominant_colors уже в шкале 0-100
    # (конвертация centers[i][0] * 100 / 255 выполнена внутри функции)
    # Взвешенный L для определения "тёмное/светлое авто"
    weighted_L = sum(lab[0] * share for lab, share in clusters)
    # Наличие тёмного кластера (кузов ≈ L<50 с > 10% пикселей)
    has_dark_cluster = any(
        lab[0] < 50.0 and share > 0.10
        for lab, share in clusters
    )
    is_dark_car = weighted_L < 62.0 or has_dark_cluster

    palette = []
    for lab, share in clusters:
        name, delta = _lab_to_color_name(lab)
        cluster_L = lab[0]   # уже в шкале 0-100
        conf = max(0.0, 1.0 - delta / 25.0)

        # Для тёмного авто: поднимаем вес тёмных кластеров (настоящий кузов),
        # опускаем яркие (блики, отражение неба, хром)
        if is_dark_car:
            if cluster_L < 50.0:
                share *= 2.0
            elif cluster_L < 60.0:
                share *= 1.2
            elif cluster_L > 68.0:
                share *= 0.25   # активно давим отражения

        palette.append((name, round(share * conf, 3)))

    # Нормализуем доли
    total_weight = sum(p[1] for p in palette)
    if total_weight > 0:
        palette = [(n, round(s / total_weight, 3)) for n, s in palette]
    palette.sort(key=lambda x: x[1], reverse=True)

    kmeans_color = palette[0][0] if palette else "Не определён"
    kmeans_conf = palette[0][1] if palette else 0.0

    # ── CLIP-валидация (если доступен) ──
    clip_color, clip_conf = _detect_color_clip(cropped)

    if clip_color and clip_conf > 0.15:
        if clip_color == kmeans_color:
            # Совпадение — повышаем уверенность
            final_color = kmeans_color
            final_conf = min(1.0, (kmeans_conf + clip_conf) / 2 + 0.15)
            method = "KMeans-LAB + CLIP (совпадение)"
        else:
            # Расхождение — берём тот, у кого выше уверенность
            if clip_conf > kmeans_conf:
                final_color = clip_color
                final_conf = clip_conf
                method = "CLIP (приоритет)"
            else:
                final_color = kmeans_color
                final_conf = kmeans_conf
                method = "KMeans-LAB (приоритет)"
    else:
        final_color = kmeans_color
        final_conf = kmeans_conf
        method = "KMeans-LAB"

    return {
        "color": final_color,
        "confidence": round(final_conf, 3),
        "palette": [(n, round(r, 3)) for n, r in palette if r > 0.01],
        "method": method,
    }


# ═══════════════════════════════════════════════
#  2. ОПРЕДЕЛЕНИЕ МОДЕЛИ ТРАНСПОРТА (CLIP)
# ═══════════════════════════════════════════════

# Популярные модели транспорта для zero-shot классификации
_VEHICLE_LABELS = [
    # Легковые
    "Toyota Camry", "Toyota Corolla", "Toyota RAV4", "Toyota Land Cruiser",
    "Hyundai Solaris", "Hyundai Tucson", "Hyundai Creta",
    "Kia Rio", "Kia Sportage", "Kia Ceed",
    "Volkswagen Polo", "Volkswagen Tiguan", "Volkswagen Golf",
    "Skoda Octavia", "Skoda Rapid", "Skoda Kodiaq",
    "BMW 3 Series", "BMW 5 Series", "BMW X5",
    "Mercedes-Benz C-Class", "Mercedes-Benz E-Class", "Mercedes-Benz GLE",
    "Audi A4", "Audi A6", "Audi Q5",
    "Lada Vesta", "Lada Granta", "Lada Niva",
    "Renault Duster", "Renault Logan",
    "Nissan Qashqai", "Nissan X-Trail",
    "Ford Focus", "Ford Kuga",
    "Chevrolet Cruze", "Chevrolet Niva",
    "Mazda 3", "Mazda CX-5",
    "Honda Civic", "Honda CR-V",
    "Mitsubishi Outlander", "Mitsubishi Lancer",
    "Subaru Forester", "Subaru Outback",
    "Lexus RX", "Lexus NX",
    "Porsche Cayenne", "Porsche 911",
    # Коммерческий / грузовой
    "GAZelle van", "Truck", "Bus", "Minivan",
    "Pickup truck", "SUV (unknown model)", "Sedan (unknown model)",
    "Hatchback (unknown model)", "Coupe (unknown model)",
]

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def _is_truthy_env(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _clip_disabled() -> bool:
    """Отключает CLIP через переменную окружения WG_DISABLE_CLIP=1."""
    return _is_truthy_env(os.getenv("WG_DISABLE_CLIP", "0"))


def _get_clip_device() -> str:
    """Выбирает устройство для CLIP с приоритетом стабильности."""
    forced = os.getenv("WG_CLIP_DEVICE", "").strip().lower()
    if forced in {"cpu", "cuda"}:
        return forced

    # На Windows CLIP+CUDA часто нестабилен (0xC0000005),
    # поэтому по умолчанию запускаем на CPU.
    if os.name == "nt":
        return "cpu"

    return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"


def _load_clip():
    """Ленивая загрузка CLIP-модели (один раз)."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    global open_clip, torch, PILImage
    if _clip_model is not None:
        return True
    if not _clip_available:
        return False
    if _clip_disabled():
        return False
    try:
        if open_clip is None or torch is None or PILImage is None:
            import open_clip as _open_clip
            import torch as _torch
            from PIL import Image as _PILImage
            open_clip = _open_clip
            torch = _torch
            PILImage = _PILImage

        device = _get_clip_device()
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k"
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        _clip_model = model
        _clip_preprocess = preprocess
        _clip_tokenizer = tokenizer
        return True
    except Exception as e:
        print(f"[VehicleAnalyzer] Ошибка загрузки CLIP: {e}")
        return False


def detect_vehicle_model(image: np.ndarray, top_k: int = 3) -> dict:
    """
    Определяет модель транспорта с помощью CLIP zero-shot.

    Args:
        image: BGR-изображение (кроп транспорта).
        top_k: количество лучших совпадений.

    Returns:
        dict с ключами:
            model       — наиболее вероятная модель (str)
            confidence  — уверенность (0-1)
            top_matches — список (модель, уверенность)
            available   — True, если CLIP доступен
    """
    if not _load_clip():
        return {
            "model": "Недоступно (установите open-clip-torch)",
            "confidence": 0.0,
            "top_matches": [],
            "available": False,
        }

    try:
        device = _get_clip_device()

        # BGR → RGB → PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)

        img_tensor = _clip_preprocess(pil_image).unsqueeze(0).to(device)

        # Ensemble: 4 промпта на марку — усредняем для устойчивости
        _model_templates = [
            "a photo of a {} on the road",
            "a {} automobile, side view",
            "a {} car parked outdoors",
            "a photo of {} driving on a street",
        ]
        all_scores = np.zeros(len(_VEHICLE_LABELS))
        with torch.no_grad():
            img_features = _clip_model.encode_image(img_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            for tmpl in _model_templates:
                prompts = [tmpl.format(label) for label in _VEHICLE_LABELS]
                text_tokens = _clip_tokenizer(prompts).to(device)
                txt_features = _clip_model.encode_text(text_tokens)
                txt_features /= txt_features.norm(dim=-1, keepdim=True)
                sim = (img_features @ txt_features.T).squeeze(0)
                all_scores += sim.cpu().numpy()
        scores = all_scores / len(_model_templates)
        # Softmax для калиброванной уверенности
        exp_scores = np.exp((scores - scores.max()) * 10)
        scores_pct = exp_scores / exp_scores.sum()
        # Масштабируем: топ-1 в пространстве N классов, base ≈ 1/N
        n = len(_VEHICLE_LABELS)
        scores_pct = np.clip((scores_pct - 1.0 / n) / (1.0 - 1.0 / n), 0.0, 1.0)

        indices = scores_pct.argsort()[::-1][:top_k]
        top_matches = [(
            _VEHICLE_LABELS[i],
            round(float(scores_pct[i]), 4),
        ) for i in indices]

        return {
            "model": top_matches[0][0],
            "confidence": top_matches[0][1],
            "top_matches": top_matches,
            "available": True,
        }

    except Exception as e:
        return {
            "model": f"Ошибка: {e}",
            "confidence": 0.0,
            "top_matches": [],
            "available": True,
        }


# ═══════════════════════════════════════════════
#  3. ЧТЕНИЕ ГОСУДАРСТВЕННОГО НОМЕРА (EasyOCR)
# ═══════════════════════════════════════════════

# Допустимые буквы на российских номерных знаках (аналоги в обоих регистрах)
_RU_PLATE_LETTERS = set('АВЕКМНОРСТУХ')

# Латинские буквы → кириллические омографы (для нормализации OCR)
_LATIN_TO_CYR = {
    'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М',
    'H': 'Н', 'O': 'О', 'P': 'Р', 'C': 'С', 'T': 'Т',
    'Y': 'У', 'X': 'Х',
}

# Цифры → буквы и буквы → цифры для позиционной коррекции
_DIGIT_TO_LETTER = {'0': 'О', '3': 'З', '6': 'Б', '8': 'В'}
_LETTER_TO_DIGIT = {'О': '0', 'З': '3', 'В': '8', 'Б': '6', 'І': '1', 'Л': '4', 'Z': '7', 'Q': '0', 'G': '6', 'S': '5', 'I': '1'}

# Формат российского номера: Б000ББ[000] или Б000ББ00
_RU_PLATE_RE = re.compile(r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$')
# Формат без региона (только 6 символов)
_RU_PLATE_SHORT_RE = re.compile(r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}$')


def _normalize_plate_text(raw: str) -> tuple:
    """
    Нормализует сырой текст OCR к формату российского номерного знака.

    Шаги:
    1. Перевод латинских омографов → кириллица
    2. Позиционная коррекция (цифра/буква в нужной позиции)
    3. Оценка соответствия стандартному формату

    Returns:
        (normalized_text, format_score) где format_score 0.0–1.0.
    """
    if not raw:
        return raw, 0.0

    # Шаг 1: Latin → Cyrillic
    s = ''.join(_LATIN_TO_CYR.get(ch, ch) for ch in raw.upper())

    # Минимальная длина для российского номера
    if len(s) < 6:
        return s, 0.0

    # Шаг 2: позиционная коррекция для формата Б000ББ[000]
    result = list(s)
    # Позиция 0 — буква
    if result[0].isdigit():
        result[0] = _DIGIT_TO_LETTER.get(result[0], result[0])
    # Позиции 1, 2, 3 — цифры
    for i in (1, 2, 3):
        if i < len(result) and not result[i].isdigit():
            result[i] = _LETTER_TO_DIGIT.get(result[i], result[i])
    # Позиции 4, 5 — буквы
    for i in (4, 5):
        if i < len(result) and result[i].isdigit():
            result[i] = _DIGIT_TO_LETTER.get(result[i], result[i])
    # Позиции 6–8 — цифры региона
    for i in range(6, min(len(result), 9)):
        if not result[i].isdigit():
            result[i] = _LETTER_TO_DIGIT.get(result[i], result[i])

    normalized = ''.join(result)

    # Шаг 3: оценка формата
    if _RU_PLATE_RE.match(normalized):
        # Все буквы должны быть из допустимого набора
        letters_ok = (normalized[0] in _RU_PLATE_LETTERS and
                      normalized[4] in _RU_PLATE_LETTERS and
                      normalized[5] in _RU_PLATE_LETTERS)
        return normalized, 1.0 if letters_ok else 0.7
    elif _RU_PLATE_SHORT_RE.match(normalized):
        letters_ok = (normalized[0] in _RU_PLATE_LETTERS and
                      normalized[4] in _RU_PLATE_LETTERS and
                      normalized[5] in _RU_PLATE_LETTERS)
        return normalized, 0.85 if letters_ok else 0.5
    else:
        # Частичное совпадение — примерная оценка по числу правильных символов
        score = 0.0
        if len(normalized) >= 6:
            correct = 0
            total = min(6, len(normalized))
            positions = [
                (0, 'letter'), (1, 'digit'), (2, 'digit'), (3, 'digit'),
                (4, 'letter'), (5, 'letter'),
            ]
            for pos, kind in positions:
                if pos < len(normalized):
                    ch = normalized[pos]
                    if kind == 'letter' and (ch in _RU_PLATE_LETTERS):
                        correct += 1
                    elif kind == 'digit' and ch.isdigit():
                        correct += 1
            score = correct / total * 0.6  # не полное совпадение
        return normalized, score

_ocr_reader = None


def _load_ocr():
    """Ленивая загрузка EasyOCR."""
    global _ocr_reader
    if _ocr_reader is not None:
        return True
    if not _easyocr_available:
        return False
    try:
        # По умолчанию держим OCR на CPU для стабильности.
        # Принудительно включить GPU можно через WG_OCR_GPU=1.
        ocr_gpu = _is_truthy_env(os.getenv("WG_OCR_GPU", "0"))
        _ocr_reader = easyocr.Reader(["en", "ru"], gpu=ocr_gpu)
        return True
    except Exception as e:
        print(f"[VehicleAnalyzer] Ошибка загрузки EasyOCR: {e}")
        return False


def _find_plate_candidates(image: np.ndarray) -> list:
    """
    Находит прямоугольные области, похожие на номерные знаки,
    с помощью анализа контуров.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape[:2]

    # Адаптивная бинаризация
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)

    # Морфология для замыкания краёв
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) >= 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / max(h, 1)
            area_ratio = (w * h) / max(w_img * h_img, 1)
            # Номерная пластина обычно имеет соотношение сторон 2:1 – 6:1
            if 1.5 <= aspect <= 7.0 and 0.002 < area_ratio < 0.15:
                candidates.append((x, y, w, h))

    # Сортируем по площади (крупнейшие первые), убираем дубли
    candidates.sort(key=lambda r: r[2] * r[3], reverse=True)
    return candidates[:5]


# ── YOLO-детектор номерных пластин (приоритетный метод) ──
_PLATE_YOLO_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Models", "license_plate_yolo.pt")
)
_plate_yolo_model = None
_plate_yolo_loaded = False


def _load_plate_yolo() -> bool:
    """Ленивая загрузка YOLO-детектора номерных пластин.
    Сначала ищет локальную модель Models/license_plate_yolo.pt,
    при отсутствии скачивает keremberke/yolov8n-license-plate-detection.
    """
    global _plate_yolo_model, _plate_yolo_loaded
    if _plate_yolo_loaded:
        return _plate_yolo_model is not None
    _plate_yolo_loaded = True
    try:
        from ultralytics import YOLO as _YOLO
        path = _PLATE_YOLO_PATH if os.path.isfile(_PLATE_YOLO_PATH) \
            else "keremberke/yolov8n-license-plate-detection"
        _plate_yolo_model = _YOLO(path)
        return True
    except Exception as e:
        print(f"[VehicleAnalyzer] YOLO для номеров недоступен: {e} — используется контурный поиск")
        return False


# ── YOLO-детектор автомобилей (для двухэтапного поиска номеров) ──
_AUTO_YOLO_MODEL = None
_AUTO_YOLO_LOADED = False
_AUTO_YOLO_VARIANT = None  # "yolov8" или "yolov11"

# Пути к доступным YOLO моделям
_YOLO8_PATHS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "yolov8n.pt")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "yolov8x.pt")),
]
_YOLO11_PATHS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "yolo11n.pt")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "yolo11x.pt")),
]


def _load_auto_yolo() -> bool:
    """Ленивая загрузка YOLOv8/11 для обнаружения области транспорта.
    Сначала пробует yolo11, потом yolo8. Для двухэтапного поиска номеров.
    """
    global _AUTO_YOLO_MODEL, _AUTO_YOLO_LOADED, _AUTO_YOLO_VARIANT
    if _AUTO_YOLO_LOADED:
        return _AUTO_YOLO_MODEL is not None
    _AUTO_YOLO_LOADED = True
    
    try:
        from ultralytics import YOLO as _YOLO
        
        # Приоритет: yolo11 > yolo8
        for path in _YOLO11_PATHS:
            if os.path.isfile(path):
                _AUTO_YOLO_MODEL = _YOLO(path)
                _AUTO_YOLO_VARIANT = "yolo11"
                print(f"[VehicleAnalyzer] Загружена YOLOv11: {os.path.basename(path)}")
                return True
        
        for path in _YOLO8_PATHS:
            if os.path.isfile(path):
                _AUTO_YOLO_MODEL = _YOLO(path)
                _AUTO_YOLO_VARIANT = "yolo8"
                print(f"[VehicleAnalyzer] Загружена YOLOv8: {os.path.basename(path)}")
                return True
        
        print("[VehicleAnalyzer] YOLOv8/11 не найдены — используется контурный поиск номеров")
        return False
    except Exception as e:
        print(f"[VehicleAnalyzer] Ошибка загрузки YOLOv8/11: {e}")
        return False


def _get_vehicle_region_yolo(image: np.ndarray) -> tuple:
    """Находит область автомобиля через YOLOv8/11.
    Возвращает (x, y, w, h) или None если автомобиль не найден.
    """
    if not _load_auto_yolo():
        return None
    
    try:
        results = _AUTO_YOLO_MODEL(image, verbose=False, conf=0.4)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                # Класс 2 = car в COCO датасете (есть и в YOLOv8, и в YOLOv11)
                if cls_id == 2:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    w, h = x2 - x1, y2 - y1
                    return (x1, y1, w, h)  # Возвращаем первый найденный автомобиль
        return None
    except Exception as e:
        print(f"[VehicleAnalyzer] Ошибка YOLO поиска авто: {e}")
        return None


def _find_plate_candidates_enhanced(image: np.ndarray) -> list:
    """Улучшенный поиск номеров: сначала YOLO находит авто, потом контуры ищут номер внутри.
    При недоступности YOLO просто применяется контурный метод к целому изображению.
    """
    vehicle_region = _get_vehicle_region_yolo(image)
    
    if vehicle_region:
        x, y, w, h = vehicle_region
        # Номер обычно находится в нижней части автомобиля
        # Берем нижние 60% области автомобиля
        plate_search_y = int(y + h * 0.4)
        plate_search_h = int(h * 0.6)
        plate_search_x2 = min(image.shape[1], x + w + 10)
        plate_search_y2 = min(image.shape[0], plate_search_y + plate_search_h + 10)
        plate_search_x1 = max(0, x - 10)
        plate_search_y1 = max(0, plate_search_y - 10)
        
        roi = image[plate_search_y1:plate_search_y2, plate_search_x1:plate_search_x2]
        
        if roi.size > 0:
            # Ищем номеры в этой области
            candidates = _find_plate_candidates(roi)
            # Сдвигаем координаты обратно на полное изображение
            return [(x + plate_search_x1, y + plate_search_y1, w, h) 
                    for x, y, w, h in candidates]
    
    # Fallback: контурный поиск по всему изображению
    return _find_plate_candidates(image)


def _find_plate_candidates_yolo(image: np.ndarray) -> list:
    """
    Находит области номерных пластин через улучшенный двухэтапный метод:
    1. YOLOv8/11 локализует автомобиль
    2. Контурный поиск находит номер внутри области авто
    При недоступности YOLO — контурный поиск по всему изображению.
    """
    return _find_plate_candidates_enhanced(image)


def detect_license_plate(image: np.ndarray) -> dict:
    """
    Детектирует и читает государственный номер.

    Args:
        image: BGR-изображение (кроп транспорта).

    Returns:
        dict с ключами:
            plate_text — распознанный текст номера (str | None)
            confidence — уверенность OCR (0-1)
            region     — (x, y, w, h) области номера или None
            message    — сообщение для пользователя
            available  — True, если EasyOCR доступен
    """
    if not _load_ocr():
        return {
            "plate_text": None,
            "confidence": 0.0,
            "region": None,
            "message": "Недоступно (установите easyocr)",
            "available": False,
        }

    try:
        candidates = _find_plate_candidates_yolo(image)

        best_text = None
        best_conf = 0.0
        best_score = 0.0  # format score
        best_region = None

        for (x, y, w, h) in candidates:
            pad = 5
            y1 = max(0, y - pad)
            y2 = min(image.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(image.shape[1], x + w + pad)
            plate_crop = image[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            # Расширенная предобработка: несколько вариантов бинаризации
            plate_variants = preprocess_plate_region(plate_crop)

            for variant in plate_variants:
                results = _ocr_reader.readtext(variant, detail=1)
                for (bbox, text, conf) in results:
                    cleaned = re.sub(r'[^A-ZА-Яa-zа-я0-9]', '', text).upper()
                    if len(cleaned) < 4:
                        continue
                    normalized, fmt_score = _normalize_plate_text(cleaned)
                    # Взвешенная оценка: формат важнее уверенности OCR
                    combined = conf * 0.4 + fmt_score * 0.6
                    if combined > best_score:
                        best_text = normalized
                        best_conf = conf
                        best_score = combined
                        best_region = (x, y, w, h)

        # Фолбэк: OCR по всему изображению
        if best_text is None:
            results = _ocr_reader.readtext(image, detail=1)
            for (bbox, text, conf) in results:
                cleaned = re.sub(r'[^A-ZА-Яa-zа-я0-9]', '', text).upper()
                if len(cleaned) < 4:
                    continue
                normalized, fmt_score = _normalize_plate_text(cleaned)
                combined = conf * 0.4 + fmt_score * 0.6
                if combined > best_score:
                    best_text = normalized
                    best_conf = conf
                    best_score = combined

        if best_text:
            return {
                "plate_text": best_text,
                "confidence": round(float(best_conf), 3),
                "region": best_region,
                "message": f"Номер: {best_text} (уверенность: {best_conf:.0%})",
                "available": True,
            }
        else:
            return {
                "plate_text": None,
                "confidence": 0.0,
                "region": None,
                "message": "Номерной знак не обнаружен на изображении",
                "available": True,
            }

    except Exception as e:
        return {
            "plate_text": None,
            "confidence": 0.0,
            "region": None,
            "message": f"Ошибка при распознавании номера: {e}",
            "available": True,
        }


# ═══════════════════════════════════════════════
#  4. КОМПЛЕКСНЫЙ АНАЛИЗ
# ═══════════════════════════════════════════════

def analyze_vehicle(image_path: str, progress_callback=None) -> dict:
    """
    Полный анализ изображения транспортного средства.

    Args:
        image_path: путь к файлу изображения.
        progress_callback: функция(step_name, step_num) -> bool.
            Возвращает False для отмены. step_num: 1=цвет, 2=модель, 3=номер.

    Returns:
        dict с результатами по всем трём аспектам + метаданные.
    """
    if not os.path.isfile(image_path):
        return {"error": f"Файл не найден: {image_path}"}

    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Не удалось прочитать изображение: {image_path}"}

    # Предобработка: CLAHE + шумоподавление + резкость
    enhanced = preprocess_image(image)

    if progress_callback and not progress_callback("Определение цвета...", 1):
        return {"error": "Отменено пользователем"}
    color_result = detect_vehicle_color(enhanced)

    if progress_callback and not progress_callback("Определение модели...", 2):
        return {"error": "Отменено пользователем"}
    model_result = detect_vehicle_model(enhanced)

    if progress_callback and not progress_callback("Чтение гос. номера...", 3):
        return {"error": "Отменено пользователем"}
    plate_result = detect_license_plate(enhanced)

    return {
        "file": os.path.basename(image_path),
        "path": image_path,
        "color": color_result,
        "model": model_result,
        "plate": plate_result,
        "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
    }


def analyze_vehicle_folder(folder_path: str) -> list:
    """
    Анализирует все изображения транспорта в папке.

    Args:
        folder_path: путь к папке с изображениями.

    Returns:
        Список результатов analyze_vehicle для каждого изображения.
    """
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    results = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith(extensions):
            full_path = os.path.join(folder_path, fname)
            result = analyze_vehicle(full_path)
            results.append(result)
    return results


def generate_report(results: list, output_path: str, elapsed_seconds: float = None) -> str:
    """
    Формирует текстовый отчёт по результатам анализа.

    Args:
        results: список словарей из analyze_vehicle.
        output_path: путь для сохранения отчёта.

    Returns:
        Путь к сохранённому файлу отчёта.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  ОТЧЁТ: АНАЛИЗ ТРАНСПОРТНЫХ СРЕДСТВ")
    lines.append(f"  Дата: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    lines.append(f"  Обработано изображений: {len(results)}")
    lines.append("=" * 60)
    lines.append("")

    for i, r in enumerate(results, 1):
        if "error" in r:
            lines.append(f"[{i}] {r.get('file', '?')} — ОШИБКА: {r['error']}")
            lines.append("")
            continue

        plate = r['plate'].get('plate_text') or r['plate']['message']
        lines.append(
            f"Объектом осмотра является файл {r['file']} на котором обнаружен транспорт "
            f"{r['model']['model']}, цвет {r['color']['color']}, "
            f"государственный номер {plate}."
        )
        lines.append("")

    lines.append("=" * 60)

    if elapsed_seconds is not None:
        mins, secs = divmod(int(elapsed_seconds), 60)
        lines.append(f"\n  Время обработки: {mins} мин {secs} сек")

    # Статистика зависимостей
    lines.append("\nСтатус модулей:")
    lines.append(f"  CLIP (модель авто):   {'✓ Доступен' if _clip_available else '✗ Не установлен (pip install open-clip-torch)'}")
    lines.append(f"  EasyOCR (номера):     {'✓ Доступен' if _easyocr_available else '✗ Не установлен (pip install easyocr)'}")
    lines.append(f"  OpenCV (цвет):        ✓ Доступен")

    report_text = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return output_path
