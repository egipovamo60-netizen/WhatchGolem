"""
Плагин анализа мобильных телефонов для Watch Golem.

Функционал:
  - Определение модели телефона (CLIP zero-shot классификация)
  - Распознавание IMEI-кода, если он виден в кадре (EasyOCR + Luhn-валидация)

Зависимости:
  pip install easyocr open-clip-torch Pillow python-docx
"""

import os
import sys
import io
import re
import cv2
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from logger import get_logger

_log = get_logger("PhoneAnalyzer")

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ─── Кэши моделей ───
_clip_model = None
_clip_preprocess = None
_tokenize = None
_torch = None
_PILImage = None
_easyocr_reader = None

# ─── Предвычисленные текстовые эмбеддинги (заполняются при загрузке CLIP) ───
_clip_brand_feats = None
_clip_category_feats = None

# ─── Модели телефонов для CLIP zero-shot ───
_PHONE_BRANDS = [
    # Apple
    "Apple iPhone 16 Pro Max", "Apple iPhone 16 Pro", "Apple iPhone 16",
    "Apple iPhone 15 Pro Max", "Apple iPhone 15 Pro", "Apple iPhone 15",
    "Apple iPhone 14 Pro Max", "Apple iPhone 14 Pro", "Apple iPhone 14",
    "Apple iPhone 13 Pro Max", "Apple iPhone 13 Pro", "Apple iPhone 13",
    "Apple iPhone 12 Pro Max", "Apple iPhone 12", "Apple iPhone 11",
    "Apple iPhone SE",
    # Samsung
    "Samsung Galaxy S25 Ultra", "Samsung Galaxy S25", "Samsung Galaxy S24 Ultra",
    "Samsung Galaxy S24", "Samsung Galaxy S23 Ultra", "Samsung Galaxy S23",
    "Samsung Galaxy S22 Ultra", "Samsung Galaxy S22", "Samsung Galaxy S21",
    "Samsung Galaxy A55", "Samsung Galaxy A54", "Samsung Galaxy A53",
    "Samsung Galaxy A35", "Samsung Galaxy A15",
    "Samsung Galaxy Z Fold 6", "Samsung Galaxy Z Flip 6",
    "Samsung Galaxy Z Fold 5", "Samsung Galaxy Z Flip 5",
    "Samsung Galaxy Note 20", "Samsung Galaxy Note 10",
    # Xiaomi
    "Xiaomi 14 Ultra", "Xiaomi 14 Pro", "Xiaomi 14",
    "Xiaomi 13 Ultra", "Xiaomi 13 Pro", "Xiaomi 13",
    "Xiaomi Redmi Note 13 Pro", "Xiaomi Redmi Note 13",
    "Xiaomi Redmi Note 12 Pro", "Xiaomi Redmi Note 12",
    "Xiaomi Poco F6 Pro", "Xiaomi Poco X6",
    # Huawei
    "Huawei P60 Pro", "Huawei P50 Pro", "Huawei Mate 60 Pro",
    "Huawei Mate 50 Pro", "Huawei Nova 12",
    # Google
    "Google Pixel 9 Pro", "Google Pixel 9", "Google Pixel 8 Pro",
    "Google Pixel 8", "Google Pixel 7 Pro", "Google Pixel 7",
    # OnePlus
    "OnePlus 12", "OnePlus 11", "OnePlus Nord 3",
    # Realme / OPPO / Vivo
    "Realme GT 5 Pro", "OPPO Find X7", "Vivo X100 Pro",
    # Honor
    "Honor Magic 6 Pro", "Honor 200 Pro", "Honor X9b",
    # Другие
    "Nokia phone", "Motorola phone", "Sony Xperia phone",
    "ASUS ROG Phone", "Nothing Phone",
    # Общие
    "unknown smartphone", "old push-button phone", "tablet device",
]

_PHONE_CATEGORIES = [
    "a smartphone", "a mobile phone",
    "a tablet", "not a phone",
]


# ═══════════════════════════════════════════════
#  Валидация IMEI (Luhn)
# ═══════════════════════════════════════════════

def _luhn_check(imei_str):
    """Проверяет IMEI (15 цифр) по алгоритму Luhn."""
    if len(imei_str) != 15 or not imei_str.isdigit():
        return False
    total = 0
    for i, ch in enumerate(imei_str):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Таблица типичных OCR-ошибок при распознавании цифровых строк IMEI.
# Буквы, которые OCR-движок часто путает с цифрами.
_OCR_CHAR_MAP = str.maketrans(
    "OoIilSsZzBbGgTtAaEeDdQq",
    "00111552228866774433990"
)


def _correct_ocr_text(text: str) -> str:
    """Заменяет типичные OCR-ошибки (O→0, I→1, S→5 и др.) в тексте."""
    return text.translate(_OCR_CHAR_MAP)


def _extract_imei_candidates(text):
    """Извлекает все возможные IMEI из текста (15 подряд идущих цифр)."""
    # Убираем пробелы/тире между группами цифр
    cleaned = re.sub(r'[\s\-/.:]+', '', text)
    # Ищем все последовательности из 15 цифр
    candidates = re.findall(r'\d{15}', cleaned)
    # Также из оригинала ищем группированные IMEI типа "35 209900 176148 2"
    spaced = re.sub(r'[^\d\s]', '', text)
    groups = re.findall(r'(?:\d[\s]*){15}', spaced)
    for g in groups:
        condensed = re.sub(r'\s+', '', g)[:15]
        if len(condensed) == 15 and condensed.isdigit() and condensed not in candidates:
            candidates.append(condensed)
    return candidates


# ═══════════════════════════════════════════════
#  Ленивая загрузка моделей
# ═══════════════════════════════════════════════

def _load_clip(progress_callback=None):
    global _clip_model, _clip_preprocess, _tokenize, _torch, _PILImage
    global _clip_brand_feats, _clip_category_feats
    if _clip_model is not None:
        return _clip_model, _clip_preprocess, _tokenize

    if progress_callback:
        progress_callback("Загрузка CLIP...", 0)

    try:
        import torch
        import open_clip
        from PIL import Image
    except ImportError:
        raise ImportError(
            "CLIP зависимости не установлены.\n"
            "Установите: pip install open-clip-torch Pillow"
        )

    _torch = torch
    _PILImage = Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    _clip_model = model
    _clip_preprocess = preprocess
    _tokenize = tokenizer

    # Предвычисляем текстовые эмбеддинги для обоих статичных списков.
    # Они никогда не меняются, поэтому достаточно один раз при загрузке.
    with torch.no_grad():
        cat_tokens = tokenizer(["a photo of {}".format(lbl) for lbl in _PHONE_CATEGORIES])
        _clip_category_feats = model.encode_text(cat_tokens)
        _clip_category_feats /= _clip_category_feats.norm(dim=-1, keepdim=True)

        brand_tokens = tokenizer(["a photo of {}".format(lbl) for lbl in _PHONE_BRANDS])
        _clip_brand_feats = model.encode_text(brand_tokens)
        _clip_brand_feats /= _clip_brand_feats.norm(dim=-1, keepdim=True)

    _log.info("CLIP загружен (ViT-B-32), текстовые эмбеддинги закешированы")
    return _clip_model, _clip_preprocess, _tokenize


def _load_easyocr(progress_callback=None):
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader

    if progress_callback:
        progress_callback("Загрузка EasyOCR...", 2)

    try:
        import easyocr
    except ImportError:
        raise ImportError(
            "EasyOCR не установлен.\n"
            "Установите: pip install easyocr"
        )

    _easyocr_reader = easyocr.Reader(['en'], gpu=False)
    _log.info("EasyOCR загружен")
    return _easyocr_reader


# ═══════════════════════════════════════════════
#  CLIP zero-shot
# ═══════════════════════════════════════════════

def _clip_classify(image_bgr, labels, prompt_template="a photo of {}"):
    """CLIP zero-shot классификация (с кодированием изображения внутри)."""
    pil_img = _PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    img_tensor = _clip_preprocess(pil_img).unsqueeze(0)
    with _torch.no_grad():
        img_feat = _clip_model.encode_image(img_tensor)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
    return _clip_classify_feat(img_feat, labels, prompt_template)


def _clip_classify_feat(img_feat, labels, prompt_template="a photo of {}", cached_txt_feats=None):
    """CLIP zero-shot классификация с уже закодированным изображением.

    Если cached_txt_feats передан — текст не кодируется повторно.
    """
    with _torch.no_grad():
        if cached_txt_feats is not None:
            txt_feat = cached_txt_feats
        else:
            prompts = [prompt_template.format(lbl) for lbl in labels]
            text_tokens = _tokenize(prompts)
            txt_feat = _clip_model.encode_text(text_tokens)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ txt_feat.T).squeeze(0)
        probs = sim.softmax(dim=0)
    idx = probs.argmax().item()
    return labels[idx], float(probs[idx])


def _detect_phone_model(image, progress_callback=None):
    """Определяет, является ли объект телефоном, и определяет модель."""
    _load_clip(progress_callback)

    if progress_callback:
        progress_callback("Определение типа устройства...", 1)

    # Кодируем изображение один раз — используем для обоих классификаторов
    pil_img = _PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_tensor = _clip_preprocess(pil_img).unsqueeze(0)
    with _torch.no_grad():
        img_feat = _clip_model.encode_image(img_tensor)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)

    # Проверяем: телефон ли это вообще (закешированные текстовые эмбеддинги)
    category, cat_conf = _clip_classify_feat(img_feat, _PHONE_CATEGORIES,
                                             cached_txt_feats=_clip_category_feats)

    if category == "not a phone":
        return {
            "is_phone": False,
            "category": "Не является телефоном",
            "category_confidence": cat_conf,
            "model": "—",
            "model_confidence": 0.0,
        }

    category_ru = {
        "a smartphone": "Смартфон",
        "a mobile phone": "Мобильный телефон",
        "a tablet": "Планшет",
    }.get(category, category)

    # Определяем конкретную модель (то же изображение, закешированные тексты)
    if progress_callback:
        progress_callback("Определение модели телефона...", 1)

    model_name, model_conf = _clip_classify_feat(img_feat, _PHONE_BRANDS,
                                                 cached_txt_feats=_clip_brand_feats)

    return {
        "is_phone": True,
        "category": category_ru,
        "category_confidence": cat_conf,
        "model": model_name,
        "model_confidence": model_conf,
    }


# ═══════════════════════════════════════════════
#  Предобработка для IMEI
# ═══════════════════════════════════════════════

def _build_imei_scan_variants(image):
    """
    Быстрый набор из 3 вариантов — только для поиска меток «IMEI».

    Не включает bilateralFilter и adaptiveThreshold (дорогие операции).
    Возвращает (scan_variants, scale, base_bgr, gray).
    """
    h, w = image.shape[:2]

    if max(h, w) < 600:
        scale = 4
    elif max(h, w) < 1200:
        scale = 2
    else:
        scale = 1

    if scale > 1:
        base = cv2.resize(image, (w * scale, h * scale),
                          interpolation=cv2.INTER_LANCZOS4)
    else:
        base = image.copy()

    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe_filter.apply(gray)

    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    scan_variants = [
        ("base",      base),
        ("clahe",     clahe_img),
        ("sharpened", sharpened),
    ]
    return scan_variants, scale, base, gray


def _extend_to_full_variants(scan_variants, base_bgr, gray):
    """
    Добавляет к scan_variants ещё 6 вариантов для детального OCR регионов.

    Вызывается только если метки «IMEI» найдены — избегает напрасной работы.
    """
    clahe_img = next(img for name, img in scan_variants if name == "clahe")

    # ── Билатеральный фильтр (размытие без смазывания краёв) ──
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # ── Otsu-бинаризация по CLAHE ──
    _, otsu = cv2.threshold(clahe_img, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Адаптивная бинаризация (для неравномерного освещения) ──
    adaptive = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # ── Гамма-коррекция: осветление (γ=0.5) и затемнение (γ=2.0) ──
    lut_light = np.array(
        [min(255, int((i / 255.0) ** 0.5 * 255)) for i in range(256)],
        dtype=np.uint8
    )
    lut_dark = np.array(
        [min(255, int((i / 255.0) ** 2.0 * 255)) for i in range(256)],
        dtype=np.uint8
    )

    return scan_variants + [
        ("otsu",         otsu),
        ("otsu_inv",     cv2.bitwise_not(otsu)),
        ("adaptive",     adaptive),
        ("adaptive_inv", cv2.bitwise_not(adaptive)),
        ("gamma_light",  cv2.LUT(gray, lut_light)),
        ("gamma_dark",   cv2.LUT(gray, lut_dark)),
    ]


def _build_imei_variants(image):
    """
    Строит полный набор предобработанных вариантов изображения для OCR IMEI.
    Возвращает (variants, scale).
    """
    scan_variants, scale, base, gray = _build_imei_scan_variants(image)
    return _extend_to_full_variants(scan_variants, base, gray), scale


# ═══════════════════════════════════════════════
#  Распознавание IMEI
# ═══════════════════════════════════════════════

def _bbox_to_orig(bbox, inv_scale):
    """Переводит bbox EasyOCR [[x,y]×4] из масштабированных в оригинальные координаты."""
    pts = np.array(bbox, dtype=np.float32) * inv_scale
    xs = pts[:, 0]
    ys = pts[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _find_boxes_for_imei(imei, raw_entries, inv_scale):
    """
    Ищет bbox-ы OCR-записей, которые содержат ≥6 цифр из найденного IMEI.
    raw_entries: список (bbox, text, conf, name)
    Возвращает список (x1,y1,x2,y2) в координатах оригинального изображения.
    """
    boxes = []
    for bbox, text, conf, name in raw_entries:
        digits_only = re.sub(r'\D', '', text)
        # Считаем совпадающими, если ≥6 цифр подряд из текста входят в IMEI
        if len(digits_only) >= 6:
            # скользящее окно: ищем любое вхождение digits_only в imei
            # или imei содержит digits_only как подстроку длиной ≥6
            for length in range(min(len(digits_only), 15), 5, -1):
                for start in range(len(digits_only) - length + 1):
                    substr = digits_only[start:start + length]
                    if substr in imei:
                        boxes.append(_bbox_to_orig(bbox, inv_scale))
                        break
                else:
                    continue
                break
    return boxes


def _ocr_region(reader, variants, inv_scale, sh, sw, lx1, ly1, lx2, ly2, ltext):
    """
    Выполняет детальный OCR в зоне вокруг одной IMEI-метки.
    Возвращает (region_texts, region_raw_entries):
      region_texts:       [(text, conf, vname), ...]
      region_raw_entries: [(bbox_orig_list, text, conf, vname), ...]
    bbox_orig_list — координаты уже в пространстве оригинального изображения.
    """
    lw = lx2 - lx1
    lh = ly2 - ly1

    # Зона поиска: метка + пространство справа (число) и небольшой отступ вверх/вниз.
    # По горизонтали: от чуть левее метки до ~16 её ширин правее.
    # По вертикали: ±3 высоты строки — покрывает случай, когда номер на следующей строке.
    pad_y = max(10, int(lh * 3.0))
    rx1 = max(0,      int(lx1 - int(lw * 0.2)))
    ry1 = max(0,      ly1 - pad_y)
    rx2 = min(sw - 1, int(lx1 + lw * 16))
    ry2 = min(sh - 1, ly2 + pad_y)

    region_texts = []
    region_raw = []

    if rx2 <= rx1 or ry2 <= ry1:
        return region_texts, region_raw

    # Число может быть уже внутри самой метки («IMEI: 356...»)
    inline = re.sub(r'[\s\-]', '', re.sub(r'IMEI', '', ltext, flags=re.IGNORECASE))
    if re.search(r'\d{10,}', inline):
        region_texts.append((inline, 1.0, "label_inline"))

    # Обходим все варианты предобработки, обрезая их до зоны
    for vname, vimg in variants:
        crop = vimg[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            continue
        try:
            res = reader.readtext(crop, detail=1, paragraph=False)
        except Exception as e:
            _log.warning(f"EasyOCR crop '{vname}': {e}")
            continue

        for cbbox, text, conf in res:
            # bbox: crop-пространство → scaled → original
            arr = np.array(cbbox, dtype=np.float32)
            arr[:, 0] = (arr[:, 0] + rx1) * inv_scale
            arr[:, 1] = (arr[:, 1] + ry1) * inv_scale
            region_texts.append((text, conf, vname))
            region_raw.append((arr.tolist(), text, conf, vname))

        # Ранний выход из вариантов для данного региона, если Luhn уже выполнен
        partial = " ".join(t for t, c, s in region_texts)
        if any(_luhn_check(c) for c in _extract_imei_candidates(partial)):
            _log.debug(f"Регион ({lx1},{ly1}): ранний выход на варианте '{vname}'")
            break

    return region_texts, region_raw


def _detect_imei(image, progress_callback=None):
    """
    Ищет IMEI-коды на изображении через OCR.

    Алгоритм:
      1. Полное сканирование изображения для поиска всех меток «IMEI».
      2. Вокруг каждой найденной метки формируется независимая зона поиска.
      3. Детальный OCR (все варианты предобработки) выполняется ТОЛЬКО внутри
         каждой зоны — числа вне зон не рассматриваются.
      4. Поддерживается произвольное количество меток IMEI на одном снимке
         (например, два IMEI у телефона с двумя SIM-слотами).
    """
    reader = _load_easyocr(progress_callback)

    if progress_callback:
        progress_callback("Поиск меток IMEI...", 3)

    # ── Шаг 0: строим только быстрые варианты для сканирования меток ──────
    scan_variants, scale, _base_bgr, _base_gray = _build_imei_scan_variants(image)
    inv_scale = 1.0 / scale
    sh, sw = scan_variants[0][1].shape[:2]

    # ── Шаг 1: обнаружение всех меток «IMEI» ─────────────────────────────
    label_scan_variants = scan_variants  # все 3 быстрых варианта
    label_regions = []        # (lx1, ly1, lx2, ly2, label_text) — scaled-координаты
    seen_label_keys = set()

    for scan_name, vimg in label_scan_variants:
        try:
            scan_res = reader.readtext(vimg, detail=1, paragraph=False)
        except Exception as e:
            _log.warning(f"EasyOCR label scan '{scan_name}': {e}")
            continue

        for bbox, text, conf in scan_res:
            if not re.search(r'IMEI', text, re.IGNORECASE):
                continue
            pts = np.array(bbox, dtype=np.float32)
            lx1, ly1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            lx2, ly2 = int(pts[:, 0].max()), int(pts[:, 1].max())
            # Группируем близкие дубли (один и тот же bbox из двух вариантов сканирования)
            key = (lx1 // 20, ly1 // 20)
            if key in seen_label_keys:
                continue
            seen_label_keys.add(key)
            label_regions.append((lx1, ly1, lx2, ly2, text))
            _log.debug(f"Метка IMEI ({scan_name}): '{text}' @ ({lx1},{ly1})-({lx2},{ly2})")

    if not label_regions:
        _log.debug("Метки IMEI не найдены на изображении")
        return {
            "found": False,
            "valid_imeis": [],
            "unverified_imeis": [],
            "imei_boxes": [],
            "message": "IMEI не обнаружен (метка «IMEI» не найдена)",
        }

    _log.debug(f"Найдено меток IMEI: {len(label_regions)}")

    # ── Шаг 2: строим полный набор вариантов ТОЛЬКО после нахождения меток ─
    # Дорогие операции (bilateralFilter, adaptiveThreshold) пропускаются,
    # если IMEI-меток нет — это главная оптимизация для изображений без IMEI.
    variants = _extend_to_full_variants(scan_variants, _base_bgr, _base_gray)

    # ── Шаг 3: OCR и валидация по каждому региону независимо ─────────────
    valid_imeis = []
    unverified_imeis = []
    imei_boxes = []
    seen_imeis = set()

    for region_idx, (lx1, ly1, lx2, ly2, ltext) in enumerate(label_regions):
        if progress_callback:
            progress_callback(f"OCR зона IMEI {region_idx + 1}/{len(label_regions)}...", 3)

        region_texts, region_raw = _ocr_region(
            reader, variants, inv_scale, sh, sw,
            lx1, ly1, lx2, ly2, ltext
        )

        region_full = " ".join(t for t, c, s in region_texts)
        _log.debug(f"Регион {region_idx + 1} текст: {region_full}")

        for imei in _extract_imei_candidates(region_full):
            if imei in seen_imeis:
                continue
            seen_imeis.add(imei)
            is_valid = _luhn_check(imei)
            boxes = _find_boxes_for_imei(imei, region_raw, 1.0)
            entry = {"imei": imei, "boxes": boxes, "luhn": is_valid,
                     "region": region_idx + 1}
            imei_boxes.append(entry)
            if is_valid:
                valid_imeis.append(imei)
            else:
                unverified_imeis.append(imei)

        # ── OCR-коррекция: пробуем исправить типичные ошибки (O→0, I→1 и др.) ──
        # Применяется только если Luhn-валидный IMEI ещё не найден в этом регионе
        region_has_valid = any(
            e["luhn"] and e["region"] == region_idx + 1 for e in imei_boxes
        )
        if not region_has_valid:
            corrected_text = _correct_ocr_text(region_full)
            if corrected_text != region_full:
                for imei in _extract_imei_candidates(corrected_text):
                    if imei in seen_imeis:
                        continue
                    if not _luhn_check(imei):
                        continue
                    seen_imeis.add(imei)
                    _log.debug(f"Регион {region_idx + 1}: IMEI найден после OCR-коррекции: {imei}")
                    boxes = _find_boxes_for_imei(imei, region_raw, 1.0)
                    entry = {"imei": imei, "boxes": boxes, "luhn": True,
                             "region": region_idx + 1}
                    imei_boxes.append(entry)
                    valid_imeis.append(imei)

    # ── Шаг 4: формирование результата ────────────────────────────────────
    result = {
        "found": len(valid_imeis) > 0 or len(unverified_imeis) > 0,
        "valid_imeis": valid_imeis,
        "unverified_imeis": unverified_imeis,
        "imei_boxes": imei_boxes,
        "imei_regions_found": len(label_regions),
    }

    if valid_imeis:
        result["message"] = f"IMEI: {', '.join(valid_imeis)} (Luhn ✓)"
    elif unverified_imeis:
        result["message"] = f"Возможный IMEI: {', '.join(unverified_imeis)} (Luhn ✗)"
    else:
        result["message"] = "IMEI не обнаружен"

    return result


def _annotate_imei(image, imei_boxes):
    """
    Рисует на копии изображения зелёные прямоугольники вокруг областей
    с Luhn-валидными IMEI. Невалидные (Luhn ✗) не отображаются.
    Возвращает аннотированное BGR-изображение.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(2, int(min(h, w) / 400))
    font_scale = max(0.5, min(w, h) / 1000)

    for entry in imei_boxes:
        if not entry["luhn"]:
            continue  # показываем только Luhn-валидные
        imei = entry["imei"]
        color = (0, 220, 0)  # зелёный
        label = f"IMEI: {imei} [OK]"

        for (x1, y1, x2, y2) in entry["boxes"]:
            # Ограничиваем координаты размерами изображения
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Рамка
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Подпись над рамкой (с тёмным фоном для читаемости)
            (tw, th), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            ty = max(y1 - 6, th + 4)
            tx = max(0, x1)
            cv2.rectangle(
                annotated,
                (tx, ty - th - baseline - 2),
                (min(w - 1, tx + tw + 4), ty + baseline),
                (20, 20, 20), cv2.FILLED
            )
            cv2.putText(
                annotated, label,
                (tx + 2, ty - baseline),
                font, font_scale, color, thickness, cv2.LINE_AA
            )

    return annotated


# ═══════════════════════════════════════════════
#  Основной анализ
# ═══════════════════════════════════════════════

def analyze_phone(image_path, progress_callback=None):
    """Анализирует изображение: модель телефона + IMEI."""
    _log.info(f"Анализ: {os.path.basename(image_path)}")

    if not os.path.isfile(image_path):
        return {"error": f"Файл не найден: {image_path}",
                "file": os.path.basename(image_path)}

    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Не удалось прочитать: {image_path}",
                "file": os.path.basename(image_path)}

    # 1. Определение модели телефона
    phone_info = _detect_phone_model(img, progress_callback)

    # 2. Поиск IMEI
    imei_info = _detect_imei(img, progress_callback)

    # 3. Аннотация: рисуем области IMEI на изображении
    annotated_path = None
    imei_boxes = imei_info.get("imei_boxes", [])
    has_boxes = any(e["luhn"] and len(e["boxes"]) > 0 for e in imei_boxes)
    if imei_boxes and has_boxes:
        annotated = _annotate_imei(img, imei_boxes)
        base_name, ext = os.path.splitext(image_path)
        annotated_path = base_name + "_imei_annotated" + (ext or ".jpg")
        cv2.imwrite(annotated_path, annotated)
        _log.info(f"Аннотированное изображение сохранено: {os.path.basename(annotated_path)}")

    if progress_callback:
        progress_callback("Готово", 4)

    result = {
        "file": os.path.basename(image_path),
        "path": os.path.abspath(image_path),
        "phone": phone_info,
        "imei": imei_info,
        "annotated_path": annotated_path,
    }

    _log.info(
        f"Результат: {phone_info['model']} ({phone_info['model_confidence']:.0%}), "
        f"IMEI: {imei_info['message']}"
    )
    return result


def analyze_phone_folder(folder_path, progress_callback=None):
    """Анализирует все изображения телефонов в папке."""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(extensions)
    )
    results = []
    for fname in files:
        full_path = os.path.join(folder_path, fname)
        r = analyze_phone(full_path, progress_callback)
        results.append(r)
    return results


# ═══════════════════════════════════════════════
#  Текстовый отчёт
# ═══════════════════════════════════════════════

def generate_report_phone(results, report_path, elapsed_seconds=None):
    """Создаёт текстовый отчёт по анализу телефонов."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    ts = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    lines = [
        "=" * 64,
        "  ОТЧЁТ: АНАЛИЗ МОБИЛЬНЫХ ТЕЛЕФОНОВ",
        f"  Дата: {ts}",
        f"  Обработано: {len(results)}",
        "=" * 64,
        "",
    ]

    for r in results:
        if "error" in r:
            lines.append(f"Файл {r.get('file', '?')} — ОШИБКА: {r['error']}")
            lines.append("")
            continue

        phone = r.get("phone", {})
        if phone.get("is_phone"):
            model_str = f"{phone.get('model', 'не определена')} ({phone.get('model_confidence', 0):.0%})"
        else:
            model_str = "модель не определена"

        imei = r.get("imei", {})
        # Показываем только Luhn-валидные IMEI — те, что отмечены зелёными метками на изображении
        imei_values = list(dict.fromkeys(imei.get("valid_imeis", [])))
        imei_str = ", ".join(imei_values) if imei_values else "не обнаружен"

        lines.append(
            f"Объектом осмотра является файл {r.get('file', '?')} на котором обнаружен "
            f"мобильный телефон — модель: {model_str}, IMEI: {imei_str}."
        )
        lines.append("")

    if elapsed_seconds is not None:
        mins, secs = divmod(int(elapsed_seconds), 60)
        lines.append("=" * 64)
        lines.append(f"  Время обработки: {mins} мин {secs} сек")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    _log.info(f"Текстовый отчёт сохранён: {report_path}")
    return report_path


# ═══════════════════════════════════════════════
#  Word-отчёт
# ═══════════════════════════════════════════════

def generate_report_word_phone(results, report_dir, elapsed_seconds=None):
    """Создаёт Word-документ с изображениями и результатами анализа телефонов."""
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from PIL import Image

    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Arial'
    style.font.size = Pt(11)

    title = doc.add_heading("Отчёт: Анализ мобильных телефонов", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    ts = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    p = doc.add_paragraph(f"Дата: {ts}\nОбработано изображений: {len(results)}")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for idx, r in enumerate(results):
        if idx > 0:
            doc.add_page_break()

        doc.add_heading(f"Изображение: {r.get('file', '?')}", level=2)

        def _insert_picture(path, label=None):
            if not (path and os.path.isfile(path)):
                return
            if label:
                lp = doc.add_paragraph(label)
                lp.runs[0].bold = True
            try:
                doc.add_picture(path, width=Inches(4.5))
            except ZeroDivisionError:
                try:
                    img_pil = Image.open(path)
                    buf = io.BytesIO()
                    img_pil.save(buf, format=img_pil.format or "JPEG", dpi=(96, 96))
                    buf.seek(0)
                    doc.add_picture(buf, width=Inches(4.5))
                except Exception as e:
                    doc.add_paragraph(f"[Ошибка вставки {label or 'изображения'}: {e}]")
            except Exception as e:
                doc.add_paragraph(f"[Ошибка вставки {label or 'изображения'}: {e}]")

        # Оригинальное изображение
        image_path = r.get("path", "")
        _insert_picture(image_path, "Оригинал:")

        # Аннотированное изображение с отмеченными IMEI (если есть)
        annotated_path = r.get("annotated_path")
        if annotated_path and os.path.isfile(annotated_path):
            _insert_picture(annotated_path, "IMEI — область считывания:")

        if "error" in r:
            doc.add_paragraph(f"Ошибка: {r['error']}")
            continue

        phone = r.get("phone", {})
        if phone.get("is_phone"):
            model_str = f"{phone.get('model', 'не определена')} ({phone.get('model_confidence', 0):.0%})"
        else:
            model_str = "модель не определена"

        imei = r.get("imei", {})
        # Показываем только Luhn-валидные IMEI — те, что отмечены зелёными метками на аннотированном изображении
        imei_values = list(dict.fromkeys(imei.get("valid_imeis", [])))
        imei_str = ", ".join(imei_values) if imei_values else "не обнаружен"

        verdict = doc.add_paragraph()
        verdict.add_run(
            f"Объектом осмотра является файл {r.get('file', '?')} на котором обнаружен "
            f"мобильный телефон — модель: {model_str}, IMEI: {imei_str}."
        )

    if elapsed_seconds is not None:
        doc.add_page_break()
        doc.add_heading("Информация об отчёте", level=2)
        mins, secs = divmod(int(elapsed_seconds), 60)
        t_para = doc.add_paragraph()
        t_para.add_run("Время обработки: ").bold = True
        t_para.add_run(f"{mins} мин {secs} сек")

    os.makedirs(report_dir, exist_ok=True)
    doc_path = os.path.join(report_dir, "PHONE_REPORT.docx")
    doc.save(doc_path)
    _log.info(f"Word отчёт сохранён: {doc_path}")
    return doc_path
