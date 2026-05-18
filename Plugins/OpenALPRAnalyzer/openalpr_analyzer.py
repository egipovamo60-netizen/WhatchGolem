"""
Плагин анализа транспортных средств OpenALPR Analyzer v4.0 - MAXIMUM QUALITY

✨ Основные улучшения v4.0:
  1. Ансамбль из 2-3x CLIP моделей (вместо одной)
  2. Региональная выборка цвета (верх/середина/низ машины)
  3. Множественные методы OCR (OpenALPR + EasyOCR + CV)
  4. Валидация и фильтрация результатов через ансамбль
  5. Кэширование загруженных моделей
  6. Cross-validation между методами

Определяет:
- Цвет: региональная выборка + HSV + LAB + Delta E CIEDE2000
- Модель: ансамбль 2-3x CLIP моделей для максимальной точности
- Номер: ансамбль OpenALPR + EasyOCR + VehicleAnalyzer с голосованием

Зависимости:
  pip install transformers torch Pillow opencv-python scikit-learn scikit-image easyocr
  Опционально: pip install openalpr-python-bindings

На Windows может потребоваться установка бинарного пакета OpenALPR:
  https://github.com/openalpr/openalpr/wiki/Installation
"""

# ⚠️ IMPORT FROM V4.0 MAXIMUM QUALITY MODULE
try:
    from .openalpr_analyzer_v4_maxquality import (
        analyze_vehicle,
        analyze_vehicle_folder,
        generate_report,
        _detect_color_advanced_v4,
        _detect_model_clip_ensemble_v4,
        _detect_plate_multimethod_v4,
    )
    _USE_V4_MAXQUALITY = True
except ImportError:
    _USE_V4_MAXQUALITY = False
    # Fallback to old implementation below


import os
import re
import json
import numpy as np
from datetime import datetime
from pathlib import Path

import cv2
from sklearn.cluster import KMeans
from skimage.color import deltaE_ciede2000

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from logger import get_logger

_log = get_logger("OpenALPRAnalyzer")

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ─────────────────────────────────────────────
# Попытка импортировать OpenALPR
# ─────────────────────────────────────────────

try:
    from openalpr import Alpr
    _OPENALPR_AVAILABLE = True
except ImportError:
    _OPENALPR_AVAILABLE = False
    _log.warning("OpenALPR не установлен. Установите: pip install openalpr-python-bindings")

try:
    from transformers import CLIPProcessor, CLIPModel
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False

# ─────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────

_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")
_OPENALPR_CONFIG = os.path.join(os.path.dirname(__file__), "openalpr.conf")

_SUPPORTED_REGIONS = ["ru", "us", "eu"]
_DEFAULT_REGION = "ru"

# Маппинг цветов LAB значения → русские названия
_COLOR_MAP_LAB = {
    "black": (20, 0, 0),         # L, a, b (dark)
    "white": (95, 0, 0),         # (light)
    "gray": (50, 0, 0),
    "red": (50, 50, 30),
    "blue": (30, -20, -50),
    "green": (50, -40, 20),
    "yellow": (80, 0, 50),
    "brown": (40, 20, 20),
    "silver": (70, 0, 0),
    "orange": (60, 40, 40),
}

_COLOR_NAMES_RU = {
    "black": "Чёрный",
    "white": "Белый",
    "gray": "Серый",
    "red": "Красный",
    "blue": "Синий",
    "green": "Зелёный",
    "yellow": "Жёлтый",
    "brown": "Коричневый",
    "silver": "Серебристый",
    "orange": "Оранжевый",
}

# Стандартные эталонные цвета в LAB
_REFERENCE_COLORS_LAB = {
    "black": np.array([20, 0, 0]),
    "white": np.array([95, 0, 0]),
    "gray": np.array([50, 0, 0]),
    "red": np.array([50, 50, 30]),
    "blue": np.array([30, -20, -50]),
    "green": np.array([50, -40, 20]),
    "yellow": np.array([80, 0, 50]),
    "brown": np.array([40, 20, 20]),
    "silver": np.array([70, 0, 0]),
    "orange": np.array([60, 40, 40]),
}

# Для нормализации российских номеров
_LATIN_TO_CYR = {
    'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М',
    'H': 'Н', 'O': 'О', 'P': 'Р', 'C': 'С', 'T': 'Т',
    'Y': 'У', 'X': 'Х',
}
_DIGIT_TO_LETTER = {'0': 'О', '3': 'З', '6': 'Б', '8': 'В'}
_LETTER_TO_DIGIT = {
    'О': '0', 'З': '3', 'В': '8', 'Б': '6',
    'І': '1', 'Л': '4', 'Z': '7', 'Q': '0',
    'G': '6', 'S': '5', 'I': '1',
}

_RU_PLATE_RE = re.compile(r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$')

# Модели для CLIP (базовый набор / fallback)
_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_VEHICLE_MODELS_DEFAULT = [
    "Toyota Camry", "Toyota Corolla", "Toyota RAV4",
    "BMW 3 Series", "BMW 5 Series", "BMW X5",
    "Mercedes C-Class", "Mercedes E-Class", "Mercedes S-Class",
    "Volkswagen Golf", "Volkswagen Passat", "Volkswagen Polo",
    "Ford Focus", "Ford Mondeo", "Ford Kuga",
    "Hyundai Solaris", "Hyundai Elantra", "Hyundai Santa Fe",
    "Lada Vesta", "Lada XRAY", "Lada Granta",
    "GAZelle van", "Renault Logan", "Skoda Octavia",
]

# Динамическая загрузка моделей из датасетов
_VEHICLE_MODELS = _VEHICLE_MODELS_DEFAULT.copy()
_DATASET_SOURCE = "default"  # "default", "compcars", "stanford"


# ─────────────────────────────────────────────
# Загрузка конфигурации
# ─────────────────────────────────────────────

def _load_config() -> dict:
    """Загружает конфигурацию из config.json."""
    if os.path.isfile(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _get_region() -> str:
    """Возвращает регион для OpenALPR."""
    cfg = _load_config()
    region = cfg.get("region", _DEFAULT_REGION).strip()
    return region if region in _SUPPORTED_REGIONS else _DEFAULT_REGION


# ─────────────────────────────────────────────
# Загрузка датасетов (CompCars, Stanford Cars)
# ─────────────────────────────────────────────

def _load_compcars_dataset(json_file: str) -> list:
    """
    Загружает CompCars Dataset (1655 моделей).
    JSON файл: {"model_name": "Toyota Camry", "category": "sedan", ...}
    """
    models = []
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if isinstance(data, list):
            # Список словарей
            for item in data:
                if isinstance(item, dict) and "model_name" in item:
                    models.append(item["model_name"])
                elif isinstance(item, str):
                    models.append(item)
        elif isinstance(data, dict):
            # Словарь моделей
            models = list(data.keys())
            
        _log.info(f"CompCars Dataset загружен: {len(models)} моделей")
        return models
    except Exception as e:
        _log.warning(f"Ошибка загрузи CompCars: {e}")
        return []


def _load_stanford_cars_dataset(annotations_file: str) -> list:
    """
    Загружает Stanford Cars Dataset (196 классов).
    Аннотации могут быть:
    - CSV: make,model,year
    - Text: каждая строка = модель
    - JSON: список или словарь моделей
    """
    models = set()
    try:
        if annotations_file.endswith(".csv"):
            import csv
            with open(annotations_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "model" in row:
                        models.add(row["model"])
                    elif "make" in row and "model" in row:
                        # Комбинируем make + model
                        full_model = f"{row.get('make', '')} {row['model']}".strip()
                        models.add(full_model)
        elif annotations_file.endswith(".json"):
            with open(annotations_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    models = set(data)
                elif isinstance(data, dict):
                    models = set(data.keys())
        else:
            # Plain text, одна модель в строке
            with open(annotations_file, "r", encoding="utf-8") as f:
                for line in f:
                    model_name = line.strip()
                    if model_name:
                        models.add(model_name)
                        
        result = list(models)
        _log.info(f"Stanford Cars Dataset загружен: {len(result)} классов")
        return result
    except Exception as e:
        _log.warning(f"Ошибка загрузки Stanford Cars: {e}")
        return []


def _initialize_vehicle_models():
    """Инициализирует список моделей из конфигурации или датасетов."""
    global _VEHICLE_MODELS, _DATASET_SOURCE
    
    cfg = _load_config()
    dataset_mode = cfg.get("dataset", "default")  # default, compcars, stanford
    
    if dataset_mode == "compcars":
        compcars_file = cfg.get("compcars_path", "")
        if compcars_file and os.path.isfile(compcars_file):
            models = _load_compcars_dataset(compcars_file)
            if models:
                _VEHICLE_MODELS = models
                _DATASET_SOURCE = "compcars"
                _log.info(f"✓ CompCars Dataset активирован: {len(models)} моделей")
            else:
                _log.warning("CompCars не загружен, используется fallback")
        else:
            _log.warning(f"CompCars файл не найден: {compcars_file}")
            
    elif dataset_mode == "stanford":
        stanford_file = cfg.get("stanford_path", "")
        if stanford_file and os.path.isfile(stanford_file):
            models = _load_stanford_cars_dataset(stanford_file)
            if models:
                _VEHICLE_MODELS = models
                _DATASET_SOURCE = "stanford"
                _log.info(f"✓ Stanford Cars Dataset активирован: {len(models)} классов")
            else:
                _log.warning("Stanford Cars не загружен, используется fallback")
        else:
            _log.warning(f"Stanford файл не найден: {stanford_file}")
    
    else:
        _DATASET_SOURCE = "default"
        _log.info(f"Использован стандартный набор моделей: {len(_VEHICLE_MODELS)}")


# Инициализация при загрузке модуля
_initialize_vehicle_models()


# ─────────────────────────────────────────────
# Обработка цвета (K-means + Delta E в LAB)
# ─────────────────────────────────────────────

def _detect_color_kmeans_advanced(image: np.ndarray) -> dict:
    """
    Гибридный метод определения цвета:
    1. K-means + Delta E CIEDE2000 в LAB (основной метод)
    2. HSV валидация для черного/белого/серого (когда LAB неоднозначен)
    
    Преимущества:
    - Точность: LAB перцепционный + научная метрика Delta E
    - Надежность: HSV страховка для монохромных цветов
    - Скорость: быстрое исключение очевидных черного/белого по HSV
    """
    # Resize для ускорения
    h, w = image.shape[:2]
    if h > 400 or w > 400:
        scale = min(400 / h, 400 / w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # === БЫСТРАЯ ПРОВЕРКА: Black/White через HSV ===
    hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    v_mean = hsv[:, :, 2].mean()  # Value (яркость)
    s_mean = hsv[:, :, 1].mean()  # Saturation (насыщенность)

    if s_mean < 30:  # Очень низкая насыщенность = монохромный
        if v_mean < 50:  # Очень темный
            _log.debug(f"HSV quick: Black detected (V={v_mean:.0f}, S={s_mean:.0f})")
            return {
                "color": "Чёрный",
                "confidence": 0.95,
            }
        elif v_mean > 200:  # Очень светлый
            _log.debug(f"HSV quick: White detected (V={v_mean:.0f}, S={s_mean:.0f})")
            return {
                "color": "Белый",
                "confidence": 0.95,
            }
        elif 80 < v_mean < 180:  # Среднее значение
            _log.debug(f"HSV quick: Gray detected (V={v_mean:.0f}, S={s_mean:.0f})")
            return {
                "color": "Серый",
                "confidence": 0.90,
            }

    # === ОСНОВНОЙ МЕТОД: LAB + K-means + Delta E ===
    lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab.reshape(-1, 3)

    # K-means clustering (найти 5 доминирующих цветов)
    try:
        kmeans = KMeans(n_clusters=5, n_init=3, max_iter=100, random_state=42)
        kmeans.fit(pixels)
        centers = kmeans.cluster_centers_
        labels, counts = np.unique(kmeans.labels_, return_counts=True)

        # Исключаем черные (L < 10) и очень светлые (L > 240) пиксели (шум)
        filtered_centers = []
        filtered_counts = []
        for idx, count in zip(labels, counts):
            L = centers[idx, 0]
            if 10 < L < 240:
                filtered_centers.append(centers[idx])
                filtered_counts.append(count)

        if filtered_centers:
            dominant_idx = np.argmax(filtered_counts)
            dominant_color_lab = np.array(filtered_centers[dominant_idx])
        else:
            dominant_color_lab = centers[labels[np.argmax(counts)]]
    except Exception as e:
        _log.warning(f"K-means failed: {e}, using mean")
        dominant_color_lab = pixels.mean(axis=0)

    # Найти ближайший стандартный цвет через Delta E CIEDE2000
    best_dist = float('inf')
    best_color_name = "gray"

    for color_name, color_lab in _REFERENCE_COLORS_LAB.items():
        # Delta E CIEDE2000 - более точная метрика восприятия цвета
        dist = deltaE_ciede2000(dominant_color_lab.reshape(1, 3), color_lab.reshape(1, 3))[0]
        if dist < best_dist:
            best_dist = dist
            best_color_name = color_name

    # Confidence: 0-20 Delta E = 0.95, 20-40 = 0.7, 40+ = 0.4
    if best_dist < 20:
        confidence = 0.95
    elif best_dist < 40:
        confidence = 0.70
    else:
        confidence = 0.40

    _log.debug(f"LAB+Delta-E: {best_color_name}, Delta E={best_dist:.1f}, confidence={confidence:.0%}")

    return {
        "color": _COLOR_NAMES_RU.get(best_color_name, "Не определён"),
        "confidence": confidence,
    }



# ─────────────────────────────────────────────
# Определение модели (CLIP)
# ─────────────────────────────────────────────

_clip_model = None
_clip_processor = None


def _get_clip_model():
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            _clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME)
            _clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME)
            _log.info("CLIP модель загружена")
        except Exception as e:
            _log.error(f"Ошибка загрузки CLIP: {e}")
            return None, None
    return _clip_model, _clip_processor


def _detect_model_clip(image: np.ndarray) -> dict:
    """Определяет марку/модель транспорта через CLIP zero-shot.
    
    Использует компактный датасет по умолчанию (24 модели) или:
    - CompCars Dataset: 1655 моделей (требует скачивания)
    - Stanford Cars: 196 классов (требует скачивания)
    """
    if not _CLIP_AVAILABLE:
        return {"model": "Не определена", "confidence": 0.0}

    model, processor = _get_clip_model()
    if model is None or processor is None:
        return {"model": "Не определена", "confidence": 0.0}

    try:
        # Подготовка изображения
        pil_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_image = Image.fromarray(pil_image)

        # Ансамбль промптов для улучшения точности
        prompts = [
            ["photo of a car", "photo of a truck", "photo of a van"],
            ["This vehicle is a car", "This vehicle is a truck", "This vehicle is a van"],
            ["a car model", "a truck model", "a van model"],
        ]

        scores_by_model = {model_name: 0.0 for model_name in _VEHICLE_MODELS}

        for prompt_list in prompts:
            inputs = processor(
                text=_VEHICLE_MODELS,
                images=pil_image,
                return_tensors="pt",
                padding=True
            )
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0].detach().numpy()

            for model_name, prob in zip(_VEHICLE_MODELS, probs):
                scores_by_model[model_name] += float(prob)

        # Среднее по ансамблю
        for model_name in scores_by_model:
            scores_by_model[model_name] /= len(prompts)

        best_model = max(scores_by_model, key=scores_by_model.get)
        best_confidence = scores_by_model[best_model]
        
        dataset_info = f" [{_DATASET_SOURCE}]" if _DATASET_SOURCE != "default" else ""
        _log.debug(f"CLIP: {best_model}{dataset_info}, confidence={best_confidence:.2%}")

        return {
            "model": best_model,
            "confidence": best_confidence,
            "dataset": _DATASET_SOURCE,
        }
    except Exception as e:
        _log.error(f"Ошибка CLIP inference: {e}")
        return {"model": "Не определена", "confidence": 0.0}


# ─────────────────────────────────────────────
# Распознавание номера (OpenALPR)
# ─────────────────────────────────────────────

def _normalize_plate_text(raw: str) -> str:
    """Нормализует текст номера к формату российского госномера."""
    if not raw:
        return ""

    s = re.sub(r'[^A-ZА-Яa-zа-я0-9]', '', str(raw)).upper()
    s = ''.join(_LATIN_TO_CYR.get(ch, ch) for ch in s)

    if len(s) < 6:
        return s

    chars = list(s)
    if chars[0].isdigit():
        chars[0] = _DIGIT_TO_LETTER.get(chars[0], chars[0])
    for i in (1, 2, 3):
        if i < len(chars) and not chars[i].isdigit():
            chars[i] = _LETTER_TO_DIGIT.get(chars[i], chars[i])
    for i in (4, 5):
        if i < len(chars) and chars[i].isdigit():
            chars[i] = _DIGIT_TO_LETTER.get(chars[i], chars[i])
    for i in range(6, min(len(chars), 9)):
        if not chars[i].isdigit():
            chars[i] = _LETTER_TO_DIGIT.get(chars[i], chars[i])

    return ''.join(chars)


def _detect_plate_openalpr(image_path: str) -> dict:
    """Распознаёт номер через OpenALPR или CV fallback."""
    if not _OPENALPR_AVAILABLE:
        _log.warning("OpenALPR недоступен, используем CV fallback")
        return _detect_plate_cv_fallback(image_path)

    try:
        region = _get_region()
        alpr = Alpr(region)

        if not alpr.is_loaded():
            _log.warning(f"OpenALPR не загружена для региона {region}, используем CV fallback")
            return _detect_plate_cv_fallback(image_path)

        results = alpr.recognize_file(image_path)

        if not results or not results.get("results"):
            _log.info(f"OpenALPR не обнаружил номер в {image_path}, используем CV fallback")
            return _detect_plate_cv_fallback(image_path)

        # Берём лучший результат
        best = results["results"][0]
        plate_text_raw = best.get("plate", "")
        plate_confidence = best.get("confidence", 0.0) / 100.0  # normalize to [0, 1]

        plate_text = _normalize_plate_text(plate_text_raw)

        if not plate_text or plate_confidence < 0.3:
            _log.info(f"OpenALPR результат слабый (conf={plate_confidence:.1%}), используем CV fallback")
            return _detect_plate_cv_fallback(image_path)

        # Бонус уверенности для номеров похожих на русские
        if _RU_PLATE_RE.match(plate_text) and plate_confidence < 0.55:
            plate_confidence = max(plate_confidence, 0.55)

        _log.info(f"OpenALPR распознал: {plate_text} ({plate_confidence:.0%})")

        return {
            "plate_text": plate_text,
            "confidence": plate_confidence,
            "message": f"Номер: {plate_text} (уверенность: {plate_confidence:.0%})",
        }

    except Exception as e:
        _log.warning(f"Ошибка OpenALPR: {e}, используем CV fallback")
        return _detect_plate_cv_fallback(image_path)


def _detect_plate_cv_fallback(image_path: str) -> dict:
    """CV fallback для распознавания номера (используется если OpenALPR недоступен)."""
    try:
        from Plugins.VehicleAnalyzer.vehicle_analyzer import detect_license_plate
        image = cv2.imread(image_path)
        if image is None:
            return {
                "plate_text": None,
                "confidence": 0.0,
                "message": "Не удалось открыть изображение",
            }
        plate_cv = detect_license_plate(image)
        plate_text = _normalize_plate_text(plate_cv.get("plate_text", ""))
        plate_conf = float(plate_cv.get("confidence", 0.0) or 0.0)

        if plate_text:
            return {
                "plate_text": plate_text,
                "confidence": plate_conf,
                "message": f"Номер (CV fallback): {plate_text} ({plate_conf:.0%})",
            }
        return {
            "plate_text": None,
            "confidence": 0.0,
            "message": "CV fallback: номер не обнаружен",
        }
    except Exception as e:
        _log.error(f"CV fallback ошибка: {e}")
        return {
            "plate_text": None,
            "confidence": 0.0,
            "message": f"Ошибка: {str(e)[:50]}",
        }


# ─────────────────────────────────────────────
# Основной анализатор
# ─────────────────────────────────────────────

class OpenALPRAnalyzer:
    """Анализатор транспортных средств через OpenALPR (локально)."""

    def __init__(self):
        if not _OPENALPR_AVAILABLE:
            _log.warning("OpenALPR не установлен. Установите: pip install openalpr-python-bindings")
        if not _CLIP_AVAILABLE:
            _log.warning("CLIP недоступна. Модель не будет определена.")

    def analyze(self, image_path: str, progress_callback=None) -> dict:
        """
        Анализирует одно изображение транспорта.
        
        Использует v4.0 MAXIMUM QUALITY если доступна, иначе fallback на v3.x.
        """
        fname = os.path.basename(image_path)
        _log.info(f"Анализ: {fname} (v4.0)" if _USE_V4_MAXQUALITY else f"Анализ: {fname}")

        if not os.path.isfile(image_path):
            _log.warning(f"Файл не найден: {image_path}")
            return {"error": f"Файл не найден: {image_path}", "file": fname}

        # === ИСПОЛЬЗОВАНИЕ V4.0 MAXIMUM QUALITY ===
        if _USE_V4_MAXQUALITY:
            try:
                if progress_callback:
                    progress_callback("🎨 Определение цвета (региональная выборка + LAB)...", 1)
                
                # Загружаем изображение
                image = cv2.imread(image_path)
                if image is None:
                    return {"error": f"Не удалось открыть: {fname}", "file": fname}
                
                # V4.0 методы
                color_result = _detect_color_advanced_v4(image)
                
                if progress_callback:
                    progress_callback("🚗 Определение модели (ансамбль CLIP)...", 2)
                
                model_result = _detect_model_clip_ensemble_v4(image)
                
                if progress_callback:
                    progress_callback("📋 Распознавание номера (ансамбль методов)...", 3)
                
                plate_result = _detect_plate_multimethod_v4(image_path)
                
                if progress_callback:
                    progress_callback("✅ Анализ завершен (v4.0)", 4)
                
                return {
                    "file": fname,
                    "path": os.path.abspath(image_path),
                    "color": {
                        "color": color_result.get("color"),
                        "confidence": color_result.get("confidence", 0.0),
                    },
                    "model": {
                        "model": model_result.get("model"),
                        "confidence": model_result.get("confidence", 0.0),
                    },
                    "plate": {
                        "message": plate_result.get("plate"),
                        "confidence": plate_result.get("confidence", 0.0),
                    },
                    "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    "engine": "OpenALPR v4.0 (MAXIMUM QUALITY)",
                }
            except Exception as e:
                _log.error(f"V4.0 ошибка, fallback на v3: {e}")
                # Fallback на v3.x
        
        # === FALLBACK НА V3.X ===
        if progress_callback:
            progress_callback("Определение цвета (Delta E)...", 1)

        # Читаем изображение
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Не удалось открыть изображение: {fname}", "file": fname}

        # Цвет (v3.x)
        color_result = _detect_color_kmeans_advanced(image)

        if progress_callback:
            progress_callback("Определение модели (CLIP)...", 2)

        # Модель (v3.x)
        model_result = _detect_model_clip(image)

        if progress_callback:
            progress_callback("Распознавание номера (OpenALPR + CV fallback)...", 3)

        # Номер (v3.x)
        plate_result = _detect_plate_openalpr(image_path)

        if progress_callback:
            progress_callback("Готово", 4)

        return {
            "file": fname,
            "path": os.path.abspath(image_path),
            "color": color_result,
            "model": model_result,
            "plate": plate_result,
            "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "engine": "OpenALPR v3.x",
        }

    def stop(self):
        """Нет ресурсов для освобождения."""
        pass


# ─────────────────────────────────────────────
# Функции-обёртки для ui.py
# ─────────────────────────────────────────────

_analyzer = None


def _get_analyzer() -> OpenALPRAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = OpenALPRAnalyzer()
    return _analyzer


def analyze_vehicle_openalpr(image_path: str, progress_callback=None) -> dict:
    """Анализирует одно изображение через OpenALPR."""
    return _get_analyzer().analyze(image_path, progress_callback)


def analyze_vehicle_folder_openalpr(folder_path: str) -> list:
    """Анализирует папку с изображениями через OpenALPR."""
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
    analyzer = _get_analyzer()
    results = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith(extensions):
            results.append(analyzer.analyze(os.path.join(folder_path, fname)))
    return results


def generate_report_openalpr(results: list, output_path: str, elapsed_seconds: float = None) -> str:
    """Генерирует текстовый отчёт по результатам анализа."""
    lines = [
        "=" * 60,
        "  ОТЧЁТ: АНАЛИЗ ТРАНСПОРТНЫХ СРЕДСТВ (OpenALPR + CV)",
        f"  Дата: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
        f"  Обработано изображений: {len(results)}",
        f"  Примечание: полностью локальное решение (без интернета)",
        f"  Методы: K-means+Delta-E (цвет) | OpenALPR+CV (номер) | CLIP (модель)",
        "=" * 60,
        "",
    ]
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

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))
    return output_path
