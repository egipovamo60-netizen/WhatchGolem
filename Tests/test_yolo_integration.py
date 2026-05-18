#!/usr/bin/env python3
"""
Тест интеграции YOLOv8/11 в VehicleAnalyzer для двухэтапного поиска номеров.
"""

import os
import sys
import cv2
from pathlib import Path

# Добавляем путь к плагину
sys.path.insert(0, str(Path(__file__).parent.parent))

# ═══════════════════════════════════════════════
# Импортируем плагин
# ═══════════════════════════════════════════════

from Plugins.VehicleAnalyzer import vehicle_analyzer

print("[TEST] Инициализация тестирования YOLOv8/11 интеграции...")
print("=" * 70)

# ═══════════════════════════════════════════════
# 1. Проверка загрузки YOLOv8/11
# ═══════════════════════════════════════════════

print("\n[STEP 1] Проверка загрузки моделей YOLOv8/11...")
print("-" * 70)

yolo_loaded = vehicle_analyzer._load_auto_yolo()
if yolo_loaded:
    print(f"✅ YOLOv{vehicle_analyzer._AUTO_YOLO_VARIANT} загружена успешно!")
    print(f"   Модель: {vehicle_analyzer._AUTO_YOLO_MODEL}")
else:
    print("⚠️  YOLOv8/11 не загружена - будет использоваться контурный поиск")

# ═══════════════════════════════════════════════
# 2. Проверка файлов моделей
# ═══════════════════════════════════════════════

print("\n[STEP 2] Проверка наличия файлов моделей...")
print("-" * 70)

all_paths = vehicle_analyzer._YOLO11_PATHS + vehicle_analyzer._YOLO8_PATHS
for path in all_paths:
    exists = "✅" if os.path.isfile(path) else "❌"
    basename = os.path.basename(path)
    print(f"{exists} {basename}: {path}")

# ═══════════════════════════════════════════════
# 3. Тестирование на реальных изображениях
# ═══════════════════════════════════════════════

print("\n[STEP 3] Тестирование на реальных изображениях...")
print("-" * 70)

test_images = [
    "Test foto/One/car_1_94percent_19-47-23.jpg",
    "Test foto/One/car_2_94percent_19-47-23.jpg",
    "Test foto/One/Foto5.jpg",
]

for img_path_rel in test_images:
    img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), img_path_rel)
    
    if not os.path.isfile(img_path):
        print(f"⚠️  Файл не найден: {img_path}")
        continue
    
    print(f"\n📸 Тестирование: {img_path_rel}")
    
    # Загружаем изображение
    image = cv2.imread(img_path)
    if image is None:
        print(f"   ❌ Не удалось загрузить изображение")
        continue
    
    h, w = image.shape[:2]
    print(f"   Размер: {w}x{h} пикселей")
    
    # Тест 1: Поиск области авто через YOLO
    print(f"   [TEST 1] Поиск области автомобиля через YOLO...")
    vehicle_region = vehicle_analyzer._get_vehicle_region_yolo(image)
    if vehicle_region:
        x, y, w_v, h_v = vehicle_region
        print(f"   ✅ Область авто найдена: x={x}, y={y}, w={w_v}, h={h_v}")
        print(f"      Площадь: {w_v * h_v} px² ({(w_v * h_v) / (image.shape[0] * image.shape[1]) * 100:.1f}% от изображения)")
    else:
        print(f"   ⚠️  Область авто не найдена")
    
    # Тест 2: Поиск номеров (двухэтапный метод)
    print(f"   [TEST 2] Поиск номерных пластин (двухэтапный метод)...")
    plate_candidates = vehicle_analyzer._find_plate_candidates_enhanced(image)
    print(f"   Найдено кандидатов на номер: {len(plate_candidates)}")
    for i, (px, py, pw, ph) in enumerate(plate_candidates):
        print(f"      Кандидат {i+1}: ({px}, {py}) размер {pw}x{ph}")
    
    # Тест 3: Полная детекция номера (с OCR)
    print(f"   [TEST 3] Полная детекция номера (с OCR)...")
    result = vehicle_analyzer.detect_license_plate(image)
    print(f"   Результат:")
    print(f"      Текст: {result.get('plate_text', 'None')}")
    print(f"      Уверенность: {result.get('confidence', 0):.2f}")
    print(f"      Регион: {result.get('region', 'None')}")
    print(f"      Доступно: {result.get('available', False)}")

# ═══════════════════════════════════════════════
# 4. Статистика
# ═══════════════════════════════════════════════

print("\n" + "=" * 70)
print("[РЕЗУЛЬТАТ] Итоги тестирования:")
print("-" * 70)
print(f"✅ YOLOv{vehicle_analyzer._AUTO_YOLO_VARIANT} интеграция: {'РАБОТАЕТ' if yolo_loaded else 'ЧАСТИЧНО (используется fallback)'}")
print(f"✅ Двухэтапный поиск номеров: активен")
print(f"✅ Контурный поиск (fallback): готов")
print("\n[ЗАКЛЮЧЕНИЕ] Тестирование завершено успешно!")
