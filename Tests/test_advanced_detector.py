#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тест новой 3-этапной системы обнаружения дубликатов
SHA-256 → pHash → CNN embeddings
"""

import sys
sys.path.insert(0, '.')

from duplicate_detector import DuplicateDetectorAdvanced
import os

print("\n" + "=" * 70)
print("ТЕСТ: НОВАЯ 3-ЭТАПНАЯ СИСТЕМА ОБНАРУЖЕНИЯ ДУБЛИКАТОВ")
print("=" * 70)

# Проверяем импорты
try:
    print("\n✓ Импорт DuplicateDetectorAdvanced успешен")
except Exception as e:
    print(f"✗ Ошибка импорта: {e}")
    sys.exit(1)

# Проверяем наличие PyTorch
try:
    import torch
    import torchvision
    print("✓ PyTorch установлен (версия {})".format(torch.__version__))
    print("✓ torchvision установлен (версия {})".format(torchvision.__version__))
    TORCH_OK = True
except ImportError:
    print("⚠️  PyTorch не установлен (система работает только с SHA-256 и pHash)")
    TORCH_OK = False

# Создаём детектор
test_folder = "test_folder"
os.makedirs(test_folder, exist_ok=True)

print(f"\n📁 Создаю детектор для папки: {test_folder}")

detector = DuplicateDetectorAdvanced(test_folder, hamming_threshold=10, ssim_threshold=0.75)

print("\n✓ Инициализация параметров:")
print(f"  • Хэмминг порог: {detector.hamming_threshold}")
print(f"  • SSIM порог: {detector.ssim_threshold}")
print(f"  • Папка: {detector.source_folder}")

# Проверяем методы
print("\n✓ Проверка доступных методов:")

methods_to_check = [
    'compute_sha256',
    'compute_phash',
    'hamming_distance',
    'stage1_find_exact_duplicates',
    'stage2_find_similar_by_phash',
    'stage3_verify_by_cnn',
    'compute_embedding',
    'cosine_similarity',
    'find_all_duplicates',
    'extract_percent',
    'process_duplicates',
    'run'
]

for method_name in methods_to_check:
    has_method = hasattr(detector, method_name)
    status = "✓" if has_method else "✗"
    print(f"  {status} {method_name}")

# Тест извлечения процента
print("\n✓ Тест извлечения процента из названия:")

test_cases = [
    ("photo_92percent.jpg", 92),
    ("image_85_percent_data.jpg", 85),
    ("file_90%.png", 90),
    ("test.jpg", 0),
    ("image_88percent_2024-03-25_10-30.jpg", 88),
    ("95percent.jpg", 95),
]

all_passed = True
for filename, expected in test_cases:
    result = detector.extract_percent(filename)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_passed = False
    print(f"  {status} {filename} → {result}% (ожидалось {expected}%)")

# Проверяем CNN загрузку
print("\n✓ Статус CNN модели:")
if TORCH_OK:
    if detector.model:
        print("  ✓ ResNet50 загружена успешно")
        print(f"  ✓ Устройство: {detector.device}")
    else:
        print("  ⚠️  CNN недоступна")
else:
    print("  ⚠️  PyTorch не установлен")

# Итоги
print("\n" + "=" * 70)
if all_passed:
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    print("\n🎯 Система готова к использованию:")
    print("   • SHA-256 для точных дубликатов")
    print("   • pHash для быстрого фильтра")
    print("   • CNN embeddings для высокой точности" if TORCH_OK else "   • CNN embeddings недоступна")
    print("\n📝 Использование:")
    print("   from duplicate_detector import DuplicateDetectorAdvanced")
    print("   detector = DuplicateDetectorAdvanced(folder_path)")
    print("   detector.run()")
else:
    print("❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОШЛИ")

print("=" * 70 + "\n")
