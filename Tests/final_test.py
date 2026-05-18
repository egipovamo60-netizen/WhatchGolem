#!/usr/bin/env python3
"""Финальный интеграционный тест"""

from duplicate_detector import detect_and_remove_duplicates
import os

print("🧪 ФИНАЛЬНЫЙ ИНТЕГРАЦИОННЫЙ ТЕСТ")
print("=" * 70)
print()

# Очищаем тестовую папку от предыдущих результатов
import shutil
if os.path.exists('test_duplicates/QunicObject'):
    shutil.rmtree('test_duplicates/QunicObject')

# Пересоздаём тестовые изображения для чистого теста
from PIL import Image

os.makedirs('test_duplicates', exist_ok=True)

# Изображение 1 - красное (100%)
img1 = Image.new('RGB', (100, 100), color=(255, 0, 0))
img1.save('test_duplicates/photo_100percent.jpg')

# Изображение 2 - то же, дубликат (90%)
img2 = Image.new('RGB', (100, 100), color=(255, 0, 0))
img2.save('test_duplicates/photo_90percent.jpg')

# Изображение 3 - разное (уникальное)
img3 = Image.new('RGB', (100, 100), color=(0, 255, 0))
img3.save('test_duplicates/photo_green.jpg')

print("✓ Тестовые данные подготовлены: 3 изображения")
print()

# Вызываем функцию
print("🔍 Запускаю обнаружение дубликатов...")
print()

result = detect_and_remove_duplicates('test_duplicates')

print()
print("=" * 70)
print("✅ РЕЗУЛЬТАТ ФУНКЦИИ:")
print("=" * 70)
print()

# Проверяем все необходимые ключи
expected_keys = {
    'duplicates_deleted': int,
    'duplicate_groups_processed': int,
    'unique_images_moved': int,
    'output_folder': str
}

all_ok = True

for key, expected_type in expected_keys.items():
    exists = key in result
    if exists:
        value = result[key]
        correct_type = isinstance(value, expected_type)
        status = "✅" if correct_type else "❌"
        type_name = type(value).__name__
        print(f"{status} result['{key}']")
        print(f"   └─ Значение: {value}")
        print(f"   └─ Тип: {type_name} (ожидается: {expected_type.__name__})")
        if not correct_type:
            all_ok = False
    else:
        print(f"❌ result['{key}'] - ОТСУТСТВУЕТ!")
        all_ok = False
    print()

print("=" * 70)
print("📊 ПРОВЕРКА ЛОГИКИ:")
print("=" * 70)
print()

# Проверяем логику результатов
checks = [
    ("Удалено >= 0", result['duplicates_deleted'] >= 0),
    ("Групп >= 0", result['duplicate_groups_processed'] >= 0),
    ("Перемещено >= 0", result['unique_images_moved'] >= 0),
    ("Папка существует", os.path.exists(result['output_folder'])),
    ("Папка не пуста", len(os.listdir(result['output_folder'])) > 0),
]

for check_name, check_result in checks:
    status = "✅" if check_result else "❌"
    print(f"{status} {check_name}")
    if not check_result:
        all_ok = False

print()
print("=" * 70)
print("📝 ИТОГОВЫЙ СТАТУС:")
print("=" * 70)
print()

if all_ok:
    print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
    print()
    print("📦 СИСТЕМА ГОТОВА К ИСПОЛЬЗОВАНИЮ")
    print()
    print("Структура результата корректна и совместима с UI.")
else:
    print("❌ ОБНАРУЖЕНЫ ПРОБЛЕМЫ!")

print()
print("=" * 70)
