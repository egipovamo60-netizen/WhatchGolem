"""
Тестовый скрипт для проверки качества OpenALPR плагина.
Помогает диагностировать проблемы с определением цвета и номера.

Использование:
    python test_openalpr_plugin.py <path_to_image.jpg>

Или для папки:
    python test_openalpr_plugin.py <path_to_folder>
"""

import os
import sys
import json
from pathlib import Path

# Добавляем корневую папку в path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Plugins.OpenALPRAnalyzer import analyze_vehicle_openalpr, analyze_vehicle_folder_openalpr
from logger import get_logger

_log = get_logger("TestOpenALPR")


def test_single_image(image_path):
    """Тестирует анализ одного изображения с подробным выводом."""
    print(f"\n{'='*70}")
    print(f"Тестирование: {os.path.basename(image_path)}")
    print(f"{'='*70}\n")

    result = analyze_vehicle_openalpr(image_path)

    if "error" in result:
        print(f"❌ ОШИБКА: {result['error']}\n")
        return False

    # Цвет
    color = result.get("color", {})
    color_name = color.get("color", "?")
    color_conf = color.get("confidence", 0.0)
    print(f"🎨 ЦВЕТ:")
    print(f"   Название:    {color_name}")
    print(f"   Уверенность: {color_conf:.0%}")
    if color_conf < 0.5:
        print(f"   ⚠️  НИЗКАЯ УВЕРЕННОСТЬ (< 50%)")
    elif color_conf >= 0.9:
        print(f"   ✅ ВЫСОКАЯ УВЕРЕННОСТЬ (>= 90%)")
    print()

    # Модель
    model = result.get("model", {})
    model_name = model.get("model", "?")
    model_conf = model.get("confidence", 0.0)
    print(f"🚗 МОДЕЛЬ:")
    print(f"   Название:    {model_name}")
    print(f"   Уверенность: {model_conf:.0%}")
    if model_conf < 0.3:
        print(f"   ⚠️  ОЧЕНЬ НИЗКАЯ УВЕРЕННОСТЬ (< 30%)")
    print()

    # Номер
    plate = result.get("plate", {})
    plate_text = plate.get("plate_text", "?")
    plate_message = plate.get("message", "?")
    plate_conf = plate.get("confidence", 0.0)
    print(f"📝 НОМЕР:")
    print(f"   Текст:       {plate_text}")
    print(f"   Сообщение:   {plate_message}")
    print(f"   Уверенность: {plate_conf:.0%}")

    # Проверка на CV fallback
    if "CV fallback" in plate_message or "cv fallback" in plate_message.lower():
        print(f"   ℹ️  OpenALPR недоступен, используется CV fallback")
    if "OpenALPR" in plate_message:
        print(f"   ℹ️  OpenALPR успешно распознал номер")

    if plate_text is None or plate_conf < 0.3:
        print(f"   ⚠️  НОМЕР НЕ РАСПОЗНАН или очень низкая уверенность")
    print()

    # Общая информация
    engine = result.get("engine", "?")
    timestamp = result.get("timestamp", "?")
    print(f"ℹ️  ИНФОРМАЦИЯ:")
    print(f"   Engine:      {engine}")
    print(f"   Timestamp:   {timestamp}")
    print(f"   Path:        {result.get('path', '?')}")
    print()

    print(f"{'='*70}\n")
    return True


def test_folder(folder_path):
    """Тестирует все изображения в папке."""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]

    if not files:
        print(f"❌ В папке {folder_path} нет изображений")
        return

    print(f"\nНайдено изображений: {len(files)}")
    results = analyze_vehicle_folder_openalpr(folder_path)

    successful = 0
    for result in results:
        if "error" not in result:
            successful += 1

    print(f"✅ Успешно обработано: {successful}/{len(files)}\n")

    # Статистика
    colors = {}
    models = {}
    plates_detected = 0

    for result in results:
        if "error" in result:
            continue

        # Цвета
        color_name = result.get("color", {}).get("color", "Unknown")
        colors[color_name] = colors.get(color_name, 0) + 1

        # Модели
        model_name = result.get("model", {}).get("model", "Unknown")
        models[model_name] = models.get(model_name, 0) + 1

        # Номера
        if result.get("plate", {}).get("plate_text"):
            plates_detected += 1

    print("📊 СТАТИСТИКА:")
    print(f"\n🎨 Цвета:")
    for color, count in sorted(colors.items(), key=lambda x: -x[1]):
        print(f"   {color}: {count}")

    print(f"\n🚗 Модели (топ 5):")
    for model, count in sorted(models.items(), key=lambda x: -x[1])[:5]:
        print(f"   {model}: {count}")

    print(f"\n📝 Номера обнаружены: {plates_detected}/{len(files)} ({100*plates_detected/len(files):.0f}%)\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Примеры:")
        print("  python test_openalpr_plugin.py car.jpg")
        print("  python test_openalpr_plugin.py ./images/")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isfile(path):
        test_single_image(path)
    elif os.path.isdir(path):
        test_folder(path)
    else:
        print(f"❌ Файл или папка не найдены: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
