#!/usr/bin/env python3
"""Скачивает YOLOv11n модель для использования в проекте."""

import urllib.request
import os
import sys

# URL для YOLOv11n
url = "https://github.com/ultralytics/assets/releases/download/v11.0/yolov11n.pt"
path = "yolov11n.pt"

print("=" * 60)
print("Загружаю YOLOv11n модель...")
print("=" * 60)

try:
    print(f"\nЗагружаю с: {url}")
    print(f"Сохраняю в: {path}\n")
    
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100)
        print(f"\rПрогресс: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} МБ)", end="")
    
    urllib.request.urlretrieve(url, path, download_progress)
    print(f"\n\n✓ Успешно загружено в {path}")
    
except urllib.error.URLError as e:
    print(f"\n✗ Ошибка сети: {e}")
    print("\nПопытаюсь через встроенное скачивание UltraLytics...")
    from ultralytics import YOLO
    model = YOLO(path)
    print("✓ Модель инициализирована через UltraLytics")
    
except Exception as e:
    print(f"\n✗ Ошибка: {e}")
    sys.exit(1)

# Проверка
print("\nПроверка загруженной модели...")
try:
    from ultralytics import YOLO
    model = YOLO(path)
    print(f"✓ YOLOv11n готов к использованию!")
    print(f"  Параметров: ~2.6M")
    print(f"  Точность выше на ~5% чем YOLOv8n")
except Exception as e:
    print(f"✗ Ошибка при инициализации: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Готово!")
print("=" * 60)
