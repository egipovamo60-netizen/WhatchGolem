"""
Тест QwenVehicleAnalyzer на одном изображении с замером времени.
"""
import os
import sys
import time
import multiprocessing

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

IMAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "Test foto", "Foto4.jpg"))

print(f"Изображение: {IMAGE_PATH}")
print(f"Существует: {os.path.isfile(IMAGE_PATH)}")
print()

# --- Шаг 1: Импорт ---
t0 = time.time()
print("[1/5] Импортируем модуль...", flush=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Plugins.QwenVehicleAnalyzer.qwen_vehicle_analyzer import QwenVehicleAnalyzer
print(f"      -> {time.time() - t0:.1f}с")

# --- Шаг 2: Создание объекта ---
t0 = time.time()
print("[2/5] Создаём QwenVehicleAnalyzer...", flush=True)
analyzer = QwenVehicleAnalyzer()
print(f"      -> {time.time() - t0:.1f}с")

# --- Шаг 3: Загрузка модели ---
t0 = time.time()
print("[3/5] Загружаем модель (может занять 1-2 минуты)...", flush=True)
try:
    analyzer._load_model()
    print(f"      -> {time.time() - t0:.1f}с  OK")
except Exception as e:
    print(f"      -> ОШИБКА загрузки модели: {e}")
    sys.exit(1)

# --- Шаг 4: Определение цвета ---
t0 = time.time()
print("[4/5] Определяем цвет автомобиля...", flush=True)
try:
    color = analyzer.detect_color(IMAGE_PATH)
    print(f"      -> {time.time() - t0:.1f}с  Результат: {color}")
except Exception as e:
    print(f"      -> ОШИБКА: {e}")

# --- Шаг 5: Определение марки ---
t0 = time.time()
print("[5/5] Определяем марку/модель...", flush=True)
try:
    model_info = analyzer.detect_model(IMAGE_PATH)
    print(f"      -> {time.time() - t0:.1f}с  Результат: {model_info}")
except Exception as e:
    print(f"      -> ОШИБКА: {e}")

print("\nТест завершён.")
