import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
import cv2
from Plugins.VehicleAnalyzer import vehicle_analyzer

print("="*70)
print("[TEST] Полная OCR детекция с YOLOv8/11")
print("="*70)

# Загрузим несколько тестовых изображений
test_images = [
    "Test foto/One/car_1_94percent_19-47-23.jpg",
    "Test foto/One/car_2_94percent_19-47-23.jpg",
    "Test foto/One/car_3_93percent_19-47-23.jpg",
]

for img_path_rel in test_images:
    img_path = os.path.join(_ROOT, img_path_rel)
    
    if not os.path.isfile(img_path):
        print(f"\n❌ Файл не найден: {img_path_rel}")
        continue
    
    print(f"\n[TEST] {img_path_rel}")
    print("-" * 70)
    
    # Загружаем изображение
    image = cv2.imread(img_path)
    if image is None:
        print(f"  ❌ Не удалось загрузить изображение")
        continue
    
    h, w = image.shape[:2]
    print(f"  Размер: {w}×{h} пикселей")
    
    # Шаг 1: Поиск региона авто через YOLO
    print(f"  [ШАГ 1] Поиск области автомобиля (YOLOv11)...")
    vehicle_region = vehicle_analyzer._get_vehicle_region_yolo(image)
    if vehicle_region:
        x, y, w_v, h_v = vehicle_region
        print(f"    ✅ Область авто: ({x}, {y}) размер {w_v}×{h_v}")
    else:
        print(f"    ⚠️  Регион авто не найден, будет использоваться всё изображение")
    
    # Шаг 2: Поиск кандидатов на номер
    print(f"  [ШАГ 2] Поиск кандидатов на номер (контуры)...")
    candidates = vehicle_analyzer._find_plate_candidates_enhanced(image)
    print(f"    ✅ Найдено кандидатов: {len(candidates)}")
    
    # Шаг 3: Полная детекция номера с OCR
    print(f"  [ШАГ 3] Детекция номера (OCR)...")
    result = vehicle_analyzer.detect_license_plate(image)
    
    print(f"    Результаты:")
    print(f"      • Текст номера: {result.get('plate_text', 'None')}")
    print(f"      • Уверенность OCR: {result.get('confidence', 0):.3f}")
    print(f"      • Регион номера: {result.get('region', 'None')}")
    print(f"      • Доступна OCR: {result.get('available', False)}")
    print(f"      • Сообщение: {result.get('message', 'N/A')}")
    
    if result.get('plate_text'):
        print(f"    ✅ Номер РАСПОЗНАН: {result['plate_text']}")
    else:
        print(f"    ⚠️  Номер не распознан")

print("\n" + "="*70)
print("[ИТОГ] Тестирование завершено!")
print("="*70)
