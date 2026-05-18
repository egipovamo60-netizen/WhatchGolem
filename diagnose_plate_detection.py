#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Диагностика проблемы определения номера машины
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Unicode fix для Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from logger import get_logger
_log = get_logger("PlateDetectionDiag")

def diagnose_plate_detection():
    """Диагностирует проблемы с определением номера."""
    
    print("\n" + "="*70)
    print("ДИАГНОСТИКА: ОПРЕДЕЛЕНИЕ НОМЕРА МАШИНЫ")
    print("="*70)
    
    issues = []
    
    # 1. Проверка OpenALPR
    print("\n📌 1. Проверка OpenALPR...")
    try:
        from openalpr import Alpr
        print("   ✅ OpenALPR установлен")
        
        # Проверка конфига
        config_path = os.path.join(os.path.dirname(__file__), "Plugins/OpenALPRAnalyzer/openalpr.conf")
        if os.path.exists(config_path):
            print(f"   ✅ openalpr.conf найден: {config_path}")
        else:
            print(f"   ❌ openalpr.conf НЕ НАЙДЕН: {config_path}")
            issues.append("OpenALPR конфиг отсутствует")
    except ImportError:
        print("   ⚠️  OpenALPR не установлен (опционально)")
        print("       Установка: pip install openalpr-python-bindings")
    except Exception as e:
        print(f"   ❌ Ошибка OpenALPR: {e}")
        issues.append(f"OpenALPR ошибка: {e}")
    
    # 2. Проверка EasyOCR
    print("\n📌 2. Проверка EasyOCR...")
    try:
        import easyocr
        print("   ✅ EasyOCR установлен")
        
        # Попытка загрузить reader
        try:
            _log.info("Загрузка EasyOCR reader для русского/английского...")
            reader = easyocr.Reader(['ru', 'en'], gpu=False)
            print("   ✅ EasyOCR reader загружен успешно")
        except Exception as e:
            print(f"   ❌ Ошибка загрузки reader: {e}")
            issues.append(f"EasyOCR reader error: {e}")
    except ImportError:
        print("   ❌ EasyOCR НЕ УСТАНОВЛЕН")
        print("       Установка: pip install easyocr")
        issues.append("EasyOCR не установлен")
    
    # 3. Проверка VehicleAnalyzer fallback
    print("\n📌 3. Проверка VehicleAnalyzer (fallback)...")
    try:
        from Plugins.VehicleAnalyzer import vehicle_analyzer
        if hasattr(vehicle_analyzer, '_detect_license_plate'):
            print("   ✅ VehicleAnalyzer.license_plate доступен (fallback OK)")
        else:
            print("   ⚠️  VehicleAnalyzer не имеет функции распознавания номера")
            issues.append("VehicleAnalyzer не имеет _detect_license_plate")
    except ImportError:
        print("   ❌ VehicleAnalyzer не импортирован")
        issues.append("VehicleAnalyzer не доступен")
    except Exception as e:
        print(f"   ❌ Ошибка VehicleAnalyzer: {e}")
        issues.append(f"VehicleAnalyzer error: {e}")
    
    # 4. Проверка v4.0 модуля
    print("\n📌 4. Проверка v4.0 модуля...")
    try:
        from Plugins.OpenALPRAnalyzer import openalpr_analyzer_v4_maxquality as v4
        print("   ✅ v4.0 модуль импортирован")
        
        # Проверка функций
        if hasattr(v4, '_detect_plate_multimethod_v4'):
            print("   ✅ _detect_plate_multimethod_v4 доступна")
        else:
            print("   ❌ _detect_plate_multimethod_v4 отсутствует!")
            issues.append("_detect_plate_multimethod_v4 не найдена")
        
        # Проверка константы _OPENALPR_CONFIG
        if hasattr(v4, '_OPENALPR_CONFIG'):
            print(f"   ✅ _OPENALPR_CONFIG определена: {v4._OPENALPR_CONFIG}")
        else:
            print("   ❌ _OPENALPR_CONFIG НЕ ОПРЕДЕЛЕНА")
            issues.append("_OPENALPR_CONFIG не определена в v4.0")
            
    except Exception as e:
        print(f"   ❌ Ошибка v4.0: {e}")
        issues.append(f"v4.0 ошибка: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Проверка конфигурации
    print("\n📌 5. Проверка конфигурации OpenALPR...")
    config_path = os.path.join(os.path.dirname(__file__), "Plugins/OpenALPRAnalyzer/openalpr.conf")
    if os.path.exists(config_path):
        print(f"   ✅ openalpr.conf существует")
        try:
            with open(config_path, 'r') as f:
                content = f.read()
                print(f"   ℹ️  Размер файла: {len(content)} байт")
                if 'country' in content or 'region' in content:
                    print("   ✅ Конфиг содержит параметры région/country")
                else:
                    print("   ⚠️  Конфиг может быть пуст или неправильный")
        except Exception as e:
            print(f"   ❌ Ошибка чтения openalpr.conf: {e}")
    else:
        print(f"   ❌ openalpr.conf НЕ НАЙДЕН: {config_path}")
        print("   ℹ️  Создайте файл openalpr.conf с содержимым config_path = /path/to/models")
        issues.append("openalpr.conf отсутствует или неправильно расположен")
    
    # 6. Тест v4.0 функции на изображении
    print("\n📌 6. Тест функции определения номера...")
    from pathlib import Path
    test_image = None
    for test_dir in ["Test foto/One", "Test video", "test_folder"]:
        test_path = Path(test_dir)
        if test_path.exists():
            for img_file in test_path.glob("*.jpg"):
                test_image = str(img_file)
                break
        if test_image:
            break
    
    if test_image:
        print(f"   📷 Тестовое изображение: {test_image}")
        try:
            from Plugins.OpenALPRAnalyzer import openalpr_analyzer_v4_maxquality as v4
            result = v4._detect_plate_multimethod_v4(test_image)
            print(f"   ℹ️  Результат: {result}")
            if result.get('plate') and result['plate'] != "Не определен":
                print(f"   ✅ Номер определен: {result['plate']} ({result.get('confidence', 0):.0%})")
            else:
                print(f"   ❌ Номер не определен (результат пуст)")
                issues.append("v4.0 не определяет номер на тестовом изображении")
        except Exception as e:
            print(f"   ❌ Ошибка при тестировании: {e}")
            import traceback
            traceback.print_exc()
            issues.append(f"Ошибка теста: {e}")
    else:
        print("   ⚠️  Тестовое изображение не найдено (невозможно протестировать)")
    
    # Итоги
    print("\n" + "="*70)
    print("ИТОГИ")
    print("="*70)
    
    if not issues:
        print("\n✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!\n")
        print("v4.0 MAXIMUM QUALITY должен работать корректно.")
        print("Если номер все равно не определяется:")
        print("  1. Проверьте качество изображения (размер, освещение, угол)")
        print("  2. Убедитесь что номер виден на изображении")
        print("  3. Посмотрите логи (Logs/ папка) для деталей")
        print("  4. Попробуйте установить OpenALPR бинарный пакет с сайта проекта")
        return True
    else:
        print(f"\n❌ НАЙДЕНО {len(issues)} ПРОБЛЕМ:\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        
        print("\n📋 РЕКОМЕНДАЦИИ:")
        if "не установлен" in str(issues).lower():
            print("  • Установите недостающий пакет:")
            print("    pip install openalpr-python-bindings easyocr")
        
        if "конфиг" in str(issues).lower():
            print("  • Создайте openalpr.conf в Plugins/OpenALPRAnalyzer/")
            print("    Пример содержимого:")
            print("    ; OpenALPR config")
            print("    [global]")
            print("    country = ru")
            print("    topn = 5")
            print("    analysis_count = 2")
        
        if "_OPENALPR_CONFIG" in str(issues):
            print("  • Уже исправлено в v4.0 (перезагрузите приложение)")
        
        return False

if __name__ == "__main__":
    success = diagnose_plate_detection()
    print("="*70 + "\n")
    sys.exit(0 if success else 1)
