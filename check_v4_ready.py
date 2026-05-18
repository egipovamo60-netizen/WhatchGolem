#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрая проверка готовности v4.0 MAXIMUM QUALITY
(без загрузки моделей - только проверка компонентов)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Управление выводом Unicode в Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_v4_components():
    """Проверяет наличие и корректность v4.0 компонентов"""
    
    print("\n" + "="*70)
    print("✓ v4.0 MAXIMUM QUALITY - ПРОВЕРКА КОМПОНЕНТОВ")
    print("="*70)
    
    results = {}
    
    # 1. Проверка основного модуля
    print("\n📌 1. Проверка основного модуля OpenALPR Analyzer...")
    try:
        from Plugins.OpenALPRAnalyzer import openalpr_analyzer
        results['main_module'] = '✅ OK'
        print("   ✅ openalpr_analyzer.py: OK")
    except Exception as e:
        results['main_module'] = f'❌ ОШИБКА: {e}'
        print(f"   ❌ ОШИБКА: {e}")
    
    # 2. Проверка v4.0 модуля
    print("\n📌 2. Проверка v4.0 MAXIMUM QUALITY модуля...")
    try:
        from Plugins.OpenALPRAnalyzer import openalpr_analyzer_v4_maxquality as v4
        results['v4_module'] = '✅ OK'
        print("   ✅ openalpr_analyzer_v4_maxquality.py: OK")
    except Exception as e:
        results['v4_module'] = f'❌ ОШИБКА: {e}'
        print(f"   ❌ ОШИБКА: {e}")
        return False
    
    # 3. Проверка v4.0 функций
    print("\n📌 3. Проверка функций v4.0...")
    v4_functions = [
        '_detect_color_advanced_v4',
        '_detect_model_clip_ensemble_v4',
        '_detect_plate_multimethod_v4',
        '_get_clip_models_ensemble',
        'analyze_vehicle',
        'analyze_vehicle_folder',
        'generate_report'
    ]
    
    missing_funcs = []
    for func_name in v4_functions:
        if hasattr(v4, func_name):
            print(f"   ✅ {func_name}: OK")
            results[f'func_{func_name}'] = '✅'
        else:
            print(f"   ❌ {func_name}: MISSING")
            missing_funcs.append(func_name)
            results[f'func_{func_name}'] = '❌ MISSING'
    
    if missing_funcs:
        print(f"\n❌ Ошибка: Отсутствуют функции: {missing_funcs}")
        return False
    
    # 4. Проверка v3.x fallback функций
    print("\n📌 4. Проверка v3.x FALLBACK функций...")
    import Plugins.OpenALPRAnalyzer.openalpr_analyzer as oaa_module
    
    v3_functions = [
        '_detect_color_kmeans_advanced',
        '_detect_model_clip',
        '_detect_plate_openalpr'
    ]
    
    missing_v3 = []
    for func_name in v3_functions:
        if hasattr(oaa_module, func_name):
            print(f"   ✅ {func_name}: OK (fallback доступен)")
            results[f'v3_func_{func_name}'] = '✅'
        else:
            print(f"   ❌ {func_name}: MISSING (fallback недоступен!)")
            missing_v3.append(func_name)
            results[f'v3_func_{func_name}'] = '❌ MISSING'
    
    # 5. Проверка механизма import
    print("\n📌 5. Проверка механизма import v4.0...")
    try:
        _USE_V4 = getattr(openalpr_analyzer, '_USE_V4_MAXQUALITY', False)
        if _USE_V4:
            print(f"   ✅ v4.0 успешно импортирован: {_USE_V4}")
            results['v4_import'] = '✅'
        else:
            print(f"   ⚠️  v4.0 импорт: {_USE_V4} (может быть отключен)")
            results['v4_import'] = '⚠️  PARTIAL'
    except Exception as e:
        print(f"   ⚠️  Ошибка проверки: {e}")
        results['v4_import'] = '⚠️  ERROR'
    
    # 6. Проверка документации
    print("\n📌 6. Проверка документации...")
    from pathlib import Path
    plugin_dir = Path("Plugins/OpenALPRAnalyzer")
    docs = ['V4_MAXQUALITY.md', 'V4_QUICKSTART.md']
    
    for doc in docs:
        doc_path = plugin_dir / doc
        if doc_path.exists():
            size = doc_path.stat().st_size
            print(f"   ✅ {doc}: OK ({size} байт)")
            results[f'doc_{doc}'] = '✅'
        else:
            print(f"   ❌ {doc}: MISSING")
            results[f'doc_{doc}'] = '❌'
    
    # 7. Проверка конфигурации
    print("\n📌 7. Проверка конфигурации...")
    config_path = plugin_dir / "config.json"
    if config_path.exists():
        import json
        try:
            with open(config_path) as f:
                config = json.load(f)
            print(f"   ✅ config.json: OK")
            if 'models' in config:
                print(f"      - Моделей: {len(config.get('models', []))} шт.")
            results['config'] = '✅'
        except Exception as e:
            print(f"   ⚠️  config.json содержит ошибку: {e}")
            results['config'] = '⚠️  ERROR'
    else:
        print(f"   ⚠️  config.json не найден (используется встроенная конфигурация)")
        results['config'] = '⚠️  USING_DEFAULT'
    
    # Итоги
    print("\n" + "="*70)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("="*70)
    
    success_count = sum(1 for v in results.values() if '✅' in str(v))
    total_count = len(results)
    
    print(f"\nусловий выполнено: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
        print("\n🎉 v4.0 MAXIMUM QUALITY готов к использованию!")
        print("\n📋 Компоненты:")
        print("   ✓ Региональное определение цвета (3 зоны)")
        print("   ✓ Ансамбль CLIP (2-3 модели)")
        print("   ✓ Мультиметодное определение номера")
        print("   ✓ Fallback механизм к v3.x")
        print("\n🚀 Может быть использован в:")
        print("   • UI (Plugins/OpenALPRAnalyzer/openalpr_analyzer.py)")
        print("   • Скриптах (from Plugins.OpenALPRAnalyzer import analyze_vehicle_openalpr)")
        print("   • Python коде (import openalpr_analyzer_v4_maxquality as v4)")
        print("\n📖 Документация:")
        print("   • V4_MAXQUALITY.md - подробное описание")
        print("   • V4_QUICKSTART.md - быстрый старт")
        return True
    else:
        print("\n❌ НЕКОТОРЫЕ ПРОВЕРКИ НЕ ПРОЙДЕНЫ")
        print("\nОшибки:")
        for key, value in results.items():
            if '❌' in str(value):
                print(f"   • {key}: {value}")
        return False

if __name__ == "__main__":
    success = check_v4_components()
    print("="*70 + "\n")
    sys.exit(0 if success else 1)
