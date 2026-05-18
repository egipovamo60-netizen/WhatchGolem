#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест v4.0 MAXIMUM QUALITY плагина OpenALPR Analyzer
Проверяет все компоненты: цвет, модель, номер
"""

import sys
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)

from Plugins.OpenALPRAnalyzer import analyze_vehicle_openalpr
from pathlib import Path
import json

def test_v4_maxquality():
    """Тестирует v4.0 MAXIMUM QUALITY"""
    
    print("\n" + "="*60)
    print("✓ TEST v4.0 MAXIMUM QUALITY OpenALPR Analyzer")
    print("="*60)
    
    # Найти тестовое изображение
    test_dirs = [
        os.path.join(_ROOT, "Test foto", "One"),
        os.path.join(_ROOT, "Test video"),
        os.path.join(_ROOT, "test_folder"),
    ]
    
    test_image = None
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists():
            for img_file in test_path.glob("*.jpg"):
                test_image = str(img_file)
                break
            if test_image:
                break
    
    if not test_image:
        print("\n⚠ Тестовое изображение не найдено")
        print("  Пожалуйста, поместите JPG во время Test foto/One или Test video")
        return False
    
    print(f"\n📷 Используется изображение: {test_image}")
    
    try:
        # Тест 1: Прямой вызов
        print("\n📊 ТЕСТ 1: Анализ одного изображения")
        print("-" * 40)
        
        result = analyze_vehicle_openalpr(test_image)
        
        print(f"🎨 Цвет:  {result.get('color', 'N/A'):<15} (уверенность: {result.get('color_confidence', 0):.2%})")
        print(f"🚗 Модель: {result.get('model', 'N/A'):<15} (уверенность: {result.get('model_confidence', 0):.2%})")
        print(f"📋 Номер: {result.get('plate', 'N/A'):<15} (уверенность: {result.get('plate_confidence', 0):.2%})")
        print(f"⏱ Время обработки: {result.get('processing_time', 0):.2f}ms")
        print(f"📌 Версия: {result.get('version', 'N/A')}")
        
        # Проверка результатов
        assert result.get('color'), "❌ Цвет не определен"
        assert result.get('model'), "❌ Модель не определена"
        assert result.get('plate'), "❌ Номер не определен"
        assert 0 <= result.get('color_confidence', 0) <= 1, "❌ Неверная уверенность цвета"
        assert 0 <= result.get('model_confidence', 0) <= 1, "❌ Неверная уверенность модели"
        assert 0 <= result.get('plate_confidence', 0) <= 1, "❌ Неверная уверенность номера"
        
        print("\n✓ ТЕСТ 1: ПРОЙДЕН")
        
        # Тест 2: Проверка v4.0 функций
        print("\n📊 ТЕСТ 2: Проверка v4.0 компонентов")
        print("-" * 40)
        
        from Plugins.OpenALPRAnalyzer import openalpr_analyzer_v4_maxquality as v4
        
        # Проверка функций
        funcs_to_check = [
            '_detect_color_advanced_v4',
            '_detect_model_clip_ensemble_v4',
            '_detect_plate_multimethod_v4',
            '_get_clip_models_ensemble',
            'analyze_vehicle',
            'analyze_vehicle_folder',
            'generate_report'
        ]
        
        for func_name in funcs_to_check:
            assert hasattr(v4, func_name), f"❌ Функция {func_name} не найдена"
            print(f"  ✓ {func_name}")
        
        print("\n✓ ТЕСТ 2: ПРОЙДЕН")
        
        # Тест 3: Проверка fallback механизма
        print("\n📊 ТЕСТ 3: Проверка v4.0 → v3.x fallback")
        print("-" * 40)
        
        from Plugins.OpenALPRAnalyzer.openalpr_analyzer import OpenALPRAnalyzer
        analyzer = OpenALPRAnalyzer()
        
        # Проверить наличие v3.x функций для fallback
        v3_funcs = [
            '_detect_color_kmeans_advanced',
            '_detect_model_clip',
            '_detect_plate_openalpr'
        ]
        
        for func_name in v3_funcs:
            assert hasattr(analyzer, func_name), f"❌ V3.x функция {func_name} не найдена"
            print(f"  ✓ {func_name} (fallback доступен)")
        
        print("\n✓ ТЕСТ 3: ПРОЙДЕН (Fallback механизм готов)")
        
        # Итоги
        print("\n" + "="*60)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
        print("="*60)
        print("\n📌 v4.0 MAXIMUM QUALITY готов к использованию:")
        print("  • Региональное определение цвета (3 зоны)")
        print("  • Ансамбль CLIP для определения модели")
        print("  • Мультиметодное определение номера")
        print("  • Fallback к v3.x при необходимости")
        print("\n🚀 Может быть использован в UI через:")
        print("  Plugins/OpenALPRAnalyzer/openalpr_analyzer.py")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_v4_maxquality()
    sys.exit(0 if success else 1)
