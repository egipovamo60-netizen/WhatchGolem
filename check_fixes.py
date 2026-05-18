#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Быстрая проверка что всё исправлено"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("\n" + "="*60)
print("ПРОВЕРКА: Исправления применены?")
print("="*60)

# 1. Проверка openalpr.conf
config_path = os.path.join(os.path.dirname(__file__), "Plugins/OpenALPRAnalyzer/openalpr.conf")
if os.path.exists(config_path):
    print(f"\n✓ openalpr.conf существует: {config_path}")
else:
    print(f"\n✗ openalpr.conf НЕ существует: {config_path}")

# 2. Проверка v4.0 импорта
try:
    from Plugins.OpenALPRAnalyzer import openalpr_analyzer_v4_maxquality as v4
    print("✓ v4.0 модуль импортирован успешно")
    
    # 3. Проверка _OPENALPR_CONFIG
    if hasattr(v4, '_OPENALPR_CONFIG'):
        print(f"✓ _OPENALPR_CONFIG определена: {v4._OPENALPR_CONFIG}")
    else:
        print("✗ _OPENALPR_CONFIG НЕ определена")
    
    # 4. Проверка функций
    if hasattr(v4, '_detect_plate_multimethod_v4'):
        print("✓ _detect_plate_multimethod_v4 доступна")
    else:
        print("✗ _detect_plate_multimethod_v4 НЕ доступна")
        
except Exception as e:
    print(f"✗ Ошибка импорта v4.0: {e}")
    import traceback
    traceback.print_exc()

print("="*60 + "\n")
