#!/usr/bin/env python3
"""Симуляция вызова UI для проверки интеграции"""

import sys
from duplicate_detector import detect_and_remove_duplicates

def simulate_ui_button_click(folder_path):
    """Симулирует нажатие кнопки 'Удалить дубликаты' в UI"""
    
    print("=" * 70)
    print("UI: Нажата кнопка 'Запустить удаление дубликатов'")
    print("=" * 70)
    print(f"📂 Выбранная папка: {folder_path}\n")
    
    # Вызываем функцию как это делает UI
    result = detect_and_remove_duplicates(folder_path)
    
    # Проверяем структуру возвращаемого значения
    print("\n" + "=" * 70)
    print("ПРОВЕРКА: Структура возвращаемого значения")
    print("=" * 70)
    
    required_keys = {
        'duplicates_deleted': int,
        'duplicate_groups_processed': int,
        'unique_images_moved': int,
        'output_folder': str
    }
    
    all_valid = True
    for key, expected_type in required_keys.items():
        if key in result:
            actual_type = type(result[key]).__name__
            expected_name = expected_type.__name__
            
            if isinstance(result[key], expected_type):
                print(f"✅ [{key}] = {result[key]:<10} (тип: {actual_type})")
            else:
                print(f"❌ [{key}] имеет неверный тип: {actual_type}, ожидалось {expected_name}")
                all_valid = False
        else:
            print(f"❌ [{key}] отсутствует в результате!")
            all_valid = False
    
    # Симуляция отображения в UI диалоге
    print("\n" + "=" * 70)
    print("UI: Диалог с результатами")
    print("=" * 70)
    
    dialog_text = f"""
┌─────────────────────────────────────────┐
│   РЕЗУЛЬТАТЫ УДАЛЕНИЯ ДУБЛИКАТОВ       │
├─────────────────────────────────────────┤
│                                         │
│  ✓ Обработано файлов:         3        │
│  ✓ Найдено групп дубликатов:  1        │
│                                         │
│  📊 УДАЛЕНО:                           │
│     Дубликатов удалено:         {result['duplicates_deleted']}        │
│     Групп обработано:           {result['duplicate_groups_processed']}        │
│                                         │
│  📁 ПЕРЕМЕЩЕНО:                        │
│     Уникальных файлов:         {result['unique_images_moved']}        │
│     Папка назначения:                  │
│     {result['output_folder']:<29} │
│                                         │
│                    [OK]      [Отмена]  │
└─────────────────────────────────────────┘
"""
    print(dialog_text)
    
    print("=" * 70)
    print("СТАТУС:", "✅ ВСЕ ПАРАМЕТРЫ КОРРЕКТНЫ" if all_valid else "❌ ОШИБКА В СТРУКТУРЕ")
    print("=" * 70)
    
    return all_valid

if __name__ == '__main__':
    # Тестируем на нашей тестовой папке
    success = simulate_ui_button_click('test_duplicates')
    
    sys.exit(0 if success else 1)
