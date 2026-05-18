"""
Примеры использования модуля duplicate_detector
"""

# ============================================================================
# Пример 1: Базовое использование в отдельном Python скрипте
# ============================================================================

from duplicate_detector import detect_and_remove_duplicates
import os

def example_1_basic_usage():
    """Самый простой пример использования"""
    folder_path = "path/to/images"
    
    if os.path.exists(folder_path):
        result = detect_and_remove_duplicates(folder_path)
        
        if result:
            print("✓ Обработка завершена!")
            print(f"  Удалено: {result['duplicates_deleted']} дубликатов")
            print(f"  Уникальных: {result['unique_images_moved']} изображений")
            print(f"  Папка результатов: {result['output_folder']}")
    else:
        print("✗ Папка не найдена")


# ============================================================================
# Пример 2: Использование с обработкой ошибок
# ============================================================================

def example_2_error_handling():
    """Пример с полной обработкой ошибок"""
    folder_path = "path/to/images"
    
    try:
        result = detect_and_remove_duplicates(folder_path)
        
        if not result:
            print("⚠ Не удалось обработать папку")
            return
        
        # Выводим подробные результаты
        print("=" * 60)
        print("РЕЗУЛЬТАТЫ ОБРАБОТКИ")
        print("=" * 60)
        print(f"Исходная папка: {result['source_folder']}")
        print(f"Папка результатов: {result['output_folder']}")
        print(f"\nСтатистика:")
        print(f"  • Найдено дубликатов: {result['total_duplicates_found']}")
        print(f"  • Удалено файлов: {result['duplicates_deleted']}")
        print(f"  • Групп дубликатов: {result['duplicate_groups_processed']}")
        print(f"  • Перемещено в QunicObject: {result['unique_images_moved']}")
        print("=" * 60)
        
    except FileNotFoundError:
        print("✗ Папка не существует")
    except PermissionError:
        print("✗ Нет прав доступа к папке")
    except Exception as e:
        print(f"✗ Ошибка: {e}")


# ============================================================================
# Пример 3: Обработка нескольких папок
# ============================================================================

def example_3_multiple_folders():
    """Обработка нескольких папок подряд"""
    folders = [
        "path/to/images1",
        "path/to/images2",
        "path/to/images3"
    ]
    
    total_deleted = 0
    total_unique = 0
    
    for folder in folders:
        if os.path.exists(folder):
            print(f"\n📁 Обработка: {folder}")
            result = detect_and_remove_duplicates(folder)
            
            if result:
                print(f"   ✓ Удалено: {result['duplicates_deleted']}")
                print(f"   ✓ Уникальных: {result['unique_images_moved']}")
                total_deleted += result['duplicates_deleted']
                total_unique += result['unique_images_moved']
        else:
            print(f"\n⚠ Папка не найдена: {folder}")
    
    print(f"\n📊 ИТОГО по всем папкам:")
    print(f"   Удалено: {total_deleted}")
    print(f"   Уникальных: {total_unique}")


# ============================================================================
# Пример 4: Создание резервной копии перед обработкой
# ============================================================================

import shutil
from datetime import datetime

def example_4_backup_before_processing():
    """Создание резервной копии перед обработкой дубликатов"""
    folder_path = "path/to/images"
    
    # Создаём резервную копию
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_folder = f"{folder_path}_backup_{timestamp}"
    
    try:
        print(f"📦 Создание резервной копии: {backup_folder}")
        shutil.copytree(folder_path, backup_folder)
        print(f"✓ Резервная копия создана")
        
        # Обрабатываем папку
        print(f"\n🔄 Обработка дубликатов...")
        result = detect_and_remove_duplicates(folder_path)
        
        if result:
            print(f"✓ Обработка завершена!")
            print(f"  Удалено: {result['duplicates_deleted']}")
            print(f"  Результаты: {result['output_folder']}")
        else:
            print("✗ Ошибка при обработке")
            
    except Exception as e:
        print(f"✗ Ошибка: {e}")


# ============================================================================
# Пример 5: Фильтрация результатов
# ============================================================================

from pathlib import Path

def example_5_filtering_results():
    """Пример обработки результатов с фильтрацией"""
    folder_path = "path/to/images"
    
    result = detect_and_remove_duplicates(folder_path)
    
    if not result:
        return
    
    # Получаем список файлов в папке результатов
    output_folder = result['output_folder']
    
    if os.path.exists(output_folder):
        # Подсчитываем файлы по расширениям
        extensions = {}
        
        for file_path in Path(output_folder).iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
        
        print(f"\n📊 Статистика файлов в {output_folder}:")
        for ext, count in sorted(extensions.items()):
            print(f"  {ext}: {count} файлов")
        
        # Получаем общий размер
        total_size = sum(
            f.stat().st_size 
            for f in Path(output_folder).iterdir() 
            if f.is_file()
        )
        print(f"  Общий размер: {total_size / (1024*1024):.2f} МБ")


# ============================================================================
# Пример 6: Логирование результатов в файл
# ============================================================================

def example_6_save_results_to_log():
    """Сохранение результатов обработки в лог-файл"""
    folder_path = "path/to/images"
    log_file = "duplicate_detection_log.txt"
    
    result = detect_and_remove_duplicates(folder_path)
    
    if result:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ОТЧЁТ О УДАЛЕНИИ ДУБЛИКАТОВ\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Исходная папка: {result['source_folder']}\n")
            f.write(f"Папка результатов: {result['output_folder']}\n\n")
            f.write("РЕЗУЛЬТАТЫ:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Всего найдено дубликатов: {result['total_duplicates_found']}\n")
            f.write(f"Удалено файлов: {result['duplicates_deleted']}\n")
            f.write(f"Групп дубликатов обработано: {result['duplicate_groups_processed']}\n")
            f.write(f"Перемещено в QunicObject: {result['unique_images_moved']}\n")
            f.write("=" * 60 + "\n")
        
        print(f"✓ Результаты сохранены в {log_file}")


# ============================================================================
# Пример 7: Использование в UI приложении (уже реализовано в ui.py)
# ============================================================================

"""
В интерфейсе приложения все необходимые функции уже реализованы:

1. Нажмите кнопку "Выбрать папку для поиска дубликатов"
2. Выберите нужную папку
3. Нажмите "Запустить удаление дубликатов"
4. Дождитесь завершения

Результаты отобразятся в информационном окне.
"""


# ============================================================================
# ЗАПУСК ПРИМЕРОВ
# ============================================================================

if __name__ == "__main__":
    print("Примеры использования модуля duplicate_detector")
    print("=" * 60)
    print("\nДля запуска примера раскомментируйте нужный вызов:")
    print()
    
    # Раскомментируйте нужный пример:
    # example_1_basic_usage()
    # example_2_error_handling()
    # example_3_multiple_folders()
    # example_4_backup_before_processing()
    # example_5_filtering_results()
    # example_6_save_results_to_log()
    
    print("Примеры готовы к использованию!")
