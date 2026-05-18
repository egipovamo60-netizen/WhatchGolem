"""
Тест функционала PersonAnalyzer плагина.

Тестирует:
  - Анализ одного изображения с человеком
  - Анализ папки с несколькими изображениями
  - Генерацию текстовых отчётов
  - Генерацию Word-отчётов
"""

import os
import sys
import json
from datetime import datetime

# Добавляем путь к плагину
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
from Plugins.PersonAnalyzer import (
    analyze_person,
    analyze_person_folder,
    generate_report_person,
    generate_report_word_person,
    reload_face_database,
)
from logger import get_logger

_log = get_logger("PersonAnalyzer_TEST")

TEST_IMAGES_ONE = os.path.join(_ROOT, "Test foto", "One")
TEST_IMAGES_GROUP = os.path.join(_ROOT, "Test foto", "Group")
OUTPUT_DIR = os.path.join(_ROOT, "Results", "PersonAnalyzer_Test")


def progress_callback(msg, step):
    """Простой callback для отслеживания прогресса."""
    print(f"  [{step}] {msg}")


def test_single_image():
    """Тест анализа одного изображения."""
    print("\n" + "=" * 70)
    print("ТЕСТ 1: Анализ одного изображения")
    print("=" * 70)

    test_image = os.path.join(TEST_IMAGES_ONE, "Foto4.jpg")
    if not os.path.isfile(test_image):
        print("Файл не найден: " + test_image)
        return None

    print("Анализируем: " + test_image)
    result = analyze_person(test_image, progress_callback)

    print("\nРезультат анализа:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    return result


def test_folder_images():
    """Тест анализа папки с изображениями."""
    print("\n" + "=" * 70)
    print("ТЕСТ 2: Анализ папки с изображениями")
    print("=" * 70)

    if not os.path.isdir(TEST_IMAGES_ONE):
        print("Папка не найдена: " + TEST_IMAGES_ONE)
        return None

    print("Анализируем папку: " + TEST_IMAGES_ONE)
    results = analyze_person_folder(TEST_IMAGES_ONE, progress_callback)

    print("\nОбработано " + str(len(results)) + " изображений")
    
    # Статистика
    success_count = sum(1 for r in results if "error" not in r)
    error_count = len(results) - success_count

    print("  Успешно: " + str(success_count))
    print("  Ошибок: " + str(error_count))

    return results


def test_text_report(results):
    """Тест генерации текстового отчёта."""
    print("\n" + "=" * 70)
    print("ТЕСТ 3: Генерация текстового отчёта")
    print("=" * 70)

    if not results:
        print("Нет результатов для отчёта")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "PERSON_ANALYSIS.txt")

    print("Создаём отчёт: " + report_path)
    generated_path = generate_report_person(results, report_path)

    if os.path.isfile(generated_path):
        with open(generated_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print("Отчёт создан (" + str(len(content)) + " символов)")
        print("\nПредпросмотр:")
        print(content[:500] + "..." if len(content) > 500 else content)
        return generated_path
    else:
        print("Ошибка: файл отчёта не создан")
        return None


def test_word_report(results):
    """Тест генерации Word-отчёта."""
    print("\n" + "=" * 70)
    print("ТЕСТ 4: Генерация Word-отчёта")
    print("=" * 70)

    if not results:
        print("Нет результатов для отчёта")
        return None

    print("Создаём Word-отчёт...")
    try:
        generated_path = generate_report_word_person(results, OUTPUT_DIR)
        if os.path.isfile(generated_path):
            size_mb = os.path.getsize(generated_path) / (1024 * 1024)
            print("Word-отчёт создан: " + generated_path)
            print("  Размер: %.2f МБ" % size_mb)
            return generated_path
        else:
            print("Ошибка: файл отчёта не создан")
            return None
    except ImportError as e:
        print("Ошибка импорта (python-docx не установлен?): " + str(e))
        return None
    except Exception as e:
        print("Ошибка при создании Word-отчёта: " + str(e))
        return None


def test_face_database():
    """Тест загрузки базы лиц."""
    print("\n" + "=" * 70)
    print("ТЕСТ 5: База лиц (FaceDB)")
    print("=" * 70)

    face_db_dir = "FaceDB"
    if not os.path.isdir(face_db_dir):
        print("Директория базы лиц не найдена: " + face_db_dir)
        print("  Это нормально - база может быть пуста на начальном этапе")
        return

    entries = [e for e in os.listdir(face_db_dir) 
               if os.path.isdir(os.path.join(face_db_dir, e)) and not e.startswith("_")]
    
    if not entries:
        print("База изображений пуста")
    else:
        print("Найдено персон: " + str(len(entries)))
        for name in entries:
            person_dir = os.path.join(face_db_dir, name)
            images = [f for f in os.listdir(person_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            print("  * " + name + ": " + str(len(images)) + " фото")

    print("\n  Перезагружаем базу лиц...")
    reload_face_database()
    print("  База перезагружена")


def main():
    """Основной тест."""
    print("\n" + "=" * 70)
    print(" " * 15 + "ТЕСТ ПЕРСОНАЛАЙЗЕРА (PersonAnalyzer)")
    print("=" * 70)

    # Тест 1: Одно изображение
    result_single = test_single_image()

    # Тест 2: Папка
    results_folder = test_folder_images()

    # Тест 3: Текстовый отчёт
    if results_folder:
        test_text_report(results_folder)

    # Тест 4: Word-отчёт
    if results_folder:
        test_word_report(results_folder)

    # Тест 5: База лиц
    test_face_database()

    # Заключение
    print("\n" + "=" * 70)
    print("ЗАКЛЮЧЕНИЕ")
    print("=" * 70)
    print("Результаты сохранены в: " + OUTPUT_DIR)
    if os.path.isdir(OUTPUT_DIR):
        files = os.listdir(OUTPUT_DIR)
        print("Файлы отчётов: " + (", ".join(files) if files else "нет"))
    
    print("\nТестирование завершено")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nОШИБКА: " + str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
