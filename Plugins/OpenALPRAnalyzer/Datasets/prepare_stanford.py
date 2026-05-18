#!/usr/bin/env python3
"""
Помощник для подготовки Stanford Cars Dataset для OpenALPR Analyzer.

Stanford Cars Dataset: http://ai.stanford.edu/~jkrause/cars/car_dataset.html
Содержит 16,185 изображений 196 классов автомобилей (1997-2012 годы).

Использование:
    1. Скачайте датасет:
       wget http://ai.stanford.edu/~jkrause/cars/car_ims.tgz
       wget http://ai.stanford.edu/~jkrause/cars/cars_annos.mat
    
    2. Распакуйте:
       tar -xzf car_ims.tgz
    
    3. Запустите скрипт:
       python prepare_stanford.py cars_annos.mat
       # Создаст stanford_cars_196.json
"""

import os
import json
from pathlib import Path

def extract_stanford_cars_from_mat(mat_file: str) -> list:
    """
    Извлекает названия 196 классов из Stanford Cars аннотаций (cars_annos.mat).
    
    Требует: scipy
    pip install scipy
    """
    try:
        import scipy.io
    except ImportError:
        print("❌ Требуется scipy: pip install scipy")
        return []
    
    try:
        print(f"📋 Читаю {mat_file}...")
        annos = scipy.io.loadmat(mat_file)
        
        # Ключи в .mat файле
        if 'class_names' in annos:
            class_names = []
            for class_data in annos['class_names'][0]:
                class_name = class_data[0] if isinstance(class_data[0], str) else str(class_data[0])
                class_names.append(class_name)
            
            print(f"✅ Извлечено {len(class_names)} классов")
            return class_names
        else:
            print("❌ 'class_names' не найдено в .mat файле")
            print("   Доступные ключи:", list(annos.keys()))
            return []
            
    except Exception as e:
        print(f"❌ Ошибка при чтении {mat_file}: {e}")
        return []


def extract_stanford_cars_from_txt(txt_file: str) -> list:
    """
    Альтернативный способ - читайте из текстового файла.
    (Если вы вручную выписали классы)
    """
    try:
        with open(txt_file, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        print(f"✅ Читаю {len(classes)} классов из {txt_file}")
        return classes
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return []


def extract_stanford_cars_manual() -> list:
    """
    196 классов Stanford Cars Dataset (в коде для fallback).
    Если у вас нет доступа к .mat файлу.
    """
    return [
        "2012 Tesla Model S Sedan",
        "2011 Volkswagen Beetle Hatchback", 
        "2012 Volvo C30 Hatchback",
        "2012 Volvo XC90 SUV",
        "2009 Volvo S60 Sedan",
        # ... добавьте все 196 классов если нужно
        # Полный список можно найти здесь:
        # http://ai.stanford.edu/~jkrause/cars/car_dataset.html
    ]


def create_stanford_json(classes: list, output_file: str):
    """Создает JSON файл с классами Stanford Cars."""
    output = {
        "source": "Stanford Cars Dataset",
        "total_classes": len(classes),
        "years": "1997-2012",
        "classes": classes,
        "usage": "Set 'dataset': 'stanford' in config.json and 'stanford_path': path/to/this/file"
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Сохранено {len(classes)} классов в {output_file}")


def create_stanford_txt(classes: list, output_file: str):
    """Создает текстовый файл с классами (альтернативно JSON)."""
    with open(output_file, "w", encoding="utf-8") as f:
        for class_name in classes:
            f.write(f"{class_name}\n")
    
    print(f"\n✅ Сохранено {len(classes)} классов в {output_file}")


if __name__ == "__main__":
    import sys
    
    print("🚗 Подготовка Stanford Cars Dataset...")
    print()
    
    if len(sys.argv) < 2:
        print("📝 Использование:")
        print("   python prepare_stanford.py cars_annos.mat")
        print("   # или")
        print("   python prepare_stanford.py classes.txt")
        print()
        print("💾 Скачайте датасет:")
        print("   wget http://ai.stanford.edu/~jkrause/cars/car_ims.tgz")
        print("   wget http://ai.stanford.edu/~jkrause/cars/cars_annos.mat")
        print("   tar -xzf car_ims.tgz")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.isfile(input_file):
        print(f"❌ Файл не найден: {input_file}")
        sys.exit(1)
    
    classes = []
    
    if input_file.endswith(".mat"):
        classes = extract_stanford_cars_from_mat(input_file)
    elif input_file.endswith(".txt"):
        classes = extract_stanford_cars_from_txt(input_file)
    else:
        print("❌ Поддерживаются только .mat и .txt файлы")
        sys.exit(1)
    
    if classes:
        json_output = "stanford_cars_196.json"
        txt_output = "stanford_cars_196.txt"
        
        create_stanford_json(classes, json_output)
        create_stanford_txt(classes, txt_output)
        
        print("\n📊 Первые 5 классов:")
        for i, cls in enumerate(classes[:5], 1):
            print(f"   {i}. {cls}")
        print(f"   ...")
        print(f"   {len(classes)}. {classes[-1]}")
        
        print("\n⚙️  Далее обновите config.json:")
        print('   {"dataset": "stanford", "stanford_path": "Datasets/stanford_cars_196.json"}')
    else:
        print("❌ Не удалось извлечь классы")
        print()
        print("💡 Попробуйте:")
        print("   1. Убедитесь, что scipy установлена: pip install scipy")
        print("   2. Проверьте формат файла cars_annos.mat")
        print("   3. Используйте альтернативный способ с текстовым файлом")
