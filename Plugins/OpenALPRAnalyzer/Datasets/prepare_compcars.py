#!/usr/bin/env python3
"""
Помощник для подготовки CompCars Dataset для OpenALPR Analyzer.

Использование:
    python prepare_compcars.py /path/to/CompCars/data/
    # Создаст compcars_1655.json
"""

import os
import json
from pathlib import Path

def extract_compcars_models(compcars_root: str) -> list:
    """
    Извлекает список моделей из CompCars датасета.
    
    CompCars структура:
        CompCars/data/
            ├─ make_model_name.txt  (список марок/моделей)
            ├─ image_list.txt       (список путей)
            └─ ...
    """
    models = set()
    root = Path(compcars_root)
    
    # Подходы к извлечению в зависимости от доступных файлов
    
    # 1. Из make_model_name.txt (если есть)
    make_model_file = root / "make_model_name.txt"
    if make_model_file.exists():
        print(f"📋 Читаю {make_model_file.name}...")
        with open(make_model_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    models.add(line)
    
    # 2. Из image_list.txt (извлечем тип машины из пути)
    image_list_file = root / "image_list.txt"
    if image_list_file.exists():
        print(f"📋 Читаю {image_list_file.name}...")
        with open(image_list_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # Формат: folder_index image_index type car_type ...
                    # car_type содержит марку/модель (обычно через _)
                    try:
                        car_type = int(parts[2])
                        # CompCars имеет ~163 типов машин
                        # Выводим тип для анализа
                        print(f"  Найден вид: {car_type}")
                    except:
                        pass
    
    # 3. Из структуры папок (если организовано по маркам)
    data_dir = root / "image"
    if data_dir.exists():
        print(f"📋 Просматриваю структуру папок {data_dir}...")
        for make_dir in data_dir.iterdir():
            if make_dir.is_dir():
                make_name = make_dir.name
                for model_dir in make_dir.iterdir():
                    if model_dir.is_dir():
                        model_name = model_dir.name
                        full_name = f"{make_name} {model_name}"
                        models.add(full_name)
                        print(f"  ✓ {full_name}")
    
    return sorted(list(models))


def create_compcars_json(models: list, output_file: str):
    """Создает JSON файл с моделями."""
    output = {
        "source": "CompCars Dataset (1655 models)",
        "total_models": len(models),
        "models": models,
        "usage": "Set 'dataset': 'compcars' in config.json and 'compcars_path': path/to/this/file"
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Сохранено {len(models)} моделей в {output_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("📝 Использование: python prepare_compcars.py /path/to/CompCars/")
        print("\n💡 Примеры:")
        print("   python prepare_compcars.py ~/CompCars/")
        print("   python prepare_compcars.py D:\\Datasets\\CompCars\\")
        sys.exit(1)
    
    compcars_path = sys.argv[1]
    
    if not os.path.isdir(compcars_path):
        print(f"❌ Папка не найдена: {compcars_path}")
        sys.exit(1)
    
    print("🚗 Подготовка CompCars Dataset...")
    models = extract_compcars_models(compcars_path)
    
    if models:
        output_file = "compcars_1655.json"
        create_compcars_json(models, output_file)
        
        print("\n📊 Первые 10 моделей:")
        for i, model in enumerate(models[:10], 1):
            print(f"   {i}. {model}")
        
        print("\n⚙️  Далее обновите config.json:")
        print('   {"dataset": "compcars", "compcars_path": "Datasets/compcars_1655.json"}')
    else:
        print("❌ Не удалось извлечь модели из датасета")
