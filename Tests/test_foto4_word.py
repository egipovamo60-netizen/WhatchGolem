"""Быстрый тест генерации Word отчёта для Foto4."""
import os
import sys

# Добавляем путь
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Plugins.Qwen2VLAnalyzer.qwen2vl_analyzer import generate_report_word_qwen2vl

# Создаём тестовый результат для Foto4
test_results = [
    {
        "file": "Foto4.jpg",
        "path": r"z:\Coding\Watch golem stable\Test foto\One\Foto4.jpg",
        "color": {"color": "Тёмно-серый", "confidence": 0.85},
        "model": {"model": "Toyota Camry", "confidence": 0.92},
        "plate": {"plate_text": "А123ВС77", "confidence": 0.88, "message": "Номер: А123ВС77 (уверенность: 88%)"},
        "timestamp": "29-03-2026 12:00:00",
        "engine": "Qwen2-VL-2B"
    }
]

try:
    output_dir = r"z:\Coding\Watch golem stable\Results\test_word_report"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Генерируем Word отчёт для Foto4...")
    doc_path = generate_report_word_qwen2vl(test_results, output_dir)
    
    if os.path.exists(doc_path):
        size = os.path.getsize(doc_path)
        print(f"✓ Word отчёт успешно создан!")
        print(f"  Путь: {doc_path}")
        print(f"  Размер: {size} байт ({size / (1024*1024):.2f} МБ)")
    else:
        print(f"❌ Файл не найден: {doc_path}")
        
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
