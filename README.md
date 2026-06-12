Watch Golem — кроссплатформенная система для анализа видеопотоков и изображений с использованием современных моделей компьютерного зрения 
(YOLO, Qwen2-VL) и специализированных анализаторов. Создано в рамках прохождение магистратуры НИЯУ МИФИ. 
**Версия:** 1.0.0  


## Основные возможности

### Обнаружение объектов
- **YOLO11** детекция объектов в видео и изображениях
- Отслеживание объектов в реальном времени
- Поддержка несколько моделей YOLO (nano, small, medium, xlarge)
- Анализ отдельных кадров и потоков видео

### Анализ транспорта
- **VehicleAnalyzer** - анализ машин и номерных знаков
- **QwenVehicleAnalyzer** - продвинутый анализ с использованием Qwen
- **OpenALPRAnalyzer** - интеграция OpenALPR для распознавания номерных знаков
- **Qwen2VLAnalyzer** - мультимодальный анализ с Qwen2.5-VL

### Анализ лиц
- **PersonAnalyzer** - обнаружение и распознавание лиц
- Встроенная база данных лиц (FaceDB) с эмбеддингами
- Поддержка групп лиц (Courteney Cox, Jennifer Aniston и т.д.)
- Экспорт отчётов в Word и Excel

### Дополнительные анализаторы
- **PhoneAnalyzer** - обнаружение мобильных телефонов
- **OpenSetAnalyzer** - анализ открытых классов
- **ModelInspector** - инспекция и анализ моделей
- **UnknownClassMerger** - объединение неизвестных классов

### Утилиты
- **Duplicate Detector** - обнаружение и удаление дубликатов изображений
- **Image Grouper** - группировка изображений по подобию
- **Parking Blocker Trainer** - обучение кастомной модели YOLO

## 📋 Системные требования

### Минимальные требования
- Python 3.8 или выше
- 8 GB оперативной памяти
- 10 GB свободного места на диске
- Windows 10/11, macOS или Linux

### Рекомендуемые требования
- Python 3.10+
- 16+ GB оперативной памяти
- NVIDIA GPU с CUDA поддержкой (для ускорения)
- 50+ GB свободного места

## Установка

### 1. Клонирование репозитория
```bash
git clone https://github.com/yourusername/WhatchGolem.git
cd WhatchGolem
```

### 2. Создание виртуального окружения
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 4. Загрузка моделей
Приложение автоматически загружает необходимые модели при первом запуске. Вы также можете загрузить их вручную:

```bash
python download_yolo11.py          # Загрузка YOLO11
python download_qwen2vl2b.py       # Загрузка Qwen2-VL-2B
python download_qwen35_4b_awq.py   # Загрузка Qwen3.5-4B-AWQ
```

## Использование

### Запуск приложения

#### Через графический интерфейс (рекомендуется)
```bash
python main.py
```

#### Через командную строку
```bash
python main.py
```

### Основные функции в UI

1. **Обработка видео** - загрузить видео для анализа с YOLO
2. **Обработка папки** - обработить все видео/изображения в папке
3. **Анализ транспорта** - использовать специализированные анализаторы
4. **Анализ лиц** - распознавание лиц из базы данных
5. **Поиск дубликатов** - найти и удалить дубликаты
6. **Обучение модели** - обучить кастомную модель parking_blocker

## 📁 Структура проекта

```
WhatchGolem/
├── main.py                          # Точка входа приложения
├── Whatch Golem.py                  # Альтернативный точка входа
├── ui.py                            # PyQt5 интерфейс
├── video_processing.py              # Обработка видео
├── model_checker.py                 # Проверка моделей
├── logger.py                        # Логирование
├── duplicate_detector.py            # Обнаружение дубликатов
├── image_grouper.py                 # Группировка изображений
├── Plugins/                         # Специализированные анализаторы
│   ├── VehicleAnalyzer/
│   ├── PersonAnalyzer/
│   ├── PhoneAnalyzer/
│   ├── Qwen2VLAnalyzer/
│   ├── QwenVehicleAnalyzer/
│   ├── OpenALPRAnalyzer/
│   ├── OpenSetAnalyzer/
│   ├── ModelInspector/
│   └── UnknownClassMerger/
├── Models/                          # Загруженные модели
│   ├── Qwen2-VL-2B-Instruct/
│   ├── Qwen2.5-VL-3B-GGUF/
│   └── Qwen3.5-4B-AWQ/
├── FaceDB/                          # База данных лиц
│   ├── _embeddings_cache.json
│   ├── Courteney Cox/
│   ├── Jennifer Aniston/
│   └── ...
├── Tests/                           # Тестовые скрипты
├── Results/                         # Результаты анализа
└── yolo*.pt                         # YOLO веса
```


## Примеры использования

### Анализ одного видео
```python
from video_processing import process_single_video
result = process_single_video("input_video.mp4", device="cuda")
```

### Анализ лиц из папки
```python
from Plugins.PersonAnalyzer import analyze_person_folder, generate_report_word_person
results = analyze_person_folder("path/to/images/")
generate_report_word_person(results, "output_report.docx")
```

### Обнаружение дубликатов
```python
from duplicate_detector import detect_and_remove_duplicates
duplicates = detect_and_remove_duplicates("path/to/images/", threshold=0.95)
```

### Обучение кастомной модели
```python
python train_parking_blocker.py \
    --source_dir "path/to/dataset" \
    --epochs 100 \
    --batch 32 \
    --device cuda
```

