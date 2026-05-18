# OpenALPR Analyzer Plugin

**Полностью локальный анализатор транспорта без интернета с автоматическим CV fallback.**

OpenALPR - это open-source движок для распознавания автомобильных номеров. Плагин использует OpenALPR вместе с локальными CV методами для определения цвета (K-means + Delta E) и модели (CLIP zero-shot).

## Особенность: Автоматический fallback

Если OpenALPR не установлен или не может распознать номер — плагин автоматически использует **CV методы VehicleAnalyzer**:
- ✅ Работает ВСЕГДА (даже без OpenALPR)
- ✅ Цвет: K-means кластеризация + метрика Delta E CIEDE2000 (из VehicleAnalyzer)
- ✅ Номер: EasyOCR + нормализация (из VehicleAnalyzer)

## Алгоритмы

### Цвет (Гибридный метод: HSV + LAB + Delta E)

**Новый гибридный подход v2.1** соединяет скорость HSV с надежностью LAB:

```
Шаг 1: Быстрая проверка через HSV
├─ Если Saturation < 30:
│  ├─ Value < 50 → Чёрный ✅
│  ├─ Value > 200 → Белый ✅
│  └─ Value 80-180 → Серый ✅
│
Шаг 2: K-means + Delta E для остального
├─ LAB пространство (перцепционное)
├─ K-means на 5 кластеров
├─ Фильтр шума (L < 10 или > 240)
└─ Delta E CIEDE2000 с эталонами
   ├─ ΔE < 20 → confidence 95%
   ├─ ΔE 20-40 → confidence 70%
   └─ ΔE > 40 → confidence 40%
```

**Преимущества гибридного метода:**
- ✅ **Быстро**: HSV быстрая проверка экономит вычисления
- ✅ **Точно**: LAB + Delta E - научный стандарт восприятия цвета
- ✅ **Надежно**: Различает монохромные цвета (серый vs черный vs белый)
- ✅ **Устойчиво**: Работает при плохом освещении и тенях

### Номер (OpenALPR → CV fallback)
1. Попытка **OpenALPR распознавания** (если установлен)
2. Если OpenALPR не может (< 30% confidence):
   - Автоматический переход на **VehicleAnalyzer CV методы**
   - EasyOCR + несколько методов OCR preprocessing
3. Нормализация к российскому формату

**Улучшение**: Гарантированный результат даже без сторонних инструментов.

### Модель (CLIP zero-shot с поддержкой датасетов)

**Три режима классификации:**

#### 1. Default (24 популярные модели) ⚡ Быстро
```
Toyota Camry, BMW X5, Mercedes S-Class,
Ford Focus, Lada Vesta, GAZelle van, ...
```
- Память: минимум
- Скорость: мгновенная
- Точность: хорошо для популярных моделей

#### 2. CompCars Dataset (1655 моделей) 🚀 Максимальная точность
**1655 4-digit моделей** из датасета Stanford/CompCars:
- Все основные азиатские, европейские, американские марки
- Включены редкие и исторические модели
- Точность классификации: ⭐⭐⭐⭐⭐ (лучшая)

**Как использовать:**
1. Скачайте компакт-список моделей:
   ```bash
   # CompCars: https://github.com/byu-object-recognition-lab/CompCars
   # Используйте примерный список из data/make_model_name.txt
   ```
2. Сохраните в JSON:
   ```json
   [
     "Acura TSX",
     "Audi A3",
     "BMW 3 Series",
     ...
   ]
   ```
3. Обновите `config.json`:
   ```json
   {
     "dataset": "compcars",
     "compcars_path": "path/to/compcars_models.json"
   }
   ```

#### 3. Stanford Cars (196 классов) 🎨 Сбалансировано
**196 официальных классов** из Stanford Cars Dataset:
- Покрывает 1997-2012 годы выпуска
- Точные фотографии каждого класса
- Хороший баланс точности/скорости

**Как использовать:**
1. Скачайте датасет: http://ai.stanford.edu/~jkrause/cars/car_dataset.html
2. Распакуйте и используйте file `cars_annos.mat` или создайте текстовый список
3. Обновите `config.json`:
   ```json
   {
     "dataset": "stanford",
     "stanford_path": "path/to/stanford_cars_list.txt"
   }
   ```

**Сравнение режимов:**

| Метрика | Default | CompCars | Stanford |
|---------|---------|----------|----------|
| **Кол-во моделей** | 24 | 1655 | 196 |
| **Скорость** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Покрытие** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Точность** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Редкие модели** | ❌ | ✅ | ✅ |

## Конфигурация

### Минимально (работает СРАЗУ)

Только Python зависимости:
```bash
pip install transformers torch Pillow opencv-python scikit-learn scikit-image
```

**Плагин будет полностью функционален с CV fallback для номеров.**

### Дополнительно (для лучшей точности номеров)

Установите OpenALPR:

#### Windows

```bash
choco install openalpr
# или скачайте исходники: https://github.com/openalpr/openalpr/releases
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get install openalpr openalpr-python-bindings
```

#### macOS

```bash
brew install openalpr
```

Затем Python bindings:
```bash
pip install openalpr-python-bindings
```

## Помощь: Плохо определяется цвет?

**Причины и решения:**

1. **Слишком темное/светлое изображение**
   - K-means фильтрует цветовой диапазон [10, 240]
   - Решение: улучшить освещение при съемке

2. **Сложные окраски (градиент, матовая краска)**
   - Алгоритм работает с средним цветом
   - Решение: близко подвести камеру, снять в хорошо освещенном месте

3. **Отражения света, тени**
   - Delta E может дать погрешность 20-40 для отраженных цветов
   - Решение: минимизировать отражения, снять несколько углов

**Тестирование:**
```python
from Plugins.OpenALPRAnalyzer import analyze_vehicle_openalpr
result = analyze_vehicle_openalpr("path/to/image.jpg")
print(result["color"])  # Проверить Delta E в логах
```

## Помощь: Плохо определяется номер?

**Если видите "CV fallback":** 
- OpenALPR не установлен или не работает
- Плагин успешно переключился на CV методы VehicleAnalyzer
- ✅ **Это нормально и ожидаемо**

**Если CV fallback тоже не работает:**

Причины:
- Номер очень маленький в кадре
- Очень плохое качество/размытие  
- Освещение против света

Решение:
- Приблизить камеру к номеру (минимум 30cm)
- Снять в хорошо освещенном месте
- Убедиться, что номер четкий

## Сравнение методов определения цвета

| | HSV гистограмма | LAB + K-means + Delta E | OpenALPR Гибридный |
|---|---|---|---|
| **Красный/Синий/Зелёный** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Серый** | ❌ Плохо | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Чёрный vs Серый** | ❌ Невозмож | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ ← **Все 3!** |
| **Белый vs Серебристый** | ❌ Невозмож | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Тени/отражения** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ ← **Лучше всех** |
| **Плохое освещение** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Скорость** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

**"Гибридный" = HSV быстрая проверка + LAB для точности** → лучший компромисс!

## Конфигурация

[config.json](config.json):
```json
{
    "region": "ru",
    "clip_model": "openai/clip-vit-base-patch32",
    "dataset": "default",
    "compcars_path": "",
    "stanford_path": ""
}
```

**Параметры:**

| Параметр | Значение | Описание |
|----------|----------|---------|
| `region` | ru, us, eu | Регион номера для OpenALPR |
| `clip_model` | openai/clip-vit-base-patch32 | CLIP модель из Hugging Face |
| `dataset` | default, compcars, stanford | Источник классов моделей |
| `compcars_path` | path/to/compcars.json | Путь к CompCars датасету (1655 моделей) |
| `stanford_path` | path/to/stanford.txt | Путь к Stanford Cars датасету (196 классов) |

**Примеры конфигурации:**

### Вариант 1: Быстро (по умолчанию)
```json
{
    "dataset": "default"
}
```

### Вариант 2: Максимальная точность (CompCars)
```json
{
    "dataset": "compcars",
    "compcars_path": "Datasets/compcars_1655.json"
}
```

### Вариант 3: Сбалансировано (Stanford)
```json
{
    "dataset": "stanford",
    "stanford_path": "Datasets/stanford_cars_196.txt"
}
```

## Подготовка датасетов

### CompCars Dataset (1655 моделей)

1. **Скачайте:**
   ```bash
   # Официальный репозиторий
   git clone https://github.com/byu-object-recognition-lab/CompCars.git
   ```

2. **Подготовьте JSON:** (используйте скрипт)
   ```python
   import json
   
   # Список всех моделей из CompCars
   models = [
       "Acura TSX", "Acura TSX Wagon", "Acura ILX",
       "Audi A3", "Audi A4", "Audi A6",
       # ... 1655 моделей
   ]
   
   with open("compcars_1655.json", "w") as f:
       json.dump(models, f, indent=2, ensure_ascii=False)
   ```

3. **Обновите config.json:**
   ```json
   {"dataset": "compcars", "compcars_path": "Datasets/compcars_1655.json"}
   ```

### Stanford Cars Dataset (196 классов)

1. **Скачайте:**
   ```bash
   # Официальный датасет
   wget http://ai.stanford.edu/~jkrause/car_ims.tgz
   tar -xzf car_ims.tgz
   ```

2. **Экспортируйте классы:**
   ```bash
   # Из распакованного архива: cars_annos.mat
   # Используйте scipy для чтения .mat файла
   python -c "
   import scipy.io
   annos = scipy.io.loadmat('cars_annos.mat')
   class_names = [str(x[0]) for x in annos['class_names'][0]]
   with open('stanford_cars_196.txt', 'w') as f:
       f.write('\n'.join(class_names))
   "
   ```

3. **Обновите config.json:**
   ```json
   {"dataset": "stanford", "stanford_path": "Datasets/stanford_cars_196.txt"}
   ```

## Логирование

Все операции логируются в `Log/app_YYYY-MM-DD.log`:
```
28-03-2026 22:00:04 [DEBUG] OpenALPRAnalyzer: Color detected: black, Delta E=15.3, confidence=95%
28-03-2026 22:00:05 [INFO] OpenALPRAnalyzer: OpenALPR распознал: А123ВС77 (92%)
28-03-2026 22:00:06 [DEBUG] OpenALPRAnalyzer: CLIP модель загружена
```

### Debug режим

Для более подробного логирования установите уровень логирования в `logger.py`:
```python
logging.basicConfig(level=logging.DEBUG)  # вместо INFO
```

## Преимущества

✅ **Полностью автономный** - интернет не требуется  
✅ **Open-source** - бесплатный, весь код открыт  
✅ **Гарантированный результат** - CV fallback работает ВСЕГДА  
✅ **Научные методы** - Delta E CIEDE2000 для цветов  
✅ **Быстро** - локальное выполнение на вашей машине  
✅ **Приватно** - никакие данные не отправляются на сервер  

## Сравнение с другими плагинами

| Плагин | И интернет | Точность | Скорость | Цена | Fallback |
|--------|---------|---------|---------|------|----------|
| **OpenALPR** | ❌ Нет | Хорошая | Быстро | Бесплатно | ✅ CV |
| Plate Recognizer | ✅ Да | Очень высокая | Медленно | 2500 запросов/мес | ❌ |
| Qwen2-VL-2B | ❌ Нет | Хорошая | Быстро | Бесплатно | ❌ |
| VehicleAnalyzer | ❌ Нет | Средняя | Быстро | Бесплатно | ✅ CV |

## Известные ограничения

- ❌ Менее точен чем облачные API (Plate Recognizer)
- ❌ CLIP модель загружается при первом запуске (~340MB)
- ❌ OpenALPR - опциональна, но рекомендуется

## История версий

**v2.0** (28.03.2026)
- ✨ Добавлен CV fallback для надежности
- ✨ Метрика Delta E CIEDE2000 для цветов (вместо Евклида)
- ✨ Фильтрация шума в K-means
- 🐛 Исправлена нестабильность определения цвета
- 🐛 Добавлено логирование для дебага

