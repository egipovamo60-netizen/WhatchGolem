# Датасеты моделей автомобилей

Эта папка содержит помощников и примеры для использования больших датасетов моделей с OpenALPR Analyzer.

## Файлы

### 📋 Скрипты подготовки

- **prepare_compcars.py** — Конвертирует CompCars Dataset (1655 моделей) в JSON
- **prepare_stanford.py** — Конвертирует Stanford Cars Dataset (196 классов) в JSON

### 📦 Примеры

- **compcars_example.json** — Пример структуры для CompCars (40 моделей)
- **stanford_example.json** — Пример структуры для Stanford (10 классов)

## Быстрый старт

### Вариант 1: CompCars (1655 моделей)

```bash
# 1. Скачайте датасет
git clone https://github.com/byu-object-recognition-lab/CompCars.git

# 2. Подготовьте JSON
python prepare_compcars.py /path/to/CompCars/data/

# 3. Получится: compcars_1655.json (в текущей папке)

# 4. Обновите конфиг
# config.json:
# {
#     "dataset": "compcars",
#     "compcars_path": "Plugins/OpenALPRAnalyzer/Datasets/compcars_1655.json"
# }
```

### Вариант 2: Stanford Cars (196 классов)

```bash
# 1. Скачайте датасет
wget http://ai.stanford.edu/~jkrause/cars/car_ims.tgz
wget http://ai.stanford.edu/~jkrause/cars/cars_annos.mat
tar -xzf car_ims.tgz

# 2. Подготовьте JSON
python prepare_stanford.py cars_annos.mat

# 3. Получится: stanford_cars_196.json и stanford_cars_196.txt

# 4. Обновите конфиг
# config.json:
# {
#     "dataset": "stanford",
#     "stanford_path": "Plugins/OpenALPRAnalyzer/Datasets/stanford_cars_196.json"
# }
```

### Вариант 3: Собственный датасет

```bash
# Создайте JSON файл:
cat > my_cars.json << 'EOF'
{
  "source": "My Custom Dataset",
  "total_models": 50,
  "models": [
    "Toyota Camry",
    "BMW X5",
    "Lada Vesta",
    ...
  ]
}
EOF

# Или текстовый файл (по одной модели в строке):
cat > my_cars.txt << 'EOF'
Toyota Camry
BMW X5
Lada Vesta
...
EOF

# Обновите config.json с путем к вашему файлу
```

## Требования

### Для CompCars
```bash
pip install numpy
python prepare_compcars.py ...
```

### Для Stanford
```bash
pip install scipy  # Важно!
python prepare_stanford.py cars_annos.mat
```

## Результаты

После подготовки вы получите:

```
Datasets/
├── compcars_1655.json       # ✅ 1655 моделей
├── stanford_cars_196.json   # ✅ 196 классов
├── stanford_cars_196.txt    # ✅ Альтернативный формат
├── compcars_example.json    # Пример структуры
├── stanford_example.json    # Пример структуры
├── prepare_compcars.py      # Скрипт подготовки
└── prepare_stanford.py      # Скрипт подготовки
```

## Активация в плагине

Обновите `config.json` в папке плагина:

```json
{
    "region": "ru",
    "clip_model": "openai/clip-vit-base-patch32",
    "dataset": "compcars",
    "compcars_path": "Plugins/OpenALPRAnalyzer/Datasets/compcars_1655.json",
    "stanford_path": ""
}
```

Затем перезагрузите UI (Ctrl+Shift+P → Reload Window) или перезагрузите Python интерпретатор.

## Производительность

| Датасет | Размер | Загрузка | Инференс |
|---------|--------|----------|---------|
| Default (24) | ~1 MB | instant | ~100ms |
| CompCars (1655) | ~80 KB JSON | 500ms | 500-800ms |
| Stanford (196) | ~15 KB JSON | 200ms | 300-500ms |

Размер JSON благодаря компактности текстовых названий очень небольшой!

## Помощь

**Обой при запуске scripts:**

```
❌ ModuleNotFoundError: No module named 'scipy'
```

Решение:
```bash
pip install scipy
```

---

**Файл не найден:**

```
❌ Файл не найден: cars_annos.mat
```

Решение:
1. Убедитесь, что скачали правильный .mat файл
2. Используйте абсолютный путь
3. Проверьте каталог файла

---

**Пустой выход из скрипта:**

Вероятно, скрипт не нашел структуру датасета. Проверьте файл вручную:

```python
import scipy.io
annos = scipy.io.loadmat('cars_annos.mat')
print(annos.keys())  # Посмотрите, какие ключи есть
```

## Ссылки

- **CompCars Dataset**: https://github.com/byu-object-recognition-lab/CompCars
- **Stanford Cars Dataset**: http://ai.stanford.edu/~jkrause/cars/car_dataset.html

