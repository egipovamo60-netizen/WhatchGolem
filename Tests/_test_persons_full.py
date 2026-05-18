import os, sys, json, traceback
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)

from Plugins.PersonAnalyzer.person_analyzer import analyze_person

folder = os.path.join(_ROOT, 'Test foto', 'Persons')
extensions = ('.jpg', '.jpeg', '.png', '.bmp')
files = sorted(f for f in os.listdir(folder) if f.lower().endswith(extensions))

results = []
ok = 0
err = 0
err_no_face = 0

print(f"Всего файлов: {len(files)}")
sys.stdout.flush()

for i, fname in enumerate(files):
    fpath = os.path.join(folder, fname)
    try:
        r = analyze_person(fpath)
        if 'error' in r:
            err += 1
            if 'Лицо не обнаружено' in r.get('error', ''):
                err_no_face += 1
        else:
            ok += 1
        results.append(r)
    except Exception:
        tb = traceback.format_exc()
        results.append({
            'file': fname,
            'path': os.path.abspath(fpath),
            'error': f'ИСКЛЮЧЕНИЕ: {tb}',
        })
        err += 1

    if (i + 1) % 10 == 0 or (i + 1) == len(files):
        print(f"  [{i+1}/{len(files)}] ok={ok}, err_no_face={err_no_face}, err_other={err - err_no_face}")
        sys.stdout.flush()

# Итог
print(f"\n=== ИТОГ ===")
print(f"Всего: {len(results)}")
print(f"Успешно проанализировано (лицо найдено): {ok}")
print(f"Нет лица: {err_no_face}")
print(f"Другие ошибки: {err - err_no_face}")
sys.stdout.flush()

# Статистика по типам объектов
from collections import Counter
type_stats = Counter()
for r in results:
    prefix = r['file'].split('_')[0]
    if 'error' in r:
        if 'Лицо не обнаружено' in r.get('error', ''):
            type_stats[prefix + ' (нет лица)'] += 1
        else:
            type_stats[prefix + ' (ошибка)'] += 1
    else:
        type_stats[prefix + ' (OK)'] += 1

print("\nСтатистика по типам:")
for k, v in sorted(type_stats.items()):
    print(f"  {k}: {v}")
sys.stdout.flush()

# Примеры успешных результатов
print("\nПримеры успешно обработанных изображений:")
shown = 0
for r in results:
    if 'error' not in r and shown < 3:
        g = r.get('gender', {}).get('gender', '?')
        age = r.get('age', '?')
        upper = r.get('clothing', {}).get('upper', {})
        lower = r.get('clothing', {}).get('lower', {})
        upper_str = f"{upper.get('color','?')} {upper.get('type','?')}" if upper else '-'
        lower_str = f"{lower.get('color','?')} {lower.get('type','?')}" if lower else '-'
        print(f"  {r['file']}: {g}, {age} лет, верх={upper_str}, низ={lower_str}")
        shown += 1
sys.stdout.flush()
