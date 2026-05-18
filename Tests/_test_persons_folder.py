import os, sys, json
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
from Plugins.PersonAnalyzer import analyze_person_folder

folder = os.path.join(_ROOT, 'Test foto', 'Persons')

def cb(msg, step):
    print(f'  [{step}] {msg}')

results = analyze_person_folder(folder, cb)

ok = [r for r in results if 'error' not in r]
err = [r for r in results if 'error' in r]

print(f'\n=== Итого: {len(results)} файлов, успешно: {len(ok)}, ошибок: {len(err)} ===')
for r in err[:20]:
    print(f'  ОШИБКА [{r["file"]}]: {r["error"]}')

if ok:
    print('\nПример успешного результата:')
    print(json.dumps(ok[0], indent=2, ensure_ascii=False))
