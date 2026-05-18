import os, sys, json, traceback
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)

# Тест только на 5 файлах
folder = os.path.join(_ROOT, 'Test foto', 'Persons')
files = [
    'clock_1_90percent_00-00-09.jpg',
    'person_1_91percent_00-00-00.jpg',
    'person_2_90percent_00-00-01.jpg',
    'keyboard_1_90percent_00-00-09.jpg',
    'umbrella_1_93percent_00-00-00.jpg',
]

from Plugins.PersonAnalyzer.person_analyzer import analyze_person

def cb(msg, step):
    sys.stdout.write(f'  [{step}] {msg}\n')
    sys.stdout.flush()

for fname in files:
    fpath = os.path.join(folder, fname)
    sys.stdout.write(f'\n--- {fname} ---\n')
    sys.stdout.flush()
    try:
        result = analyze_person(fpath, cb)
        sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        sys.stdout.flush()
    except Exception:
        sys.stdout.write('EXCEPTION:\n')
        sys.stdout.write(traceback.format_exc())
        sys.stdout.flush()
