import os, sys, json
sys.path.insert(0, '.')
from Plugins.PersonAnalyzer.person_analyzer import analyze_person

r = analyze_person(r'Test foto\Persons\keyboard_1_90percent_00-00-09.jpg')
print("Keys:", list(r.keys()))
print(json.dumps(r, indent=2, ensure_ascii=False))
