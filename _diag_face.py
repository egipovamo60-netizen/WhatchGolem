import os, sys
sys.path.insert(0, '.')
import cv2, numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

img_path = r"Test foto\Persons\person_1_91percent_00-00-00.jpg"
img = cv2.imread(img_path)
faces = app.get(img)
print(f"Найдено лиц: {len(faces)}")
if faces:
    f = faces[0]
    print(f"Тип: {type(f)}")
    print(f"isinstance dict: {isinstance(f, dict)}")
    print(f"Ключи dict: {list(f.keys()) if isinstance(f, dict) else 'N/A'}")
    print(f"f.gender = {getattr(f, 'gender', 'NO_ATTR')}")
    print(f"f.get gender = {f.get('gender', 'NO_KEY') if isinstance(f, dict) else 'N/A'}")
    print(f"f.sex = {getattr(f, 'sex', 'NO_ATTR')}")
    print(f"f.age = {getattr(f, 'age', 'NO_ATTR')}")
    print(f"f.det_score = {getattr(f, 'det_score', 'NO_ATTR')}")
    print(f"f.embedding shape = {f.embedding.shape if f.embedding is not None else None}")
    if isinstance(f, dict):
        for k, v in f.items():
            if not k.startswith('_'):
                vr = repr(v)[:60] if v is not None else 'None'
                print(f"  [{k}] = {vr}")
