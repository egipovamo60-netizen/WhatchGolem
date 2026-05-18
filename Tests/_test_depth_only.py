import os, time, sys, numpy as np
os.environ['TRANSFORMERS_VERBOSITY'] = 'warning'

import cv2, torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

print("Loading depth model...")
t0 = time.time()
try:
    proc = AutoImageProcessor.from_pretrained("Intel/dpt-hybrid-midas", local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", local_files_only=True)
    print(f"  loaded local cache in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  local_files_only failed: {e}")
    proc = AutoImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
    print(f"  loaded from HF in {time.time()-t0:.1f}s")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Moving to {device}...")
model = model.to(device).eval()
print(f"  ready in {time.time()-t0:.1f}s")

img = cv2.imread(r"z:\Coding\Watch golem stable\Test foto\Transport\One\Car_one__test_foto_001.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
print(f"Image: {w}x{h}")

scale = 512 / max(h, w)
if scale < 1.0:
    img = cv2.resize(img, (int(w*scale), int(h*scale)))
pil = Image.fromarray(img)
inputs = proc(images=pil, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
print("Running inference...")
t1 = time.time()
with torch.no_grad():
    out = model(**inputs)
    depth = out.predicted_depth.squeeze().detach().cpu().numpy()
print(f"  done in {time.time()-t1:.2f}s, shape={depth.shape}")
print("SUCCESS")
