"""
Скачивает Qwen2-VL-2B-Instruct (transformers format) с HuggingFace.
Размер: ~4.5 GB (fp16 safetensors)

Использование:
    python download_qwen2vl2b.py
"""
from huggingface_hub import snapshot_download
import os

REPO_ID = "Qwen/Qwen2-VL-2B-Instruct"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models", "Qwen2-VL-2B-Instruct")

if os.path.isdir(LOCAL_DIR) and os.path.isfile(os.path.join(LOCAL_DIR, "config.json")):
    print(f"Модель уже скачана: {LOCAL_DIR}")
else:
    print(f"Скачиваем {REPO_ID} -> {LOCAL_DIR}")
    print("Размер: ~4.5 GB, может занять несколько минут...\n")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        ignore_patterns=["*.bin"],  # только safetensors
    )
    print(f"\nГотово: {LOCAL_DIR}")
