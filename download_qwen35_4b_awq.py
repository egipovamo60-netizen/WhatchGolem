"""
Скачивает Qwen3.5-4B-AWQ-4bit с HuggingFace.
Размер: ~2.5-3 GB (AWQ 4-bit)
Требует: pip install autoawq (для инференса)

Использование:
    python download_qwen35_4b_awq.py
"""
from huggingface_hub import snapshot_download
import os

REPO_ID = "cyankiwi/Qwen3.5-4B-AWQ-4bit"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models", "Qwen3.5-4B-AWQ")

if os.path.isdir(LOCAL_DIR) and os.path.isfile(os.path.join(LOCAL_DIR, "config.json")):
    print(f"Модель уже скачана: {LOCAL_DIR}")
else:
    print(f"Скачиваем {REPO_ID} -> {LOCAL_DIR}")
    print("Размер: ~2.5-3 GB, может занять несколько минут...\n")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
    )
    print(f"\nГотово: {LOCAL_DIR}")
