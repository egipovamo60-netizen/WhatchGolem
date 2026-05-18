"""Диагностика загрузки Qwen2-VL-2B вне Qt."""
import os, sys, traceback
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE, "Models", "Qwen2-VL-2B-Instruct")

print(f"Модель: {MODEL_DIR}")
print(f"Файлы: {os.listdir(MODEL_DIR)}")

import torch
print(f"torch: {torch.__version__}")
print(f"bfloat16 поддержка: {torch.backends.cpu.get_cpu_capability()}")

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

print("\nЗагружаем processor...", flush=True)
try:
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print("Processor OK")
except Exception as e:
    print(f"ОШИБКА processor: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nЗагружаем модель (bfloat16, no device_map)...", flush=True)
try:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    print("Модель загружена OK")
    print(f"Параметры: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
except Exception as e:
    print(f"ОШИБКА загрузки: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nТест inference...")
try:
    import glob
    imgs = glob.glob(os.path.join(BASE, "Test foto", "*.jpg"))
    if not imgs:
        print("Нет тестовых изображений в 'Test foto'")
        sys.exit(0)
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": imgs[0]},
        {"type": "text", "text": "What color is this vehicle? Reply: {\"color\": \"...\"}"},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt")
    model_dtype = next(model.parameters()).dtype
    inputs = {k: (v.to(dtype=model_dtype) if v.is_floating_point() else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    result = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Ответ: {result}")
except Exception as e:
    print(f"ОШИБКА inference: {e}")
    traceback.print_exc()
