"""Диагностика Qwen2.5-VL GGUF."""
import os, base64, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
mmproj = os.path.join(BASE, "Models", "Qwen2.5-VL-3B-GGUF", "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf")
model_f = os.path.join(BASE, "Models", "Qwen2.5-VL-3B-GGUF", "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf")

# Ищем тестовое изображение
test_img = None
test_dir = os.path.join(BASE, "Test foto")
if os.path.isdir(test_dir):
    for f in os.listdir(test_dir):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            test_img = os.path.join(test_dir, f)
            break

if not test_img:
    print("ОШИБКА: тестовое изображение не найдено в 'Test foto'")
    sys.exit(1)

print(f"Изображение: {test_img}")

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

print("Создаём handler...")
handler = Qwen25VLChatHandler(clip_model_path=mmproj, verbose=False)
print("Handler создан")

print("Загружаем модель...")
m = Llama(model_path=model_f, chat_handler=handler, n_ctx=512, n_threads=4, verbose=False)
print("Модель загружена, запускаем inference...")

with open(test_img, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
data_uri = f"data:image/jpeg;base64,{b64}"

resp = m.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": 'What color is this vehicle? Reply JSON only: {"color": "..."}'},
            ],
        }
    ],
    max_tokens=40,
    temperature=0.1,
)
print("Ответ:", resp["choices"][0]["message"]["content"])
