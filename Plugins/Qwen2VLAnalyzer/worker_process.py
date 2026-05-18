"""
Subprocess-воркер для Qwen2-VL-2B.
Запускается как отдельный процесс, полностью изолированный от Qt.

Протокол:
  stdin  -> JSON-строки с командами: {"action": "analyze", "image": "/path/to/img.jpg"}
  stdout -> JSON-строки с ответами: {"status": "progress", "step": "...", "num": 0}
                                    {"status": "result", "data": {...}}
                                    {"status": "error", "message": "..."}
  Специальная строка "READY" выводится когда модель загружена.
"""
import os
import sys
import json

# Форсируем UTF-8 для stdout/stdin на Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Добавляем корень проекта в путь
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

MODEL_DIR = os.path.join(BASE_DIR, "Models", "Qwen2-VL-2B-Instruct")


def emit(obj: dict):
    """Отправляет JSON-объект в stdout (построчно)."""
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def load_model():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def query(model, processor, image_path: str, prompt: str, max_new_tokens: int = 200) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    # Уменьшаем размер изображения для ускорения на CPU
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path, "min_pixels": 224 * 224, "max_pixels": 448 * 448},
        {"type": "text", "text": prompt},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")

    model_dtype = next(model.parameters()).dtype
    inputs = {
        k: (v.to(dtype=model_dtype) if v.is_floating_point() else v)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    return processor.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()


def analyze(model, processor, image_path: str) -> dict:
    import os, re, json as _json

    def parse_json(text: str) -> dict:
        """Извлекает JSON из ответа модели, включая вложенные объекты."""
        text = text.strip()
        # Удаляем markdown-блоки
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = text.replace('```', '').strip()
        # Прямой парсинг
        try:
            return _json.loads(text)
        except Exception:
            pass
        # Ищем самый внешний JSON-объект с вложенными {}
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        return _json.loads(text[start:i+1])
                    except Exception:
                        start = -1
        return {}

    def parse_nested_json(text: str, key: str) -> dict:
        """Извлекает вложенный JSON-объект по ключу из текста."""
        # Ищем паттерн "key": { ... }
        pattern = rf'"{key}"\s*:\s*(\{{[^}}]+\}})'
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return _json.loads(m.group(1))
            except Exception:
                pass
        return {}

    # Один запрос вместо трёх — в 3 раза быстрее
    emit({"status": "progress", "step": "Анализ изображения...", "num": 1})
    combined_prompt = (
        "Проанализируй ТОЛЬКО транспортное средство на изображении и ответь СТРОГО валидным JSON без пояснений. "
        "Используй формат: "
        "{\"color\": {\"color\": \"цвет\", \"confidence\": 0-100}, "
        "\"model\": {\"brand\": \"марка\", \"model\": \"модель\", \"confidence\": 0-100}, "
        "\"plate\": {\"plate\": \"номер или null\", \"confidence\": 0-100}}. "
        "Цвет выбирай из: черный, антрацитовый, графитовый, темно-серый, серый, серебристый, белый, синий, красный, зеленый, желтый, оранжевый, коричневый, бежевый. "
        "Для номера используй формат российского госномера типа А123ВС77 или А123ВС777, без пробелов. "
        "Если символ не уверен, выбирай наиболее вероятный, но не добавляй лишних символов."
    )
    try:
        resp = query(model, processor, image_path, combined_prompt, max_new_tokens=220)
        # Отправляем сырой ответ для отладки
        emit({"status": "debug", "raw": resp})
        d = parse_json(resp)
    except Exception as e:
        d = {}
        emit({"status": "progress", "step": f"Ошибка анализа: {e}", "num": 1})

    # --- Цвет ---
    color_d = d.get("color", {})
    if isinstance(color_d, dict) and color_d:
        color = {"color": color_d.get("color", "Не определён"),
                 "confidence": min(max(float(color_d.get("confidence", 0)) / 100.0, 0.0), 1.0)}
    else:
        color = {"color": "Не определён", "confidence": 0.0}

    emit({"status": "progress", "step": "Обработка результатов...", "num": 2})

    # --- Марка/модель ---
    model_d = d.get("model", {})
    if isinstance(model_d, dict) and model_d:
        brand = model_d.get("brand", "Не определена")
        mname = model_d.get("model", "")
        full = f"{brand} {mname}".strip() if mname else brand
        car_model = {"model": full,
                     "confidence": min(max(float(model_d.get("confidence", 0)) / 100.0, 0.0), 1.0)}
    else:
        car_model = {"model": "Не определена", "confidence": 0.0}

    # --- Номер ---
    plate_d = d.get("plate", {})
    if isinstance(plate_d, dict) and plate_d:
        plate_raw = plate_d.get("plate")
        conf = min(max(float(plate_d.get("confidence", 0)) / 100.0, 0.0), 1.0)
    else:
        plate_raw = None
        conf = 0.0

    if plate_raw and plate_raw != "null":
        pc = re.sub(r"[^A-ZА-Я0-9]", "", str(plate_raw).upper())
        plate = {"plate_text": pc if len(pc) >= 3 else None, "confidence": conf,
                 "message": f"Номер: {pc} (уверенность: {conf:.0%})" if len(pc) >= 3 else "Номерной знак не распознан"}
    else:
        plate = {"plate_text": None, "confidence": 0.0, "message": "Номерной знак не обнаружен"}

    from datetime import datetime
    return {
        "file": os.path.basename(image_path),
        "path": image_path,
        "color": color,
        "model": car_model,
        "plate": plate,
        "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "engine": "Qwen2-VL-2B",
    }


def main():
    if not os.path.isdir(MODEL_DIR):
        emit({"status": "error", "message": f"Модель не найдена: {MODEL_DIR}"})
        sys.exit(1)

    emit({"status": "progress", "step": "Загрузка Qwen2-VL-2B...", "num": 0})
    try:
        model, processor = load_model()
    except Exception as e:
        emit({"status": "error", "message": f"Ошибка загрузки модели: {e}"})
        sys.exit(1)

    # Сигнал готовности
    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except Exception:
            continue

        action = cmd.get("action")
        if action == "analyze":
            image_path = cmd.get("image", "")
            if not os.path.isfile(image_path):
                emit({"status": "result", "data": {"error": f"Файл не найден: {image_path}",
                                                    "file": os.path.basename(image_path)}})
                continue
            try:
                result = analyze(model, processor, image_path)
                emit({"status": "result", "data": result})
            except Exception as e:
                emit({"status": "error", "message": str(e)})
        elif action == "quit":
            break


if __name__ == "__main__":
    main()
