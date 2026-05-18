"""
Плагин open-set детекции объектов для Watch Golem.

Подход:
  1. Генерация candidate boxes через YOLO и generic proposals.
  2. Оценка "известности" каждого кандидата через OpenCLIP.
    3. Оценка относительной глубины сцены через monocular depth model.
    4. Объекты с низкой похожестью на известные классы помечаются как Unknown.

Зависимости:
    pip install ultralytics open-clip-torch Pillow python-docx transformers
"""

import os
# Safetensors на Windows падает с ACCESS_VIOLATION при прямой загрузке на CUDA.
# Устанавливаем до любого импорта safetensors/open_clip/transformers.
os.environ["SAFETENSORS_FAST_GPU"] = "0"
# Ограничиваем таймаут загрузки моделей с HuggingFace Hub.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

import sys
import io
import json
import shutil
import threading
import concurrent.futures
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from logger import get_logger

_log = get_logger("OpenSetAnalyzer")

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_yolo_model = None
_yolo_weights_path = None
_clip_model = None
_clip_preprocess = None
_tokenize = None
_torch = None
_PILImage = None
_clip_device = None
_text_feature_cache = {}
_depth_model = None
_depth_image_processor = None
_depth_device = None
_depth_load_error = None

_YOLO_WEIGHT_CANDIDATES = (
    "yolo11x.pt",
    "yolo11n.pt",
    "yolov8n.pt",
)
_DEPTH_MODEL_ID = os.environ.get("OPENSET_DEPTH_MODEL", "Intel/dpt-hybrid-midas")
_CLIP_TEMPLATES = (
    "a photo of a {}",
    "a close-up photo of a {}",
    "a cropped photo of a {}",
    "a blurry photo of a {}",
    "an image of a {}",
)

_DEFAULT_DET_CONFIDENCE = 0.15
_DEFAULT_KNOWN_SIM_THRESHOLD = 0.24
_DEFAULT_STRONG_SIM_THRESHOLD = 0.30
_DEFAULT_MARGIN_THRESHOLD = 0.015
_DEFAULT_YOLO_SUPPORT_CONFIDENCE = 0.35
_DEFAULT_GENERIC_TOPK = 12
_DEFAULT_MAX_PROPOSALS = 24
_DEFAULT_VIDEO_SAMPLE_INTERVAL = 1.0
_DEFAULT_VIDEO_MAX_KEYFRAMES = 12
_DEFAULT_UNKNOWN_CROP_MARGIN_SCALE = 0.12
_DEFAULT_UNKNOWN_CLASS_SIMILARITY = 0.88
_UNKNOWN_CLASS_DB_DIRNAME = "UnknownClassDB"
_UNKNOWN_CLASS_DB_CLASSES_DIRNAME = "Classes"
_UNKNOWN_CLASS_DB_REPORT_NAME = "UNKNOWN_CLASS_DATABASE_REPORT.txt"
_UNKNOWN_CLASS_DB_JSON_NAME = "UNKNOWN_CLASS_DATABASE_REPORT.json"

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v")


def _resolve_weights_path():
    for candidate in _YOLO_WEIGHT_CANDIDATES:
        full_path = os.path.join(_BASE_DIR, candidate)
        if os.path.isfile(full_path):
            return full_path
    raise FileNotFoundError(
        "Не найдены веса YOLO для open-set плагина. "
        f"Ожидались: {', '.join(_YOLO_WEIGHT_CANDIDATES)}"
    )


def _load_yolo(progress_callback=None):
    global _yolo_model, _yolo_weights_path
    if _yolo_model is not None:
        return _yolo_model

    if progress_callback:
        progress_callback("Загрузка YOLO для proposals...", 0)

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "Ultralytics не установлен. Установите: pip install ultralytics"
        ) from e

    _yolo_weights_path = _resolve_weights_path()
    _yolo_model = YOLO(_yolo_weights_path)
    _log.info(f"YOLO загружен для OpenSetAnalyzer: {os.path.basename(_yolo_weights_path)}")
    return _yolo_model


def _get_clip_device():
    global _clip_device
    if _clip_device is not None:
        return _clip_device
    try:
        import torch
        if torch.cuda.is_available():
            _clip_device = "cuda"
            _log.info(f"OpenSetAnalyzer: GPU — {torch.cuda.get_device_name(0)}")
        else:
            _clip_device = "cpu"
    except Exception:
        _clip_device = "cpu"
    return _clip_device


def _load_clip(progress_callback=None):
    global _clip_model, _clip_preprocess, _tokenize, _torch, _PILImage
    if _clip_model is not None:
        return _clip_model, _clip_preprocess, _tokenize

    if progress_callback:
        progress_callback("Загрузка OpenCLIP...", 1)

    try:
        import os as _os
        import torch
        import open_clip
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "OpenCLIP зависимости не установлены. Установите: pip install open-clip-torch Pillow"
        ) from e

    _torch = torch
    _PILImage = Image
    device = _get_clip_device()

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # Прогрев: запускаем dummy forward pass чтобы сразу скомпилировать CUDA кернелы
    if device == "cuda":
        try:
            with torch.no_grad():
                _dummy = model.encode_image(torch.zeros(1, 3, 224, 224, device=device))
                _dummy = model.encode_text(tokenizer(["warmup"]).to(device))
            _log.info("OpenCLIP GPU warmup завершён")
        except Exception:
            pass

    _clip_model = model
    _clip_preprocess = preprocess
    _tokenize = tokenizer
    _log.info(f"OpenCLIP загружен (ViT-B-32, device={device})")
    return _clip_model, _clip_preprocess, _tokenize


def _load_depth_model(progress_callback=None, allow_failure=True):
    global _depth_model, _depth_image_processor, _depth_device, _depth_load_error
    global _torch, _PILImage

    if _depth_model is not None and _depth_image_processor is not None:
        return _depth_model, _depth_image_processor

    if _depth_load_error is not None and allow_failure:
        return None, None

    if progress_callback:
        progress_callback("Загрузка depth model...", 1)

    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as e:
        message = (
            "Depth model зависимости не установлены. "
            "Установите: pip install transformers Pillow torch"
        )
        if allow_failure:
            _depth_load_error = message
            _log.warning(message)
            return None, None
        raise ImportError(message) from e

    _torch = torch
    _PILImage = Image
    # Пробуем GPU, при ошибке откатываемся на CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # DPT-Hybrid на CPU слишком медленный (30–120 с/кадр) — отключаем.
    if device == "cpu":
        message = (
            f"Depth model ({_DEPTH_MODEL_ID}) пропущен: GPU недоступен. "
            "Оценка глубины работает только с CUDA."
        )
        _depth_load_error = message
        _log.warning(message)
        return None, None

    try:
        image_processor = AutoImageProcessor.from_pretrained(
            _DEPTH_MODEL_ID, local_files_only=True
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            _DEPTH_MODEL_ID, local_files_only=True
        )
    except Exception:
        # local_files_only не нашёл кэш — загружаем с HuggingFace
        try:
            image_processor = AutoImageProcessor.from_pretrained(_DEPTH_MODEL_ID)
            model = AutoModelForDepthEstimation.from_pretrained(_DEPTH_MODEL_ID)
        except Exception as exc:
            message = f"Не удалось загрузить depth model {_DEPTH_MODEL_ID}: {exc}"
            if allow_failure:
                _depth_load_error = message
                _log.warning(message)
                return None, None
            raise RuntimeError(message) from exc

    for target_device in ([device, "cpu"] if device != "cpu" else ["cpu"]):
        try:
            model = model.to(target_device).eval()
            device = target_device
            break
        except Exception as exc:
            if target_device == "cpu":
                message = f"Depth model не удалось загрузить на CPU: {exc}"
                if allow_failure:
                    _depth_load_error = message
                    _log.warning(message)
                    return None, None
                raise RuntimeError(message) from exc
            _log.warning(f"Depth model: GPU недоступен ({exc}), используем CPU")

    _depth_model = model
    _depth_image_processor = image_processor
    _depth_device = device
    _depth_load_error = None
    _log.info(f"Depth model загружен ({_DEPTH_MODEL_ID}, device={device})")
    return _depth_model, _depth_image_processor


def _get_known_labels(yolo_model, known_labels=None):
    if known_labels:
        labels = [str(label).strip() for label in known_labels if str(label).strip()]
        if labels:
            return labels

    names = yolo_model.names
    if isinstance(names, dict):
        return [names[idx] for idx in sorted(names)]
    return [str(name) for name in names]


def _sanitize_label(label):
    return str(label).replace("_", " ").strip()


def _get_text_features(labels):
    _load_clip()
    key = tuple(labels)
    if key in _text_feature_cache:
        return _text_feature_cache[key]

    prompts = []
    label_indices = []
    for idx, label in enumerate(labels):
        clean_label = _sanitize_label(label)
        for template in _CLIP_TEMPLATES:
            prompts.append(template.format(clean_label))
            label_indices.append(idx)

    device = _get_clip_device()
    text_tokens = _tokenize(prompts).to(device)

    with _torch.no_grad():
        prompt_features = _clip_model.encode_text(text_tokens)
        prompt_features /= prompt_features.norm(dim=-1, keepdim=True)

    by_label = []
    for idx in range(len(labels)):
        gathered = prompt_features[[i for i, owner in enumerate(label_indices) if owner == idx]]
        feature = gathered.mean(dim=0)
        feature /= feature.norm(dim=-1, keepdim=True)
        by_label.append(feature)

    features = _torch.stack(by_label, dim=0)
    _text_feature_cache[key] = features
    return features


def _encode_crop(crop_bgr):
    _load_clip()
    device = _get_clip_device()

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = _PILImage.fromarray(rgb)
    img_tensor = _clip_preprocess(pil_image).unsqueeze(0).to(device)

    with _torch.no_grad():
        img_feature = _clip_model.encode_image(img_tensor)
        img_feature /= img_feature.norm(dim=-1, keepdim=True)

    return img_feature.squeeze(0)


def _clip_scores(crop_bgr, known_labels):
    text_features = _get_text_features(known_labels)
    image_feature = _encode_crop(crop_bgr)

    with _torch.no_grad():
        similarities = (image_feature @ text_features.T).float()
        probabilities = (similarities * 10.0).softmax(dim=0)

    order = similarities.argsort(descending=True)
    top_idx = int(order[0].item())
    second_idx = int(order[1].item()) if len(known_labels) > 1 else top_idx

    return {
        "top_label": known_labels[top_idx],
        "top_similarity": float(similarities[top_idx].item()),
        "second_similarity": float(similarities[second_idx].item()),
        "top_probability": float(probabilities[top_idx].item()),
        "all_similarities": similarities.cpu().numpy(),
    }


_DEPTH_MAX_SIDE = 512  # ограничение входного разрешения depth model

def _compute_depth_map(image_bgr):
    model, image_processor = _load_depth_model(allow_failure=True)
    if model is None or image_processor is None:
        return None

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb.shape[:2]

    # Уменьшаем до _DEPTH_MAX_SIDE для ускорения инференса
    scale = _DEPTH_MAX_SIDE / max(orig_h, orig_w)
    if scale < 1.0:
        small = cv2.resize(rgb, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = rgb
        scale = 1.0

    pil_image = _PILImage.fromarray(small)
    inputs_raw = image_processor(images=pil_image, return_tensors="pt")
    inputs_on_device = {k: v.to(_depth_device) for k, v in inputs_raw.items()}

    _DEPTH_INFER_TIMEOUT = 20.0  # секунд; GPU инференс не должен занимать больше

    def _run_inference():
        with _torch.inference_mode():
            return model(**inputs_on_device)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_inference)
            outputs = future.result(timeout=_DEPTH_INFER_TIMEOUT)
    except concurrent.futures.TimeoutError:
        _log.warning(
            f"Depth inference превысил {_DEPTH_INFER_TIMEOUT:.0f}с — карта глубины пропущена."
        )
        return None
    except Exception as exc:
        _log.warning(f"Depth inference failed: {exc}")
        return None

    predicted_depth = outputs.predicted_depth
    depth_map = predicted_depth.squeeze().detach().cpu().numpy().astype(np.float32)
    depth_map[~np.isfinite(depth_map)] = 0.0
    # Возвращаем маленькую карту и коэффициент масштаба для пересчёта боксов
    return {"map": depth_map, "scale": float(scale)}


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def _nms(proposals, iou_threshold=0.45):
    proposals = sorted(proposals, key=lambda item: item["score"], reverse=True)
    kept = []
    for proposal in proposals:
        if all(_iou(proposal["box"], existing["box"]) < iou_threshold for existing in kept):
            kept.append(proposal)
    return kept


def _clip_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    return x1, y1, x2, y2


def _expand_box(box, width, height, scale=0.08):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * scale)
    pad_y = int(bh * scale)
    return _clip_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def _generate_yolo_proposals(image, det_confidence, progress_callback=None):
    model = _load_yolo(progress_callback)
    if progress_callback:
        progress_callback("Генерация proposals через YOLO...", 2)

    result = model(image, verbose=False)[0]
    proposals = []
    for box in result.boxes:
        conf = float(box.conf[0].item())
        if conf < det_confidence:
            continue
        cls_id = int(box.cls[0].item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        proposals.append({
            "box": (x1, y1, x2, y2),
            "source": "yolo",
            "score": conf + 0.35,
            "det_confidence": conf,
            "yolo_label": str(model.names[cls_id]),
        })
    return proposals


def _generate_generic_proposals(image, top_k=_DEFAULT_GENERIC_TOPK, progress_callback=None):
    if progress_callback:
        progress_callback("Генерация generic proposals...", 2)

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(32 * 32, int(h * w * 0.003))

    proposals = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < min_area:
            continue
        if bw < 24 or bh < 24:
            continue

        aspect = bw / max(bh, 1)
        if aspect > 8.0 or aspect < 0.2:
            continue

        fill_ratio = cv2.contourArea(contour) / max(area, 1)
        if fill_ratio < 0.15:
            continue

        proposals.append({
            "box": (x, y, x + bw, y + bh),
            "source": "generic",
            "score": min(0.45, 0.10 + fill_ratio * 0.15 + area / float(h * w)),
            "det_confidence": 0.0,
            "yolo_label": None,
        })

    proposals.sort(key=lambda item: ((item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1])), reverse=True)
    return proposals[:top_k]


def _merge_proposals(yolo_proposals, generic_proposals, width, height, max_proposals):
    merged = []
    for proposal in yolo_proposals + generic_proposals:
        proposal = dict(proposal)
        proposal["box"] = _expand_box(proposal["box"], width, height)
        merged.append(proposal)

    merged = _nms(merged, iou_threshold=0.45)
    return merged[:max_proposals]


def _classify_proposal(crop, proposal, known_labels, thresholds):
    clip_result = _clip_scores(crop, known_labels)
    top_similarity = clip_result["top_similarity"]
    second_similarity = clip_result["second_similarity"]
    top_probability = clip_result["top_probability"]
    margin = top_similarity - second_similarity
    yolo_label = proposal.get("yolo_label")
    det_confidence = float(proposal.get("det_confidence", 0.0) or 0.0)
    source = proposal.get("source", "generic")

    known_sim_threshold = thresholds["known_similarity"]
    strong_sim_threshold = thresholds["strong_similarity"]
    margin_threshold = thresholds["margin"]
    yolo_support_confidence = thresholds["yolo_support_confidence"]

    yolo_supports_known = (
        source == "yolo"
        and det_confidence >= yolo_support_confidence
        and yolo_label in known_labels
        and (clip_result["top_label"] == yolo_label or top_similarity >= known_sim_threshold)
    )

    if source == "generic":
        is_known = (
            top_similarity >= strong_sim_threshold
            and top_probability >= 0.20
            and margin >= margin_threshold
        )
    else:
        is_known = yolo_supports_known or (
            top_similarity >= strong_sim_threshold and margin >= 0.005
        )

    novelty_from_similarity = max(0.0, strong_sim_threshold - top_similarity) / max(strong_sim_threshold, 1e-6)
    novelty_from_margin = max(0.0, margin_threshold - margin) / max(margin_threshold, 1e-6)
    unknown_score = float(np.clip(0.65 * novelty_from_similarity + 0.35 * novelty_from_margin, 0.0, 1.0))
    if is_known:
        unknown_score = max(0.0, min(1.0, 1.0 - top_probability))

    label = clip_result["top_label"] if is_known else "Unknown"
    return {
        "label": label,
        "is_known": is_known,
        "unknown_score": round(unknown_score, 3),
        "clip_label": clip_result["top_label"],
        "clip_similarity": round(top_similarity, 4),
        "clip_probability": round(top_probability, 4),
        "clip_margin": round(margin, 4),
        "detector_label": yolo_label,
        "detector_confidence": round(det_confidence, 4),
        "source": source,
    }


def _estimate_box_depth(depth_map, box):
    if depth_map is None:
        return None

    dm = depth_map["map"]
    scale = depth_map["scale"]
    height, width = dm.shape[:2]

    # Масштабируем координаты бокса к размеру depth map
    sx1 = int(box[0] * scale)
    sy1 = int(box[1] * scale)
    sx2 = int(box[2] * scale)
    sy2 = int(box[3] * scale)
    x1, y1, x2, y2 = _clip_box((sx1, sy1, sx2, sy2), width, height)
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)

    inset_x = max(1, int(box_width * 0.2))
    inset_y = max(1, int(box_height * 0.2))
    inner_x1 = min(x2 - 1, x1 + inset_x)
    inner_y1 = min(y2 - 1, y1 + inset_y)
    inner_x2 = max(inner_x1 + 1, x2 - inset_x)
    inner_y2 = max(inner_y1 + 1, y2 - inset_y)

    region = dm[inner_y1:inner_y2, inner_x1:inner_x2]
    if region.size == 0:
        region = dm[y1:y2, x1:x2]
    if region.size == 0:
        return None

    finite_values = region[np.isfinite(region)]
    if finite_values.size == 0:
        return None
    return float(np.median(finite_values))


def _attach_depth_to_detections(detections, depth_map):
    depth_info = {
        "enabled": depth_map is not None,
        "model": _DEPTH_MODEL_ID if depth_map is not None else None,
        "error": _depth_load_error,
    }

    if depth_map is None or not detections:
        return depth_info

    finite_map = depth_map["map"][np.isfinite(depth_map["map"])]
    if finite_map.size == 0:
        depth_info["enabled"] = False
        depth_info["error"] = depth_info["error"] or "Depth map пуста"
        return depth_info

    low = float(np.percentile(finite_map, 5))
    high = float(np.percentile(finite_map, 95))
    denom = max(high - low, 1e-6)

    scored_items = []
    for item in detections:
        raw_depth = _estimate_box_depth(depth_map, item["box"])
        if raw_depth is None:
            item["result"]["depth_score"] = None
            item["result"]["depth_bucket"] = None
            item["result"]["depth_rank"] = None
            continue

        depth_score = float(np.clip((raw_depth - low) / denom, 0.0, 1.0))
        if depth_score >= 0.66:
            depth_bucket = "near"
        elif depth_score >= 0.33:
            depth_bucket = "mid"
        else:
            depth_bucket = "far"

        item["result"]["depth_score"] = round(depth_score, 3)
        item["result"]["depth_bucket"] = depth_bucket
        item["result"]["depth_rank"] = None
        scored_items.append((raw_depth, item))

    for rank, (_, item) in enumerate(sorted(scored_items, key=lambda entry: entry[0], reverse=True), 1):
        item["result"]["depth_rank"] = rank

    depth_info.update({
        "enabled": True,
        "model": _DEPTH_MODEL_ID,
        "map_min": round(float(np.min(finite_map)), 4),
        "map_max": round(float(np.max(finite_map)), 4),
    })
    return depth_info


def _annotate_detections(image, detections):
    annotated = image.copy()
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        is_known = item["result"]["is_known"]
        color = (0, 200, 0) if is_known else (0, 140, 255)
        label = item["result"]["label"]
        confidence = item["result"]["clip_probability"] if is_known else item["result"]["unknown_score"]
        depth_bucket = item["result"].get("depth_bucket")
        depth_suffix = f" {depth_bucket}" if depth_bucket else ""
        caption = f"{label}{depth_suffix} [{confidence:.2f}]"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(y1 - th - baseline - 6, 0)
        cv2.rectangle(annotated, (x1, top), (x1 + tw + 6, top + th + baseline + 6), (20, 20, 20), cv2.FILLED)
        cv2.putText(
            annotated,
            caption,
            (x1 + 3, top + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _format_timestamp(seconds):
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total_ms, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def _build_engine_info():
    return {
        "proposal_detector": os.path.basename(_yolo_weights_path or "unknown"),
        "novelty_model": "OpenCLIP ViT-B-32",
        "depth_model": _DEPTH_MODEL_ID if _depth_model is not None else None,
    }


def _write_unknown_timestamps_file(frame_results, output_path):
    lines = [
        "UNKNOWN FRAMES",
        "=" * 64,
        "",
    ]

    unknown_frames = [frame for frame in frame_results if frame.get("unknown_count", 0) > 0]
    if not unknown_frames:
        lines.append("Unknown-объекты не найдены.")
    else:
        for index, frame in enumerate(unknown_frames, 1):
            lines.append(
                f"[{index}] frame={frame.get('frame_index', 0)} | "
                f"time={frame.get('timestamp', '00:00:00.000')} | "
                f"unknown={frame.get('unknown_count', 0)} | "
                f"known={frame.get('known_count', 0)} | "
                f"saved={frame.get('saved_frame_path') or '—'}"
            )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return output_path


def _cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def _normalize_embedding(vector):
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector
    return vector / norm


def _get_unknown_class_database_paths(db_root=None):
    root = os.path.abspath(db_root or os.path.join(_BASE_DIR, _UNKNOWN_CLASS_DB_DIRNAME))
    classes_dir = os.path.join(root, _UNKNOWN_CLASS_DB_CLASSES_DIRNAME)
    return {
        "root": root,
        "classes_dir": classes_dir,
        "report_path": os.path.join(root, _UNKNOWN_CLASS_DB_REPORT_NAME),
        "json_path": os.path.join(root, _UNKNOWN_CLASS_DB_JSON_NAME),
    }


def get_unknown_class_database_dir(db_root=None):
    return _get_unknown_class_database_paths(db_root)["root"]


def get_unknown_class_database_classes_dir(db_root=None):
    return _get_unknown_class_database_paths(db_root)["classes_dir"]


def _iter_recursive_images(folder_path):
    paths = []
    if not folder_path or not os.path.isdir(folder_path):
        return paths

    for current_root, dir_names, file_names in os.walk(folder_path):
        dir_names.sort()
        for file_name in sorted(file_names):
            if file_name.lower().endswith(_IMAGE_EXTENSIONS):
                paths.append(os.path.join(current_root, file_name))
    return paths


def _safe_source_token(value):
    token = str(value or "unknown").replace("/", "__").replace("\\", "__")
    token = token.replace(":", "_").strip("._ ")
    return token or "unknown"


def _unique_destination_path(directory, file_name):
    base_name, ext = os.path.splitext(file_name)
    candidate = os.path.join(directory, file_name)
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base_name}_{suffix:03d}{ext}")
        suffix += 1
    return candidate


def _parse_unknown_class_cluster_id(class_name, fallback):
    if class_name.startswith("UnknownClass_"):
        try:
            return int(class_name.split("_")[-1])
        except ValueError:
            return fallback
    return fallback


def _load_unknown_class_database_metadata(json_path):
    if not os.path.isfile(json_path):
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    lookup = {}
    for cluster in payload.get("clusters", []):
        for item in cluster.get("items", []):
            path = item.get("path") or item.get("copied_path")
            if path:
                lookup[os.path.abspath(path)] = item
    return lookup


def _load_unknown_class_database_clusters(db_root=None):
    db_paths = _get_unknown_class_database_paths(db_root)
    os.makedirs(db_paths["classes_dir"], exist_ok=True)
    metadata_lookup = _load_unknown_class_database_metadata(db_paths["json_path"])

    clusters = []
    total_images = 0
    for class_name in sorted(os.listdir(db_paths["classes_dir"])):
        class_dir = os.path.join(db_paths["classes_dir"], class_name)
        if not os.path.isdir(class_dir):
            continue

        image_paths = _iter_recursive_images(class_dir)
        if not image_paths:
            continue

        items = []
        embeddings = []
        source_dirs = set()
        source_labels = set()
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                continue

            embedding = _encode_crop(image).detach().cpu().numpy().astype(np.float32)
            embedding = _normalize_embedding(embedding)
            item_meta = metadata_lookup.get(os.path.abspath(image_path), {})
            item = {
                "path": os.path.abspath(image_path),
                "source_path": item_meta.get("source_path") or os.path.abspath(image_path),
                "source_dir": item_meta.get("source_dir") or "database",
                "source_label": item_meta.get("source_label") or "database",
                "similarity_to_cluster": float(item_meta.get("similarity_to_cluster", 1.0)),
            }
            items.append(item)
            embeddings.append(embedding)
            source_dirs.add(item["source_dir"])
            source_labels.add(item["source_label"])

        if not items:
            continue

        centroid = _normalize_embedding(np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32))
        cluster_id = _parse_unknown_class_cluster_id(class_name, len(clusters) + 1)
        clusters.append({
            "cluster_id": cluster_id,
            "class_name": class_name,
            "dir": class_dir,
            "items": items,
            "embeddings": embeddings,
            "centroid": centroid,
            "source_dirs": source_dirs,
            "source_labels": source_labels,
        })
        total_images += len(items)

    clusters.sort(key=lambda item: (item["cluster_id"], item["class_name"]))
    return clusters, total_images


def _serialize_unknown_class_database_clusters(clusters, similarity_threshold):
    serialized = []
    for cluster in sorted(clusters, key=lambda item: (item["cluster_id"], item["class_name"])):
        serialized.append({
            "class_name": cluster["class_name"],
            "dir": cluster["dir"],
            "size": len(cluster["items"]),
            "similarity_threshold": float(similarity_threshold),
            "source_dirs": sorted(cluster.get("source_dirs", set())),
            "source_labels": sorted(cluster.get("source_labels", set())),
            "items": [
                {
                    "path": item["path"],
                    "source_path": item.get("source_path") or item["path"],
                    "source_dir": item.get("source_dir") or "database",
                    "source_label": item.get("source_label") or "database",
                    "similarity_to_cluster": round(float(item.get("similarity_to_cluster", 1.0)), 4),
                }
                for item in cluster["items"]
            ],
        })
    return serialized


def _write_unknown_class_database_report(result, output_path):
    lines = [
        "UNKNOWN CLASS DATABASE",
        "=" * 72,
        "",
        f"Дата: {result.get('timestamp', datetime.now().strftime('%d-%m-%Y %H:%M:%S'))}",
        f"База: {result.get('db_root', '-')}",
        f"Папка классов: {result.get('classes_dir', '-')}",
        f"Всего классов: {result.get('class_count', 0)}",
        f"Всего изображений в базе: {result.get('total_image_count', 0)}",
        f"Добавлено изображений: {result.get('added_image_count', 0)}",
        f"Порог схожести: {result.get('similarity_threshold', _DEFAULT_UNKNOWN_CLASS_SIMILARITY):.2f}",
        "",
    ]

    skipped_files = result.get("skipped_files", [])
    if skipped_files:
        lines.append("Пропущенные файлы:")
        for skipped in skipped_files:
            lines.append(f"  - {skipped['path']} | {skipped['reason']}")
        lines.append("")

    for cluster in result.get("clusters", []):
        lines.append(
            f"{cluster['class_name']} | size={cluster['size']} | sources={len(cluster['source_dirs'])} | threshold={cluster['similarity_threshold']:.2f}"
        )
        for item in cluster["items"]:
            lines.append(
                f"  - sim={item.get('similarity_to_cluster', 1.0):.4f} | {item.get('source_dir', 'database')} | {os.path.basename(item.get('source_path') or item['path'])}"
            )
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output_path


def _write_unknown_class_database_json(result, output_path):
    payload = {
        "timestamp": result.get("timestamp"),
        "db_root": result.get("db_root"),
        "classes_dir": result.get("classes_dir"),
        "class_count": int(result.get("class_count", 0)),
        "total_image_count": int(result.get("total_image_count", 0)),
        "added_image_count": int(result.get("added_image_count", 0)),
        "similarity_threshold": float(result.get("similarity_threshold", _DEFAULT_UNKNOWN_CLASS_SIMILARITY)),
        "source_dirs": list(result.get("source_dirs", [])),
        "source_dir_count": int(result.get("source_dir_count", 0)),
        "input_root": result.get("input_root"),
        "engine": dict(result.get("engine", {})),
        "skipped_files": list(result.get("skipped_files", [])),
        "clusters": list(result.get("clusters", [])),
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return output_path


def supplement_unknown_class_database_from_items(
    saved_items,
    similarity_threshold=_DEFAULT_UNKNOWN_CLASS_SIMILARITY,
    db_root=None,
    progress_callback=None,
    source_name=None,
):
    db_paths = _get_unknown_class_database_paths(db_root)
    os.makedirs(db_paths["classes_dir"], exist_ok=True)

    if not saved_items:
        existing_clusters, existing_image_count = _load_unknown_class_database_clusters(db_paths["root"])
        serialized_clusters = _serialize_unknown_class_database_clusters(existing_clusters, similarity_threshold)
        empty_result = {
            "db_root": db_paths["root"],
            "classes_dir": db_paths["classes_dir"],
            "report_path": db_paths["report_path"] if os.path.isfile(db_paths["report_path"]) else None,
            "json_report_path": db_paths["json_path"] if os.path.isfile(db_paths["json_path"]) else None,
            "class_count": len(serialized_clusters),
            "total_image_count": existing_image_count,
            "added_image_count": 0,
            "similarity_threshold": float(similarity_threshold),
            "clusters": serialized_clusters,
            "skipped_files": [],
            "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "engine": {
                "embedding_model": "OpenCLIP ViT-B-32",
                "strategy": "persistent-unknownclass-database",
            },
        }
        return empty_result

    _load_clip(progress_callback)
    if progress_callback:
        progress_callback("Загрузка базы UnknownClass...", 1)

    clusters, existing_image_count = _load_unknown_class_database_clusters(db_paths["root"])
    next_cluster_id = max((cluster["cluster_id"] for cluster in clusters), default=0) + 1
    timestamp_token = datetime.now().strftime("%Y%m%d_%H%M%S")
    skipped_files = []
    prepared_items = []

    total_items = len(saved_items)
    for index, raw_item in enumerate(saved_items, start=1):
        if progress_callback:
            progress_callback(
                f"Подготовка UnknownClass DB: {index}/{total_items}",
                min(40, int(index * 40 / max(total_items, 1))),
            )

        try:
            source_path = os.path.abspath(raw_item["path"])
            embedding = raw_item.get("embedding")
            if embedding is None:
                crop = cv2.imread(source_path)
                if crop is None:
                    raise ValueError(f"Не удалось прочитать изображение: {source_path}")
                embedding = _encode_crop(crop).detach().cpu().numpy().astype(np.float32)
            embedding = _normalize_embedding(np.asarray(embedding, dtype=np.float32))
        except Exception as exc:
            skipped_files.append({
                "path": raw_item.get("path") or "?",
                "reason": str(exc),
            })
            continue

        source_dir = raw_item.get("relative_dir") or raw_item.get("source_dir") or source_name or os.path.basename(os.path.dirname(source_path)) or "unknown"
        source_label = raw_item.get("parent_label") or raw_item.get("source_label") or source_name or source_dir
        prepared_items.append({
            "path": source_path,
            "embedding": embedding,
            "source_dir": source_dir,
            "source_label": source_label,
            "detection": raw_item.get("detection"),
        })

    if progress_callback:
        progress_callback("Обновление базы UnknownClass...", 60)

    added_items = 0
    for item in prepared_items:
        best_cluster = None
        best_similarity = -1.0
        for cluster in clusters:
            similarity = _cosine_similarity(item["embedding"], cluster["centroid"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        if best_cluster is None or best_similarity < similarity_threshold:
            class_name = f"UnknownClass_{next_cluster_id:03d}"
            class_dir = os.path.join(db_paths["classes_dir"], class_name)
            os.makedirs(class_dir, exist_ok=True)
            best_cluster = {
                "cluster_id": next_cluster_id,
                "class_name": class_name,
                "dir": class_dir,
                "items": [],
                "embeddings": [],
                "centroid": item["embedding"].copy(),
                "source_dirs": set(),
                "source_labels": set(),
            }
            clusters.append(best_cluster)
            item_similarity = 1.0
            next_cluster_id += 1
        else:
            item_similarity = float(best_similarity)

        file_name = f"{timestamp_token}_{_safe_source_token(item['source_dir'])}__{os.path.basename(item['path'])}"
        destination_path = _unique_destination_path(best_cluster["dir"], file_name)
        if os.path.abspath(item["path"]) != os.path.abspath(destination_path):
            shutil.copy2(item["path"], destination_path)

        db_item = {
            "path": os.path.abspath(destination_path),
            "source_path": item["path"],
            "source_dir": item["source_dir"],
            "source_label": item["source_label"],
            "similarity_to_cluster": round(float(item_similarity), 4),
        }
        best_cluster["items"].append(db_item)
        best_cluster["embeddings"].append(item["embedding"])
        best_cluster["centroid"] = _normalize_embedding(
            np.mean(np.stack(best_cluster["embeddings"], axis=0), axis=0).astype(np.float32)
        )
        best_cluster["source_dirs"].add(item["source_dir"])
        best_cluster["source_labels"].add(item["source_label"])
        added_items += 1

        detection = item.get("detection")
        if detection is not None:
            detection["result"]["unknown_class_name"] = best_cluster["class_name"]
            detection["result"]["unknown_class_dir"] = best_cluster["dir"]
            detection["result"]["unknown_class_database_dir"] = db_paths["root"]

    serialized_clusters = _serialize_unknown_class_database_clusters(clusters, similarity_threshold)
    source_dirs = sorted({item.get("source_dir") for item in prepared_items if item.get("source_dir")})
    result = {
        "db_root": db_paths["root"],
        "classes_dir": db_paths["classes_dir"],
        "output_dir": db_paths["root"],
        "merged_classes_dir": db_paths["classes_dir"],
        "input_root": None,
        "class_count": len(serialized_clusters),
        "total_image_count": existing_image_count + added_items,
        "added_image_count": added_items,
        "image_count": added_items,
        "similarity_threshold": float(similarity_threshold),
        "source_dirs": source_dirs,
        "source_dir_count": len(source_dirs),
        "clusters": serialized_clusters,
        "skipped_files": skipped_files,
        "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "engine": {
            "embedding_model": "OpenCLIP ViT-B-32",
            "strategy": "persistent-unknownclass-database",
        },
    }

    report_path = _write_unknown_class_database_report(result, db_paths["report_path"])
    json_report_path = _write_unknown_class_database_json(result, db_paths["json_path"])
    result["report_path"] = report_path
    result["json_report_path"] = json_report_path

    if progress_callback:
        progress_callback("Готово", 100)

    return result


def supplement_unknown_class_database_from_folder(
    folder_path,
    similarity_threshold=_DEFAULT_UNKNOWN_CLASS_SIMILARITY,
    db_root=None,
    progress_callback=None,
):
    if not os.path.isdir(folder_path):
        return {"error": f"Папка не найдена: {folder_path}", "input_root": os.path.abspath(folder_path)}

    image_paths = _iter_recursive_images(folder_path)
    if not image_paths:
        return {
            "error": "В указанной папке не найдено изображений Unknown class.",
            "input_root": os.path.abspath(folder_path),
        }

    _load_clip(progress_callback)
    prepared_items = []
    skipped_files = []
    total_images = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        if progress_callback:
            progress_callback(
                f"Подготовка изображений для UnknownClass DB: {index}/{total_images}",
                min(55, int(index * 55 / max(total_images, 1))),
            )
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Не удалось прочитать изображение: {image_path}")
            embedding = _encode_crop(image).detach().cpu().numpy().astype(np.float32)
            embedding = _normalize_embedding(embedding)
        except Exception as exc:
            skipped_files.append({
                "path": image_path,
                "reason": str(exc),
            })
            continue

        relative_dir = os.path.relpath(os.path.dirname(image_path), folder_path)
        if relative_dir == ".":
            relative_dir = "root"
        source_label = os.path.basename(os.path.dirname(image_path)) or relative_dir
        prepared_items.append({
            "path": image_path,
            "embedding": embedding,
            "relative_dir": relative_dir,
            "parent_label": source_label,
            "source_dir": relative_dir,
            "source_label": source_label,
        })

    if not prepared_items:
        return {
            "error": "Не удалось получить эмбеддинги ни для одного изображения.",
            "input_root": os.path.abspath(folder_path),
            "skipped_files": skipped_files,
        }

    result = supplement_unknown_class_database_from_items(
        prepared_items,
        similarity_threshold=similarity_threshold,
        db_root=db_root,
        progress_callback=progress_callback,
        source_name=os.path.basename(os.path.normpath(folder_path)) or "import",
    )
    result["input_root"] = os.path.abspath(folder_path)
    result["skipped_files"] = list(result.get("skipped_files", [])) + skipped_files
    result["source_dirs"] = sorted({item["source_dir"] for item in prepared_items})
    result["source_dir_count"] = len(result["source_dirs"])
    result["image_count"] = len(prepared_items)

    report_path = _write_unknown_class_database_report(result, result["report_path"])
    json_report_path = _write_unknown_class_database_json(result, result["json_report_path"])
    result["report_path"] = report_path
    result["json_report_path"] = json_report_path
    return result


def consolidate_unknown_class_database(
    similarity_threshold=_DEFAULT_UNKNOWN_CLASS_SIMILARITY,
    db_root=None,
    progress_callback=None,
):
    """Объединяет похожие классы внутри базы UnknownClassDB.

    Быстрый алгоритм на основе матричного умножения (BLAS):
      1. Строит нормализованную матрицу центроидов (N × D).
      2. Вычисляет ВСЕ попарные косинусные схожести за один матричный проход
         по чанкам (≤ ~300 МБ RAM на чанк), используя dot(u, v) = cosine_sim
         для нормализованных векторов.
      3. Собирает все пары >= similarity_threshold через np.nonzero (векторизовано),
         сортирует по убыванию схожести.
      4. Жадно объединяет пары: для каждой пары верифицирует актуальную схожесть
         (центроид мог измениться после предыдущих слияний в том же проходе),
         затем сливает меньший кластер в больший.
      5. Повторяет до сходимости (0 слияний за проход; обычно 2–3 прохода).

    Сложность: O(N²·D / chunk) на BLAS вместо O(N²) Python-итераций за слияние.
    Для N=18 000, D=512 каждый проход занимает секунды вместо часов.
    Файлы перемещаются, пустые папки удаляются. Обновляет JSON и TXT отчёты базы.
    """
    _CHUNK_SIZE = 2000  # строк за чанк: 2000 × 18000 × 4 B ≈ 144 МБ

    db_paths = _get_unknown_class_database_paths(db_root)

    if progress_callback:
        progress_callback("Загрузка OpenCLIP...", 0)
    _load_clip(progress_callback)

    if progress_callback:
        progress_callback("Загрузка базы UnknownClass...", 1)
    clusters, _ = _load_unknown_class_database_clusters(db_paths["root"])

    if not clusters:
        return {
            "db_root": db_paths["root"],
            "classes_dir": db_paths["classes_dir"],
            "class_count": 0,
            "merged_count": 0,
            "similarity_threshold": float(similarity_threshold),
            "clusters": [],
            "skipped_files": [],
            "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "engine": {"embedding_model": "OpenCLIP ViT-B-32", "strategy": "fast_consolidation"},
        }

    total_before = len(clusters)
    merged_count = 0
    pass_num = 0

    while True:
        pass_num += 1
        n = len(clusters)
        if progress_callback:
            pct = min(5 + pass_num * 25, 85)
            progress_callback(
                f"Проход {pass_num}: {n} классов, объединено {merged_count}...",
                pct,
            )

        # --- 1. Нормализованная матрица центроидов (N × D) ---
        centroid_matrix = np.stack([c["centroid"] for c in clusters], axis=0).astype(np.float32)
        norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
        centroid_matrix /= np.where(norms > 0, norms, 1.0)

        # --- 2. Поиск всех пар >= threshold через чанковое матричное умножение ---
        pair_is = []
        pair_js = []
        pair_sims_list = []

        for chunk_start in range(0, n, _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, n)
            chunk = centroid_matrix[chunk_start:chunk_end]           # (K × D)
            sims_chunk = (chunk @ centroid_matrix.T).astype(np.float32)  # (K × N)

            chunk_rows = chunk_end - chunk_start
            # Маска: только верхний треугольник (j > i) выше порога
            row_local = np.arange(chunk_rows, dtype=np.int32)[:, None]   # (K, 1)
            col_global = np.arange(n, dtype=np.int32)[None, :]            # (1, N)
            global_i = chunk_start + row_local                             # (K, 1)
            mask = (col_global > global_i) & (sims_chunk >= float(similarity_threshold))

            local_is_found, js_found = np.nonzero(mask)
            if len(local_is_found) == 0:
                continue

            pair_is.extend((local_is_found + chunk_start).tolist())
            pair_js.extend(js_found.tolist())
            pair_sims_list.extend(sims_chunk[local_is_found, js_found].tolist())

        if not pair_is:
            break  # нет пар выше порога — сходимость

        # --- 3. Сортируем по убыванию схожести (greedy: сначала самые похожие) ---
        order = np.argsort(pair_sims_list)[::-1]

        # --- 4. Жадное слияние ---
        active = [True] * n
        pass_merges = 0

        for k in order:
            idx_i = pair_is[k]
            idx_j = pair_js[k]

            if not active[idx_i] or not active[idx_j]:
                continue

            # Верифицируем актуальную схожесть (центроид мог измениться в этом проходе)
            actual_sim = float(np.dot(clusters[idx_i]["centroid"], clusters[idx_j]["centroid"]))
            if actual_sim < float(similarity_threshold):
                continue

            # Сливаем меньший кластер в больший по числу изображений
            if len(clusters[idx_j]["items"]) > len(clusters[idx_i]["items"]):
                keep_idx, drop_idx = idx_j, idx_i
            else:
                keep_idx, drop_idx = idx_i, idx_j

            keep = clusters[keep_idx]
            drop = clusters[drop_idx]

            _log.info(
                f"Консолидация: {drop['class_name']} -> {keep['class_name']} "
                f"(sim={actual_sim:.4f}, sizes: {len(drop['items'])} + {len(keep['items'])})"
            )

            # Перемещаем файлы drop -> keep
            for db_item in drop["items"]:
                src = db_item["path"]
                if not os.path.isfile(src):
                    continue
                dst = _unique_destination_path(keep["dir"], os.path.basename(src))
                shutil.move(src, dst)
                db_item["path"] = dst
                db_item["source_dir"] = db_item.get("source_dir", "merged")
                keep["items"].append(db_item)

            keep["embeddings"].extend(drop["embeddings"])
            keep["centroid"] = _normalize_embedding(
                np.mean(np.stack(keep["embeddings"], axis=0), axis=0).astype(np.float32)
            )
            keep["source_dirs"] |= drop["source_dirs"]
            keep["source_labels"] |= drop["source_labels"]

            # Удаляем пустую папку drop
            try:
                if os.path.isdir(drop["dir"]):
                    remaining = [
                        f for f in os.listdir(drop["dir"])
                        if os.path.isfile(os.path.join(drop["dir"], f))
                    ]
                    if not remaining:
                        shutil.rmtree(drop["dir"])
            except Exception as exc:
                _log.warning(f"Не удалось удалить {drop['dir']}: {exc}")

            active[drop_idx] = False
            merged_count += 1
            pass_merges += 1

        # Убираем слитые кластеры из списка
        clusters = [c for c, a in zip(clusters, active) if a]

        if pass_merges == 0:
            break  # Сходимость: в этом проходе не было ни одного слияния

    if progress_callback:
        progress_callback("Обновление отчётов...", 96)

    serialized_clusters = _serialize_unknown_class_database_clusters(clusters, similarity_threshold)
    total_images = sum(len(c["items"]) for c in clusters)
    result = {
        "db_root": db_paths["root"],
        "classes_dir": db_paths["classes_dir"],
        "report_path": db_paths["report_path"],
        "json_report_path": db_paths["json_path"],
        "class_count": len(clusters),
        "class_count_before": total_before,
        "total_image_count": total_images,
        "added_image_count": 0,
        "merged_count": merged_count,
        "similarity_threshold": float(similarity_threshold),
        "source_dirs": [],
        "source_dir_count": 0,
        "clusters": serialized_clusters,
        "skipped_files": [],
        "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "engine": {"embedding_model": "OpenCLIP ViT-B-32", "strategy": "fast_consolidation"},
    }
    _write_unknown_class_database_report(result, db_paths["report_path"])
    _write_unknown_class_database_json(result, db_paths["json_path"])

    if progress_callback:
        progress_callback("Готово", 100)

    return result


def build_yolo_dataset_from_clusters(
    unknown_classes: dict,
    output_dir: str,
    val_split: float = 0.15,
    log_callback=None,
) -> dict:
    """Формирует YOLO-датасеты из кластеров неизвестных объектов (Пункт 5 архитектуры).

    По каждому кластеру (UnknownClass_001, ...) создаёт отдельный датасет:

        output_dir/
          UnknownClass_001/
            images/train/   ← чистые кадры (без боксов)
            images/val/
            labels/train/   ← YOLO-метки (координаты объекта в полном кадре)
            labels/val/
            data.yaml
          UnknownClass_002/
            ...

    Кропы в датасет не входят — только кадры и метки.
    Если в одном кадре несколько объектов одного класса, они объединяются в один label-файл.

    Возвращает:
        {
          "datasets": [{"class_name": ..., "dataset_dir": ..., "data_yaml": ...,
                        "train_count": int, "val_count": int}, ...],
          "output_dir": str,
          "total_classes": int,
        }
    """
    import yaml as _yaml
    import random as _rng
    from collections import defaultdict

    classes = unknown_classes.get("classes", [])
    if not classes:
        return {"datasets": [], "output_dir": output_dir, "total_classes": 0}

    os.makedirs(output_dir, exist_ok=True)
    datasets = []
    _rng.seed(42)

    for cluster in classes:
        class_name   = cluster["class_name"]
        frame_labels = cluster.get("frame_labels", [])

        # Группируем YOLO-метки по кадрам: frame_path → [label_line, ...]
        frame_map: dict = defaultdict(list)
        for fl in frame_labels:
            fp  = fl.get("clean_frame_path")
            lbl = fl.get("yolo_label")
            if fp and lbl and os.path.isfile(fp):
                frame_map[fp].append(lbl)

        if not frame_map:
            if log_callback:
                log_callback(f"[{class_name}] Нет кадров с известным путём — пропуск датасета")
            continue

        # Разбиваем на train / val
        all_frames = list(frame_map.keys())
        _rng.shuffle(all_frames)
        split_idx    = max(1, int(len(all_frames) * (1.0 - val_split)))
        train_frames = all_frames[:split_idx]
        val_frames   = all_frames[split_idx:]

        # Создаём структуру папок
        ds_dir = os.path.join(output_dir, class_name)
        for split in ("train", "val"):
            os.makedirs(os.path.join(ds_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(ds_dir, "labels", split), exist_ok=True)

        def _write_split(frames, split):
            for frame_path in frames:
                frame_name = os.path.basename(frame_path)
                stem       = os.path.splitext(frame_name)[0]
                # Кадр
                dst_img = os.path.join(ds_dir, "images", split, frame_name)
                shutil.copy2(frame_path, dst_img)
                # Метка (несколько объектов в кадре → несколько строк)
                dst_lbl = os.path.join(ds_dir, "labels", split, stem + ".txt")
                with open(dst_lbl, "w", encoding="utf-8") as _lf:
                    _lf.write("\n".join(frame_map[frame_path]) + "\n")

        _write_split(train_frames, "train")
        _write_split(val_frames,   "val")

        # data.yaml
        data_cfg = {
            "path":  ds_dir,
            "train": "images/train",
            "val":   "images/val",
            "nc":    1,
            "names": {0: class_name},
        }
        yaml_path = os.path.join(ds_dir, "data.yaml")
        with open(yaml_path, "w", encoding="utf-8") as _yf:
            _yaml.dump(data_cfg, _yf, allow_unicode=True, default_flow_style=False)

        if log_callback:
            log_callback(
                f"[{class_name}] YOLO датасет: train={len(train_frames)}, "
                f"val={len(val_frames)} кадров → {ds_dir}"
            )

        datasets.append({
            "class_name":  class_name,
            "dataset_dir": ds_dir,
            "data_yaml":   yaml_path,
            "train_count": len(train_frames),
            "val_count":   len(val_frames),
        })

    return {
        "datasets":     datasets,
        "output_dir":   output_dir,
        "total_classes": len(datasets),
    }


def _save_unknown_object_crops(image, detections, output_dir, file_prefix, margin_scale,
                               clean_frame_path=None):
    """Сохраняет кропы неизвестных объектов и YOLO-метки рядом с ними.

    Для каждого неизвестного объекта создаёт:
      {prefix}_unknown_NNN.jpg  — кроп (с небольшим отступом)
      {prefix}_unknown_NNN.txt  — YOLO-метка (координаты в ПОЛНОМ кадре, без отступа)

    clean_frame_path — путь к чистому кадру (без боксов), который будет использован
                       при сборке датасета для тренировки.
    """
    saved_paths = []
    saved_items = []
    directory_created = False
    image_height, image_width = image.shape[:2]

    for unknown_index, item in enumerate(
        (entry for entry in detections if not entry["result"].get("is_known")),
        start=1,
    ):
        # Оригинальные координаты bbox — используются для YOLO-метки
        bx1, by1, bx2, by2 = [int(v) for v in item["box"]]
        bx1 = max(0, min(bx1, image_width - 1))
        by1 = max(0, min(by1, image_height - 1))
        bx2 = max(bx1 + 1, min(bx2, image_width))
        by2 = max(by1 + 1, min(by2, image_height))

        # Расширенные координаты — для кропа (с запасом)
        x1, y1, x2, y2 = _expand_box(item["box"], image_width, image_height, scale=margin_scale)
        crop = image[y1:y2, x1:x2].copy()
        if crop.size == 0:
            item["result"]["unknown_object_path"] = None
            continue

        if not directory_created:
            os.makedirs(output_dir, exist_ok=True)
            directory_created = True

        output_path = os.path.join(output_dir, f"{file_prefix}_unknown_{unknown_index:03d}.jpg")
        cv2.imwrite(output_path, crop)
        item["result"]["unknown_object_path"] = output_path
        saved_paths.append(output_path)

        # ── YOLO-метка: нормализованные координаты объекта в полном кадре ──
        cx  = ((bx1 + bx2) / 2.0) / image_width
        cy  = ((by1 + by2) / 2.0) / image_height
        bw  = (bx2 - bx1) / float(image_width)
        bh  = (by2 - by1) / float(image_height)
        yolo_label = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        label_path = os.path.splitext(output_path)[0] + ".txt"
        with open(label_path, "w", encoding="utf-8") as _lf:
            _lf.write(yolo_label + "\n")

        try:
            embedding = _encode_crop(crop).detach().cpu().numpy().astype(np.float32)
        except Exception:
            embedding = None
        saved_items.append({
            "path":             output_path,
            "label_path":       label_path,
            "yolo_label":       yolo_label,
            "clean_frame_path": clean_frame_path,
            "embedding":        embedding,
            "detection":        item,
        })

    return {
        "dir":   output_dir if saved_paths else None,
        "paths": saved_paths,
        "items": [it for it in saved_items if it["embedding"] is not None],
    }


def _try_import_hdbscan():
    """Пытается импортировать HDBSCAN из scikit-learn (≥1.3) или hdbscan пакета."""
    try:
        from sklearn.cluster import HDBSCAN as _HDBSCAN
        return _HDBSCAN
    except ImportError:
        pass
    try:
        import hdbscan as _hdbscan_pkg
        return _hdbscan_pkg.HDBSCAN
    except ImportError:
        return None


def _cluster_with_hdbscan(embeddings, min_cluster_size=3, min_samples=2):
    """Кластеризует эмбеддинги с HDBSCAN + cosine distance.

    Возвращает массив меток (int): -1 = шум, 0..N = кластер.
    """
    HDBSCAN = _try_import_hdbscan()
    if HDBSCAN is None:
        raise ImportError(
            "HDBSCAN недоступен. Установите: pip install scikit-learn>=1.3 или pip install hdbscan"
        )
    matrix = np.stack(embeddings, axis=0).astype(np.float32)
    # HDBSCAN с cosine метрикой через предварительно нормализованные векторы
    # cosine distance = 1 - cosine_similarity, работает правильно на нормализованных векторах
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="cosine",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(matrix)
    return labels


def _cluster_unknown_crops_to_folder(
    saved_items,
    output_dir,
    similarity_threshold,
    use_hdbscan=False,
    hdbscan_min_cluster_size=3,
    hdbscan_min_samples=2,
):
    """Кластеризует кропы Unknown по схожести в подпапки внутри output_dir.

    Работает полностью локально — глобальная база UnknownClassDB не изменяется.
    Возвращает словарь с информацией о созданных классах.

    Параметры:
        use_hdbscan: если True — использует HDBSCAN + cosine distance (лучшее
            качество, требует scikit-learn>=1.3 или пакет hdbscan).
            Объекты без кластера (шум, label=-1) помещаются в UnknownClass_Noise.
            Если False — жадный алгоритм на основе порога схожести (работает
            онлайн, не требует дополнительных пакетов).
        hdbscan_min_cluster_size: минимум объектов для формирования кластера.
        hdbscan_min_samples: чем меньше, тем мягче граница шума.

    Для каждого кропа также копирует рядом YOLO-метку (.txt) если она существует.
    В поле frame_labels каждого класса хранятся пары (clean_frame_path, yolo_label)
    для последующей сборки YOLO-датасета через build_yolo_dataset_from_clusters().
    """
    if not saved_items:
        return {"dir": None, "class_count": 0, "classes": []}

    # Фильтруем items без эмбеддингов
    valid_items = []
    valid_embeddings = []
    for item in saved_items:
        emb = item.get("embedding")
        if emb is None:
            continue
        emb = _normalize_embedding(np.asarray(emb, dtype=np.float32))
        valid_items.append(item)
        valid_embeddings.append(emb)

    if not valid_items:
        return {"dir": None, "class_count": 0, "classes": []}

    os.makedirs(output_dir, exist_ok=True)

    # ── Определяем метки кластеров ─────────────────────────────────────────
    if use_hdbscan and len(valid_items) >= 2:
        try:
            raw_labels = _cluster_with_hdbscan(
                valid_embeddings, hdbscan_min_cluster_size, hdbscan_min_samples
            )
            # Переводим в словарь cluster_id → [indices]
            from collections import defaultdict as _dd
            cluster_map = _dd(list)
            for i, lbl in enumerate(raw_labels):
                cluster_map[int(lbl)].append(i)
            # Сортируем по размеру кластера (убывание), шум (-1) в конец
            sorted_ids = sorted(
                [k for k in cluster_map if k >= 0],
                key=lambda k: -len(cluster_map[k]),
            )
            # Назначаем имена
            named_clusters = []
            for rank, cid in enumerate(sorted_ids, start=1):
                named_clusters.append((f"UnknownClass_{rank:03d}", cluster_map[cid]))
            noise_indices = cluster_map.get(-1, [])
            if noise_indices:
                named_clusters.append(("UnknownClass_Noise", noise_indices))
            _log.info(
                f"HDBSCAN: {len(sorted_ids)} кластеров, {len(noise_indices)} шумовых объектов"
            )
        except Exception as e:
            _log.warning(f"HDBSCAN не удался ({e}), откат на жадный алгоритм")
            use_hdbscan = False

    if not use_hdbscan or len(valid_items) < 2:
        # ── Жадный алгоритм (онлайн, не требует HDBSCAN) ──────────────────
        local_clusters = []
        next_id = 1
        for i, (item, embedding) in enumerate(zip(valid_items, valid_embeddings)):
            best_cluster = None
            best_similarity = -1.0
            for cluster in local_clusters:
                sim = _cosine_similarity(embedding, cluster["centroid"])
                if sim > best_similarity:
                    best_similarity = sim
                    best_cluster = cluster

            if best_cluster is None or best_similarity < similarity_threshold:
                class_name = f"UnknownClass_{next_id:03d}"
                class_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                best_cluster = {
                    "class_name": class_name,
                    "dir": class_dir,
                    "indices": [],
                    "embeddings": [],
                    "centroid": embedding.copy(),
                }
                local_clusters.append(best_cluster)
                next_id += 1

            best_cluster["indices"].append(i)
            best_cluster["embeddings"].append(embedding)
            best_cluster["centroid"] = _normalize_embedding(
                np.mean(np.stack(best_cluster["embeddings"], axis=0), axis=0).astype(np.float32)
            )

        named_clusters = [
            (c["class_name"], c["indices"]) for c in local_clusters
        ]

    # ── Копируем файлы в папки кластеров ──────────────────────────────────
    result_classes = []
    for class_name, indices in named_clusters:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        cluster_items = []

        for i in indices:
            item = valid_items[i]
            src_path = item["path"]
            dst_name = os.path.basename(src_path)
            dst_path = _unique_destination_path(class_dir, dst_name)
            if os.path.abspath(src_path) != os.path.abspath(dst_path):
                shutil.copy2(src_path, dst_path)

            src_txt = os.path.splitext(src_path)[0] + ".txt"
            dst_txt = os.path.splitext(dst_path)[0] + ".txt"
            if os.path.isfile(src_txt) and not os.path.isfile(dst_txt):
                shutil.copy2(src_txt, dst_txt)

            cluster_items.append({
                "crop_path":        dst_path,
                "label_path":       dst_txt if os.path.isfile(dst_txt) else None,
                "yolo_label":       item.get("yolo_label"),
                "clean_frame_path": item.get("clean_frame_path"),
            })

        result_classes.append({
            "class_name":  class_name,
            "dir":         class_dir,
            "count":       len(cluster_items),
            "frame_labels": [
                {
                    "clean_frame_path": it.get("clean_frame_path"),
                    "yolo_label":       it.get("yolo_label"),
                }
                for it in cluster_items
            ],
        })

    if not result_classes:
        return {"dir": None, "class_count": 0, "classes": []}

    return {
        "dir":         output_dir,
        "class_count": len(result_classes),
        "classes":     result_classes,
    }


def _analyze_open_set_image_array(
    image,
    labels,
    det_confidence,
    known_similarity_threshold,
    strong_similarity_threshold,
    margin_threshold,
    yolo_support_confidence,
    generic_top_k,
    max_proposals,
    progress_callback=None,
):
    yolo_proposals = _generate_yolo_proposals(image, det_confidence, progress_callback)
    generic_proposals = _generate_generic_proposals(image, generic_top_k, progress_callback)
    proposals = _merge_proposals(
        yolo_proposals,
        generic_proposals,
        image.shape[1],
        image.shape[0],
        max_proposals,
    )

    if progress_callback:
        progress_callback("Оценка novelty и depth...", 3)

    thresholds = {
        "known_similarity": known_similarity_threshold,
        "strong_similarity": strong_similarity_threshold,
        "margin": margin_threshold,
        "yolo_support_confidence": yolo_support_confidence,
    }
    depth_map = _compute_depth_map(image)
    detections = []
    for proposal in proposals:
        x1, y1, x2, y2 = proposal["box"]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        result = _classify_proposal(crop, proposal, labels, thresholds)
        detections.append({
            "box": proposal["box"],
            "result": result,
        })

    depth_info = _attach_depth_to_detections(detections, depth_map)

    detections.sort(
        key=lambda item: item["result"]["unknown_score"]
        if not item["result"]["is_known"]
        else item["result"]["clip_probability"],
        reverse=True,
    )

    known_count = sum(1 for item in detections if item["result"]["is_known"])
    unknown_count = sum(1 for item in detections if not item["result"]["is_known"])

    return {
        "proposal_count": len(proposals),
        "known_count": known_count,
        "unknown_count": unknown_count,
        "detections": detections,
        "depth_info": depth_info,
        "thresholds": thresholds,
        "annotated": _annotate_detections(image, detections),
    }


def analyze_open_set(
    image_path,
    progress_callback=None,
    known_labels=None,
    det_confidence=_DEFAULT_DET_CONFIDENCE,
    known_similarity_threshold=_DEFAULT_KNOWN_SIM_THRESHOLD,
    strong_similarity_threshold=_DEFAULT_STRONG_SIM_THRESHOLD,
    margin_threshold=_DEFAULT_MARGIN_THRESHOLD,
    yolo_support_confidence=_DEFAULT_YOLO_SUPPORT_CONFIDENCE,
    generic_top_k=_DEFAULT_GENERIC_TOPK,
    max_proposals=_DEFAULT_MAX_PROPOSALS,
    unknown_crop_margin_scale=_DEFAULT_UNKNOWN_CROP_MARGIN_SCALE,
    unknown_class_similarity_threshold=_DEFAULT_UNKNOWN_CLASS_SIMILARITY,
    use_hdbscan=False,
    _skip_cluster_build=False,
):
    """Анализирует изображение в постановке open-set detection."""
    _log.info(f"Open-set анализ: {os.path.basename(image_path)}")

    if not os.path.isfile(image_path):
        return {"error": f"Файл не найден: {image_path}", "file": os.path.basename(image_path)}

    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Не удалось прочитать: {image_path}", "file": os.path.basename(image_path)}

    yolo_model = _load_yolo(progress_callback)
    labels = _get_known_labels(yolo_model, known_labels)
    _load_clip(progress_callback)
    _load_depth_model(progress_callback, allow_failure=True)

    frame_result = _analyze_open_set_image_array(
        image,
        labels,
        det_confidence,
        known_similarity_threshold,
        strong_similarity_threshold,
        margin_threshold,
        yolo_support_confidence,
        generic_top_k,
        max_proposals,
        progress_callback=progress_callback,
    )

    base_name, ext = os.path.splitext(image_path)
    annotated_path = base_name + "_openset_annotated" + (ext or ".jpg")
    cv2.imwrite(annotated_path, frame_result["annotated"])
    unknown_objects = _save_unknown_object_crops(
        image,
        frame_result["detections"],
        base_name + "_openset_unknown_objects",
        os.path.basename(base_name),
        unknown_crop_margin_scale,
        clean_frame_path=image_path,   # оригинальный файл = чистый кадр для датасета
    )

    if _skip_cluster_build:
        # Режим сбора данных: возвращаем сырые items для межфайловой кластеризации
        if progress_callback:
            progress_callback("Готово", 4)
        return {
            "type": "image",
            "file": os.path.basename(image_path),
            "path": os.path.abspath(image_path),
            "annotated_path": annotated_path,
            "unknown_objects_dir": unknown_objects["dir"],
            "unknown_object_paths": unknown_objects["paths"],
            "_unknown_object_items": unknown_objects["items"],
            "known_labels": labels,
            "known_label_count": len(labels),
            "proposal_count": frame_result["proposal_count"],
            "known_count": frame_result["known_count"],
            "unknown_count": frame_result["unknown_count"],
            "detections": frame_result["detections"],
            "depth_info": frame_result["depth_info"],
            "thresholds": frame_result["thresholds"],
            "unknown_crop_margin_scale": float(unknown_crop_margin_scale),
            "unknown_class_similarity_threshold": float(unknown_class_similarity_threshold),
            "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "engine": _build_engine_info(),
        }

    unknown_classes = _cluster_unknown_crops_to_folder(
        unknown_objects["items"],
        base_name + "_openset_unknown_classes",
        unknown_class_similarity_threshold,
        use_hdbscan=use_hdbscan,
    )

    # ── Пункт 5: Формируем YOLO-датасеты по каждому кластеру ──────────────
    yolo_datasets_dir = base_name + "_openset_yolo_datasets"
    yolo_datasets = build_yolo_dataset_from_clusters(
        unknown_classes,
        yolo_datasets_dir,
        log_callback=lambda msg: _log.info(msg),
    )

    if progress_callback:
        progress_callback("Готово", 4)

    return {
        "type": "image",
        "file": os.path.basename(image_path),
        "path": os.path.abspath(image_path),
        "annotated_path": annotated_path,
        "unknown_objects_dir": unknown_objects["dir"],
        "unknown_object_paths": unknown_objects["paths"],
        "unknown_classes_dir": unknown_classes["dir"],
        "unknown_class_count": unknown_classes["class_count"],
        "unknown_classes": unknown_classes["classes"],
        "yolo_datasets_dir": yolo_datasets_dir if yolo_datasets.get("total_classes", 0) > 0 else None,
        "yolo_datasets":     yolo_datasets.get("datasets", []),
        "known_labels": labels,
        "known_label_count": len(labels),
        "proposal_count": frame_result["proposal_count"],
        "known_count": frame_result["known_count"],
        "unknown_count": frame_result["unknown_count"],
        "detections": frame_result["detections"],
        "depth_info": frame_result["depth_info"],
        "thresholds": frame_result["thresholds"],
        "unknown_crop_margin_scale": float(unknown_crop_margin_scale),
        "unknown_class_similarity_threshold": float(unknown_class_similarity_threshold),
        "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "engine": _build_engine_info(),
    }


def analyze_open_set_video(
    video_path,
    progress_callback=None,
    known_labels=None,
    det_confidence=_DEFAULT_DET_CONFIDENCE,
    known_similarity_threshold=_DEFAULT_KNOWN_SIM_THRESHOLD,
    strong_similarity_threshold=_DEFAULT_STRONG_SIM_THRESHOLD,
    margin_threshold=_DEFAULT_MARGIN_THRESHOLD,
    yolo_support_confidence=_DEFAULT_YOLO_SUPPORT_CONFIDENCE,
    generic_top_k=_DEFAULT_GENERIC_TOPK,
    max_proposals=_DEFAULT_MAX_PROPOSALS,
    sample_interval_seconds=_DEFAULT_VIDEO_SAMPLE_INTERVAL,
    max_keyframes=_DEFAULT_VIDEO_MAX_KEYFRAMES,
    unknown_crop_margin_scale=_DEFAULT_UNKNOWN_CROP_MARGIN_SCALE,
    unknown_class_similarity_threshold=_DEFAULT_UNKNOWN_CLASS_SIMILARITY,
    use_hdbscan=False,
    _skip_cluster_build=False,
):
    """Анализирует видео, прогоняя open-set детекцию по выборочным кадрам."""
    _log.info(f"Open-set анализ видео: {os.path.basename(video_path)}")

    if not os.path.isfile(video_path):
        return {"error": f"Файл не найден: {video_path}", "file": os.path.basename(video_path)}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Не удалось открыть видео: {video_path}", "file": os.path.basename(video_path)}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frame_step = max(1, int(round((fps if fps > 0 else 1.0) * max(sample_interval_seconds, 0.1))))
    estimated_samples = (frame_count + frame_step - 1) // frame_step if frame_count > 0 else 0
    total_steps = max(estimated_samples + 3, 4)

    if progress_callback:
        progress_callback("Загрузка YOLO для видео...", 0, total_steps)
    yolo_model = _load_yolo()
    labels = _get_known_labels(yolo_model, known_labels)

    if progress_callback:
        progress_callback("Загрузка OpenCLIP...", 1, total_steps)
    _load_clip()
    _load_depth_model(allow_failure=True)

    base_name, _ = os.path.splitext(video_path)
    annotated_video_path = base_name + "_openset_annotated.mp4"
    saved_frames_dir = base_name + "_openset_frames"
    annotated_frames_dir = base_name + "_openset_annotated_frames"
    unknown_timestamps_path = base_name + "_openset_unknown_timestamps.txt"
    unknown_objects_dir = base_name + "_openset_unknown_objects"
    os.makedirs(saved_frames_dir, exist_ok=True)

    writer = None
    if width > 0 and height > 0:
        writer_fps = fps / frame_step if fps > 0 else 1.0
        writer = cv2.VideoWriter(
            annotated_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(writer_fps, 1.0),
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            writer = None

    frame_results = []
    sampled_frames = 0
    saved_keyframes = 0
    saved_annotated_frames = 0
    total_known = 0
    total_unknown = 0
    total_proposals = 0
    frames_with_unknown = 0
    current_frame_index = 0
    unknown_object_paths = []
    unknown_object_items = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if current_frame_index % frame_step != 0:
                current_frame_index += 1
                continue

            sampled_frames += 1
            timestamp_seconds = current_frame_index / fps if fps > 0 else (sampled_frames - 1) * sample_interval_seconds
            timestamp_text = _format_timestamp(timestamp_seconds)

            if progress_callback:
                progress_callback(
                    f"[{sampled_frames}/{estimated_samples or '?'}] Анализ кадра {timestamp_text}",
                    min(sampled_frames + 1, total_steps - 1),
                    total_steps,
                )

            frame_result = _analyze_open_set_image_array(
                frame,
                labels,
                det_confidence,
                known_similarity_threshold,
                strong_similarity_threshold,
                margin_threshold,
                yolo_support_confidence,
                generic_top_k,
                max_proposals,
                progress_callback=None,
            )

            annotated_frame = frame_result.pop("annotated")
            if writer is not None:
                writer.write(annotated_frame)

            # ── Сохраняем ЧИСТЫЙ кадр (без боксов) только если есть неизвестные ──
            # Пункт 2 архитектуры: кадр для датасета не должен содержать боксы
            clean_frame_path = None
            if frame_result["unknown_count"] > 0:
                clean_frame_name = f"frame_{current_frame_index:06d}_{timestamp_text.replace(':', '-').replace('.', '_')}.jpg"
                clean_frame_path = os.path.join(saved_frames_dir, clean_frame_name)
                cv2.imwrite(clean_frame_path, frame)  # чистый кадр, не annotated_frame!
                saved_keyframes += 1

            # ── Сохраняем аннотированный кадр (с bbox) если есть любые детекции ──
            if frame_result["known_count"] > 0 or frame_result["unknown_count"] > 0:
                os.makedirs(annotated_frames_dir, exist_ok=True)
                ann_frame_name = f"frame_{current_frame_index:06d}_{timestamp_text.replace(':', '-').replace('.', '_')}_ann.jpg"
                cv2.imwrite(os.path.join(annotated_frames_dir, ann_frame_name), annotated_frame)
                saved_annotated_frames += 1

            saved_frame_path = clean_frame_path

            if frame_result["unknown_count"] > 0:
                frames_with_unknown += 1

            saved_unknown_objects = _save_unknown_object_crops(
                frame,
                frame_result["detections"],
                unknown_objects_dir,
                f"frame_{current_frame_index:06d}_{timestamp_text.replace(':', '-').replace('.', '_')}",
                unknown_crop_margin_scale,
                clean_frame_path=clean_frame_path,
            )
            unknown_object_paths.extend(saved_unknown_objects["paths"])
            unknown_object_items.extend(saved_unknown_objects["items"])

            total_known += frame_result["known_count"]
            total_unknown += frame_result["unknown_count"]
            total_proposals += frame_result["proposal_count"]

            frame_results.append({
                "frame_index": current_frame_index,
                "timestamp_seconds": round(timestamp_seconds, 3),
                "timestamp": timestamp_text,
                "proposal_count": frame_result["proposal_count"],
                "known_count": frame_result["known_count"],
                "unknown_count": frame_result["unknown_count"],
                "saved_frame_path": saved_frame_path,
                "unknown_object_paths": saved_unknown_objects["paths"],
                "detections": frame_result["detections"],
                "depth_info": frame_result["depth_info"],
            })

            current_frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    _write_unknown_timestamps_file(frame_results, unknown_timestamps_path)

    if _skip_cluster_build:
        # Режим сбора данных: кластеризацию и сборку датасета выполнит вызывающий код
        if progress_callback:
            progress_callback("Готово", total_steps, total_steps)
        return {
            "type": "video",
            "file": os.path.basename(video_path),
            "path": os.path.abspath(video_path),
            "annotated_video_path": annotated_video_path if os.path.isfile(annotated_video_path) else None,
            "saved_frames_dir": saved_frames_dir,
            "annotated_frames_dir": annotated_frames_dir if saved_annotated_frames > 0 else None,
            "saved_annotated_frames": saved_annotated_frames,
            "unknown_timestamps_path": unknown_timestamps_path,
            "unknown_objects_dir": unknown_objects_dir if unknown_object_paths else None,
            "unknown_object_paths": unknown_object_paths,
            "_unknown_object_items": unknown_object_items,  # сырые данные для межвидеовой кластеризации
            "known_labels": labels,
            "known_label_count": len(labels),
            "proposal_count": total_proposals,
            "known_count": total_known,
            "unknown_count": total_unknown,
            "frame_count": frame_count,
            "fps": round(fps, 3),
            "sampled_frames": sampled_frames,
            "frame_step": frame_step,
            "sample_interval_seconds": float(sample_interval_seconds),
            "frames_with_unknown": frames_with_unknown,
            "frame_results": frame_results,
            "unknown_crop_margin_scale": float(unknown_crop_margin_scale),
            "unknown_class_similarity_threshold": float(unknown_class_similarity_threshold),
            "depth_info": {
                "enabled": _depth_model is not None,
                "model": _DEPTH_MODEL_ID if _depth_model is not None else None,
                "error": _depth_load_error,
            },
            "thresholds": {
                "known_similarity": known_similarity_threshold,
                "strong_similarity": strong_similarity_threshold,
                "margin": margin_threshold,
                "yolo_support_confidence": yolo_support_confidence,
            },
            "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "engine": _build_engine_info(),
        }

    unknown_classes_dir_local = base_name + "_openset_unknown_classes"
    unknown_classes = _cluster_unknown_crops_to_folder(
        unknown_object_items,
        unknown_classes_dir_local,
        unknown_class_similarity_threshold,
        use_hdbscan=use_hdbscan,
    )

    # ── Пункт 5: Формируем YOLO-датасеты по каждому кластеру ──────────────
    yolo_datasets_dir = base_name + "_openset_yolo_datasets"
    yolo_datasets = build_yolo_dataset_from_clusters(
        unknown_classes,
        yolo_datasets_dir,
        log_callback=lambda msg: _log.info(msg),
    )

    if progress_callback:
        progress_callback("Готово", total_steps, total_steps)

    return {
        "type": "video",
        "file": os.path.basename(video_path),
        "path": os.path.abspath(video_path),
        "annotated_video_path": annotated_video_path if os.path.isfile(annotated_video_path) else None,
        "saved_frames_dir": saved_frames_dir,
        "annotated_frames_dir": annotated_frames_dir if saved_annotated_frames > 0 else None,
        "saved_annotated_frames": saved_annotated_frames,
        "unknown_timestamps_path": unknown_timestamps_path,
        "unknown_objects_dir": unknown_objects_dir if unknown_object_paths else None,
        "unknown_object_paths": unknown_object_paths,
        "unknown_classes_dir": unknown_classes["dir"],
        "unknown_class_count": unknown_classes["class_count"],
        "unknown_classes": unknown_classes["classes"],
        "known_labels": labels,
        "known_label_count": len(labels),
        "proposal_count": total_proposals,
        "known_count": total_known,
        "unknown_count": total_unknown,
        "frame_count": frame_count,
        "fps": round(fps, 3),
        "sampled_frames": sampled_frames,
        "frame_step": frame_step,
        "sample_interval_seconds": float(sample_interval_seconds),
        "frames_with_unknown": frames_with_unknown,
        "frame_results": frame_results,
        "yolo_datasets_dir": yolo_datasets_dir if yolo_datasets.get("total_classes", 0) > 0 else None,
        "yolo_datasets":     yolo_datasets.get("datasets", []),
        "unknown_crop_margin_scale": float(unknown_crop_margin_scale),
        "unknown_class_similarity_threshold": float(unknown_class_similarity_threshold),
        "depth_info": {
            "enabled": _depth_model is not None,
            "model": _DEPTH_MODEL_ID if _depth_model is not None else None,
            "error": _depth_load_error,
        },
        "thresholds": {
            "known_similarity": known_similarity_threshold,
            "strong_similarity": strong_similarity_threshold,
            "margin": margin_threshold,
            "yolo_support_confidence": yolo_support_confidence,
        },
        "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "engine": _build_engine_info(),
    }


def analyze_open_set_multi_video(
    video_paths,
    output_dir,
    progress_callback=None,
    known_labels=None,
    det_confidence=_DEFAULT_DET_CONFIDENCE,
    known_similarity_threshold=_DEFAULT_KNOWN_SIM_THRESHOLD,
    strong_similarity_threshold=_DEFAULT_STRONG_SIM_THRESHOLD,
    margin_threshold=_DEFAULT_MARGIN_THRESHOLD,
    yolo_support_confidence=_DEFAULT_YOLO_SUPPORT_CONFIDENCE,
    generic_top_k=_DEFAULT_GENERIC_TOPK,
    max_proposals=_DEFAULT_MAX_PROPOSALS,
    sample_interval_seconds=_DEFAULT_VIDEO_SAMPLE_INTERVAL,
    max_keyframes=_DEFAULT_VIDEO_MAX_KEYFRAMES,
    unknown_crop_margin_scale=_DEFAULT_UNKNOWN_CROP_MARGIN_SCALE,
    unknown_class_similarity_threshold=_DEFAULT_UNKNOWN_CLASS_SIMILARITY,
    use_hdbscan=False,
):
    """Анализирует несколько файлов (видео и/или изображений) и формирует единый
    набор классов из всех неизвестных объектов.

    Каждый файл обрабатывается с _skip_cluster_build=True — кропы, чистые кадры
    и YOLO-метки сохраняются, но кластеризация откладывается до конца. После
    обработки всех файлов raw items объединяются и кластеризуются за один проход.

    Возвращает:
        {
          "type": "multi_video",
          "file_results": [...],           # результаты по каждому файлу
          "combined_unknown_classes": ..., # единый словарь кластеров
          "yolo_datasets": [...],
          "yolo_datasets_dir": str,
          "combined_classes_dir": str,
          "total_unknown_count": int,
          ...
        }
    """
    _log.info(f"Unified open-set анализ: {len(video_paths)} файлов")
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = _IMAGE_EXTENSIONS
    video_extensions = _VIDEO_EXTENSIONS

    total_phases = len(video_paths) * 4 + 2  # +2 для кластеризации и датасета
    if progress_callback:
        progress_callback(f"Обработка файлов (0/{len(video_paths)})...", 0, total_phases)

    file_results = []
    all_unknown_items = []

    for idx, file_path in enumerate(video_paths):
        fname = os.path.basename(file_path)
        base_offset = idx * 4

        def _step_cb(step_name, step_num, maximum=None,
                     _off=base_offset, _i=idx, _n=len(video_paths), _fn=fname):
            if progress_callback is None:
                return True
            if maximum:
                mapped = _off + min(4, int(round((step_num / max(maximum, 1)) * 4)))
            else:
                mapped = _off + min(step_num, 4)
            progress_callback(
                f"[{_i + 1}/{_n}] {_fn}\n{step_name}",
                mapped, total_phases
            )
            return True

        lower = file_path.lower()
        if lower.endswith(video_extensions):
            result = analyze_open_set_video(
                file_path,
                progress_callback=_step_cb,
                known_labels=known_labels,
                det_confidence=det_confidence,
                known_similarity_threshold=known_similarity_threshold,
                strong_similarity_threshold=strong_similarity_threshold,
                margin_threshold=margin_threshold,
                yolo_support_confidence=yolo_support_confidence,
                generic_top_k=generic_top_k,
                max_proposals=max_proposals,
                sample_interval_seconds=sample_interval_seconds,
                max_keyframes=max_keyframes,
                unknown_crop_margin_scale=unknown_crop_margin_scale,
                unknown_class_similarity_threshold=unknown_class_similarity_threshold,
                _skip_cluster_build=True,
            )
            items = result.pop("_unknown_object_items", [])
        else:
            # Изображение — анализируем, забираем items напрямую
            result = analyze_open_set(
                file_path,
                progress_callback=_step_cb,
                known_labels=known_labels,
                det_confidence=det_confidence,
                known_similarity_threshold=known_similarity_threshold,
                strong_similarity_threshold=strong_similarity_threshold,
                margin_threshold=margin_threshold,
                yolo_support_confidence=yolo_support_confidence,
                generic_top_k=generic_top_k,
                max_proposals=max_proposals,
                unknown_crop_margin_scale=unknown_crop_margin_scale,
                unknown_class_similarity_threshold=unknown_class_similarity_threshold,
                _skip_cluster_build=True,
            )
            items = result.pop("_unknown_object_items", [])

        file_results.append(result)
        all_unknown_items.extend(items)

    # Единая кластеризация по всем файлам
    if progress_callback:
        progress_callback(
            f"Кластеризация {len(all_unknown_items)} unknown-объектов...",
            len(video_paths) * 4, total_phases
        )
    combined_classes_dir = os.path.join(output_dir, "combined_unknown_classes")
    unknown_classes = _cluster_unknown_crops_to_folder(
        all_unknown_items,
        combined_classes_dir,
        unknown_class_similarity_threshold,
        use_hdbscan=use_hdbscan,
    )

    if progress_callback:
        progress_callback(
            f"Формирование YOLO-датасетов ({unknown_classes['class_count']} классов)...",
            len(video_paths) * 4 + 1, total_phases
        )
    yolo_datasets_dir = os.path.join(output_dir, "combined_yolo_datasets")
    yolo_datasets = build_yolo_dataset_from_clusters(
        unknown_classes,
        yolo_datasets_dir,
        log_callback=lambda msg: _log.info(msg),
    )

    if progress_callback:
        progress_callback("Готово", total_phases, total_phases)

    total_unknown = sum(r.get("unknown_count", 0) for r in file_results)
    total_known   = sum(r.get("known_count",   0) for r in file_results)
    total_frames  = sum(r.get("sampled_frames", 0) + (1 if r.get("type") == "image" else 0) for r in file_results)

    return {
        "type":                     "multi_video",
        "output_dir":               output_dir,
        "file_results":             file_results,
        "video_results":            file_results,  # обратная совместимость
        "combined_classes_dir":     unknown_classes["dir"],
        "combined_class_count":     unknown_classes["class_count"],
        "combined_unknown_classes": unknown_classes["classes"],
        "yolo_datasets_dir":        yolo_datasets_dir if yolo_datasets.get("total_classes", 0) > 0 else None,
        "yolo_datasets":            yolo_datasets.get("datasets", []),
        "total_unknown_count":      total_unknown,
        "total_known_count":        total_known,
        "total_sampled_frames":     total_frames,
        "video_count":              len(video_paths),
        "file_count":               len(video_paths),
        "timestamp":                datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
    }


def analyze_open_set_folder(folder_path, progress_callback=None, known_labels=None):
    """Анализирует изображения и видео в папке в open-set режиме."""
    results = []
    for fname in sorted(os.listdir(folder_path)):
        lower_name = fname.lower()
        if lower_name.endswith(_IMAGE_EXTENSIONS):
            full_path = os.path.join(folder_path, fname)
            results.append(analyze_open_set(full_path, progress_callback=progress_callback, known_labels=known_labels))
        elif lower_name.endswith(_VIDEO_EXTENSIONS):
            full_path = os.path.join(folder_path, fname)
            results.append(analyze_open_set_video(full_path, progress_callback=progress_callback, known_labels=known_labels))
        else:
            continue
    return results


def generate_report_open_set(results, report_path):
    """Создаёт текстовый отчёт по open-set анализу."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    ts = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    lines = [
        "=" * 72,
        "  ОТЧЁТ: OPEN-SET DETECTION",
        f"  Дата: {ts}",
        f"  Обработано элементов: {len(results)}",
        "=" * 72,
        "",
    ]

    for result in results:
        lines.append("-" * 72)
        lines.append(f"Файл: {result.get('file', '?')}")
        lines.append("-" * 72)

        if "error" in result:
            lines.append(f"ОШИБКА: {result['error']}")
            lines.append("")
            continue

        result_type = result.get("type", "image")
        lines.append(f"Тип: {result_type}")
        lines.append(f"Proposal boxes: {result.get('proposal_count', 0)}")
        lines.append(f"Known objects:  {result.get('known_count', 0)}")
        lines.append(f"Unknown objects:{result.get('unknown_count', 0)}")
        lines.append(f"Скрины Unknown:  {len(result.get('unknown_object_paths', []))}")
        lines.append(f"Запас unknown crop: {result.get('unknown_crop_margin_scale', _DEFAULT_UNKNOWN_CROP_MARGIN_SCALE):.2f}")
        lines.append(f"Классов в базе UnknownClass: {result.get('unknown_class_count', 0)}")
        lines.append(f"Порог схожести классов: {result.get('unknown_class_similarity_threshold', _DEFAULT_UNKNOWN_CLASS_SIMILARITY):.2f}")
        lines.append(f"Known label set size: {result.get('known_label_count', 0)}")
        depth_info = result.get("depth_info", {})
        lines.append(f"Depth model: {depth_info.get('model') or 'disabled'}")
        if depth_info.get("error"):
            lines.append(f"Depth status: {depth_info['error']}")

        if result_type == "video":
            lines.append(f"Всего кадров: {result.get('frame_count', 0)}")
            lines.append(f"FPS: {result.get('fps', 0)}")
            lines.append(f"Проанализировано кадров: {result.get('sampled_frames', 0)}")
            lines.append(f"Шаг по кадрам: {result.get('frame_step', 0)}")
            lines.append(f"Интервал выборки: {result.get('sample_interval_seconds', 0)} сек")
            lines.append(f"Кадров с Unknown: {result.get('frames_with_unknown', 0)}")
            lines.append(f"Аннотированное видео: {result.get('annotated_video_path') or '—'}")
            lines.append(f"Папка сохранённых кадров: {result.get('saved_frames_dir') or '—'}")
            lines.append(f"Файл таймкодов Unknown: {result.get('unknown_timestamps_path') or '—'}")
            lines.append(f"Папка скринов Unknown: {result.get('unknown_objects_dir') or '—'}")
            lines.append(f"Папка классов Unknown (локальная): {result.get('unknown_classes_dir') or '—'}")
            lines.append("")

            for frame_idx, frame in enumerate(result.get("frame_results", []), 1):
                lines.append(
                    f"[Frame {frame_idx}] idx={frame.get('frame_index', 0)} | t={frame.get('timestamp', '00:00:00.000')} | "
                    f"proposals={frame.get('proposal_count', 0)} | known={frame.get('known_count', 0)} | "
                    f"unknown={frame.get('unknown_count', 0)} | saved={frame.get('saved_frame_path') or '—'}"
                )
                for det_idx, item in enumerate(frame.get("detections", []), 1):
                    x1, y1, x2, y2 = item["box"]
                    det = item["result"]
                    state = "KNOWN" if det["is_known"] else "UNKNOWN"
                    lines.append(
                        f"     [{det_idx}] {state} | box=({x1},{y1},{x2},{y2}) | label={det['label']} | "
                        f"clip_top={det['clip_label']} | sim={det['clip_similarity']:.4f} | "
                        f"p={det['clip_probability']:.4f} | novelty={det['unknown_score']:.4f} | "
                        f"depth={det.get('depth_score', '—')} ({det.get('depth_bucket') or '—'}) | "
                        f"src={det['source']}"
                    )
                    if det.get("unknown_class_name"):
                        lines.append(f"          unknown_class={det['unknown_class_name']}")
                    if det.get("unknown_object_path"):
                        lines.append(f"          unknown_crop={det['unknown_object_path']}")
                    if det.get("detector_label"):
                        lines.append(
                            f"          detector={det['detector_label']} ({det['detector_confidence']:.4f}), "
                            f"margin={det['clip_margin']:.4f}"
                        )
        else:
            lines.append(f"Аннотированное изображение: {result.get('annotated_path', '—')}")
            lines.append(f"Папка скринов Unknown: {result.get('unknown_objects_dir') or '—'}")
            lines.append(f"Папка классов Unknown (локальная): {result.get('unknown_classes_dir') or '—'}")
            lines.append("")

            for idx, item in enumerate(result.get("detections", []), 1):
                x1, y1, x2, y2 = item["box"]
                det = item["result"]
                state = "KNOWN" if det["is_known"] else "UNKNOWN"
                lines.append(
                    f"[{idx}] {state} | box=({x1},{y1},{x2},{y2}) | label={det['label']} | "
                    f"clip_top={det['clip_label']} | sim={det['clip_similarity']:.4f} | "
                    f"p={det['clip_probability']:.4f} | novelty={det['unknown_score']:.4f} | "
                    f"depth={det.get('depth_score', '—')} ({det.get('depth_bucket') or '—'}) | "
                    f"src={det['source']}"
                )
                if det.get("unknown_class_name"):
                    lines.append(f"     unknown_class={det['unknown_class_name']}")
                if det.get("unknown_object_path"):
                    lines.append(f"     unknown_crop={det['unknown_object_path']}")
                if det.get("detector_label"):
                    lines.append(
                        f"     detector={det['detector_label']} ({det['detector_confidence']:.4f}), "
                        f"margin={det['clip_margin']:.4f}"
                    )
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    _log.info(f"Open-set отчёт сохранён: {report_path}")
    return report_path


def generate_report_word_open_set(results, report_dir):
    """Создаёт Word-отчёт по open-set анализу."""
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from PIL import Image

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    title = doc.add_heading("Отчёт: Open-set detection", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = doc.add_paragraph(
        f"Дата: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n"
        f"Обработано элементов: {len(results)}"
    )
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for idx, result in enumerate(results):
        if idx > 0:
            doc.add_page_break()

        doc.add_heading(f"Изображение: {result.get('file', '?')}", level=2)

        def _insert_picture(path, label=None):
            if not (path and os.path.isfile(path)):
                return
            if label:
                paragraph = doc.add_paragraph(label)
                paragraph.runs[0].bold = True
            try:
                doc.add_picture(path, width=Inches(4.5))
            except ZeroDivisionError:
                try:
                    img_pil = Image.open(path)
                    buf = io.BytesIO()
                    img_pil.save(buf, format=img_pil.format or "JPEG", dpi=(96, 96))
                    buf.seek(0)
                    doc.add_picture(buf, width=Inches(4.5))
                except Exception as exc:
                    doc.add_paragraph(f"[Ошибка вставки изображения: {exc}]")
            except Exception as exc:
                doc.add_paragraph(f"[Ошибка вставки изображения: {exc}]")

        if "error" in result:
            doc.add_paragraph(f"Ошибка: {result['error']}")
            continue

        result_type = result.get("type", "image")
        doc.add_paragraph(f"Тип: {result_type}")
        depth_info = result.get("depth_info", {})
        doc.add_paragraph(f"Depth model: {depth_info.get('model') or 'disabled'}")
        if depth_info.get("error"):
            doc.add_paragraph(f"Depth status: {depth_info['error']}")

        if result_type == "image":
            _insert_picture(result.get("path"), "Оригинал:")
            _insert_picture(result.get("annotated_path"), "Аннотация open-set:")
            doc.add_paragraph(f"Папка скринов Unknown: {result.get('unknown_objects_dir') or '—'}")
            doc.add_paragraph(f"Классы базы UnknownClass: {result.get('unknown_classes_dir') or '—'}")
            doc.add_paragraph(f"Отчёт по базе UnknownClass: {result.get('unknown_classes_report_path') or '—'}")
        else:
            doc.add_paragraph(f"Исходное видео: {result.get('path', '—')}")
            doc.add_paragraph(f"Аннотированное видео: {result.get('annotated_video_path') or '—'}")
            doc.add_paragraph(f"Папка кадров: {result.get('saved_frames_dir') or '—'}")
            doc.add_paragraph(f"Файл таймкодов Unknown: {result.get('unknown_timestamps_path') or '—'}")
            doc.add_paragraph(f"Папка скринов Unknown: {result.get('unknown_objects_dir') or '—'}")
            doc.add_paragraph(f"Классы базы UnknownClass: {result.get('unknown_classes_dir') or '—'}")
            doc.add_paragraph(f"Отчёт по базе UnknownClass: {result.get('unknown_classes_report_path') or '—'}")

        table = doc.add_table(rows=0, cols=2)
        table.style = "Light Grid Accent 1"

        def add_row(label, value):
            row = table.add_row()
            row.cells[0].text = label
            row.cells[1].text = str(value)

        add_row("Proposal boxes", result.get("proposal_count", 0))
        add_row("Known objects", result.get("known_count", 0))
        add_row("Unknown objects", result.get("unknown_count", 0))
        add_row("Скрины Unknown", len(result.get("unknown_object_paths", [])))
        add_row("Запас unknown crop", result.get("unknown_crop_margin_scale", _DEFAULT_UNKNOWN_CROP_MARGIN_SCALE))
        add_row("Классов в базе UnknownClass", result.get("unknown_class_count", 0))
        add_row("Порог схожести классов", result.get("unknown_class_similarity_threshold", _DEFAULT_UNKNOWN_CLASS_SIMILARITY))
        add_row("Known label set size", result.get("known_label_count", 0))
        add_row("Depth model", depth_info.get("model") or "disabled")

        if result_type == "video":
            add_row("Всего кадров", result.get("frame_count", 0))
            add_row("FPS", result.get("fps", 0))
            add_row("Проанализировано кадров", result.get("sampled_frames", 0))
            add_row("Интервал выборки, сек", result.get("sample_interval_seconds", 0))
            add_row("Кадров с Unknown", result.get("frames_with_unknown", 0))

            inserted_frames = 0
            for frame in result.get("frame_results", []):
                if inserted_frames >= 6:
                    break
                saved_frame_path = frame.get("saved_frame_path")
                if not saved_frame_path:
                    continue
                doc.add_paragraph(
                    f"Кадр {frame.get('frame_index', 0)} ({frame.get('timestamp', '00:00:00.000')}), "
                    f"unknown={frame.get('unknown_count', 0)}, known={frame.get('known_count', 0)}"
                )
                _insert_picture(saved_frame_path)
                inserted_frames += 1
        else:
            for det_idx, item in enumerate(result.get("detections", []), 1):
                det = item["result"]
                prefix = f"Object #{det_idx}"
                add_row(prefix, f"{det['label']} ({'known' if det['is_known'] else 'unknown'})")
                add_row(f"{prefix} similarity", det.get("clip_similarity", 0.0))
                add_row(f"{prefix} novelty", det.get("unknown_score", 0.0))
                add_row(f"{prefix} depth", f"{det.get('depth_bucket') or '—'} ({det.get('depth_score', '—')})")
                add_row(f"{prefix} unknown class", det.get("unknown_class_name") or "—")

    os.makedirs(report_dir, exist_ok=True)
    doc_path = os.path.join(report_dir, "OPEN_SET_REPORT.docx")
    doc.save(doc_path)
    _log.info(f"Word open-set отчёт сохранён: {doc_path}")
    return doc_path