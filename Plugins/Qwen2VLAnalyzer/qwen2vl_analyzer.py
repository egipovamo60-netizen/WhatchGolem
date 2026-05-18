"""
Плагин анализа транспортных средств на базе Qwen2-VL-2B-Instruct.

Модель запускается в отдельном subprocess, полностью изолированном от Qt.
Это устраняет крэши OpenMP/MKL внутри QThread на Windows.

Зависимости:
  pip install transformers torch Pillow qwen-vl-utils
"""

import os
import sys
import json
import subprocess
import re
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from logger import get_logger

_log = get_logger("Qwen2VLAnalyzer")

_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "worker_process.py")
_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Models", "Qwen2-VL-2B-Instruct")
)


class Qwen2VLSubprocessAnalyzer:
    """
    Управляет subprocess'ом с моделью Qwen2-VL-2B.
    Общение через stdin/stdout JSON.
    """

    def __init__(self):
        self._proc = None

    def _ensure_started(self, progress_callback=None):
        """Запускает subprocess если не запущен."""
        if self._proc is not None and self._proc.poll() is None:
            return  # уже работает
        _log.info("Запуск subprocess Qwen2-VL-2B...")

        if not os.path.isdir(_MODEL_DIR) or not os.path.isfile(
            os.path.join(_MODEL_DIR, "config.json")
        ):
            raise FileNotFoundError(
                f"Модель не найдена: {_MODEL_DIR}\n"
                "Запустите: python download_qwen2vl2b.py"
            )

        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        self._proc = subprocess.Popen(
            [sys.executable, _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        # Читаем вывод до сигнала READY или ошибки
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("Subprocess завершился неожиданно при загрузке модели")
            line = line.strip()
            if not line:
                continue
            if line == "READY":
                _log.info("Модель Qwen2-VL-2B загружена и готова")
                break
            try:
                msg = json.loads(line)
                if msg.get("status") == "progress" and progress_callback:
                    progress_callback(msg.get("step", ""), msg.get("num", 0))
                elif msg.get("status") == "error":
                    raise RuntimeError(msg.get("message", "Неизвестная ошибка"))
            except json.JSONDecodeError:
                pass  # служебный вывод transformers — игнорируем

    def analyze(self, image_path: str, progress_callback=None) -> dict:
        """Анализирует изображение через subprocess."""
        _log.info(f"Анализ: {os.path.basename(image_path)}")
        if not os.path.isfile(image_path):
            _log.warning(f"Файл не найден: {image_path}")
            return {"error": f"Файл не найден: {image_path}",
                    "file": os.path.basename(image_path)}

        if progress_callback:
            progress_callback("Загрузка Qwen2-VL-2B...", 0)

        try:
            self._ensure_started(progress_callback)
        except Exception as e:
            _log.error(f"Ошибка загрузки модели: {e}")
            return {"error": f"Ошибка загрузки модели: {e}",
                    "file": os.path.basename(image_path)}

        # Отправляем команду
        cmd = json.dumps({"action": "analyze", "image": image_path}, ensure_ascii=False)
        try:
            self._proc.stdin.write(cmd + "\n")
            self._proc.stdin.flush()
        except Exception as e:
            _log.error(f"Ошибка связи с subprocess: {e}")
            self._proc = None
            return {"error": f"Ошибка связи с subprocess: {e}",
                    "file": os.path.basename(image_path)}

        # Читаем ответы пока не получим result или error
        while True:
            line = self._proc.stdout.readline()
            if not line:
                self._proc = None
                return {"error": "Subprocess завершился неожиданно",
                        "file": os.path.basename(image_path)}
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            status = msg.get("status")
            if status == "progress":
                if progress_callback:
                    progress_callback(msg.get("step", ""), msg.get("num", 0))
            elif status == "debug":
                raw = msg.get("raw", "")
                _log.debug(f"RAW ответ модели [{os.path.basename(image_path)}]: {raw}")
                _log_debug(image_path, raw)
            elif status == "result":
                _log.info(f"Результат получен для {os.path.basename(image_path)}")
                data = msg.get("data", {})
                return _enrich_with_cv_fallback(image_path, data, progress_callback)
            elif status == "error":
                _log.error(f"Ошибка анализа {os.path.basename(image_path)}: {msg.get('message')}")
                return {"error": msg.get("message", "Неизвестная ошибка"),
                        "file": os.path.basename(image_path)}

    def stop(self):
        """Завершает subprocess."""
        if self._proc and self._proc.poll() is None:
            _log.info("Остановка subprocess Qwen2-VL-2B")
            try:
                self._proc.stdin.write(json.dumps({"action": "quit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.terminate()
        self._proc = None


# ─────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEBUG_LOG = os.path.join(_BASE_DIR, "Results", "qwen2vl_debug.log")
_RU_PLATE_RE = re.compile(r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$')

_LATIN_TO_CYR = {
    'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М',
    'H': 'Н', 'O': 'О', 'P': 'Р', 'C': 'С', 'T': 'Т',
    'Y': 'У', 'X': 'Х',
}
_DIGIT_TO_LETTER = {'0': 'О', '3': 'З', '6': 'Б', '8': 'В'}
_LETTER_TO_DIGIT = {'О': '0', 'З': '3', 'В': '8', 'Б': '6', 'І': '1', 'Л': '4', 'Z': '7', 'Q': '0', 'G': '6', 'S': '5', 'I': '1'}


def _log_debug(image_path: str, raw_response: str):
    """Записывает сырой ответ модели в лог-файл для отладки."""
    try:
        os.makedirs(os.path.dirname(_DEBUG_LOG), exist_ok=True)
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%d-%m-%Y %H:%M:%S}] {os.path.basename(image_path)}\n")
            f.write(f"  RAW: {raw_response}\n\n")
    except Exception:
        pass


def _normalize_plate_text(raw: str) -> str:
    """Нормализует текст номера к формату российского госномера."""
    if not raw:
        return ""

    s = re.sub(r'[^A-ZА-Яa-zа-я0-9]', '', str(raw)).upper()
    s = ''.join(_LATIN_TO_CYR.get(ch, ch) for ch in s)
    if len(s) < 6:
        return s

    chars = list(s)
    if chars[0].isdigit():
        chars[0] = _DIGIT_TO_LETTER.get(chars[0], chars[0])
    for i in (1, 2, 3):
        if i < len(chars) and not chars[i].isdigit():
            chars[i] = _LETTER_TO_DIGIT.get(chars[i], chars[i])
    for i in (4, 5):
        if i < len(chars) and chars[i].isdigit():
            chars[i] = _DIGIT_TO_LETTER.get(chars[i], chars[i])
    for i in range(6, min(len(chars), 9)):
        if not chars[i].isdigit():
            chars[i] = _LETTER_TO_DIGIT.get(chars[i], chars[i])

    return ''.join(chars)


def _is_good_plate(plate_text: str) -> bool:
    return bool(plate_text and _RU_PLATE_RE.match(plate_text))


def _is_weak_color(color_result: dict) -> bool:
    color = str(color_result.get("color", "")).strip().lower()
    conf = float(color_result.get("confidence", 0.0) or 0.0)
    bad_names = {"", "не определён", "не определен", "unknown", "n/a", "none"}
    return color in bad_names or conf < 0.35


def _choose_better_color(color_qwen: dict, color_cv: dict) -> dict:
    """Выбирает более надёжный цвет между Qwen и CV на тёмных авто."""
    q_name = str(color_qwen.get("color", "")).strip().lower()
    c_name = str(color_cv.get("color", "")).strip().lower()
    q_conf = float(color_qwen.get("confidence", 0.0) or 0.0)
    c_conf = float(color_cv.get("confidence", 0.0) or 0.0)

    dark_colors = {"чёрный", "черный", "антрацитовый", "графитовый", "тёмно-серый", "темно-серый"}
    light_gray_colors = {"серый", "светло-серый", "серебристый", "жемчужный"}

    # Частый кейс: Qwen даёт "серый", а CV распознаёт тёмный кузов.
    if q_name in light_gray_colors and c_name in dark_colors and c_conf >= 0.18:
        return {
            "color": color_cv.get("color", "Не определён"),
            "confidence": c_conf,
        }

    if q_conf >= c_conf:
        return {
            "color": color_qwen.get("color", "Не определён"),
            "confidence": q_conf,
        }
    return {
        "color": color_cv.get("color", "Не определён"),
        "confidence": c_conf,
    }


def _hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return 999
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def _enrich_with_cv_fallback(image_path: str, result: dict, progress_callback=None) -> dict:
    """Уточняет цвет и номер через классический CV/OCR, если ответ Qwen слабый."""
    if not isinstance(result, dict) or "error" in result:
        return result

    try:
        import cv2
        from Plugins.VehicleAnalyzer.vehicle_analyzer import preprocess_image, detect_vehicle_color, detect_license_plate
    except Exception:
        return result

    image = cv2.imread(image_path)
    if image is None:
        return result

    if progress_callback:
        progress_callback("Уточнение цвета и номера (CV fallback)...", 3)
    enhanced = preprocess_image(image)

    color_qwen = result.get("color", {}) if isinstance(result.get("color"), dict) else {}
    color_cv = detect_vehicle_color(enhanced)
    cv_color_as_dict = {
        "color": color_cv.get("color", "Не определён"),
        "confidence": float(color_cv.get("confidence", 0.0) or 0.0),
    }

    if _is_weak_color(color_qwen):
        if cv_color_as_dict["confidence"] >= 0.20:
            result["color"] = cv_color_as_dict
    else:
        result["color"] = _choose_better_color(color_qwen, cv_color_as_dict)

    plate_qwen = result.get("plate", {}) if isinstance(result.get("plate"), dict) else {}
    q_plate_text = _normalize_plate_text(plate_qwen.get("plate_text"))
    q_plate_conf = float(plate_qwen.get("confidence", 0.0) or 0.0)

    plate_cv = detect_license_plate(enhanced)
    cv_plate_text = _normalize_plate_text(plate_cv.get("plate_text"))
    cv_plate_conf = float(plate_cv.get("confidence", 0.0) or 0.0)

    q_good = _is_good_plate(q_plate_text)
    cv_good = _is_good_plate(cv_plate_text)

    final_plate = None
    final_conf = 0.0

    if q_good and cv_good:
        if q_plate_text == cv_plate_text:
            final_plate = q_plate_text
            final_conf = max(q_plate_conf, cv_plate_conf, 0.55)
        else:
            # Если номера различаются на 1 символ и первые 6 символов совпадают,
            # доверяем CV (чаще лучше для OCR зоны региона, где путаются 7/2).
            one_char_diff = _hamming_distance(q_plate_text, cv_plate_text) == 1
            same_prefix = q_plate_text[:6] == cv_plate_text[:6]
            if one_char_diff and same_prefix and cv_plate_conf >= 0.20:
                final_plate = cv_plate_text
                final_conf = max(cv_plate_conf, 0.55)
            elif cv_plate_conf > q_plate_conf + 0.05:
                final_plate = cv_plate_text
                final_conf = cv_plate_conf
            else:
                final_plate = q_plate_text
                final_conf = max(q_plate_conf, 0.55)
    elif cv_good:
        final_plate = cv_plate_text
        final_conf = max(cv_plate_conf, 0.55)
    elif q_good:
        final_plate = q_plate_text
        final_conf = max(q_plate_conf, 0.55)
    elif cv_plate_conf >= q_plate_conf and cv_plate_text:
        final_plate = cv_plate_text
        final_conf = cv_plate_conf
    elif q_plate_text:
        final_plate = q_plate_text
        final_conf = q_plate_conf

    if final_plate:
        result["plate"] = {
            "plate_text": final_plate,
            "confidence": final_conf,
            "message": f"Номер: {final_plate} (уверенность: {final_conf:.0%})",
        }
    else:
        result["plate"] = {
            "plate_text": None,
            "confidence": 0.0,
            "message": "Номерной знак не обнаружен",
        }

    return result


# ─────────────────────────────────
# Функции-обёртки для ui.py
# ─────────────────────────────────

_analyzer = None


def _get_analyzer() -> Qwen2VLSubprocessAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = Qwen2VLSubprocessAnalyzer()
    return _analyzer


def analyze_vehicle_qwen2vl(image_path: str, progress_callback=None) -> dict:
    return _get_analyzer().analyze(image_path, progress_callback)


def analyze_vehicle_folder_qwen2vl(folder_path: str) -> list:
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
    analyzer = _get_analyzer()
    results = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith(extensions):
            results.append(analyzer.analyze(os.path.join(folder_path, fname)))
    return results


def generate_report_qwen2vl(results: list, output_path: str, elapsed_seconds: float = None) -> str:
    lines = [
        "=" * 60,
        "  ОТЧЁТ: АНАЛИЗ ТРАНСПОРТНЫХ СРЕДСТВ (Qwen2-VL-2B)",
        f"  Дата: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
        f"  Обработано изображений: {len(results)}",
        "=" * 60,
        "",
    ]
    for i, r in enumerate(results, 1):
        if "error" in r:
            lines.append(f"[{i}] {r.get('file', '?')} — ОШИБКА: {r['error']}")
            lines.append("")
            continue
        plate = r['plate'].get('plate_text') or r['plate']['message']
        lines.append(
            f"Объектом осмотра является файл {r['file']} на котором обнаружен транспорт "
            f"{r['model']['model']}, цвет {r['color']['color']}, "
            f"государственный номер {plate}."
        )
        lines.append("")
    lines.append("=" * 60)
    if elapsed_seconds is not None:
        mins, secs = divmod(int(elapsed_seconds), 60)
        lines.append(f"\n  Время обработки: {mins} мин {secs} сек")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))
    return output_path


def generate_report_word_qwen2vl(results: list, output_dir: str, elapsed_seconds: float = None) -> str:
    """Генерирует один общий Word документ с изображениями и результатами анализа Qwen2-VL-2B.
    
    Args:
        results: список результатов анализа
        output_dir: директория для сохранения документа
        
    Returns:
        путь к созданному Word документу
    """
    try:
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        _log.error("python-docx не установлен. Установите: pip install python-docx")
        raise ImportError(
            "Для создания Word документов требуется python-docx.\n"
            "Установите: pip install python-docx"
        )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаём один документ для всех результатов
    doc = Document()
    
    # Добавляем главный заголовок
    title = doc.add_heading("Отчёт анализа транспортных средств (Qwen2-VL-2B)", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Общая информация
    info_para = doc.add_paragraph()
    info_para.add_run(f"Дата: ").bold = True
    info_para.add_run(datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
    
    count_para = doc.add_paragraph()
    count_para.add_run(f"Обработано изображений: ").bold = True
    count_para.add_run(str(len([r for r in results if "error" not in r])))
    
    # Обработка каждого результата
    for i, r in enumerate(results, 1):
        # Добавляем разделитель (новая страница после первого элемента)
        if i > 1:
            doc.add_page_break()
        
        if "error" in r:
            error_heading = doc.add_heading(f"[{i}] {r.get('file', '?')} — ОШИБКА", level=2)
            error_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            error_para = doc.add_paragraph()
            error_para.add_run("Ошибка: ").bold = True
            error_para.add_run(r['error'])
            continue
        
        # Заголовок для этого транспорта
        file_heading = doc.add_heading(f"[{i}] {r.get('file', '?')}", level=2)
        file_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Информация о файле и времени
        info_para = doc.add_paragraph()
        info_para.add_run(f"Дата анализа: ").bold = True
        timestamp = r.get("timestamp", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        info_para.add_run(timestamp)
        
        # Добавляем изображение
        image_path = r.get("path")
        image_added = False
        
        if image_path:
            # Преобразуем в абсолютный путь, если относительный
            if not os.path.isabs(image_path):
                image_path = os.path.abspath(image_path)
            
            _log.info(f"Попытка добавить изображение: {image_path}")
            
            if os.path.isfile(image_path):
                try:
                    # Проверяем, что это изображение
                    from PIL import Image
                    img = Image.open(image_path)
                    img_format = img.format
                    img_size = img.size
                    _log.info(f"Изображение найдено: {img_format} {img_size}")
                    
                    # Попробуем добавить изображение
                    # Используем height вместо width, чтобы избежать деления на ноль при отсутствии DPI
                    try:
                        doc.add_picture(image_path, width=Inches(5.5))
                    except ZeroDivisionError:
                        # Если нет DPI метаданных, установим стандартный DPI
                        _log.warning(f"Нет DPI метаданных для {os.path.basename(image_path)}, устанавливаем DPI")
                        try:
                            import io
                            img = Image.open(image_path)
                            # Сохраняем в памяти с установленным DPI (96 DPI - стандартный)
                            img_bytes = io.BytesIO()
                            img.save(img_bytes, format='JPEG', dpi=(96, 96))
                            img_bytes.seek(0)
                            doc.add_picture(img_bytes, width=Inches(5.5))
                        except Exception as e_inner:
                            # Последний вариант - используем height
                            _log.warning(f"Не удалось установить DPI, пробуем height: {e_inner}")
                            doc.add_picture(image_path, height=Inches(3.5))
                    
                    last_paragraph = doc.paragraphs[-1]
                    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    image_added = True
                    _log.info(f"✓ Изображение успешно добавлено: {os.path.basename(image_path)}")
                    
                except Exception as e:
                    error_msg = f"Ошибка добавления изображения {os.path.basename(image_path)}: {type(e).__name__}: {str(e)}"
                    _log.error(error_msg)
                    img_error = doc.add_paragraph()
                    img_error_run = img_error.add_run(f"⚠ {error_msg}")
                    img_error_run.italic = True
                    img_error_run.font.color.rgb = RGBColor(255, 0, 0)
            else:
                # Файл не найден
                error_msg = f"Файл не найден: {image_path}"
                _log.warning(error_msg)
                img_missing = doc.add_paragraph()
                img_missing_run = img_missing.add_run(f"⚠ {error_msg}")
                img_missing_run.italic = True
                img_missing_run.font.color.rgb = RGBColor(255, 0, 0)
        else:
            _log.warning(f"Путь к изображению не указан для результата: {r.get('file', '?')}")
        
        # Итоговая запись осмотра
        doc.add_paragraph()  # пустая строка
        plate_info = r.get("plate", {})
        plate_text = plate_info.get("plate_text") or plate_info.get("message", "не установлен")
        color_info = r.get("color", {})
        model_info = r.get("model", {})
        verdict_para = doc.add_paragraph()
        verdict_para.add_run(
            f"Объектом осмотра является файл {r.get('file', '?')} на котором обнаружен транспорт "
            f"{model_info.get('model', 'не определена')}, цвет {color_info.get('color', 'не определён')}, "
            f"государственный номер {plate_text}."
        )
    
    # Информация об анализаторе в конце
    doc.add_page_break()
    footer_heading = doc.add_heading("Информация об отчёте", level=2)
    
    engine_para = doc.add_paragraph()
    engine_para.add_run(f"Анализатор: ").bold = True
    engine_para.add_run("Qwen2-VL-2B")
    
    total_para = doc.add_paragraph()
    total_para.add_run(f"Всего обработано изображений: ").bold = True
    total_para.add_run(str(len(results)))
    
    errors_count = len([r for r in results if "error" in r])
    if errors_count > 0:
        errors_para = doc.add_paragraph()
        errors_para.add_run(f"Ошибок при обработке: ").bold = True
        errors_para.add_run(str(errors_count))

    if elapsed_seconds is not None:
        mins, secs = divmod(int(elapsed_seconds), 60)
        elapsed_para = doc.add_paragraph()
        elapsed_para.add_run("Время обработки: ").bold = True
        elapsed_para.add_run(f"{mins} мин {secs} сек")
    
    # Сохраняем один документ
    doc_filename = f"Qwen2VL_Analysis_Report.docx"
    doc_path = os.path.join(output_dir, doc_filename)
    
    try:
        doc.save(doc_path)
        _log.info(f"Word документ создан: {doc_path}")
    except Exception as e:
        _log.error(f"Ошибка при сохранении Word документа: {e}")
        raise
    
    return doc_path
