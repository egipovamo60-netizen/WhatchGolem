import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import time
from datetime import datetime
from ultralytics import YOLO
from PyQt5.QtGui import QImage, QPixmap
from collections import deque
import numpy as np

# Ограничиваем потоки PyTorch для стабильности с Qt
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Загрузка модели YOLO11 (xlarge)
model = YOLO("yolo11x.pt")

# Расширения для видео и изображений
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

# Глобальный счетчик ID для уникальной идентификации объектов
object_id_counter = {} # Словарь для хранения счетчика ID по типам объектов

def get_center(box):
    """Получить центр боксера (x1, y1, x2, y2)"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def box_distance(box1, box2):
    """Вычислить расстояние между центрами двух боксов"""
    center1 = get_center(box1)
    center2 = get_center(box2)
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def match_detections(prev_detections, current_detections, max_distance=50):
    """
    Сопоставить текущие детекции с предыдущими на основе расстояния
    prev_detections: {obj_id: (x1, y1, x2, y2, obj_name, class_id, ...)}
    current_detections: [(x1, y1, x2, y2, confidence, class_id)]
    Returns: список (matched_id, detection_idx) или None для новых объектов
    """
    matches = []
    used_indices = set()
    
    for obj_id, obj_data in prev_detections.items():
        prev_x1, prev_y1, prev_x2, prev_y2, obj_name, class_id = obj_data[:6]
        prev_box = (prev_x1, prev_y1, prev_x2, prev_y2)
        best_dist = max_distance
        best_idx = None
        
        for idx, detection in enumerate(current_detections):
            if idx in used_indices:
                continue
            x1, y1, x2, y2, confidence, det_class_id = detection
            
            # Проверяем, что это объект того же класса
            if int(det_class_id) == int(class_id):
                curr_box = (x1, y1, x2, y2)
                dist = box_distance(prev_box, curr_box)
                
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
        
        if best_idx is not None:
            matches.append((obj_id, best_idx))
            used_indices.add(best_idx)
    
    return matches, used_indices

def generate_new_id(obj_name):
    """Генерировать новый уникальный ID для объекта"""
    if obj_name not in object_id_counter:
        object_id_counter[obj_name] = 0
    object_id_counter[obj_name] += 1
    return f"{obj_name}_{object_id_counter[obj_name]}"

def reset_id_counter():
    """Сбросить счетчик ID (используется при начале новой обработки)"""
    global object_id_counter
    object_id_counter = {}


def get_video_metadata(video_path, output_dir=None):
    """
    Получить метаданные видеофайла и сохранить их в txt файл
    
    Args:
        video_path (str): Путь к видеофайлу
        output_dir (str, optional): Директория для сохранения txt файла. 
                                   Если None, файл сохраняется в той же директории что и видео
    
    Returns:
        str: Путь к созданному txt файлу с метаданными
    
    Raises:
        ValueError: Если видеофайл не может быть открыт
    """
    try:
        # Открываем видеофайл
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Не удается открыть видеофайл: {video_path}")
        
        # Получаем метаданные видео
        filename = os.path.basename(video_path)
        filename_without_ext = os.path.splitext(filename)[0]
        
        # Получаем формат видео (кодек)
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
        
        # Получаем количество кадров и fps для расчета продолжительности
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Расчет продолжительности в секундах
        if fps > 0:
            duration_seconds = frame_count / fps
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            milliseconds = int((duration_seconds % 1) * 1000)
            duration_formatted = f"{minutes}:{seconds:02d}.{milliseconds:03d}"
        else:
            duration_formatted = "Неизвестна"
        
        # Получаем разрешение видео
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Закрываем видеофайл
        cap.release()
        
        # Определяем директорию для сохранения txt файла
        if output_dir is None:
            output_dir = os.path.dirname(video_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Создаем txt файл с метаданными
        metadata_filename = f"{filename_without_ext}_metadata.txt"
        metadata_path = os.path.join(output_dir, metadata_filename)
        
        # Записываем метаданные в файл
        with open(metadata_path, "w", encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("МЕТАДАННЫЕ ВИДЕОФАЙЛА\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Имя файла: {filename}\n")
            f.write(f"Имя без расширения: {filename_without_ext}\n")
            f.write(f"Формат записи (кодек): {codec if codec.strip() else 'Неизвестен'}\n")
            f.write(f"Продолжительность: {duration_formatted} сек\n")
            f.write(f"Количество кадров: {frame_count}\n")
            f.write(f"FPS: {fps:.2f}\n")
            f.write(f"Разрешение: {width}x{height}\n")
            f.write(f"\nДата создания отчета: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
        
        print(f"[METADATA] Метаданные сохранены: {metadata_path}")
        
        return metadata_path
        
    except Exception as e:
        print(f"[ERROR] Ошибка при получении метаданных видео: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def process_frame(cap, tracking_objects, trail_points, log_file, out, show_video_checkbox, label, confidence_threshold=0.5):
    ret, frame = cap.read()

    if not ret:
        cap.release()
        out.release()  # Закрываем файл с сохраненным видео
        return False

    # Получение текущего времени видео
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Время в секундах
    time_formatted = time.strftime('%H:%M:%S', time.gmtime(current_time))

    # Обработка кадра с помощью YOLOv8
    results = model(frame)

    # Постобработка и размечивание объектов
    frame = draw_boxes_and_track(frame, results, time_formatted, tracking_objects, trail_points, log_file, confidence_threshold)

    # Сохранение обработанного кадра в видеофайл
    out.write(frame)

    if show_video_checkbox.isChecked():
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        step = channel * width
        q_img = QImage(frame_rgb.data, width, height, step, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img))

    return True

def draw_boxes_and_track(frame, results, time_formatted, tracking_objects, trail_points, log_file, confidence_threshold=0.5):
    current_detections = []
    current_ids = set()
    
    # Собираем все текущие детекции
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = result
        if confidence > confidence_threshold:  # Фильтрация по порогу уверенности
            current_detections.append((x1, y1, x2, y2, confidence, class_id))
    
    # Сопоставляем текущие детекции с отслеживаемыми объектами
    matches, used_indices = match_detections(tracking_objects, current_detections, max_distance=100)
    
    # Обновляем существующие объекты
    for obj_id, detection_idx in matches:
        x1, y1, x2, y2, confidence, class_id = current_detections[detection_idx]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Обновляем позицию объекта с сохранением confidence
        prev_x1, prev_y1, prev_x2, prev_y2, obj_name, _, _ = tracking_objects[obj_id]
        tracking_objects[obj_id] = (x1, y1, x2, y2, obj_name, class_id, confidence)
        current_ids.add(obj_id)
        
        # Добавляем точку в трейл
        if obj_id in trail_points:
            trail_points[obj_id].append(((x1 + x2) // 2, (y1 + y2) // 2))
    
    # Обрабатываем новые объекты
    for idx, detection in enumerate(current_detections):
        if idx not in used_indices:
            x1, y1, x2, y2, confidence, class_id = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Получаем название класса
            obj_name = model.names[int(class_id)]
            
            # Генерируем новый ID
            obj_id = generate_new_id(obj_name)
            current_ids.add(obj_id)
            
            # Добавляем новый объект в отслеживание с сохранением confidence
            tracking_objects[obj_id] = (x1, y1, x2, y2, obj_name, class_id, confidence)
            trail_points[obj_id] = deque(maxlen=30)
            trail_points[obj_id].append(((x1 + x2) // 2, (y1 + y2) // 2))
            
            # Логируем появление нового объекта с его ID
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(f"{time_formatted} - {obj_id} ({obj_name}, уверенность: {confidence:.2f})\n")
            
            # Сохраняем обрезанный кадр с обнаруженным объектом
            cropped_frame = frame[y1:y2, x1:x2].copy()
            accuracy_percent = int(confidence * 100)
            time_for_filename = time_formatted.replace(':', '-')
            frame_filename = f"{os.path.dirname(log_file)}/{obj_id}_{accuracy_percent}percent_{time_for_filename}.jpg"
            cv2.imwrite(frame_filename, cropped_frame)
    
    # Удаляем объекты, которые больше не отслеживаются
    for obj_id in list(tracking_objects.keys()):
        if obj_id not in current_ids:
            del tracking_objects[obj_id]
            if obj_id in trail_points:
                del trail_points[obj_id]
    
    # Рисуем рамки и ID на объектах
    for obj_id, obj_data in tracking_objects.items():
        if obj_id in current_ids:  # Только если объект есть в текущем кадре
            x1, y1, x2, y2, obj_name, class_id, confidence = obj_data
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{obj_id} [{confidence*100:.0f}%]"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Рисуем следы объектов
    for obj_id, points in trail_points.items():
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            cv2.line(frame, points[i - 1], points[i], (0, 0, 255), 2)
    
    return frame

def process_videos_in_folder(folder_path, tracking_objects, trail_points, show_video_checkbox, label, confidence_threshold=0.5):
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        # Сбрасываем счетчик ID для каждого нового видео
        reset_id_counter()
        cap = cv2.VideoCapture(video_path)
        
        # Получение имени файла без расширения и текущей даты/времени
        video_name = os.path.splitext(video_file)[0]
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        result_folder = os.path.join("Results", f"{video_name}_{timestamp}")
        os.makedirs(result_folder, exist_ok=True)
        
        # Сохраняем метаданные видео
        get_video_metadata(video_path, result_folder)
        
        log_file = os.path.join(result_folder, f"detect_{video_name}.txt")
        out = cv2.VideoWriter(os.path.join(result_folder, f"output_{video_file}"), cv2.VideoWriter_fourcc(*'XVID'), 20.0, (int(cap.get(3)), int(cap.get(4))))

        with open(log_file, "w", encoding='utf-8') as f:
            f.write("Обнаруженные объекты:\n")

        while process_frame(cap, tracking_objects, trail_points, log_file, out, show_video_checkbox, label, confidence_threshold):
            pass

def process_single_video(video_path, tracking_objects, trail_points, show_video_checkbox, label, confidence_threshold=0.5):
    """Обработка одного видеофайла"""
    # Сбрасываем счетчик ID для новой обработки
    reset_id_counter()
    cap = cv2.VideoCapture(video_path)
    
    # Получение имени файла без расширения и текущей даты/времени
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    # Создание папки для результатов с датой и временем
    result_folder = os.path.join("Results", f"{video_name}_{timestamp}")
    os.makedirs(result_folder, exist_ok=True)
    
    # Сохраняем метаданные видео
    get_video_metadata(video_path, result_folder)
    
    # Путь до лог-файла и видео-файла
    log_file = os.path.join(result_folder, f"detect_{video_name}.txt")
    output_video_path = os.path.join(result_folder, f"output_{video_filename}")
    
    # Инициализация VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Инициализация лог-файла
    with open(log_file, "w", encoding='utf-8') as f:
        f.write("Обнаруженные объекты с индивидуальными ID:\n")
    
    # Обработка видео кадр за кадром
    while process_frame(cap, tracking_objects, trail_points, log_file, out, show_video_checkbox, label, confidence_threshold):
        pass

def process_single_image(image_path, tracking_objects, trail_points, show_video_checkbox, label, confidence_threshold=0.5):
    """Обработка одного изображения"""
    try:
        # Сбрасываем счетчик ID для новой обработки
        reset_id_counter()
        
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка: не удается загрузить изображение {image_path}")
            return
        
        print(f"[IMAGE] Загружено изображение: {image_path}")
        
        # Получение имени файла без расширения и текущей даты/времени
        image_filename = os.path.basename(image_path)
        image_name = os.path.splitext(image_filename)[0]
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        
        # Создание папки для результатов с датой и временем
        result_folder = os.path.join("Results", f"{image_name}_{timestamp}")
        os.makedirs(result_folder, exist_ok=True)
        print(f"[IMAGE] Создана папка результатов: {result_folder}")
        
        # Путь до лог-файла и обработанного изображения
        log_file = os.path.join(result_folder, f"detect_{image_name}.txt")
        output_image_path = os.path.join(result_folder, f"output_{image_filename}")
        
        # Инициализация лог-файла
        with open(log_file, "w", encoding='utf-8') as f:
            f.write("Обнаруженные объекты с индивидуальными ID:\n")
        
        print(f"[IMAGE] Начинается обработка изображения...")
        # Обработка изображения
        processed_image = process_image_frame(image, log_file, show_video_checkbox, label, confidence_threshold)
        
        # Сохранение обработанного изображения
        cv2.imwrite(output_image_path, processed_image)
        print(f"[IMAGE] Сохранено обработанное изображение: {output_image_path}")
        print(f"[IMAGE] Обработка завершена!")
    except Exception as e:
        print(f"[ERROR] Ошибка при обработке изображения {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()

def process_images_in_folder(folder_path, tracking_objects, trail_points, show_video_checkbox, label, confidence_threshold=0.5):
    """Обработка всех изображений в папке"""
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTENSIONS)]
    print(f"[FOLDER] Найдено изображений: {len(image_files)}")
    
    for image_file in image_files:
        # Сбрасываем счетчик ID для каждого нового изображения
        reset_id_counter()
        image_path = os.path.join(folder_path, image_file)
        print(f"[FOLDER] Обрабатываю: {image_file}")
        process_single_image(image_path, tracking_objects, trail_points, show_video_checkbox, label, confidence_threshold)

def process_image_frame(frame, log_file, show_video_checkbox, label, confidence_threshold=0.5):
    """Обработка одного кадра/изображения для детекции объектов"""
    try:
        # Обработка кадра с помощью YOLOv8
        print(f"[PROCESS] Отправляю изображение в YOLO...")
        results = model(frame)
        print(f"[PROCESS] YOLO обработал изображение, найдено результатов: {len(results)}")
        
        # Постобработка и размечивание объектов
        frame = detect_and_draw(frame, results, log_file, confidence_threshold)
        
        # Показ результата в UI
        if show_video_checkbox.isChecked():
            print(f"[PROCESS] Отображаю результат в UI...")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_rgb.shape
            step = channel * width
            q_img = QImage(frame_rgb.data, width, height, step, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(q_img))
        
        return frame
    except Exception as e:
        print(f"[ERROR] Ошибка в process_image_frame: {str(e)}")
        import traceback
        traceback.print_exc()
        return frame

def detect_and_draw(frame, results, log_file, confidence_threshold=0.5):
    """Детекция объектов и рисование рамок с уникальными ID"""
    try:
        # Получаем текущее время детекции
        current_time = datetime.now().strftime('%H-%M-%S')
        
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            if confidence > confidence_threshold:  # Фильтрация по порогу уверенности
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Получение названия класса объекта
                obj_name = model.names[int(class_id)]
                
                # Генерируем индивидуальный ID для объекта
                obj_id = generate_new_id(obj_name)
                
                print(f"[DETECT] Найден объект: {obj_id} ({obj_name}) с уверенностью {confidence:.2f}")
                
                # Логируем обнаруженный объект с его ID
                with open(log_file, "a", encoding='utf-8') as f:
                    f.write(f"{current_time} - {obj_id} ({obj_name}): {confidence:.2f}\n")
                
                # Сохраняем обрезанный фрагмент с обнаруженным объектом (только область внутри рамки)
                cropped_frame = frame[y1:y2, x1:x2].copy()
                # Формируем имя файла с ID, процентом точности и временем детекции
                accuracy_percent = int(confidence * 100)
                frame_filename = f"{os.path.dirname(log_file)}/{obj_id}_{accuracy_percent}percent_{current_time}.jpg"
                cv2.imwrite(frame_filename, cropped_frame)
                print(f"[DETECT] Сохранен скриншот: {frame_filename}")
                
                # Рисуем рамку и ID на кадре
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"ID:{obj_id} [{confidence*100:.0f}%]"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    except Exception as e:
        print(f"[ERROR] Ошибка в detect_and_draw: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return frame