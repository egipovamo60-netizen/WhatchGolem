import cv2
import sys
import numpy as np
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QCheckBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
from collections import deque
import time

# Загрузка модели YOLO11 (xlarge)
model = YOLO("yolo11x.pt")

# Глобальный счетчик ID для уникальной идентификации объектов
object_id_counter = {}

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
    """Сбросить счетчик ID"""
    global object_id_counter
    object_id_counter = {}

class VideoObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Watch Golem")
        self.setGeometry(100, 100, 1000, 800)

        self.video_path = None
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)

        self.tracking_objects = {}  # Словарь для отслеживаемых объектов
        self.trail_points = {}  # Следы объектов

        self.log_file = "object_detection_log.txt"  # Имя файла для сохранения логов

        # Переменные для отслеживания прогресса
        self.total_frames = 0
        self.current_frame = 0
        self.start_time = None
        self.fps = 30.0

        # Очищаем лог-файл при запуске
        with open(self.log_file, "w", encoding='utf-8') as f:
            f.write("Обнаруженные объекты с индивидуальными ID:\n")

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("Выберите видео для анализа")
        self.label.setScaledContents(True)
        layout.addWidget(self.label)

        self.btn_open = QPushButton("Открыть видео")
        self.btn_open.clicked.connect(self.open_video)
        layout.addWidget(self.btn_open)

        self.btn_start = QPushButton("Начать анализ")
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_start.setEnabled(False)
        layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Остановить анализ")
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.btn_stop.setEnabled(False)
        layout.addWidget(self.btn_stop)

        self.show_video_checkbox = QCheckBox("Показывать видео во время обработки")
        self.show_video_checkbox.setChecked(True)
        layout.addWidget(self.show_video_checkbox)

        # Информация о прогрессе
        layout.addWidget(QLabel("─" * 50))  # Разделитель
        
        self.info_total_frames = QLabel("Всего кадров: 0")
        layout.addWidget(self.info_total_frames)

        self.info_processed_frames = QLabel("Обработано кадров: 0")
        layout.addWidget(self.info_processed_frames)

        self.info_progress = QLabel("Прогресс: 0%")
        layout.addWidget(self.info_progress)

        self.info_elapsed_time = QLabel("Прошедшее время: 00:00:00")
        layout.addWidget(self.info_elapsed_time)

        self.info_remaining_time = QLabel("Оставшееся время: --:--:--")
        layout.addWidget(self.info_remaining_time)

        self.setLayout(layout)

        # Создаем папку для сохранения кадров, если она не существует
        if not os.path.exists("FRAMES"):
            os.makedirs("FRAMES")

    def open_video(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Видео файлы (*.mp4 *.avi *.mov *.mkv)", options=options)

        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(self.video_path)
            
            # Получаем информацию о видео
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Обновляем информацию о кадрах
            self.info_total_frames.setText(f"Всего кадров: {self.total_frames}")
            
            self.btn_start.setEnabled(True)

            # Создаем объект VideoWriter для сохранения видео
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(self.cap.get(3)), int(self.cap.get(4))))

    def start_analysis(self):
        if self.cap:
            # Сбрасываем счетчик ID перед началом анализа
            reset_id_counter()
            
            # Инициализируем переменные для отслеживания прогресса
            self.current_frame = 0
            self.start_time = time.time()
            
            self.timer.start(30)  # Обработка кадров каждые 30 мс
            self.btn_stop.setEnabled(True)

    def stop_analysis(self):
        self.timer.stop()
        self.btn_stop.setEnabled(False)
        self.cap.release()
        self.out.release()  # Закрываем файл с сохраненным видео

    def process_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            self.timer.stop()
            self.cap.release()
            self.out.release()  # Закрываем файл с сохраненным видео
            self.info_remaining_time.setText("Оставшееся время: 00:00:00")
            return

        # Увеличиваем счетчик обработанных кадров
        self.current_frame += 1

        # Получение текущего времени видео
        current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Время в секундах
        time_formatted = time.strftime('%H:%M:%S', time.gmtime(current_time))

        # Обработка кадра с помощью YOLOv11
        results = model(frame)

        # Постобработка и размечивание объектов
        frame = self.draw_boxes_and_track(frame, results, time_formatted)

        # Сохранение обработанного кадра в видеофайл
        self.out.write(frame)

        # Обновление информации о прогрессе
        self.update_progress_info()

        if self.show_video_checkbox.isChecked():
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_rgb.shape
            step = channel * width
            q_img = QImage(frame_rgb.data, width, height, step, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(q_img))

    def update_progress_info(self):
        """Обновление информации о прогрессе обработки"""
        if self.total_frames == 0:
            return

        # Обновляем количество обработанных кадров
        self.info_processed_frames.setText(f"Обработано кадров: {self.current_frame}/{self.total_frames}")

        # Вычисляем процент завершенности
        progress_percent = int((self.current_frame / self.total_frames) * 100)
        self.info_progress.setText(f"Прогресс: {progress_percent}%")

        # Вычисляем прошедшее время
        elapsed_time = time.time() - self.start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        self.info_elapsed_time.setText(f"Прошедшее время: {hours:02d}:{minutes:02d}:{seconds:02d}")

        # Вычисляем оставшееся время (если уже обработано хотя бы несколько кадров)
        if self.current_frame > 0:
            time_per_frame = elapsed_time / self.current_frame
            remaining_frames = self.total_frames - self.current_frame
            remaining_time = time_per_frame * remaining_frames
            
            hours_rem = int(remaining_time // 3600)
            minutes_rem = int((remaining_time % 3600) // 60)
            seconds_rem = int(remaining_time % 60)
            self.info_remaining_time.setText(f"Оставшееся время: {hours_rem:02d}:{minutes_rem:02d}:{seconds_rem:02d}")

    def draw_boxes_and_track(self, frame, results, time_formatted):
        current_detections = []
        current_ids = set()
        
        # Собираем все текущие детекции
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            if confidence > 0.5:  # Фильтрация по порогу уверенности
                current_detections.append((x1, y1, x2, y2, confidence, class_id))
        
        # Сопоставляем текущие детекции с отслеживаемыми объектами
        matches, used_indices = match_detections(self.tracking_objects, current_detections, max_distance=100)
        
        # Обновляем существующие объекты
        for obj_id, detection_idx in matches:
            x1, y1, x2, y2, confidence, class_id = current_detections[detection_idx]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Обновляем позицию объекта с сохранением confidence
            prev_x1, prev_y1, prev_x2, prev_y2, obj_name, _, _ = self.tracking_objects[obj_id]
            self.tracking_objects[obj_id] = (x1, y1, x2, y2, obj_name, class_id, confidence)
            current_ids.add(obj_id)
            
            # Добавляем точку в трейл
            if obj_id in self.trail_points:
                self.trail_points[obj_id].append(((x1 + x2) // 2, (y1 + y2) // 2))
        
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
                self.tracking_objects[obj_id] = (x1, y1, x2, y2, obj_name, class_id, confidence)
                self.trail_points[obj_id] = deque(maxlen=30)
                self.trail_points[obj_id].append(((x1 + x2) // 2, (y1 + y2) // 2))
                
                # Логируем появление нового объекта с его ID
                with open(self.log_file, "a", encoding='utf-8') as f:
                    f.write(f"{time_formatted} - {obj_id} ({obj_name}, уверенность: {confidence:.2f})\n")
                
                # Сохраняем кадр с обнаруженным объектом
                frame_copy = frame.copy()
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"ID:{obj_id} [{confidence*100:.0f}%]"
                cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                frame_filename = f"FRAMES/{obj_id}_{time_formatted.replace(':', '-')}.jpg"
                cv2.imwrite(frame_filename, frame_copy)
        
        # Удаляем объекты, которые больше не отслеживаются
        for obj_id in list(self.tracking_objects.keys()):
            if obj_id not in current_ids:
                del self.tracking_objects[obj_id]
                if obj_id in self.trail_points:
                    del self.trail_points[obj_id]
        
        # Рисуем рамки и ID на объектах
        for obj_id, obj_data in self.tracking_objects.items():
            if obj_id in current_ids:  # Только если объект есть в текущем кадре
                x1, y1, x2, y2, obj_name, class_id, confidence = obj_data
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{obj_id} [{confidence*100:.0f}%]"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Рисование следов объектов
        for obj_id, points in self.trail_points.items():
            for i in range(1, len(points)):
                if points[i - 1] is None or points[i] is None:
                    continue
                cv2.line(frame, points[i - 1], points[i], (0, 0, 255), 2)
        
        return frame

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())
