#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Продвинутая система обнаружения дубликатов изображений
Архитектура: SHA-256 → pHash → CNN embeddings
Автор: Улучшенная версия
"""

import os
import re
import shutil
import hashlib
import logging
from pathlib import Path
from PIL import Image
import imagehash
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict

# Для CNN embeddings
try:
    from torchvision import models, transforms
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch не установлен - CNN embeddings недоступны")
    print("   Но система работает с SHA-256 и pHash!")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class DuplicateDetectorAdvanced:
    """Продвинутый класс для обнаружения дубликатов с 3-этапной архитектурой"""
    
    def __init__(self, source_folder, hamming_threshold=10, ssim_threshold=0.75):
        """
        Args:
            source_folder: Папка с изображениями
            hamming_threshold: Порог для pHash (5-25)
            ssim_threshold: Порог для структурного сходства (0.60-1.00)
        """
        self.source_folder = source_folder
        self.output_folder = os.path.join(source_folder, "QunicObject")
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
        self.hamming_threshold = hamming_threshold
        self.ssim_threshold = ssim_threshold
        
        # CNN модель
        self.device = None
        self.model = None
        self.transform = None
        self.embedding_cache = {}
        
        if TORCH_AVAILABLE:
            self._init_cnn_model()
    
    def _init_cnn_model(self):
        """Инициализирует ResNet50 для CNN embeddings"""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"\n📦 Загружаю ResNet50 (использую {str(self.device).upper()})...")
            
            # Загружаем предобученную модель
            self.model = models.resnet50(pretrained=True)
            # Убираем последний слой (classification head)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.to(self.device)
            self.model.eval()
            
            # Подготовка изображений
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            print("✓ ResNet50 готова!\n")
        except Exception as e:
            print(f"⚠️  Ошибка инициализации CNN: {e}\n")
            self.model = None
    
    # ═══════════════════════════════════════════════════════════════════
    # ЭТАП 1: SHA-256 - ТОЧНЫЕ ДУБЛИКАТЫ
    # ═══════════════════════════════════════════════════════════════════
    
    @staticmethod
    def compute_sha256(file_path, chunk_size=65536):
        """Вычисляет SHA-256 хеш файла"""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(chunk_size), b''):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Ошибка SHA-256: {e}")
            return None
    
    def stage1_find_exact_duplicates(self, image_files):
        """
        ЭТАП 1: Находит точные дубликаты по SHA-256
        Самый быстрый и точный метод
        """
        print("\n" + "=" * 70)
        print("[ЭТАП 1] 🔍 Поиск ТОЧНЫХ дубликатов (SHA-256)...")
        print("=" * 70)
        
        sha256_map = defaultdict(list)
        
        for img_path in image_files:
            sha256 = self.compute_sha256(img_path)
            if sha256:
                sha256_map[sha256].append(img_path)
        
        # Извлекаем только группы с дубликатами
        exact_groups = [files for files in sha256_map.values() if len(files) > 1]
        remaining = [f for files in sha256_map.values() if len(files) == 1 for f in files]
        
        print(f"✓ Точных дубликатов найдено: {len(exact_groups)} групп")
        print(f"✓ Уникальных файлов: {len(remaining)}")
        
        if exact_groups:
            for i, group in enumerate(exact_groups, 1):
                print(f"  └─ Группа {i}: {len(group)} файлов")
                for f in group:
                    print(f"     • {os.path.basename(f)}")
        
        return exact_groups, remaining
    
    # ═══════════════════════════════════════════════════════════════════
    # ЭТАП 2: pHASH - БЫСТРЫЙ ФИЛЬТР ПОХОЖИХ
    # ═══════════════════════════════════════════════════════════════════
    
    @staticmethod
    def compute_phash(image_path):
        """Вычисляет perceptual hash"""
        try:
            image = Image.open(image_path)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            return imagehash.phash(image)
        except Exception as e:
            return None
    
    @staticmethod
    def hamming_distance(hash1, hash2):
        """Расстояние Хэмминга (0-64 бита)"""
        if hash1 is None or hash2 is None:
            return 64
        try:
            # Используем встроенный метод imagehash для вычисления расстояния
            return hash1 - hash2
        except:
            # Fallback если что-то пошло не так
            return 64
    
    def stage2_find_similar_by_phash(self, remaining_images):
        """
        ЭТАП 2: Находит похожие изображения по pHash
        Быстрый фильтр перед CNN обработкой
        """
        print("\n" + "=" * 70)
        print("[ЭТАП 2] ⚡ Поиск похожих по pHash (быстрый фильтр)...")
        print("=" * 70)
        
        if not remaining_images:
            print("✓ Нет изображений для проверки")
            return []
        
        # Вычисляем pHash для всех
        phash_map = {}
        for img_path in remaining_images:
            phash = self.compute_phash(img_path)
            if phash:
                phash_map[img_path] = phash
        
        print(f"✓ Вычислено pHash: {len(phash_map)} изображений")
        
        # Находим кластеры похожих
        visited = set()
        phash_groups = []
        
        for i, img_path1 in enumerate(list(phash_map.keys())):
            if img_path1 in visited:
                continue
            
            group = [img_path1]
            visited.add(img_path1)
            
            for img_path2 in list(phash_map.keys())[i+1:]:
                if img_path2 in visited:
                    continue
                
                distance = self.hamming_distance(phash_map[img_path1], phash_map[img_path2])
                
                # Более мягкий порог для начального фильтра
                if distance <= self.hamming_threshold + 5:
                    group.append(img_path2)
                    visited.add(img_path2)
            
            if len(group) > 1:
                phash_groups.append(group)
        
        print(f"✓ Kandidaten групп найдено: {len(phash_groups)}")
        
        if phash_groups:
            for i, group in enumerate(phash_groups, 1):
                print(f"  └─ Кандидат {i}: {len(group)} файлов")
        
        return phash_groups
    
    # ═══════════════════════════════════════════════════════════════════
    # ЭТАП 3: CNN EMBEDDINGS - ФИНАЛЬНАЯ ПРОВЕРКА
    # ═══════════════════════════════════════════════════════════════════
    
    def compute_embedding(self, image_path):
        """Вычисляет CNN embedding (2048-мерный вектор)"""
        if not self.model:
            return None
        
        if image_path in self.embedding_cache:
            return self.embedding_cache[image_path]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(image_tensor)
                embedding = embedding.squeeze().cpu().numpy()
                self.embedding_cache[image_path] = embedding
                return embedding
        except Exception as e:
            return None
    
    @staticmethod
    def cosine_similarity(vec1, vec2):
        """Косинусное расстояние (0-1)"""
        if vec1 is None or vec2 is None:
            return 0.0
        
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def stage3_verify_by_cnn(self, phash_groups):
        """
        ЭТАП 3: Финальная проверка по CNN embeddings
        Высокая точность, проверяет семантическое сходство
        """
        if not TORCH_AVAILABLE or not self.model:
            print("\n" + "=" * 70)
            print("[ЭТАП 3] ⚠️  CNN недоступна - пропускаю финальную проверку")
            print("=" * 70)
            return phash_groups
        
        print("\n" + "=" * 70)
        print("[ЭТАП 3] 🧠 Финальная проверка по CNN embeddings...")
        print("=" * 70)
        
        final_groups = []
        
        for group in phash_groups:
            if len(group) < 2:
                continue
            
            # Вычисляем embeddings
            embeddings = {}
            for img_path in group:
                emb = self.compute_embedding(img_path)
                if emb is not None:
                    embeddings[img_path] = emb
            
            if len(embeddings) < 2:
                continue
            
            # Берём первый как эталон
            ref_path = list(embeddings.keys())[0]
            ref_emb = embeddings[ref_path]
            
            verified_group = [ref_path]
            
            # Проверяем остальные
            for img_path, emb in list(embeddings.items())[1:]:
                similarity = self.cosine_similarity(ref_emb, emb)
                
                # Высокий порог для финальной проверки (0.85+)
                if similarity > 0.85:
                    verified_group.append(img_path)
            
            if len(verified_group) > 1:
                final_groups.append(verified_group)
        
        print(f"✓ CNN подтвердила: {len(final_groups)} групп дубликатов")
        
        return final_groups
    
    # ═══════════════════════════════════════════════════════════════════
    # ГЛАВНЫЙ PIPELINE
    # ═══════════════════════════════════════════════════════════════════
    
    def find_all_duplicates(self):
        """Главный pipeline: SHA256 → pHash → CNN"""
        
        # Найти все изображения
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(Path(self.source_folder).glob(f'**/*{ext}'))
        image_files = [str(f) for f in image_files]
        
        if not image_files:
            print("❌ Изображения не найдены")
            return []
        
        print(f"\n📊 Найдено изображений: {len(image_files)}\n")
        
        # Этап 1: SHA-256
        exact_groups, remaining = self.stage1_find_exact_duplicates(image_files)
        
        # Этап 2: pHash
        phash_groups = self.stage2_find_similar_by_phash(remaining)
        
        # Этап 3: CNN
        final_groups = self.stage3_verify_by_cnn(phash_groups)
        
        # Объединяем результаты
        all_groups = exact_groups + final_groups
        
        print("\n" + "=" * 70)
        print(f"📊 ИТОГО: {len(all_groups)} групп дубликатов найдено")
        print("=" * 70)
        
        return all_groups
    
    def extract_percent(self, filename):
        """Извлекает процент из названия файла"""
        patterns = [
            r'(\d+)%',
            r'(\d+)[\s_]*percent'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return 0
    
    def process_duplicates(self, duplicate_groups):
        """Обрабатывает группы: оставляет с максимальным процентом"""
        
        os.makedirs(self.output_folder, exist_ok=True)
        total_deleted = 0
        total_groups = len(duplicate_groups)
        
        print("\n" + "=" * 70)
        print("🗑️  ОБРАБОТКА ДУБЛИКАТОВ")
        print("=" * 70)
        
        for group_idx, group in enumerate(duplicate_groups, 1):
            # Сортируем по проценту
            files_data = []
            for img_path in group:
                img_name = os.path.basename(img_path)
                percent = self.extract_percent(img_name)
                files_data.append((img_path, img_name, percent))
            
            files_data.sort(key=lambda x: x[2], reverse=True)
            
            print(f"\n💾 Группа {group_idx}:")
            
            # Сохраняем лучший
            best_path, best_name, best_percent = files_data[0]
            print(f"  ✓ СОХРАНЁН: {best_name} ({best_percent}%)")
            
            # Удаляем остальные
            for del_path, del_name, del_percent in files_data[1:]:
                try:
                    os.remove(del_path)
                    print(f"  ✗ УДАЛЁН:   {del_name} ({del_percent}%)")
                    total_deleted += 1
                except Exception as e:
                    print(f"  ! ОШИБКА:  {del_name} - {e}")
        
        print("\n" + "=" * 70)
        print(f"✅ Удалено файлов: {total_deleted}")
        print("=" * 70)
        
        return total_deleted, total_groups
    
    def move_unique_images(self):
        """Перемещает оставшиеся уникальные изображения в QunicObject"""
        os.makedirs(self.output_folder, exist_ok=True)
        moved_count = 0
        
        print("\n" + "=" * 70)
        print("📁 ПЕРЕМЕЩЕНИЕ УНИКАЛЬНЫХ ИЗОБРАЖЕНИЙ")
        print("=" * 70)
        
        for ext in self.image_extensions:
            for img_path in Path(self.source_folder).glob(f'**/*{ext}'):
                img_path_str = str(img_path)
                # Пропускаем если уже в итоговой папке
                if self.output_folder in img_path_str:
                    continue
                
                try:
                    dest_path = os.path.join(self.output_folder, os.path.basename(img_path_str))
                    shutil.copy2(img_path_str, dest_path)
                    moved_count += 1
                except Exception as e:
                    print(f"  ⚠️  Ошибка перемещения {os.path.basename(img_path_str)}: {e}")
        
        print(f"✓ Перемещено изображений: {moved_count}")
        print("=" * 70)
        
        return moved_count
    
    def run(self):
        """Запускает весь процесс"""
        
        print("\n" + "█" * 70)
        print("█  ПРОДВИНУТАЯ СИСТЕМА ОБНАРУЖЕНИЯ ДУБЛИКАТОВ ИЗОБРАЖЕНИЙ")
        print("█  Архитектура: SHA-256 → pHash → CNN embeddings")
        print("█" * 70)
        
        duplicates = self.find_all_duplicates()
        deleted = 0
        total_groups = 0
        
        if duplicates:
            deleted, total_groups = self.process_duplicates(duplicates)
        
        # Перемещаем оставшиеся уникальные изображения
        moved = self.move_unique_images()
        
        print(f"\n✨ Обработка завершена!")
        print(f"   • Групп дубликатов обработано: {total_groups}")
        print(f"   • Файлов удалено: {deleted}")
        print(f"   • Файлов перемещено: {moved}")
        print(f"   • Результаты в: {self.output_folder}")
        
        print("\n" + "█" * 70 + "\n")
        
        return {
            'duplicates_deleted': deleted,
            'duplicate_groups_processed': total_groups,
            'unique_images_moved': moved,
            'output_folder': self.output_folder
        }


# Функция-обёртка для совместимости
def detect_and_remove_duplicates(folder_path, hamming_threshold=10, ssim_threshold=0.75):
    """
    Обёртка для использования из UI и других модулей
    Использует новую 3-этапную архитектуру
    
    Returns:
        dict с ключами:
        - duplicates_deleted: количество удаленных файлов
        - duplicate_groups_processed: количество групп дубликатов
        - unique_images_moved: количество перемещенных уникальных файлов
        - output_folder: папка для результатов
    """
    if not os.path.exists(folder_path):
        print(f"❌ Папка не существует: {folder_path}")
        return None
    
    detector = DuplicateDetectorAdvanced(
        folder_path,
        hamming_threshold=hamming_threshold,
        ssim_threshold=ssim_threshold
    )
    
    # Запускаем и получаем результаты
    result = detector.run()
    
    return result
