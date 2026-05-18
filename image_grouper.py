#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Модуль для группировки похожих изображений в отдельные папки.
Создаёт структуру GroupBy/Group_1, GroupBy/Group_2 и т.д.
"""

import os
import shutil
from PIL import Image
import imagehash
from collections import defaultdict
from datetime import datetime


class ImageGrouper:
    """Класс для группировки похожих изображений"""
    
    def __init__(self, hamming_threshold=10, ssim_threshold=0.75):
        """
        Инициализация группировщика изображений
        
        Args:
            hamming_threshold: Порог расстояния Хэмминга (0-64, меньше = строже)
            ssim_threshold: Порог SSIM (0-1, больше = строже)
        """
        self.hamming_threshold = hamming_threshold
        self.ssim_threshold = ssim_threshold
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
        
    def get_image_hash(self, image_path):
        """
        Получить усреднённый хеш изображения
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Хеш изображения или None если ошибка
        """
        try:
            img = Image.open(image_path)
            return imagehash.average_hash(img)
        except Exception as e:
            print(f"⚠ Ошибка при обработке {os.path.basename(image_path)}: {e}")
            return None

    def calculate_hash_distance(self, hash1, hash2):
        """
        Вычислить расстояние Хэмминга между двумя хешами
        
        Args:
            hash1: Первый хеш
            hash2: Второй хеш
            
        Returns:
            Расстояние Хэмминга (0-64)
        """
        if hash1 is None or hash2 is None:
            return float('inf')
        # Используем встроенный оператор вычитания imagehash для вычисления расстояния Хэмминга
        return hash1 - hash2

    def group_images(self, folder_path):
        """
        Сгруппировать изображения в папке по схожести
        
        Args:
            folder_path: Путь к папке с изображениями
            
        Returns:
            Словарь с информацией о группах:
            {
                'groups': [
                    {'id': 1, 'images': [...], 'size': 3},
                    {'id': 2, 'images': [...], 'size': 2}
                ],
                'total_groups': 2,
                'total_images': 5,
                'ungrouped': [...]
            }
        """
        if not os.path.exists(folder_path):
            print(f"❌ Папка не найдена: {folder_path}")
            return None
        
        # Собираем все изображения
        image_files = [
            f for f in os.listdir(folder_path) 
            if f.lower().endswith(self.image_extensions)
        ]
        
        if not image_files:
            print(f"⚠ В папке нет изображений: {folder_path}")
            return {
                'groups': [],
                'total_groups': 0,
                'total_images': 0,
                'ungrouped': []
            }
        
        print(f"📁 Найдено изображений: {len(image_files)}")
        print("🔄 Вычисляю хеши изображений...")
        
        # Вычисляем хеши для всех изображений
        image_hashes = {}
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            img_hash = self.get_image_hash(image_path)
            if img_hash is not None:
                image_hashes[image_file] = img_hash
        
        print(f"✓ Хеши вычислены для {len(image_hashes)} изображений")
        
        # Группируем похожие изображения
        print("🔗 Группирую похожие изображения...")
        groups = []
        processed = set()
        
        for image_file, img_hash in image_hashes.items():
            if image_file in processed:
                continue
            
            # Создаём новую группу
            group = [image_file]
            processed.add(image_file)
            
            # Ищем похожие изображения
            for other_file, other_hash in image_hashes.items():
                if other_file in processed:
                    continue
                
                distance = self.calculate_hash_distance(img_hash, other_hash)
                
                if distance <= self.hamming_threshold:
                    group.append(other_file)
                    processed.add(other_file)
            
            if len(group) > 0:
                groups.append(group)
        
        # Сортируем группы по размеру (большие первыми)
        groups.sort(key=len, reverse=True)
        
        # Формируем результат
        result = {
            'groups': [
                {
                    'id': idx + 1,
                    'images': group,
                    'size': len(group)
                }
                for idx, group in enumerate(groups)
            ],
            'total_groups': len(groups),
            'total_images': len(image_hashes),
            'ungrouped': [f for f in image_files if f not in image_hashes]
        }
        
        print(f"✓ Создано {result['total_groups']} групп")
        
        return result

    def save_groups(self, folder_path, group_data, base_output_dir=None):
        """
        Сохранить сгруппированные изображения в структуру папок
        
        Args:
            folder_path: Исходная папка с изображениями
            group_data: Результат функции group_images
            base_output_dir: Папка для сохранения (по умолчанию сохраняется в той же папке, где изображения)
            
        Returns:
            Информация о сохранённых данных или None при ошибке
        """
        if not group_data:
            print("❌ Данные группировки не предоставлены")
            return None
        
        # Если base_output_dir не указан, сохраняем в папке со скриншотами
        if base_output_dir is None:
            base_output_dir = folder_path
        
        # Создаём основную папку GroupBy с временной меткой в той же папке что и изображения
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        output_dir = os.path.join(base_output_dir, "GroupBy", f"grouped_{timestamp}")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n📁 Создана папка: {output_dir}")
            
            total_copied = 0
            
            # Создаём папки для каждой группы и копируем изображения
            for group_info in group_data['groups']:
                group_id = group_info['id']
                group_folder = os.path.join(output_dir, f"Group_{group_id}")
                os.makedirs(group_folder, exist_ok=True)
                
                print(f"\n📂 Группа {group_id} ({group_info['size']} изображений):")
                
                for image_file in group_info['images']:
                    src_path = os.path.join(folder_path, image_file)
                    dst_path = os.path.join(group_folder, image_file)
                    
                    try:
                        shutil.copy2(src_path, dst_path)
                        print(f"   ✓ {image_file}")
                        total_copied += 1
                    except Exception as e:
                        print(f"   ✗ Ошибка копирования {image_file}: {e}")
            
            # Сохраняем отчёт
            report_path = os.path.join(output_dir, "REPORT.txt")
            self._save_report(report_path, group_data, total_copied)
            
            return {
                'success': True,
                'output_dir': output_dir,
                'total_groups': group_data['total_groups'],
                'total_images': group_data['total_images'],
                'copied_images': total_copied,
                'ungrouped_images': len(group_data['ungrouped']),
                'report_file': report_path
            }
            
        except Exception as e:
            print(f"❌ Ошибка при сохранении групп: {e}")
            return None

    def _save_report(self, report_path, group_data, total_copied):
        """
        Сохранить отчёт о группировке в файл
        
        Args:
            report_path: Путь для сохранения отчёта
            group_data: Результат функции group_images
            total_copied: Количество скопированных файлов
        """
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("ОТЧЁТ О ГРУППИРОВКЕ ИЗОБРАЖЕНИЙ\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Дата и время: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n\n")
                
                f.write("ПАРАМЕТРЫ:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Порог Хэмминга: {self.hamming_threshold}\n")
                f.write(f"Порог SSIM: {self.ssim_threshold}\n\n")
                
                f.write("РЕЗУЛЬТАТЫ:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Всего групп: {group_data['total_groups']}\n")
                f.write(f"Всего изображений: {group_data['total_images']}\n")
                f.write(f"Скопировано изображений: {total_copied}\n")
                f.write(f"Не сгруппированных: {len(group_data['ungrouped'])}\n\n")
                
                f.write("ДЕТАЛИ ПО ГРУППАМ:\n")
                f.write("-" * 70 + "\n")
                for group_info in group_data['groups']:
                    f.write(f"\nГруппа {group_info['id']} ({group_info['size']} изображений):\n")
                    for image in group_info['images']:
                        f.write(f"  • {image}\n")
                
                if group_data['ungrouped']:
                    f.write(f"\nНе сгруппированные изображения ({len(group_data['ungrouped'])}):\n")
                    for image in group_data['ungrouped']:
                        f.write(f"  • {image}\n")
                
                f.write("\n" + "=" * 70 + "\n")
            
            print(f"✓ Отчёт сохранён: {report_path}")
        except Exception as e:
            print(f"⚠ Ошибка при сохранении отчёта: {e}")


def group_images_by_similarity(folder_path, hamming_threshold=10, ssim_threshold=0.75):
    """
    Главная функция для группировки изображений
    
    Args:
        folder_path: Путь к папке с изображениями
        hamming_threshold: Порог Хэмминга (по умолчанию 10)
        ssim_threshold: Порог SSIM (по умолчанию 0.75)
        
    Returns:
        Информация о результатах группировки или None при ошибке
        
    Examples:
        >>> result = group_images_by_similarity("path/to/images")
        >>> if result and result['success']:
        ...     print(f"Создано {result['total_groups']} групп")
        ...     print(f"Результаты в: {result['output_dir']}")
    """
    print("=" * 70)
    print("ГРУППИРОВКА ПОХОЖИХ ИЗОБРАЖЕНИЙ")
    print("=" * 70)
    print()
    
    grouper = ImageGrouper(hamming_threshold, ssim_threshold)
    
    # Группируем изображения
    group_data = grouper.group_images(folder_path)
    
    if not group_data:
        return None
    
    # Выводим статистику
    print("\n📊 СТАТИСТИКА:")
    print(f"  Групп: {group_data['total_groups']}")
    print(f"  Изображений: {group_data['total_images']}")
    if group_data['ungrouped']:
        print(f"  Не сгруппировано: {len(group_data['ungrouped'])}")
    
    print("\n📋 РАЗМЕР ГРУПП:")
    for group_info in group_data['groups'][:10]:  # Показываем первые 10
        print(f"  Группа {group_info['id']}: {group_info['size']} изображений")
    
    if len(group_data['groups']) > 10:
        print(f"  ... и ещё {len(group_data['groups']) - 10} групп")
    
    # Сохраняем группы
    print("\n💾 Сохраняю группы...")
    result = grouper.save_groups(folder_path, group_data)
    
    if result and result['success']:
        print("\n✓ ГОТОВО!")
        print(f"  Результаты сохранены в: {result['output_dir']}")
        print(f"  Отчёт: {result['report_file']}")
    
    print("\n" + "=" * 70)
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        hamming = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        ssim = float(sys.argv[3]) if len(sys.argv) > 3 else 0.75
        
        result = group_images_by_similarity(folder, hamming, ssim)
        
        if result and result['success']:
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        print("Использование:")
        print("  python image_grouper.py <папка_с_изображениями> [порог_хэмминга] [порог_ssim]")
        print("\nПримеры:")
        print("  python image_grouper.py ./photos")
        print("  python image_grouper.py ./photos 12")
        print("  python image_grouper.py ./photos 10 0.75")
