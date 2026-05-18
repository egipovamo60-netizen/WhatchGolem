#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Диагностический инструмент для выявления почему дубликаты не обнаруживаются
"""

import sys
sys.path.insert(0, '.')

from PIL import Image
import imagehash
import cv2
from skimage.metrics import structural_similarity as ssim
import os

def diagnose_images(image_path1, image_path2):
    """
    Диагностирует почему два похожих изображения не обнаруживаются как дубликаты
    
    Args:
        image_path1: Путь к первому изображению
        image_path2: Путь ко второму изображению
    """
    
    if not os.path.exists(image_path1) or not os.path.exists(image_path2):
        print("❌ Один из файлов не найден!")
        return
    
    print("=" * 70)
    print("ДИАГНОСТИКА ДУБЛИКАТОВ")
    print("=" * 70)
    print(f"\nИзображение 1: {os.path.basename(image_path1)}")
    print(f"Изображение 2: {os.path.basename(image_path2)}")
    
    # Загружаем изображения
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)
    
    print(f"\n📊 РАЗМЕРЫ И ФОРМАТЫ:")
    print(f"  Изображение 1: {img1.size} ({img1.format}) {img1.mode}")
    print(f"  Изображение 2: {img2.size} ({img2.format}) {img2.mode}")
    
    # Генерируем хеши
    print(f"\n🔐 ХЕШИ (расстояние Хэмминга):")
    
    hash_avg1 = imagehash.average_hash(img1)
    hash_avg2 = imagehash.average_hash(img2)
    dist_avg = bin((hash_avg1.hash ^ hash_avg2.hash)).count('1')
    print(f"  Average Hash: {dist_avg} бит (< 8 = совпадает)")
    
    hash_phash1 = imagehash.phash(img1)
    hash_phash2 = imagehash.phash(img2)
    dist_phash = bin((hash_phash1.hash ^ hash_phash2.hash)).count('1')
    print(f"  Perceptual Hash: {dist_phash} бит (< 8 = совпадает)")
    
    hash_dhash1 = imagehash.dhash(img1)
    hash_dhash2 = imagehash.dhash(img2)
    dist_dhash = bin((hash_dhash1.hash ^ hash_dhash2.hash)).count('1')
    print(f"  Difference Hash: {dist_dhash} бит (< 8 = совпадает)")
    
    hash_whash1 = imagehash.whash(img1)
    hash_whash2 = imagehash.whash(img2)
    dist_whash = bin((hash_whash1.hash ^ hash_whash2.hash)).count('1')
    print(f"  Wavelet Hash: {dist_whash} бит (< 8 = совпадает)")
    
    # Считаем сколько методов совпадает
    matches = sum([
        dist_avg <= 10,
        dist_phash <= 10,
        dist_dhash <= 10,
        dist_whash <= 10
    ])
    print(f"\n  ✓ {matches}/4 методов совпадают (нужно 2+)")
    
    # SSIM
    print(f"\n📐 СТРУКТУРНОЕ СХОДСТВО (SSIM):")
    
    try:
        img1_cv = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img2_cv = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
        
        if img1_cv is not None and img2_cv is not None:
            h = min(img1_cv.shape[0], img2_cv.shape[0])
            w = min(img1_cv.shape[1], img2_cv.shape[1])
            
            img1_cv = img1_cv[:h, :w]
            img2_cv = img2_cv[:h, :w]
            
            ssim_score = ssim(img1_cv, img2_cv, data_range=255)
            print(f"  SSIM: {ssim_score:.3f} (> 0.75 = совпадает)")
        else:
            print(f"  SSIM: Не удалось вычислить")
    except Exception as e:
        print(f"  SSIM: Ошибка - {e}")
    
    # Рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    print(f"  Если это ДЕЙСТВИТЕЛЬНО дубликаты, то:")
    
    if matches < 2:
        print(f"    └─ Уменьшьте Хэмминг с 10 до 12-15")
    else:
        print(f"    ✓ Хеши совпадают достаточно")
    
    if 'ssim_score' in locals() and ssim_score < 0.75:
        print(f"    └─ Уменьшьте SSIM с 0.75 до 0.60-0.70")
    
    if img1.size != img2.size:
        print(f"    └─ Размеры отличаются - это нормально, уменьшьте SSIM")
    
    if img1.format != img2.format:
        print(f"    └─ Разные форматы - уменьшьте Хэмминг на 3-5")
    
    # Коэффициенты для разных сценариев
    print(f"\n🎯 РЕКОМЕНДУЕМЫЕ ПАРАМЕТРЫ:")
    print(f"  Для этого случая:")
    
    suggested_hamming = 10
    suggested_ssim = 0.75
    
    if matches < 2:
        suggested_hamming = 15
        print(f"    • Хэмминг: {suggested_hamming} (было много различий в хешах)")
    else:
        print(f"    • Хэмминг: {suggested_hamming} (хеши совпадают)")
    
    if 'ssim_score' in locals() and ssim_score < 0.75:
        suggested_ssim = 0.65
        print(f"    • SSIM: {suggested_ssim} (было низкое структурное сходство)")
    else:
        print(f"    • SSIM: {suggested_ssim}")
    
    print(f"\n  Примечание: попробуйте эти значения в UI слайдерах")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    print("ДИАГНОСТИКА ДУБЛИКАТОВ")
    print("\nИспользование:")
    print("  python diagnose_duplicates.py image1.jpg image2.jpg")
    print("\nПримеры:")
    print("  python diagnose_duplicates.py photo_92percent.jpg photo_85percent.jpg")
    print("  python diagnose_duplicates.py image_v1.png image_v2.jpg")
    
    if len(sys.argv) == 3:
        img1 = sys.argv[1]
        img2 = sys.argv[2]
        diagnose_images(img1, img2)
    else:
        print("\n❌ Укажите два изображения для сравнения")
