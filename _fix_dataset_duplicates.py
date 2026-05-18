"""
Исправляет дубликаты в YOLO-датасете Parking_border:
- Находит файлы вида frame_XXX_1.txt, frame_XXX_2.txt
- Объединяет их bbox в base-файл frame_XXX.txt
- Удаляет дубль-файлы (.txt и .jpg)
"""
import os
import re

DATASET_DIR = r"z:\Coding\Watch golem stable\Results\OpenSetFolder_28-04-2026_15-36-10\combined_yolo_datasets\Parking_border"
SPLITS = ["train", "val"]


def fix_split(split):
    labels_dir = os.path.join(DATASET_DIR, "labels", split)
    images_dir = os.path.join(DATASET_DIR, "images", split)

    all_txts = set(os.listdir(labels_dir))
    dup_pattern = re.compile(r'^(.+)_(\d+)\.txt$')

    # base_stem -> [dup_fname, ...]
    duplicates = {}
    for fname in sorted(all_txts):
        m = dup_pattern.match(fname)
        if m:
            base_stem = m.group(1)
            if (base_stem + ".txt") in all_txts:
                duplicates.setdefault(base_stem, []).append(fname)

    if not duplicates:
        print(f"[{split}] Дубликаты не найдены.")
        return

    print(f"[{split}] Найдено {len(duplicates)} кадров с дубликатами:")

    merged_count = 0
    deleted_count = 0

    for base_stem, dup_files in duplicates.items():
        base_txt = os.path.join(labels_dir, base_stem + ".txt")

        # Пропускаем если base уже удалён в предыдущей итерации
        if not os.path.isfile(base_txt):
            continue

        # Читаем bbox из base
        with open(base_txt, "r") as f:
            base_lines = [l.strip() for l in f if l.strip()]

        # Читаем bbox из дублей
        extra_lines = []
        for dup_fname in dup_files:
            with open(os.path.join(labels_dir, dup_fname), "r") as f:
                extra_lines.extend(l.strip() for l in f if l.strip())

        # Объединяем уникальные bbox
        all_lines = list(dict.fromkeys(base_lines + extra_lines))

        with open(base_txt, "w") as f:
            f.write("\n".join(all_lines) + "\n")

        print(f"  {base_stem}: {len(base_lines)} + {len(extra_lines)} bbox -> {len(all_lines)}")
        merged_count += 1

        # Удаляем дубль .txt и дубль .jpg
        for dup_fname in dup_files:
            dup_path = os.path.join(labels_dir, dup_fname)
            if not os.path.isfile(dup_path):
                continue
            os.remove(dup_path)
            deleted_count += 1

            dup_stem = dup_fname[:-4]  # убираем .txt
            for img_fname in os.listdir(images_dir):
                if img_fname.startswith(dup_stem + "."):
                    os.remove(os.path.join(images_dir, img_fname))
                    deleted_count += 1
                    print(f"    Удалён образ: {img_fname}")

    remaining_imgs = len([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png"))])
    remaining_txts = len([f for f in os.listdir(labels_dir) if f.endswith(".txt")])
    print(f"[{split}] Итого: {merged_count} кадров объединены, {deleted_count} файлов удалено.")
    print(f"[{split}] Осталось: {remaining_imgs} изображений, {remaining_txts} label-файлов.\n")


if __name__ == "__main__":
    for split in SPLITS:
        fix_split(split)
    print("Готово. Датасет исправлен.")
