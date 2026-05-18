"""
Утилита управления базой лиц для Watch Golem.

Использование:
  python manage_face_db.py add "Иван Петров" path/to/photo.jpg
  python manage_face_db.py list
  python manage_face_db.py remove "Иван Петров"
  python manage_face_db.py rebuild
"""

import os
import sys
import shutil

FACE_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FaceDB")


def add_person(name, image_path):
    """Добавляет фото человека в базу лиц."""
    if not os.path.isfile(image_path):
        print(f"Файл не найден: {image_path}")
        return

    folder_name = name.replace(" ", "_")
    folder = os.path.join(FACE_DB_DIR, folder_name)
    os.makedirs(folder, exist_ok=True)

    dst = os.path.join(folder, os.path.basename(image_path))
    shutil.copy2(image_path, dst)

    # Сбрасываем кэш эмбеддингов
    cache = os.path.join(FACE_DB_DIR, "_embeddings_cache.json")
    if os.path.isfile(cache):
        os.remove(cache)

    print(f"✓ Добавлено: {name} ← {os.path.basename(image_path)}")
    print(f"  Папка: {folder}")


def list_persons():
    """Выводит список всех людей в базе."""
    if not os.path.isdir(FACE_DB_DIR):
        print("База лиц пуста (директория FaceDB не найдена)")
        return

    persons = []
    for entry in sorted(os.listdir(FACE_DB_DIR)):
        d = os.path.join(FACE_DB_DIR, entry)
        if os.path.isdir(d) and not entry.startswith("_"):
            photos = [
                f for f in os.listdir(d)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            persons.append((entry.replace("_", " "), len(photos)))

    if not persons:
        print("База лиц пуста")
        return

    print(f"База лиц ({len(persons)} персон):")
    for name, count in persons:
        print(f"  • {name} — {count} фото")


def remove_person(name):
    """Удаляет человека из базы лиц."""
    folder_name = name.replace(" ", "_")
    folder = os.path.join(FACE_DB_DIR, folder_name)

    if os.path.isdir(folder):
        shutil.rmtree(folder)
        cache = os.path.join(FACE_DB_DIR, "_embeddings_cache.json")
        if os.path.isfile(cache):
            os.remove(cache)
        print(f"✓ Удалён: {name}")
    else:
        print(f"✗ Не найден: {name}")
        print(f"  Искали папку: {folder}")


def rebuild_cache():
    """Удаляет кэш эмбеддингов (пересоздаётся при следующем анализе)."""
    cache = os.path.join(FACE_DB_DIR, "_embeddings_cache.json")
    if os.path.isfile(cache):
        os.remove(cache)
        print("✓ Кэш удалён. Будет пересоздан при следующем анализе.")
    else:
        print("Кэш не найден (будет создан при первом анализе).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "add":
        if len(sys.argv) < 4:
            print('Использование: python manage_face_db.py add "Имя Фамилия" path/to/photo.jpg')
            sys.exit(1)
        add_person(sys.argv[2], sys.argv[3])

    elif cmd == "list":
        list_persons()

    elif cmd == "remove":
        if len(sys.argv) < 3:
            print('Использование: python manage_face_db.py remove "Имя Фамилия"')
            sys.exit(1)
        remove_person(sys.argv[2])

    elif cmd == "rebuild":
        rebuild_cache()

    else:
        print(f"Неизвестная команда: {cmd}")
        print(__doc__)
