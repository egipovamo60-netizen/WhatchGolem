#!/usr/bin/env python3
"""Тест функции удаления дубликатов"""

from duplicate_detector import detect_and_remove_duplicates
import os

print('🔍 Запускаю обнаружение дубликатов...\n')

result = detect_and_remove_duplicates('test_duplicates')

print('\n✅ РЕЗУЛЬТАТЫ:')
print(f'├─ Удалено дубликатов: {result["duplicates_deleted"]}')
print(f'├─ Обработано групп: {result["duplicate_groups_processed"]}')
print(f'├─ Перемещено уникальных: {result["unique_images_moved"]}')
print(f'└─ Папка результата: {result["output_folder"]}')

print('\n📂 Проверка файлов:')
if os.path.exists('test_duplicates'):
    files = os.listdir('test_duplicates')
    print(f'test_duplicates: {files} ({len(files)} файл(ов))')
else:
    print('test_duplicates: [папка удалена/не существует]')

if os.path.exists(result['output_folder']):
    files = os.listdir(result['output_folder'])
    print(f'{result["output_folder"]}: {files} ({len(files)} файл(ов))')
else:
    print(f'{result["output_folder"]}: [не существует]')

print('\n✓ Тест завершён!')
