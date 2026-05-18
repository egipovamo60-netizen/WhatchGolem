"""
Скрипт запуска тренировки детектора парковочных блокираторов.

Использование:
  python train_parking_blocker.py
  python train_parking_blocker.py --epochs 100 --batch 16 --imgsz 640
  python train_parking_blocker.py --weights yolo11x.pt --device 0

После завершения модель сохраняется в:
  Models/parking_blocker.pt
"""

import os
import sys
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

from Plugins.ParkingBlockerTrainer.parking_blocker_trainer import (
    prepare_dataset,
    train,
    LEARN_DATASET_DIR,
    OUTPUT_MODEL_PATH,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Тренировка YOLO-детектора парковочных блокираторов"
    )
    parser.add_argument(
        "--source", default=LEARN_DATASET_DIR,
        help="Папка с изображениями блокираторов (по умолчанию: LearnDataset/)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Количество эпох (по умолчанию: 50)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Размер изображения для обучения (по умолчанию: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=8,
        help="Размер батча (по умолчанию: 8)"
    )
    parser.add_argument(
        "--weights", default=None,
        help="Базовые веса YOLO (по умолчанию: автовыбор из корня проекта)"
    )
    parser.add_argument(
        "--output", default=OUTPUT_MODEL_PATH,
        help=f"Куда сохранить итоговую модель (по умолчанию: {OUTPUT_MODEL_PATH})"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Устройство: auto | cpu | 0 | 0,1 (по умолчанию: auto)"
    )
    parser.add_argument(
        "--overwrite-dataset", action="store_true",
        help="Пересоздать YOLO-датасет, даже если он уже существует"
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Только подготовить датасет, без тренировки"
    )
    parser.add_argument(
        "--data", default=None,
        help="Путь к готовому data.yaml (пропускает генерацию датасета)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Тренировка детектора парковочных блокираторов")
    print("=" * 60)
    if args.data:
        print(f"  Готовый датасет      : {args.data}")
    else:
        print(f"  Источник изображений : {args.source}")
    print(f"  Эпохи                : {args.epochs}")
    print(f"  Размер изображения   : {args.imgsz}")
    print(f"  Батч                 : {args.batch}")
    print(f"  Device               : {args.device}")
    print(f"  Итоговая модель      : {args.output}")
    print("=" * 60)

    if args.prepare_only:
        data_yaml = prepare_dataset(
            source_dir=args.source,
            overwrite=args.overwrite_dataset,
        )
        print(f"\nДатасет готов: {data_yaml}")
        return

    model_path = train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        pretrained_weights=args.weights,
        output_path=args.output,
        overwrite_dataset=args.overwrite_dataset,
        device=args.device,
        source_dir=args.source,
        data_yaml=args.data,
    )

    print("\n" + "=" * 60)
    print(f"  Тренировка завершена!")
    print(f"  Модель сохранена: {model_path}")
    print("=" * 60)
    print("\nДля использования модели:")
    print("  from Plugins.ParkingBlockerTrainer import detect")
    print("  detections = detect(image)  # image — np.ndarray BGR или путь к файлу")
    print("  # -> [{\"bbox\": [x1,y1,x2,y2], \"conf\": 0.87, \"class\": \"parking_blocker\"}]")


if __name__ == "__main__":
    main()
