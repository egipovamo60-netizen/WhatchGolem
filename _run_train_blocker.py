"""
Вспомогательный скрипт — запускается как отдельный subprocess из TrainBlockerWorker.
Изолирует PyTorch/CUDA от Qt GUI-процесса, что предотвращает WinError 1455 / 0xC0000005.
Принимает путь к JSON-файлу с параметрами как единственный аргумент.
"""
import os
import sys
import json

os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["PYTHONUNBUFFERED"]       = "1"

# Форсируем line-buffered режим для stdout/stderr на уровне файловых дескрипторов.
# os.environ[PYTHONUNBUFFERED] влияет только если задан ДО старта Python;
# здесь мы уже внутри процесса, поэтому пересоздаём обёртки явно.
try:
    sys.stdout = open(sys.stdout.fileno(), "w", buffering=1, closefd=False, encoding="utf-8", errors="replace")
    sys.stderr = open(sys.stderr.fileno(), "w", buffering=1, closefd=False, encoding="utf-8", errors="replace")
except Exception:
    pass  # если не поддерживается — продолжаем без изменений

def _patch_device(device: str) -> str:
    """Передаём device как есть; auto разрешится внутри train()."""
    return device

def main():
    if len(sys.argv) < 2:
        print("[ERROR] Не передан путь к файлу аргументов.", flush=True)
        sys.exit(1)

    args_file = sys.argv[1]
    with open(args_file, "r", encoding="utf-8") as f:
        args = json.load(f)

    source_dir        = args["source_dir"]
    epochs            = int(args["epochs"])
    imgsz             = int(args["imgsz"])
    batch             = int(args["batch"])
    weights           = args["weights"] or None
    output_path       = args["output_path"]
    overwrite_dataset = bool(args["overwrite_dataset"])
    use_synthetic     = bool(args.get("use_synthetic", False))
    n_train           = int(args.get("n_train", 2000))
    n_val             = int(args.get("n_val", 300))
    patience          = int(args.get("patience", 50))
    data_yaml         = args.get("data_yaml") or None
    class_name        = args.get("class_name") or "parking_blocker"
    device            = _patch_device(args["device"])

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)

    from Plugins.ParkingBlockerTrainer.parking_blocker_trainer import (
        prepare_dataset, prepare_synthetic_dataset, train
    )

    def log_cb(msg):
        print(msg, flush=True)

    # Подготовка датасета
    if data_yaml:
        if not os.path.isfile(data_yaml):
            print(f"[ERROR] data.yaml не найден: {data_yaml}", flush=True)
            sys.exit(1)
        print(f"Используется готовый YOLO-датасет: {data_yaml}", flush=True)
    elif use_synthetic:
        print(f"Генерация синтетического датасета (train={n_train}, val={n_val})...", flush=True)
        data_yaml = prepare_synthetic_dataset(
            source_dir=source_dir,
            n_train=n_train,
            n_val=n_val,
            overwrite=overwrite_dataset,
            log_callback=log_cb,
        )
        print(f"Датасет готов: {data_yaml}", flush=True)
    else:
        print("Подготовка датасета (кропы с bbox на весь кадр)...", flush=True)
        data_yaml = prepare_dataset(
            source_dir=source_dir,
            overwrite=overwrite_dataset,
            class_name=class_name,
        )
        print(f"Датасет готов: {data_yaml}", flush=True)

    print(
        f"Запуск тренировки: epochs={epochs}, imgsz={imgsz}, "
        f"batch={batch}, device={device}",
        flush=True,
    )

    def log_cb(msg):
        print(msg, flush=True)

    model_path = train(
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        pretrained_weights=weights,
        output_path=output_path,
        overwrite_dataset=False,
        device=device,
        patience=patience,
        log_callback=log_cb,
        data_yaml=data_yaml,
        class_name=class_name,
    )

    print(f"Модель сохранена: {model_path}", flush=True)


if __name__ == "__main__":
    main()
