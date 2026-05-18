import sys
import io
import os
import multiprocessing

# Принудительно UTF-8 в консоли Windows (исправляет кракозябры в PowerShell)
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Регистрируем DLL-директории onnxruntime в главном потоке,
# чтобы они были доступны во всех фоновых QThread на Windows.
if hasattr(os, "add_dll_directory"):
    import importlib.util as _ilu
    for _pkg in ("onnxruntime", "onnxruntime.capi"):
        _spec = _ilu.find_spec(_pkg)
        if _spec and _spec.origin:
            try:
                os.add_dll_directory(os.path.dirname(_spec.origin))
            except Exception:
                pass
    del _ilu

import sys
from PyQt5.QtWidgets import QApplication
from ui import VideoObjectDetectionApp
from logger import get_logger

log = get_logger("main")

if __name__ == "__main__":
    log.info("=== Запуск приложения Watch Golem ===")
    app = QApplication(sys.argv)
    window = VideoObjectDetectionApp()
    window.show()
    code = app.exec_()
    log.info(f"=== Приложение завершено (код: {code}) ===")
    sys.exit(code)