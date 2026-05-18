from huggingface_hub import hf_hub_download
import os

gguf_dir = r'z:\Coding\Watch golem stable\Models\Qwen2.5-VL-3B-GGUF'
os.makedirs(gguf_dir, exist_ok=True)

log = open(os.path.join(gguf_dir, 'download_log.txt'), 'w', buffering=1)

main_gguf = os.path.join(gguf_dir, 'Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf')
if os.path.isfile(main_gguf) and os.path.getsize(main_gguf) > 1_800_000_000:
    log.write(f'[1/2] Основная модель уже скачана: {main_gguf}\n')
else:
    log.write('[1/2] Скачиваем основную модель Q4_K_M...\n')
    path1 = hf_hub_download(
        repo_id='ggml-org/Qwen2.5-VL-3B-Instruct-GGUF',
        filename='Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf',
        local_dir=gguf_dir
    )
    log.write(f'OK: {path1}\n')

mmproj_gguf = os.path.join(gguf_dir, 'mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf')
if os.path.isfile(mmproj_gguf) and os.path.getsize(mmproj_gguf) > 800_000_000:
    log.write(f'[2/2] mmproj уже скачан: {mmproj_gguf}\n')
else:
    log.write('[2/2] Скачиваем mmproj...\n')
    path2 = hf_hub_download(
        repo_id='ggml-org/Qwen2.5-VL-3B-Instruct-GGUF',
        filename='mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf',
        local_dir=gguf_dir
    )
    log.write(f'OK: {path2}\n')

log.write('=== ВСЕ ФАЙЛЫ СКАЧАНЫ ===\n')
log.close()
