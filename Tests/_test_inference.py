"""
Тест полного инференса через воркер subprocess.
Посылаем analyze команду с тестовым изображением.
"""
import os
import sys
import json
import subprocess
import time

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKER = os.path.join(BASE, "Plugins", "Qwen35AWQAnalyzer", "worker_process.py")
TEST_IMG = os.path.join(BASE, "Test foto", "Foto2.jpg")

env = os.environ.copy()
proc = subprocess.Popen(
    [sys.executable, WORKER],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True, encoding="utf-8", bufsize=1, env=env,
)

print("Waiting for READY...", flush=True)
start = time.time()
while True:
    line = proc.stdout.readline()
    if not line:
        print("DIED, stderr:", proc.stderr.read()[-2000:])
        sys.exit(1)
    line = line.rstrip()
    if line:
        try:
            msg = json.loads(line)
            if msg.get("status") == "error":
                print("LOAD ERROR:", msg.get("message"))
                proc.kill()
                sys.exit(1)
            if msg.get("status") == "progress":
                print(f"  [{round(time.time()-start,1)}s] {msg.get('step')}", flush=True)
        except (json.JSONDecodeError, AttributeError):
            print("  raw:", line[:200])
    if line == "READY":
        print(f"READY in {round(time.time()-start,1)}s!", flush=True)
        break
    if time.time() - start > 120:
        print("TIMEOUT")
        proc.kill(); sys.exit(1)

print(f"\nSending analyze for: {os.path.basename(TEST_IMG)}", flush=True)
cmd = json.dumps({"action": "analyze", "image": TEST_IMG}, ensure_ascii=False)
proc.stdin.write(cmd + "\n")
proc.stdin.flush()

print("Waiting for result...", flush=True)
t2 = time.time()
while True:
    line = proc.stdout.readline()
    if not line:
        print("DIED during analyze, stderr:", proc.stderr.read()[-2000:])
        break
    line = line.rstrip()
    if not line:
        continue
    try:
        msg = json.loads(line)
        if msg.get("status") == "result":
            elapsed = round(time.time()-t2, 1)
            print(f"\nRESULT ({elapsed}s):")
            print(json.dumps(msg.get("data", {}), ensure_ascii=False, indent=2))
            break
        elif msg.get("status") == "progress":
            print(f"  analysis: {msg.get('step')}", flush=True)
        elif msg.get("status") == "debug":
            print(f"  raw model output: {msg.get('raw', '')[:500]}")
        elif msg.get("status") == "error":
            print("ANALYZE ERROR:", msg.get("message"))
            break
    except (json.JSONDecodeError, AttributeError):
        print("  raw:", line[:200])
    if time.time() - t2 > 120:
        print("INFERENCE TIMEOUT")
        break

proc.stdin.write(json.dumps({"action": "quit"}) + "\n")
proc.stdin.flush()
proc.wait(timeout=5)
print("\nDone.")
