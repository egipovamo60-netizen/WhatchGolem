"""Quick smoke test for Qwen35AWQAnalyzer worker."""
import os
import subprocess
import sys
import json
import time

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

WORKER = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "Plugins", "Qwen35AWQAnalyzer", "worker_process.py")

env = os.environ.copy()

proc = subprocess.Popen(
    [sys.executable, WORKER],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding="utf-8",
    bufsize=1,
    env=env,
)

print("Worker started, waiting for READY...", flush=True)
start = time.time()

while True:
    line = proc.stdout.readline()
    if not line:
        stderr_out = proc.stderr.read()
        print("== PROCESS DIED ==")
        print("STDERR:")
        print(stderr_out[-4000:])
        sys.exit(1)

    line = line.rstrip()
    if line:
        print(">>>", line[:300])

    if line == "READY":
        elapsed = round(time.time() - start, 1)
        print(f"SUCCESS: Worker is READY in {elapsed}s")
        proc.stdin.write(json.dumps({"action": "quit"}) + "\n")
        proc.stdin.flush()
        proc.wait(timeout=10)
        break

    try:
        msg = json.loads(line)
        if msg.get("status") == "error":
            print("ERROR from worker:", msg.get("message"))
            proc.kill()
            stderr_out = proc.stderr.read()
            print("STDERR:", stderr_out[-2000:])
            sys.exit(1)
    except (json.JSONDecodeError, AttributeError):
        pass

    if time.time() - start > 150:
        print("TIMEOUT (150s)")
        proc.kill()
        stderr_out = proc.stderr.read()
        print("STDERR:", stderr_out[-2000:])
        sys.exit(1)
