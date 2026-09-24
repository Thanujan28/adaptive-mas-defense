"""Temporary helper: run each test file separately with a timeout."""
import glob
import subprocess
import sys
import time

files = sorted(glob.glob("tests/test_*.py"))
total_passed = 0
total_failed = 0
problems = []

for path in files:
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", path, "-q", "--no-header"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        elapsed = time.time() - start
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        print(f"{path}: rc={proc.returncode} {elapsed:.1f}s :: {tail}", flush=True)
        if proc.returncode != 0:
            problems.append((path, out[-4000:]))
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"{path}: TIMEOUT after {elapsed:.1f}s")
        problems.append((path, "TIMEOUT"))

print("\n===== PROBLEMS =====")
for path, out in problems:
    print(f"\n--- {path} ---")
    print(out)
