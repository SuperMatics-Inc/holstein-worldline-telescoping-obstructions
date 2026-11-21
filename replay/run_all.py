#!/usr/bin/env python3
"""Standard-library-only driver for the Barriers IV reproducibility supplement."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time


CHECKS: list[tuple[str, list[list[str]]]] = [
    ("replay-charged-skeleton.py", [[], ["--mutants"]]),
    ("replay-telescope.py", [[], ["--mutants"]]),
    ("replay-annealed-spectral.py", [[], ["--mutants"]]),
    ("replay-annealed-endpoint.py", [[], ["--mutants"], ["--tables"]]),
    ("replay-neutral-dipole.py", [[], ["--mutants"]]),
]


def main() -> int:
    supp_dir = Path(__file__).resolve().parent
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    t_total_start = time.monotonic()
    all_ok = True
    script_results: list[tuple[str, bool, float]] = []

    for script, modes in CHECKS:
        script_ok = True
        t_script_start = time.monotonic()
        for args in modes:
            label = " ".join([script, *args])
            cmd = [sys.executable, "-B", str(supp_dir / script), *args]
            t0 = time.monotonic()
            proc = subprocess.run(
                cmd,
                cwd=supp_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            dt = time.monotonic() - t0
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            tail = lines[-1] if lines else "(no output)"
            if proc.returncode == 0:
                print(f"PASS  {label}  rc=0  {dt:.2f}s  | {tail}", flush=True)
            else:
                script_ok = False
                all_ok = False
                print(f"FAIL  {label}  rc={proc.returncode}  {dt:.2f}s  | {tail}", flush=True)
                for extra in lines[-10:]:
                    print(f"      {extra}", flush=True)
        script_dt = time.monotonic() - t_script_start
        status = "PASS" if script_ok else "FAIL"
        print(f"SCRIPT {status}  {script}  ({script_dt:.2f}s)", flush=True)
        script_results.append((script, script_ok, script_dt))

    total_dt = time.monotonic() - t_total_start
    passed = sum(1 for _, ok, _ in script_results if ok)
    print(f"SUMMARY: {passed}/{len(script_results)} scripts passed in {total_dt:.2f}s", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
