#!/usr/bin/env python3
"""Bounded, no-retry formal runner for the locked Night-8B matrix."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = Path("/root/autodl-fs/night8b_raw_runs_20260820")
BASE = RAW / "formal/base"
ADAPTER = RAW / "formal/adapter"
TRANSFORM = RAW / "formal/transforms"
LOGS = RAW / "logs/formal"
STATE = RAW / "formal_runner_state.json"
RUNNER = REPO / "scripts/night8b_train.py"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def run_cell(kind: str, seed: int, command: list[str], timeout_s: int,
             state: dict) -> bool:
    key = f"{kind}:seed_{seed}"
    if key in state["attempts"]:
        raise RuntimeError(f"no-retry runner refuses a second attempt for {key}")
    log = LOGS / f"{kind}_seed_{seed}.log"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1"}
    started = time.time(); status = "failed"; rc = None
    try:
        with log.open("wb") as handle:
            completed = subprocess.run(command, cwd=REPO, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, timeout=timeout_s)
        rc = int(completed.returncode)
        status = "success" if rc == 0 else "failed"
    except subprocess.TimeoutExpired:
        status = "resource_censored_timeout"
    row = {"kind": kind, "seed": seed, "status": status, "returncode": rc,
           "timeout_seconds": timeout_s, "wall_seconds": time.time() - started,
           "log": str(log), "scientific_retry": False, "fallback": False}
    state["attempts"][key] = row; atomic_json(STATE, state)
    print(json.dumps(row, sort_keys=True), flush=True)
    return status == "success"


def main() -> None:
    if STATE.exists():
        raise RuntimeError("formal runner state already exists; no automatic resume/retry")
    LOGS.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "started_unix": time.time(), "attempts": {},
             "scientific_retry": 0, "fallback": 0, "label_access": False,
             "wall_limit_seconds": 21600}
    atomic_json(STATE, state)
    py = sys.executable
    for seed in range(10):
        if time.time() - state["started_unix"] >= 21600:
            state["status"] = "time_budget_exhausted"; atomic_json(STATE, state); return
        ok = run_cell("base_train", seed,
                      [py, str(RUNNER), "base", "--seed", str(seed), "--root", str(BASE)],
                      3600, state)
        if ok:
            run_cell("base_reload", seed,
                     [py, str(RUNNER), "reload-base", "--seed", str(seed), "--root", str(BASE)],
                     1200, state)
    for seed in range(10):
        base_audit = BASE / f"seed_{seed}/attempt_001/base_checkpoint_reload_audit.json"
        if not base_audit.is_file() or json.loads(base_audit.read_text()).get("status") != "PASS":
            state["attempts"][f"adapter_train:seed_{seed}"] = {
                "kind":"adapter_train","seed":seed,"status":"skipped_invalid_base",
                "scientific_retry":False,"fallback":False}
            atomic_json(STATE, state); continue
        run_cell("adapter_train", seed,
                 [py, str(RUNNER), "adapter", "--seed", str(seed),
                  "--base-root", str(BASE), "--adapter-root", str(ADAPTER)], 3600, state)
    for method in ("U00", "F00"):
        for seed in range(10):
            if method == "F00":
                audit = ADAPTER / f"formal/seed_{seed}/attempt_001/worker/reload_forward_audit.json"
                if not audit.is_file() or json.loads(audit.read_text()).get("status") != "PASS":
                    state["attempts"][f"transform_{method}:seed_{seed}"] = {
                        "kind":f"transform_{method}","seed":seed,"status":"skipped_invalid_adapter",
                        "scientific_retry":False,"fallback":False}
                    atomic_json(STATE, state); continue
            run_cell(f"transform_{method}", seed,
                     [py, str(RUNNER), "transform", "--seed", str(seed), "--method", method,
                      "--base-root", str(BASE), "--adapter-root", str(ADAPTER),
                      "--transform-root", str(TRANSFORM)], 1200, state)
    state["elapsed_seconds"] = time.time() - state["started_unix"]
    success = sum(x["status"] == "success" for x in state["attempts"].values())
    state["status"] = "complete" if success == 50 else "incomplete"
    atomic_json(STATE, state)
    print(json.dumps({"status":state["status"],"successful_processes":success,
                      "elapsed_seconds":state["elapsed_seconds"]},sort_keys=True))


if __name__ == "__main__":
    main()
