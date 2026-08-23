#!/usr/bin/env python3
"""Run the frozen Night-15A candidates on unseen backbone seeds 3-7."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/root/autodl-fs/night15a_multimodal_contribution_and_score_stability_20260823")
GRID = REPO / "configs/night15a/candidate_grid.json"
INPUTS = {
    ("RAW_SVD_V2", "P22"): ROOT / "raw_preprocess_v2/P22/preprocessed_feature_level.npz",
    ("RAW_SVD_V2", "MISAR_E15_5_S1"): ROOT / "raw_preprocess_v2/MISAR_E15_5_S1/preprocessed_feature_level.npz",
    ("MORAN_RAW_BANK_V1", "P22"): ROOT / "raw_preprocess_moran_v1/P22/preprocessed_feature_level.npz",
    ("MORAN_RAW_BANK_V1", "MISAR_E15_5_S1"): ROOT / "raw_preprocess_moran_v1/MISAR_E15_5_S1/preprocessed_feature_level.npz",
}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def one(task: dict, output_root: Path, log_root: Path) -> dict:
    candidate = task["candidate_id"]
    bank = task["preprocess_bank"]
    dataset = task["dataset"]
    seed = int(task["seed"])
    input_path = INPUTS[(bank, dataset)]
    output = output_root / bank / candidate / dataset / f"seed_{seed}"
    log = log_root / f"{bank}__{candidate}__{dataset}__seed_{seed}.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = [
        sys.executable,
        str(REPO / "scripts/night15a/night15a_train.py"),
        "train",
        "--input", str(input_path),
        "--grid", str(GRID),
        "--candidate", candidate,
        "--seed", str(seed),
        "--output", str(output),
    ]
    with log.open("w", encoding="utf-8", newline="\n") as handle:
        completed = subprocess.run(command, cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT)
    return {
        **task,
        "input_path": str(input_path),
        "output": str(output),
        "log": str(log),
        "return_code": int(completed.returncode),
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "wall_seconds": time.perf_counter() - started,
    }


def run(freeze_path: Path, output: Path, max_parallel: int, candidates: set[str]) -> None:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    tasks = []
    for candidate in freeze["candidates"]:
        if candidates and candidate["candidate_id"] not in candidates:
            continue
        for dataset in freeze["datasets"]:
            for seed in freeze["unseen_backbone_seeds"]:
                tasks.append({
                    "candidate_id": candidate["candidate_id"],
                    "preprocess_bank": candidate["preprocess_bank"],
                    "dataset": dataset,
                    "seed": int(seed),
                })
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=int(max_parallel)) as executor:
        futures = {
            executor.submit(one, task, output / "runs", output / "logs"): task
            for task in tasks
        }
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                task = futures[future]
                rows.append({
                    **task,
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
            atomic_json(output / "unseen_training_manifest.partial.json", {"rows": rows})
    manifest = {
        "freeze_path": str(freeze_path),
        "freeze_id": freeze["freeze_id"],
        "task_count": len(tasks),
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "fail_count": sum(row["status"] == "FAIL" for row in rows),
        "max_parallel": int(max_parallel),
        "labels_in_training": False,
        "wall_seconds": time.perf_counter() - started,
        "rows": sorted(rows, key=lambda row: (row["candidate_id"], row["dataset"], row["seed"])),
    }
    atomic_json(output / "unseen_training_manifest.json", manifest)
    if manifest["fail_count"]:
        raise RuntimeError("one or more frozen unseen-seed trainings failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--candidate", action="append", default=[])
    args = parser.parse_args()
    run(Path(args.freeze), Path(args.output), args.max_parallel, set(args.candidate))


if __name__ == "__main__":
    main()
