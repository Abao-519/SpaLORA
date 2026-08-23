#!/usr/bin/env python3
"""Run every frozen endpoint on the preregistered unseen backbone seeds."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def evaluate(task: dict[str, object], evaluator: Path) -> dict[str, object]:
    output = Path(str(task["output"]))
    command = [
        sys.executable,
        str(evaluator),
        "--dataset",
        str(task["dataset"]),
        "--run-dir",
        str(task["run_dir"]),
        "--config",
        str(task["config_path"]),
        "--output",
        str(output),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    return {
        **task,
        "returncode": completed.returncode,
        "wall_seconds": time.perf_counter() - started,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "log": completed.stdout,
    }


def run(
    registry_path: Path,
    training_roots: list[Path],
    output: Path,
    max_parallel: int,
    candidates: set[str],
    recover_existing: bool,
) -> None:
    if recover_existing:
        if not output.is_dir():
            raise RuntimeError("recovery output directory is missing")
    else:
        output.mkdir(parents=True, exist_ok=False)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    evaluator = Path(__file__).with_name("night15a_frozen_eval.py")
    tasks: list[dict[str, object]] = []
    for entry in registry["entries"]:
        if candidates and entry["candidate_id"] not in candidates:
            continue
        for seed in registry["unseen_backbone_seeds"]:
            relative = (
                Path("runs") / entry["preprocess_bank"] / entry["candidate_id"]
                / entry["dataset"] / f"seed_{seed}"
            )
            matches = [root / relative for root in training_roots if (root / relative).exists()]
            if len(matches) != 1:
                raise RuntimeError(f"expected one unseen training run for {relative}, found {matches}")
            run_dir = matches[0]
            reload_path = run_dir / "fresh_process_reload.json"
            if not reload_path.exists():
                raise RuntimeError(f"missing unseen fresh-process reload: {reload_path}")
            tasks.append(
                {
                    "frozen_id": entry["frozen_id"],
                    "dataset": entry["dataset"],
                    "candidate_id": entry["candidate_id"],
                    "cluster_k": entry["cluster_k"],
                    "model_seed": seed,
                    "run_dir": str(run_dir),
                    "config_path": entry["config_path"],
                    "output": str(
                        output
                        / entry["preprocess_bank"]
                        / entry["candidate_id"]
                        / entry["dataset"]
                        / f"K{entry['cluster_k']}"
                        / f"seed_{seed}"
                    ),
                }
            )
    started = time.perf_counter()
    results: list[dict[str, object]] = []
    if recover_existing:
        for task in tasks:
            task_output = Path(str(task["output"]))
            manifest_path = task_output / "frozen_endpoint_manifest.json"
            ledger_path = task_output / "frozen_endpoint_ledger.csv"
            if not manifest_path.exists() or not ledger_path.exists():
                raise RuntimeError(f"incomplete existing evaluation: {task_output}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            row_count = sum(1 for _ in ledger_path.open("r", encoding="utf-8")) - 1
            if row_count != 11 or manifest["rows"] != 11:
                raise RuntimeError(f"existing evaluation row count mismatch: {task_output}")
            if (
                manifest["dataset"] != task["dataset"]
                or manifest["candidate_id"] != task["candidate_id"]
                or int(manifest["model_seed"]) != int(task["model_seed"])
            ):
                raise RuntimeError(f"existing evaluation identity mismatch: {task_output}")
            results.append({
                **task,
                "returncode": 0,
                "wall_seconds": float(manifest["wall_seconds"]),
                "status": "PASS",
                "recovered_from_complete_atomic_unit_artifacts": True,
                "individual_manifest_path": str(manifest_path),
            })
    else:
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {executor.submit(evaluate, task, evaluator): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                log_path = output / "logs" / f"{result['frozen_id']}__seed_{result['model_seed']}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(str(result.pop("log")), encoding="utf-8")
                result["log_path"] = str(log_path)
    results.sort(key=lambda row: (str(row["frozen_id"]), int(row["model_seed"])))
    atomic_json(
        output / "unseen_evaluation_batch_manifest.json",
        {
            "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
            "task_count": len(results),
            "passed": sum(row["status"] == "PASS" for row in results),
            "failed": sum(row["status"] != "PASS" for row in results),
            "max_parallel": max_parallel,
            "recovered_from_complete_atomic_unit_artifacts": recover_existing,
            "wall_seconds": time.perf_counter() - started,
            "registry_path": str(registry_path),
            "tasks": results,
        },
    )
    if any(row["status"] != "PASS" for row in results):
        raise RuntimeError("one or more frozen unseen evaluations failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--training-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--recover-existing", action="store_true")
    args = parser.parse_args()
    run(
        Path(args.registry),
        [Path(value) for value in args.training_root],
        Path(args.output),
        args.max_parallel,
        set(args.candidate),
        args.recover_existing,
    )


if __name__ == "__main__":
    main()
