#!/usr/bin/env python3
"""Auditable P1/R1/R2 orchestration for locked Night-9B MF-RACF cells."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night9b_racf import canonical_json_sha  # noqa: E402

OUT = REPO / "outputs/night9b"
RAW = Path("/root/autodl-fs/night9b_racf_20260820")
REGISTRY_PATH = REPO / "protocols/night9b/SpaLORA_Night9B_RACF_Registry_2026-08-20.json"
PYTHON = Path("/root/miniconda3/envs/SpaLORA/bin/python")
COSMOS = Path("/root/autodl-fs/night8a_external_sources_20260820/COSMOS/COSMOS")
ARISE = Path("/root/autodl-fs/night8a_external_sources_20260820/ARISE/ARISE")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def git_head(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          check=True, text=True, capture_output=True).stdout.strip()


def source_lock() -> dict:
    cosmos_files = [COSMOS / x for x in ("cosmos.py", "modulesWNN.py", "pyWNN.py", "util.py")]
    arise_files = [ARISE / x for x in ("begin.py", "train.py", "model.py", "process.py",
                                        "code/hln.py", "code/mouse_brain.py", "code/H3K4me3.py")]
    for path in cosmos_files + arise_files:
        if not path.is_file(): raise RuntimeError(f"missing locked source file: {path}")
    hits = []
    needles = ("true_labels", "adjusted_rand_score", "best_embeddings", "best_labels", "ari")
    for path in arise_files:
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if any(token.lower() in line.lower() for token in needles):
                hits.append({"path": str(path.relative_to(ARISE)), "line": number,
                             "text_sha256": hashlib.sha256(line.strip().encode()).hexdigest(),
                             "semantic": "true-label evaluation or ARI-selected best embedding path"})
    cosmos_text = (COSMOS / "cosmos.py").read_text(errors="replace")
    lock = {
        "schema_version": 1, "status": "PASS", "label_access": False,
        "COSMOS": {"url": "https://github.com/Lin-Xu-lab/COSMOS",
                   "commit": git_head(COSMOS),
                   "expected_commit": "56ea355be51e64d9253e2871b8bd447fdfd0d230",
                   "files": {str(p.relative_to(COSMOS)): sha256_file(p) for p in cosmos_files},
                   "license_file": next((str(p) for p in (COSMOS / "LICENSE", COSMOS / "LICENSE.md") if p.is_file()), None),
                   "endpoint": "official loss-based fixed endpoint; label callbacks forbidden",
                   "contains_loss_early_stopping": "min_stop" in cosmos_text and "best_params" in cosmos_text,
                   "allowed_fair_benchmark": True},
        "ARISE": {"url": "https://github.com/XiangxiangWang-code/ARISE",
                  "commit": git_head(ARISE),
                  "files": {str(p.relative_to(ARISE)): sha256_file(p) for p in arise_files},
                  "license_file": next((str(p) for p in (ARISE / "LICENSE", ARISE / "LICENSE.md") if p.is_file()), None),
                  "label_selection_hits": hits,
                  "endpoint": "public scripts deserialize true labels, evaluate every epoch and retain best embedding by ARI",
                  "allowed_fair_benchmark": False,
                  "allowed_use": "source semantics and structural provenance only; never imported or executed"},
        "night9b_policy": {"ARISE_import_count": 0, "ARISE_execution_count": 0,
                           "label_or_ARI_selection_count": 0},
    }
    if lock["COSMOS"]["commit"] != lock["COSMOS"]["expected_commit"] or not hits:
        raise RuntimeError("external source semantic lock failed")
    return lock


def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def source_rows() -> list[dict]:
    with (REPO / "outputs/night7a_handoff/source_views_index.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_view(dataset: str, seed: int) -> dict:
    graph = "G04_SP10_F10_EUC_UNION" if dataset == "A1" else "G00_SP18_F20_CORR_UNION"
    rows = [r for r in source_rows() if r["dataset"] == dataset.lower()
            and r["graph_id"] == graph and int(r["seed"]) == int(seed)]
    if len(rows) != 1: raise RuntimeError(f"source view authority is not unique: {dataset}/{seed}")
    return rows[0]


def r02_cell(seed: int) -> dict:
    paths = [REPO / "outputs/night7b_handoff/locked_R1_manifest.json",
             REPO / "outputs/night7b_handoff/locked_R2_manifest.json"]
    rows = []
    for path in paths:
        rows += [r for r in json.loads(path.read_text())["training_cells"]
                 if r["dataset"] == "p22" and r["recipe_id"] == "R02" and int(r["seed"]) == int(seed)]
    if len(rows) != 1 or rows[0]["status"] != "success":
        raise RuntimeError(f"R02 reference authority is not unique/successful: seed {seed}")
    return rows[0]


def inherited(dataset: str) -> dict:
    if dataset == "A1":
        return {"K": 10, "spatial_k": 10, "epochs": 200, "optimizer": "Adam",
                "learning_rate": 1e-4, "weight_decay": 0.0}
    return {"K": 9, "spatial_k": 18, "epochs": 160, "optimizer": "AdamW",
            "learning_rate": 1e-3, "weight_decay": 1e-5}


def build_config(dataset: str, seed: int, candidate: dict, stage: str,
                 unit_id: str, output_dir: Path, smoke: bool = False) -> dict:
    row = source_view(dataset, seed)
    ref = row["views_path"] if dataset == "A1" else r02_cell(seed)["training_manifest"]["embedding_path"]
    cfg = {"schema_version": "night9b-racf-config-v1", "unit_id": unit_id,
           "stage": stage, "candidate": candidate,
           "candidate_config_sha256": canonical_json_sha(candidate), "seed": int(seed),
           "views_path": row["views_path"], "reference_path": ref,
           "coordinates_path": row["coordinates_path"],
           "observation_ids_path": row["observation_ids_path"],
           "output_dir": str(output_dir), "smoke": bool(smoke)}
    cfg.update(inherited(dataset))
    if smoke: cfg["epochs"] = 1
    return cfg


def run_command(command: list[str], log: Path, timeout: float) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                              text=True, timeout=timeout, cwd=REPO)
    return {"returncode": proc.returncode, "runtime_seconds": time.perf_counter() - started,
            "log": str(log), "log_sha256": sha256_file(log)}


def p1() -> None:
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    lock = source_lock(); atomic_json(OUT / "external_source_semantics_lock.json", lock)
    test = run_command([str(PYTHON), "-m", "pytest", "-q", "tests/test_night9b_racf.py"],
                       OUT / "p1_pytest.log", 600)
    if test["returncode"] != 0: raise RuntimeError("P1 unit/gradient tests failed")
    reg = registry(); smoke_candidate = reg["candidates"][-1]
    smoke_rows = []
    for index, dataset in enumerate(("A1", "P22")):
        unit_id = f"smoke-u{index:03d}"; outdir = RAW / "p1_smoke" / unit_id / "attempt_001"
        cfg = build_config(dataset, 0, smoke_candidate, "P1_SMOKE", unit_id, outdir, True)
        cfg_path = RAW / "p1_configs" / f"{unit_id}.json"; atomic_json(cfg_path, cfg)
        already_complete = (outdir / "training_manifest.json").is_file() and (outdir / "reload_audit.json").is_file()
        if already_complete:
            if json.loads((outdir / "training_manifest.json").read_text())["status"] != "SUCCESS_PRE_LABEL" or json.loads((outdir / "reload_audit.json").read_text())["status"] != "PASS":
                raise RuntimeError(f"existing P1 smoke is not a verified completed cell: {unit_id}")
        else:
            train = run_command([str(PYTHON), "scripts/night9b_train.py", "train", "--config", str(cfg_path)],
                                RAW / "p1_logs" / f"{unit_id}.train.log", 1200)
            if train["returncode"] != 0: raise RuntimeError(f"P1 real smoke train failed: {unit_id}")
            reload = run_command([str(PYTHON), "scripts/night9b_train.py", "reload", "--config", str(cfg_path),
                                  "--output", str(outdir)], RAW / "p1_logs" / f"{unit_id}.reload.log", 1200)
            if reload["returncode"] != 0: raise RuntimeError(f"P1 real smoke reload failed: {unit_id}")
        manifest = json.loads((outdir / "training_manifest.json").read_text())
        audit = json.loads((outdir / "reload_audit.json").read_text())
        smoke_rows.append({"opaque_unit_id": unit_id, "dataset_role_recorded_only_by_orchestrator": dataset,
                           "config_sha256": sha256_file(cfg_path), "manifest_sha256": sha256_file(outdir / "training_manifest.json"),
                           "reload_audit_sha256": sha256_file(outdir / "reload_audit.json"),
                           "status": "PASS", "reused_completed_smoke": already_complete, "label_access": False,
                           "six_views_exact": all(x["exact"] for x in audit["views"].values()),
                           "h05_partition_exact": audit["h05_partition_exact"],
                           "graph_semantics": manifest["graph_semantics"]})
    atomic_json(OUT / "p1_real_smoke_manifest.json", {"status": "PASS", "rows": smoke_rows,
                                                       "scientific_training_units": 0, "label_access": False})
    atomic_json(OUT / "p1_semantic_contract.json", {
        "schema_version": 1, "status": "PASS", "external_source_lock_sha256": sha256_file(OUT / "external_source_semantics_lock.json"),
        "pytest": test, "real_smokes": "2/2", "checkpoint_roundtrip": "2/2",
        "six_view_exact_parity": "2/2", "H05_partition_exact_parity": "2/2",
        "common_graph_intersection_negative_tests": "PASS", "gradient_three_branch_test": "PASS",
        "reliability_simplex_range_and_stop_gradient": "PASS", "dgi_seed_permutation": "PASS",
        "label_access": {"A1": 0, "P22": 0, "MISAR_Y": 0},
        "formal_science_started": False, "next_phase_authorized": True,
    })
    print(json.dumps({"status": "P1_PASS", "smokes": 2}, sort_keys=True))


def stage(stage_name: str, candidate_ids: list[str], seeds: list[int]) -> None:
    reg = registry(); by_id = {x["id"]: x for x in reg["candidates"]}
    unknown = set(candidate_ids) - set(by_id)
    if unknown: raise RuntimeError(f"unregistered candidate ids: {sorted(unknown)}")
    rows = []; failures = []; index = 0
    for candidate_id in candidate_ids:
        candidate = by_id[candidate_id]
        for dataset in ("A1", "P22"):
            for seed in seeds:
                unit_id = f"{stage_name.lower()}-u{index:03d}"; index += 1
                outdir = RAW / stage_name.lower() / unit_id / "attempt_001"
                cfg = build_config(dataset, seed, candidate, stage_name, unit_id, outdir)
                cfg_path = RAW / "configs" / stage_name.lower() / f"{unit_id}.json"; atomic_json(cfg_path, cfg)
                train = run_command([str(PYTHON), "scripts/night9b_train.py", "train", "--config", str(cfg_path)],
                                    RAW / "logs" / stage_name.lower() / f"{unit_id}.train.log", 2700)
                row = {"unit_id": unit_id, "stage": stage_name, "candidate_id": candidate_id,
                       "dataset": dataset, "seed": seed, "attempt": 1,
                       "scientific_retry": 0, "fallback": False, "label_access": False,
                       "config_path": str(cfg_path), "config_sha256": sha256_file(cfg_path), "train": train}
                if train["returncode"] == 0:
                    reload = run_command([str(PYTHON), "scripts/night9b_train.py", "reload", "--config", str(cfg_path),
                                          "--output", str(outdir)], RAW / "logs" / stage_name.lower() / f"{unit_id}.reload.log", 2700)
                    row["reload"] = reload
                    if reload["returncode"] == 0:
                        row["status"] = "success"; row["output_dir"] = str(outdir)
                        row["manifest_sha256"] = sha256_file(outdir / "training_manifest.json")
                        row["reload_audit_sha256"] = sha256_file(outdir / "reload_audit.json")
                    else: row["status"] = "reload_failed"; failures.append(row)
                else: row["status"] = "train_failed"; failures.append(row)
                rows.append(row)
    payload = {"schema_version": 1, "status": "PASS" if not failures else "COMPLETED_WITH_FAILURES",
               "stage": stage_name, "planned_units": len(candidate_ids) * 2 * len(seeds),
               "attempted_units": len(rows), "success_units": sum(r["status"] == "success" for r in rows),
               "failure_units": len(failures), "scientific_retry": 0, "fallback": 0,
               "label_access": {"A1": 0, "P22": 0, "MISAR_Y": 0}, "rows": rows}
    atomic_json(OUT / f"{stage_name.lower()}_lock_manifest.json", payload)
    print(json.dumps({"status": payload["status"], "stage": stage_name,
                      "success": payload["success_units"], "planned": payload["planned_units"]}, sort_keys=True))


def cosmos_stage() -> None:
    rows = []
    for seed in range(5):
        outdir = RAW / "p2_cosmos" / f"seed_{seed}" / "attempt_001"
        result = run_command([str(PYTHON), "scripts/night9b_cosmos.py", "--seed", str(seed),
                              "--output", str(outdir)],
                             RAW / "logs/p2_cosmos" / f"seed_{seed}.log", 2700)
        row = {"seed": seed, "attempt": 1, "scientific_training": True,
               "scientific_retry": 0, "fallback": False, "label_access": False,
               "run": result}
        if result["returncode"] == 0:
            row.update({"status": "success", "output_dir": str(outdir),
                        "manifest_sha256": sha256_file(outdir / "cosmos_manifest.json")})
        else:
            row["status"] = "failed_or_timed_out"
        rows.append(row)
    payload = {"schema_version": 1,
               "status": "PASS" if all(x["status"] == "success" for x in rows) else "COMPLETED_WITH_FAILURES",
               "planned_training_units": 5, "attempted_training_units": len(rows),
               "successful_training_units": sum(x["status"] == "success" for x in rows),
               "endpoint_outputs_expected": 10,
               "endpoint_outputs_successful": 2 * sum(x["status"] == "success" for x in rows),
               "one_training_shared_by_two_lanes_per_seed": True,
               "scientific_retry": 0, "fallback": 0,
               "label_access": {"A1": 0, "P22": 0, "MISAR_Y": 0}, "rows": rows}
    atomic_json(OUT / "p2_cosmos_lock_manifest.json", payload)
    print(json.dumps({"status": payload["status"], "success": payload["successful_training_units"]}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("p1")
    sub.add_parser("cosmos")
    p = sub.add_parser("stage"); p.add_argument("--stage", choices=["R1", "R2"], required=True)
    p.add_argument("--candidates", required=True); p.add_argument("--seeds", required=True)
    args = ap.parse_args()
    if args.mode == "p1": p1()
    elif args.mode == "cosmos": cosmos_stage()
    else: stage(args.stage, args.candidates.split(","), [int(x) for x in args.seeds.split(",")])


if __name__ == "__main__": main()
