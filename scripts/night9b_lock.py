#!/usr/bin/env python3
"""Create the Night-9B pre-label total lock after P2 and R1."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/night9b"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n"); os.replace(tmp, path)


def main() -> None:
    p1 = json.loads((OUT / "p1_semantic_contract.json").read_text())
    cosmos = json.loads((OUT / "p2_cosmos_lock_manifest.json").read_text())
    r1 = json.loads((OUT / "r1_lock_manifest.json").read_text())
    if p1["status"] != "PASS": raise RuntimeError("P1 is not PASS")
    if cosmos["successful_training_units"] != 5 or cosmos["endpoint_outputs_successful"] != 10:
        raise RuntimeError("COSMOS 5/5 and endpoints 10/10 required")
    if r1["planned_units"] != 42 or r1["success_units"] != 42 or r1["failure_units"]:
        raise RuntimeError("R1 42/42 required")
    raw_rows = []
    for row in r1["rows"]:
        out = Path(row["output_dir"]); manifest = json.loads((out / "training_manifest.json").read_text())
        reload = json.loads((out / "reload_audit.json").read_text())
        if reload["status"] != "PASS" or not reload["h05_partition_exact"] or not all(v["exact"] for v in reload["views"].values()):
            raise RuntimeError(f"R1 reload contract failed: {row['unit_id']}")
        for key, item in manifest["artifacts"].items():
            path = Path(item["path"])
            if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha(path) != item["sha256"]:
                raise RuntimeError(f"artifact mismatch: {row['unit_id']}/{key}")
            raw_rows.append({"stage": "R1", "unit_id": row["unit_id"], "artifact": key,
                             "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha(path)})
    for row in cosmos["rows"]:
        manifest = json.loads((Path(row["output_dir"]) / "cosmos_manifest.json").read_text())
        for key, item in manifest["artifacts"].items():
            path = Path(item["path"])
            if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha(path) != item["sha256"]:
                raise RuntimeError(f"COSMOS artifact mismatch: seed {row['seed']}/{key}")
            raw_rows.append({"stage": "P2_COSMOS", "unit_id": f"cosmos-seed-{row['seed']}",
                             "artifact": key, "path": str(path), "size_bytes": path.stat().st_size,
                             "sha256": sha(path)})
    raw_path = OUT / "raw_artifact_manifest.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "unit_id", "artifact", "path", "size_bytes", "sha256"])
        writer.writeheader(); writer.writerows(raw_rows)
    code_paths = [REPO / p for p in ("SpaLORA/night9b_racf.py", "scripts/night9b_train.py",
                                      "scripts/night9b_orchestrate.py", "scripts/night9b_cosmos.py",
                                      "scripts/night9b_lock.py", "scripts/night9b_evaluate.py",
                                      "tests/test_night9b_racf.py",
                                      "protocols/night9b/SpaLORA_Night9B_RACF_Registry_2026-08-20.json")]
    if any(not p.is_file() for p in code_paths): raise RuntimeError("prelabel source set incomplete")
    atomic_json(OUT / "prelabel_total_lock.json", {
        "schema_version": 1, "status": "LOCKED_PRE_LABEL", "P1": "PASS",
        "R1_training": "42/42", "R1_checkpoint_reload": "42/42",
        "R1_six_views_and_H05_exact": "42/42", "COSMOS_training": "5/5",
        "COSMOS_endpoint_outputs": "10/10", "scientific_retry": 0, "fallback": 0,
        "raw_artifact_rows": len(raw_rows), "raw_artifact_manifest_sha256": sha(raw_path),
        "source_sha256": {str(p.relative_to(REPO)): sha(p) for p in code_paths},
        "label_access": {"A1": 0, "P22": 0, "MISAR_Y": 0},
        "evaluator_locked_before_label_access": True,
        "return_to_model_structure_or_evaluator_change_after_lock": False,
    })
    print(json.dumps({"status": "LOCKED_PRE_LABEL", "raw_rows": len(raw_rows)}, sort_keys=True))


if __name__ == "__main__": main()
