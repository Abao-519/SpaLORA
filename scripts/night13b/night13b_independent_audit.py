#!/usr/bin/env python3
"""Independent, fail-closed audit of the frozen Night-13B handoff tables."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/night13b_handoff"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    formal = pd.read_csv(OUT / "absolute_metrics.csv")
    references = pd.read_csv(OUT / "strong_internal_reference_board.csv")
    artifacts = pd.read_csv(OUT / "historical_reference_artifact_audit.csv")
    decision = json.loads((OUT / "night13b_decision.json").read_text(encoding="utf-8"))
    source = (REPO / "SpaLORA/night13b_unified.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_literals = [value.value.lower() for value in ast.walk(tree)
                          if isinstance(value, ast.Constant) and isinstance(value.value, str)
                          and any(key in value.value.lower() for key in
                                  ["a1", "p22", "tonsil", "misar"])]
    checks = {
        "formal_rows_35": len(formal) == 35,
        "seven_datasets_five_seeds_each": formal.groupby("dataset")["seed"].nunique().eq(5).all(),
        "registered_seeds_exact": all(set(group["seed"]) == set(range(5)) for _, group in formal.groupby("dataset")),
        "all_metrics_finite": np.isfinite(formal[["absolute_ari", "absolute_nmi", "ami", "fmi", "morans_i", "gearys_c"]].to_numpy()).all(),
        "gpu_time_positive": (formal["gpu_seconds"] > 0).all(),
        "partition_seed_rows_preserved": formal.groupby("dataset")["partition_sha256"].nunique().eq(1).all(),
        "reference_board_has_required_methods": {"RNA_ONLY", "SECOND_MODALITY_ONLY", "C00_G04_COMMON_BRIDGE", "F00_R02_COMMON_BRIDGE", "N02_HIER_ONLY_COMMON_BRIDGE"} <= set(references["method"]),
        "artifact_rows_sha_recomputed": len(artifacts) == 140 and artifacts["status"].eq("PRESENT_SHA_RECOMPUTED").all(),
        "artifact_files_still_match": all(Path(row.absolute_path).is_file() and Path(row.absolute_path).stat().st_size == int(row.size) and sha(row.absolute_path) == row.sha256 for row in artifacts.itertuples()),
        "model_source_dataset_identity_literals_zero": len(forbidden_literals) == 0,
        "decision_local_signal": decision["classification"] == "LOCAL SIGNAL" and not decision["confirmed_milestone_gate"],
        "raw_immutability": decision["raw_immutability_passed"],
        "p0_2_of_2": decision["p0_passed"] == decision["p0_expected"] == 2,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    result = {"schema": "spalora.night13b.independent_audit.v1", "checks": checks,
              "passed": all(checks.values()), "failed_checks": [key for key, value in checks.items() if not value],
              "formal_sha256": sha(OUT / "absolute_metrics.csv"),
              "reference_board_sha256": sha(OUT / "strong_internal_reference_board.csv"),
              "artifact_audit_sha256": sha(OUT / "historical_reference_artifact_audit.csv")}
    path = OUT / "independent_audit.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
