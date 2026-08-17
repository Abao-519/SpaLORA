#!/usr/bin/env python3
"""Run and lock Night-6D semantic tests before any scientific training."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night3af_cache import sha256_file
from SpaLORA.night6d_pipeline import atomic_json


def main() -> None:
    out = REPO / "outputs/night6d_handoff"
    graph = json.loads((out / "graph_cache_manifest_index.json").read_text())
    data = json.loads((out / "d1_label_free_manifest.json").read_text())
    p22 = json.loads((out / "p22_cache_reuse_audit.json").read_text())
    firewall = json.loads((out / "data_role_and_label_firewall.json").read_text())
    if graph["status"] != "LOCKED" or graph["entry_count"] != 4:
        raise RuntimeError("four locked graph caches required")
    if data["status"] != "PASS" or p22["status"] != "PASS" or firewall["status"] != "PASS":
        raise RuntimeError("data/cache/firewall contract is not locked")
    if any((REPO / "outputs/night6d_handoff").glob("locked_training_manifest.json")):
        raise RuntimeError("formal training existed before P0 semantic gate")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_night6d_semantic.py"],
        cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    log = out / "tests/p0_semantic_pytest.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"P0 semantic tests failed; see {log}")
    payload = {
        "status": "P0_SEMANTIC_PASS",
        "gate_message": "P0-SEMANTIC PASS; BEGIN LOCKED 40-UNIT TRAINING",
        "pytest_exit_code": 0,
        "pytest_log": str(log),
        "pytest_log_sha256": sha256_file(log),
        "locked_graphs": "2/2",
        "locked_heads": "2/2",
        "graph_caches": "4/4",
        "toy_checkpoint_round_trip": "fresh-process PASS",
        "formal_training_started_before_gate": False,
        "label_values_deserialized": False,
        "covered": [
            "Night-6C source SHAs", "dataset contracts", "kNN lexical tie-break",
            "self-tuning sigma and kernel", "three-affinity arithmetic mean",
            "spectral fixed implementation", "original H5AD and ground-truth denial",
            "early evaluator denial", "label/metric payload rejection",
            "fresh-process checkpoint round-trip",
        ],
        "silent_fallback": False,
    }
    atomic_json(out / "p0_semantic_contract.json", payload)
    atomic_json(out / "tests_and_invariance_audit.json", {
        "status": "P0_PASS_FORMAL_RESULTS_PENDING",
        "p0_semantic": payload,
        "fixed_seeds": list(range(10)),
        "fixed_order": "dataset_graph_seed_then_dataset_graph_head_seed",
        "parameter_tuning": False,
        "seed_search": False,
        "label_checkpoint_selection": False,
        "protected_dataset_access": False,
    })
    print(json.dumps({"status": payload["status"],
                      "pytest_log_sha256": payload["pytest_log_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
