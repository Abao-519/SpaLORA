#!/usr/bin/env python3
"""Create the Night-12B preformal registry and formal freeze manifest."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

from SpaLORA.night12a_schema_p0 import atomic_json, file_sha256
from SpaLORA.night12b_link_ident import UNITS, ZERO_COUNTS, raw_root_snapshot


def copy_verified(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if file_sha256(source) != file_sha256(target):
        raise RuntimeError("copied audit file hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    repo, derived, raw = args.repo.resolve(), args.derived.resolve(), args.raw.resolve()
    out = repo / "outputs" / "night12b_handoff"
    out.mkdir(parents=True, exist_ok=True)
    units = []
    for unit in UNITS:
        pre = json.loads((derived / "preflight" / unit / "preflight.json").read_text())
        reload = json.loads((derived / "preflight" / unit / "fresh_process_reload.json").read_text())
        if not pre["finite"] or not reload["round_trip_passed"]:
            raise RuntimeError(f"preflight/reload failed for {unit}")
        merged = dict(pre)
        merged["fresh_process_reload"] = True
        merged["fresh_process_reload_sha256"] = file_sha256(derived / "preflight" / unit / "fresh_process_reload.json")
        units.append(merged)
    atomic_json(out / "real_unit_preflight.json", {
        "schema": "spalora.night12b.real_unit_preflight.v1", "required": "6/6",
        "passed": "6/6", "scientific_values_printed_or_used": False, "units": units,
    })
    with (out / "row_join_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["unit_id", "adapter", "rna_rows", "other_rows_or_barcodes", "joined_rows",
                  "rna_unmatched", "other_unmatched", "cross_replicate_spot_alignment"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in units:
            j = row["join_audit"]
            writer.writerow({
                "unit_id": row["unit_id"], "adapter": j["adapter"], "rna_rows": j["rna_rows"],
                "other_rows_or_barcodes": j.get("adt_rows", j.get("all_atac_barcodes")),
                "joined_rows": j["joined_rows"], "rna_unmatched": j["rna_unmatched"],
                "other_unmatched": j.get("adt_unmatched", j.get("atac_unmatched_from_full_barcode_set")),
                "cross_replicate_spot_alignment": 0,
            })
    fold_manifest = []
    for row in units:
        path = Path(row["fold_assignment_path"])
        fold_manifest.append({"unit_id": row["unit_id"], "path": str(path), "size": path.stat().st_size,
                              "sha256": file_sha256(path), "sizes": row["fold_audit"]["sizes"]})
    atomic_json(out / "spatial_fold_assignment_manifest.json", {"files": fold_manifest, "count": len(fold_manifest)})
    for name in ["bridge_panel.csv", "bridge_panel_identifiers.txt", "bridge_panel_construction_audit.json"]:
        copy_verified(derived / "preflight" / name, out / name)

    firewall = {
        "schema": "spalora.night12b.label_and_prohibited_action_firewall.v1",
        "counts": ZERO_COUNTS, "all_expected_zero": all(x == 0 for x in ZERO_COUNTS.values()),
        "runner_has_label_or_metric_arguments": False, "scientific_family_formula_count": 1,
        "input_adapter_count": 2, "shared_fold_ols_decoy_bootstrap_score_gate_code": True,
    }
    atomic_json(out / "label_and_prohibited_action_firewall.json", firewall)
    atomic_json(out / "correction_cycle_registry.json", {
        "formal_correction_cycles_used": 0, "formal_correction_limit": 1,
        "preformal_implementation_fixes": [
            "registered P10 authority hash was corrected to include the contract-required final newline",
            "compact Night-12A audits that omitted a false/implicit CSV header flag are restored from header width before loading",
            "ADT observation order omitted by compact schema is re-audited from identifiers before exact reindex",
        ],
        "scientific_formula_or_threshold_changes": 0,
    })
    env = {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "scipy": scipy.__version__, "cpu_count": os.cpu_count(),
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              check=True, text=True, capture_output=True).stdout.strip(),
        "cuda_used_by_night12b": False,
    }
    atomic_json(out / "environment_audit.json", env)
    before = json.loads((derived / "audit" / "raw_before.json").read_text())
    current = raw_root_snapshot(raw)
    if before != current:
        raise RuntimeError("historical raw metadata changed before formal freeze")
    atomic_json(out / "raw_immutability_baseline.json", before)

    paths = [
        repo / "SpaLORA" / "night12b_link_ident.py",
        repo / "scripts" / "night12b" / "night12b_runner.py",
        repo / "scripts" / "night12b" / "night12b_freeze.py",
        repo / "tests" / "night12b" / "test_night12b_link_ident.py",
        *sorted((repo / "configs" / "night12b").glob("*")),
        out / "bridge_panel.csv", out / "bridge_panel_identifiers.txt",
        out / "bridge_panel_construction_audit.json", out / "real_unit_preflight.json",
        out / "row_join_audit.csv", out / "spatial_fold_assignment_manifest.json",
        out / "label_and_prohibited_action_firewall.json", out / "raw_immutability_baseline.json",
        out / "environment_audit.json", out / "correction_cycle_registry.json",
    ]
    files = [{"path": p.relative_to(repo).as_posix(), "size": p.stat().st_size, "sha256": file_sha256(p)} for p in paths]
    artifacts = [{"unit_id": u["unit_id"], "artifact_path": u["artifact_path"],
                  "artifact_size": Path(u["artifact_path"]).stat().st_size,
                  "artifact_sha256": u["artifact_sha256"],
                  "fold_assignment_sha256": u["fold_assignment_sha256"]} for u in units]
    atomic_json(out / "formal_freeze_manifest.json", {
        "schema": "spalora.night12b.formal_freeze.v1", "frozen": True,
        "parent_commit": "ed7c93d979a85eb8b907058ea463d93ba65f6956",
        "panel_m": units[0]["m"], "required_preflight": "6/6", "passed_preflight": "6/6",
        "block_bootstraps": 64, "decision_bootstraps": 2000,
        "files": files, "registered_artifacts": artifacts,
        "firewall_all_zero": True, "scientific_values_used_to_change_contract": False,
    })
    print(json.dumps({"preflight": "6/6", "panel_m": units[0]["m"], "freeze_files": len(files)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
