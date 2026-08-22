#!/usr/bin/env python3
"""Independent fail-closed audit for Night-11B tracked outputs."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from SpaLORA.night11b_discordance import atomic_json, file_sha

OUT = REPO / "outputs/night11b_handoff"
RAW = Path("/root/autodl-fs/night11b_rna_protein_discordance_identifiability_20260822")
PARENT = "e6e85fc378b46a2aba34636ac999bfa9c976b87f"


def finite(value):
    if isinstance(value, float): return math.isfinite(value)
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, list): return all(finite(v) for v in value)
    return True


def main():
    checks = {}
    json_files = sorted(OUT.glob("*.json"))
    parsed = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in json_files}
    checks["all_json_strict_finite"] = all(finite(v) for v in parsed.values())
    preflight = parsed["real_input_preflight.json"]
    checks["real_smoke_3_of_3"] = preflight["status"] == "3_OF_3_PASS" and len(preflight["rows"]) == 3 and all(r["fresh_reload"]["status"] == "PASS" for r in preflight["rows"])
    contract = parsed["night11b_contract.json"]
    checks["contract_frozen"] = contract["bootstrap_count"] == 32 and contract["permutation_count"] == 199 and len(contract["linked_features"]) == 29 and contract["combined_evidence"] == "sqrt(p_boot * p_space)"
    with (OUT / "evidence_by_dataset_feature.csv").open(encoding="utf-8", newline="") as handle:
        evidence = list(csv.DictReader(handle))
    checks["evidence_rows_complete"] = len(evidence) == 522 and sum(r["control"] == "REAL_RESIDUAL" for r in evidence) == 87 and sum(r["control"] != "REAL_RESIDUAL" for r in evidence) == 435
    checks["dataset_feature_grid_complete"] = all(sum(r["dataset"] == d and r["control"] == "REAL_RESIDUAL" for r in evidence) == 29 for d in ("a1", "tonsil", "d1"))
    decision = parsed["night11b_decision.json"]
    gates = decision["gates"]
    expected = "NIGHT11B_EVIDENCE_AXES_IDENTIFIABLE" if gates["controls_gate"] and gates["full_evidence_better_than_each_axis_gate"] and gates["a1_d1_reproducibility_gate"] else ("NIGHT11B_SYNTHETIC_ONLY_NO_REAL_IDENTIFIABILITY" if gates["controls_gate"] and gates["full_evidence_better_than_each_axis_gate"] else "NIGHT11B_EVIDENCE_AXES_NOT_IDENTIFIABLE")
    checks["terminal_matches_frozen_gates"] = decision["terminal_status"] == expected
    checks["constant_rank_fail_closed"] = gates["a1_d1_rank_constant"] and gates["a1_d1_spearman"] == 0.0 and gates["a1_d1_permutation_p"] == 1.0
    firewall = parsed["label_firewall_audit.json"]
    checks["all_forbidden_counters_zero"] = firewall["all_zero"] and all(v == 0 for v in firewall["counters"].values())
    checks["no_gpu"] = parsed["resource_audit.json"]["gpu_peak_attributed_mib"] == 0
    checks["correction_cycle_closed"] = parsed["formal_correction_audit.json"]["formal_correction_cycles"] == 1 and parsed["formal_correction_audit.json"]["all_three_datasets_rerun"]
    checks["cycles_preserved"] = all((RAW / ("formal_cycle_%d" % c) / d / "formal_manifest.json").is_file() for c in (0, 1) for d in ("a1", "tonsil", "d1")) and (RAW / "formal_cycle_0_summary_invalid/night11b_decision.json").is_file()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    ancestor = subprocess.call(["git", "merge-base", "--is-ancestor", PARENT, head], cwd=str(REPO)) == 0
    checks["git_parent_is_ancestor"] = ancestor
    checks["targeted_tests_pass"] = (OUT / "targeted_tests.log").is_file() and "6 passed" in (OUT / "targeted_tests.log").read_text(encoding="utf-8")
    status = "PASS" if all(checks.values()) else "FAIL"
    rows = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in {"independent_audit.json", "repo_delivery_file_index.json"}:
            rows.append({"path": str(path.relative_to(REPO)), "size": path.stat().st_size, "sha256": file_sha(path)})
    atomic_json(OUT / "independent_audit.json", {
        "schema": "spalora.night11b.independent_audit.v1", "status": status,
        "checks": checks, "terminal_status": decision["terminal_status"],
        "classification": decision["classification"], "audited_head": head,
        "files": rows,
        "source_files": [{"path": str(p.relative_to(REPO)), "size": p.stat().st_size, "sha256": file_sha(p)} for p in (
            REPO / "SpaLORA/night11b_discordance.py", REPO / "scripts/night11b/night11b_runner.py",
            REPO / "scripts/night11b/night11b_reload.py", REPO / "scripts/night11b/night11b_audit.py",
            REPO / "tests/test_night11b_discordance.py", REPO / "configs/night11b/night11b_contract.json",
        )],
    })
    if status != "PASS":
        raise SystemExit("independent audit failed: %s" % [k for k, v in checks.items() if not v])
    print(json.dumps({"event": "independent_audit_pass", "checks": len(checks)}, sort_keys=True))


if __name__ == "__main__":
    main()
