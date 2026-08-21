#!/usr/bin/env python3
"""Independent fail-closed audit for the frozen Night-11A delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--formal-index", required=True)
    p.add_argument("--smoke-index", required=True); p.add_argument("--freeze", required=True)
    p.add_argument("--decision", required=True); p.add_argument("--output", required=True)
    a = p.parse_args()
    fi = json.loads(Path(a.formal_index).read_text()); si = json.loads(Path(a.smoke_index).read_text())
    freeze = json.loads(Path(a.freeze).read_text()); decision = json.loads(Path(a.decision).read_text())
    expected = {(u, c, s, d, arm) for u in ["u000", "u020"]
                for c in ["CLEAN_HOLDOUT", "LOCAL_TARGET_DAMAGE", "LOCAL_AUXILIARY_CONFLICT"]
                for s in [17, 29, 43]
                for d in ["MODALITY2_TO_MODALITY1", "MODALITY1_TO_MODALITY2"]
                for arm in ["B0_SELF_ONLY", "B1_ALWAYS_TRANSFER", "B2_UNCERTAINTY_ONLY", "B3_SELECTIVE_NULL"]}
    actual = []; manifest_hashes = []; checkpoint_hashes = []
    all_status = True; all_firewall = True; corruption_ok = True
    for item in fi["cases"]:
        mp = Path(item["manifest"]); all_status &= mp.is_file() and sha(mp) == item["manifest_sha256"]
        m = json.loads(mp.read_text()); manifest_hashes.append({"path": str(mp), "sha256": sha(mp)})
        cp = Path(m["checkpoint"]); all_status &= cp.is_file() and sha(cp) == m["checkpoint_sha256"]
        checkpoint_hashes.append({"path": str(cp), "sha256": sha(cp)})
        all_firewall &= all(v == 0 for v in m["forbidden_counters"].values())
        for d, c in m["corruption"].items():
            corruption_ok &= c["permutation_fixed_points"] == 0 and c["patch_size"] == (697 if m["unit_id"] == "u000" else 1840)
        for r in m["rows"]:
            actual.append((r["unit_id"], r["condition"], r["replicate_seed"], r["direction"], r["arm"]))
            all_status &= r["status"] == "PASS" and r["metrics"]["exact_null_pass"]
    counts = Counter(actual)
    formal_matrix = set(actual) == expected and len(actual) == 144 and all(v == 1 for v in counts.values())
    smoke_roundtrip = 0
    for item in si["cases"]:
        m = json.loads(Path(item["manifest"]).read_text())
        if m.get("fresh_process_roundtrip", {}).get("status") == "PASS": smoke_roundtrip += 1
        all_firewall &= all(v == 0 for v in m["forbidden_counters"].values())
    implementation_frozen = all(Path(path).is_file() and sha(path) == digest
                                for path, digest in freeze["implementation_hashes"].items())
    terminal_valid = decision["terminal"] in {
        "NIGHT11A_SELECTIVE_TRANSFER_IDENTIFIABILITY_PASS",
        "NIGHT11A_NO_IDENTIFIABLE_TRANSFER_UTILITY"}
    checks = {"formal_matrix_144_unique": formal_matrix, "all_runtime_rows_pass": all_status,
              "smoke_fresh_process_roundtrip_2_of_2": smoke_roundtrip == 2,
              "implementation_and_config_frozen": implementation_frozen,
              "sparse_patch_permutations_no_fixed_points": corruption_ok,
              "forbidden_counters_all_zero": all_firewall,
              "decision_terminal_contract_valid": terminal_valid,
              "correction_cycles_within_limit": decision["global_correction_cycles"] <= 1,
              "resource_wall_within_4h": decision["resources"]["remote_wall_seconds"] <= 14400,
              "resource_gpu_within_8192mib": decision["resources"]["peak_gpu_mib"] <= 8192}
    audit = {"schema": "spalora.night11a.independent_audit.v1",
             "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
             "formal_rows": len(actual), "formal_expected_rows": len(expected),
             "smoke_roundtrip_count": smoke_roundtrip, "decision_sha256": sha(a.decision),
             "freeze_sha256": sha(a.freeze), "formal_index_sha256": sha(a.formal_index),
             "smoke_index_sha256": sha(a.smoke_index), "manifest_hashes": manifest_hashes,
             "checkpoint_hashes": checkpoint_hashes, "forbidden_counters": decision["forbidden_counters"]}
    atomic_json(a.output, audit)
    print(json.dumps({"status": audit["status"], "audit_sha256": sha(a.output),
                      "checks": checks}, sort_keys=True))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__": raise SystemExit(main())
