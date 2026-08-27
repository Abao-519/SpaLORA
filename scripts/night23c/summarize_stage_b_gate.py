"""Mechanically apply the frozen Night-23C learned-method gate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from SpaLORA.night23c_tristate_bridge import ARMS


FULL = "FULL_NESTED_CALIBRATED_TRISTATE_SIGNED"
CARRIER = "CARRIER_ONLY"


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output-table", required=True); p.add_argument("--output-decision", required=True); args = p.parse_args()
    summary = []
    for path in args.inputs:
        rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
        by = {row["candidate_id"]: row for row in rows}
        if set(by) != set(ARMS): raise RuntimeError(f"candidate schema mismatch: {path}")
        full, carrier = by[FULL], by[CARRIER]
        controls = [by[name] for name in ARMS if name != FULL]
        full_ari, full_nmi = float(full["ari"]), float(full["nmi"])
        best_ari = max(float(row["ari"]) for row in controls); best_nmi = max(float(row["nmi"]) for row in controls)
        pass_all = all(full_ari > float(row["ari"]) and full_nmi > float(row["nmi"]) for row in controls)
        summary.append({"lane": full["lane"], "full_ari": full_ari, "full_nmi": full_nmi,
                        "carrier_ari": float(carrier["ari"]), "carrier_nmi": float(carrier["nmi"]),
                        "delta_ari_vs_carrier": full_ari - float(carrier["ari"]), "delta_nmi_vs_carrier": full_nmi - float(carrier["nmi"]),
                        "coordinate_best_control_ari": best_ari, "coordinate_best_control_nmi": best_nmi,
                        "delta_ari_vs_coordinate_best": full_ari - best_ari, "delta_nmi_vs_coordinate_best": full_nmi - best_nmi,
                        "independent_dual_win_over_every_control": pass_all, "partition_sha256": full["partition_sha256"],
                        "min_cluster_size_full": int(full["min_cluster_size_full"])})
    passes = sum(row["independent_dual_win_over_every_control"] for row in summary)
    macro_ari = float(np.mean([row["delta_ari_vs_carrier"] for row in summary])); macro_nmi = float(np.mean([row["delta_nmi_vs_carrier"] for row in summary]))
    safety = all(row["delta_ari_vs_carrier"] >= -0.03 for row in summary if not row["independent_dual_win_over_every_control"])
    method_gate = passes >= 2 and macro_ari > 0 and macro_nmi > 0 and safety
    narrow = "CROSS_STUDY_RELATION_BRIDGE_SIGNAL" if method_gate else ("LOCAL_RELATION_BRIDGE_SIGNAL" if passes == 1 else "NO_CALIBRATED_RELATION_BRIDGE_SIGNAL")
    decision = {"schema": "night23c-stage-b-decision-v1", "primary_independent_passes": passes, "primary_total": len(summary),
                "study_balanced_delta_ari_vs_carrier": macro_ari, "study_balanced_delta_nmi_vs_carrier": macro_nmi,
                "failed_lane_safety": safety, "method_gate_passed": method_gate, "placenta_confirmation_authorized": method_gate,
                "narrow_classification": narrow, "night23_mainline_permanently_closed": passes == 0,
                "selection_rule": "FULL strictly exceeds every matched control in both ARI and NMI; then macro and failed-lane safety gates"}
    out = Path(args.output_table); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    Path(args.output_decision).write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__": main()
