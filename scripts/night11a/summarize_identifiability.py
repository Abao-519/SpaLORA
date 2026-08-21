#!/usr/bin/env python3
"""Independent, label-free Night-11A formal aggregation and decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


ARMS = ["B0_SELF_ONLY", "B1_ALWAYS_TRANSFER", "B2_UNCERTAINTY_ONLY", "B3_SELECTIVE_NULL"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def metrics(score, utility):
    score = np.concatenate(score); utility = np.concatenate(utility)
    margin = max(1e-8, .01 * float(np.median(np.abs(utility))))
    keep = np.abs(utility) > margin; y = utility[keep] > 0; s = score[keep]
    pos = int(y.sum()); neg = int((~y).sum())
    if not pos or not neg: auroc = auprc = float("nan")
    else:
        ranks = rankdata(s, method="average")
        auroc = float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))
        order = np.argsort(-s, kind="mergesort"); yy = y[order].astype(float)
        auprc = float(((np.cumsum(yy) / np.arange(1, len(yy) + 1)) * yy).sum() / pos)
    rs = rankdata(score); ru = rankdata(utility)
    rho = float(np.corrcoef(rs, ru)[0, 1]) if np.std(rs) and np.std(ru) else float("nan")
    sep = float(np.median(score[utility > margin]) - np.median(score[utility < -margin])) if pos and neg else float("nan")
    return {"auroc": auroc, "auprc": auprc, "spearman": rho,
            "gate_separation": sep, "positive_events": pos, "negative_events": neg,
            "excluded_fraction": float(1 - keep.mean()), "gate_std": float(score.std()),
            "gate_bins": [float(np.mean(score <= .1)), float(np.mean((score > .1) & (score < .9))), float(np.mean(score >= .9))]}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--formal-index", required=True)
    p.add_argument("--smoke-index", required=True); p.add_argument("--output", required=True)
    p.add_argument("--remote-start-epoch", type=float, required=True); a = p.parse_args()
    fi = json.loads(Path(a.formal_index).read_text()); si = json.loads(Path(a.smoke_index).read_text())
    if fi["case_count"] != 18 or fi["row_count"] != 144: raise RuntimeError("formal matrix incomplete")
    manifests = [json.loads(Path(x["manifest"]).read_text()) for x in fi["cases"]]
    rows = [r for m in manifests for r in m["rows"]]
    families = sorted(set(r["family"] for r in rows))
    family_arrays = {f: {"gate": [], "utility": []} for f in families}
    pooled = {"gate": [], "utility": []}; mse_table = {}
    for m in manifests:
        z = np.load(m["checkpoint"])
        for direction in ["MODALITY2_TO_MODALITY1", "MODALITY1_TO_MODALITY2"]:
            gate = z[direction + "__gate"]; utility = z[direction + "__oracle_utility"]
            family_arrays[m["family"]]["gate"].append(gate); family_arrays[m["family"]]["utility"].append(utility)
            pooled["gate"].append(gate); pooled["utility"].append(utility)
        for row in m["rows"]:
            key = (row["family"], row["condition"], row["arm"])
            mse_table.setdefault(key, []).append(row["metrics"]["mse_mean"])
    ident = {f: metrics(v["gate"], v["utility"]) for f, v in family_arrays.items()}
    ident["POOLED"] = metrics(pooled["gate"], pooled["utility"])
    mse_summary = {}
    for (fam, cond, arm), values in mse_table.items():
        mse_summary.setdefault(fam, {}).setdefault(cond, {})[arm] = float(np.mean(values))
    gates = {}
    gates["event_counts"] = all(v["positive_events"] >= 100 and v["negative_events"] >= 100 for f, v in ident.items() if f != "POOLED")
    gates["pooled_auroc"] = ident["POOLED"]["auroc"] >= .75
    gates["per_family_auroc"] = all(ident[f]["auroc"] >= .70 for f in families)
    gates["pooled_spearman"] = ident["POOLED"]["spearman"] >= .35
    gates["per_family_spearman"] = all(ident[f]["spearman"] >= .25 for f in families)
    gates["per_family_gate_separation"] = all(ident[f]["gate_separation"] >= .25 for f in families)
    gates["clean_safety"] = all(mse_summary[f]["CLEAN_HOLDOUT"]["B3_SELECTIVE_NULL"] <= 1.02 * mse_summary[f]["CLEAN_HOLDOUT"]["B0_SELF_ONLY"] for f in families)
    gates["target_damage_usefulness"] = all(mse_summary[f]["LOCAL_TARGET_DAMAGE"]["B3_SELECTIVE_NULL"] <= .99 * min(mse_summary[f]["LOCAL_TARGET_DAMAGE"]["B0_SELF_ONLY"], mse_summary[f]["LOCAL_TARGET_DAMAGE"]["B1_ALWAYS_TRANSFER"]) for f in families)
    gates["auxiliary_conflict_safety"] = all(mse_summary[f]["LOCAL_AUXILIARY_CONFLICT"]["B3_SELECTIVE_NULL"] <= 1.01 * mse_summary[f]["LOCAL_AUXILIARY_CONFLICT"]["B0_SELF_ONLY"] and mse_summary[f]["LOCAL_AUXILIARY_CONFLICT"]["B3_SELECTIVE_NULL"] <= .90 * mse_summary[f]["LOCAL_AUXILIARY_CONFLICT"]["B1_ALWAYS_TRANSFER"] for f in families)
    norm_regret = {arm: [] for arm in ARMS}
    for r in rows: norm_regret[r["arm"]].append(r["metrics"]["normalized_regret_mean"])
    pooled_norm_regret = {arm: float(np.mean(x)) for arm, x in norm_regret.items()}
    gates["uncertainty_ablation"] = pooled_norm_regret["B3_SELECTIVE_NULL"] <= .90 * pooled_norm_regret["B2_UNCERTAINTY_ONLY"]
    gates["no_gate_collapse"] = all(ident[f]["gate_std"] >= .05 and max(ident[f]["gate_bins"]) <= .95 for f in families)
    smoke = [json.loads(Path(x["manifest"]).read_text()) for x in si["cases"]]
    roundtrip_count = sum(m.get("fresh_process_roundtrip", {}).get("status") == "PASS" for m in smoke)
    gates["roundtrip"] = roundtrip_count == 2
    wall = time.time() - a.remote_start_epoch; peak_gpu = max([0] + [m.get("peak_gpu_mib", 0) for m in manifests + smoke])
    gates["resource"] = wall <= 4 * 3600 and peak_gpu <= 8192
    firewall = fi["forbidden_counters"]
    gates["firewall"] = all(v == 0 for v in firewall.values())
    all_pass = all(gates.values())
    terminal = "NIGHT11A_SELECTIVE_TRANSFER_IDENTIFIABILITY_PASS" if all_pass else "NIGHT11A_NO_IDENTIFIABLE_TRANSFER_UTILITY"
    classification = "LOCAL SIGNAL" if all_pass else "SCIENTIFIC NEGATIVE"
    decision = {"schema": "spalora.night11a.decision.v1", "terminal": terminal,
                "classification": classification, "formal_rows": len(rows),
                "formal_cases": len(manifests), "identifiability": ident,
                "mse_by_family_condition_arm": mse_summary,
                "pooled_normalized_regret": pooled_norm_regret, "gates": gates,
                "roundtrip_count": roundtrip_count, "global_correction_cycles": 0,
                "resources": {"remote_wall_seconds": wall, "peak_gpu_mib": peak_gpu,
                              "peak_rss_kib": max(m["peak_rss_kib"] for m in manifests + smoke),
                              "formal_cpu_seconds_sum": sum(m["cpu_seconds"] for m in manifests)},
                "forbidden_counters": firewall,
                "claim_boundary": "Label-free mechanism identifiability only; no clustering performance or paper-readiness claim."}
    atomic_json(a.output, decision)
    print(json.dumps({"terminal": terminal, "classification": classification,
                      "decision_sha256": sha(a.output), "gates": gates}, sort_keys=True))


if __name__ == "__main__": main()
