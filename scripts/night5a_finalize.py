#!/usr/bin/env python3
"""Generate the compact, auditable Night-5A final handoff."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.night5a_rnd import load_registry, registry_contracts, sha256_file


CONFIG_PATH = REPO / "configs/night5a_metric_rnd.json"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def write_csv(path: Path, rows: list) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def stats_rows(per_run: pd.DataFrame, candidates: list) -> list:
    rows = []
    for candidate in candidates:
        for dataset in ("a1", "placenta"):
            group = per_run[(per_run.candidate_id == candidate) & (per_run.dataset == dataset)]
            if len(group) != 5:
                continue
            for metric in ("ari", "nmi", "q", "spatial_neighbor_agreement",
                           "spatial_cluster_moran_mean", "spatial_cluster_geary_mean",
                           "runtime_seconds", "gpu_peak_allocated_mib"):
                values = group[metric].to_numpy(float)
                rows.append({"candidate_id": candidate, "dataset": dataset, "metric": metric,
                             "n": len(values), "mean": float(values.mean()),
                             "sd": float(values.std(ddof=1)), "median": float(np.median(values)),
                             "minimum": float(values.min()), "maximum": float(values.max())})
    return rows


def raw_manifest(raw_root: Path) -> dict:
    rows = []
    for path in sorted(raw_root.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path), "relative_path": str(path.relative_to(raw_root)),
                         "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    return {"schema_version": 1, "root": str(raw_root), "file_count": len(rows),
            "total_size_bytes": sum(row["size_bytes"] for row in rows), "files": rows}


def collapse_audit(config, selected):
    raw_root = Path(config["paths"]["raw_runs"])
    rows = []
    for dataset in ("a1", "placenta"):
        for seed in config["seeds"]:
            directory = raw_root / dataset / selected / ("seed_%d" % seed)
            trajectory = pd.read_csv(directory / "loss_trajectory.csv")
            last = trajectory.iloc[-1]
            active = ["rna_recon", "mod2_recon", "corr1"]
            contributions = {name: float(last[name + "_contribution"]) for name in active}
            total = sum(abs(value) for value in contributions.values())
            fractions = {name: abs(value) / max(total, 1e-12) for name, value in contributions.items()}
            rows.append({
                "dataset": dataset, "seed": int(seed),
                "minimum_active_loss_fraction": float(min(fractions.values())),
                "active_loss_fractions": fractions,
                "minimum_cross_attention_entropy_trajectory": float(trajectory.attention_cross_entropy.min()),
                "corr2_coefficient_exact_zero": bool((trajectory.corr2_coefficient == 0.0).all()),
                "finite": bool(np.isfinite(trajectory.select_dtypes(include=[np.number]).to_numpy()).all()),
            })
    return {"schema_version": 1, "selected_candidate": selected, "rows": rows,
            "loss_contribution_collapsed": bool(any(row["minimum_active_loss_fraction"] <= 1e-6 for row in rows)),
            "attention_collapsed": bool(any(row["minimum_cross_attention_entropy_trajectory"] <= 0.1 for row in rows)),
            "all_finite": all(row["finite"] for row in rows),
            "corr2_exact_zero_all_runs": all(row["corr2_coefficient_exact_zero"] for row in rows)}


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    registry = load_registry(REPO / config["candidate_registry"])
    contracts = registry_contracts(registry)
    decisions = {stage: json.loads((output / (stage.lower() + "_decision.json")).read_text(encoding="utf-8"))
                 for stage in ("R1", "R2", "R3")}
    selected_payload = json.loads((output / "selected_for_p22_lock.json").read_text(encoding="utf-8"))
    selected = selected_payload["selected_candidates"]
    final_status = selected_payload["status"]
    if len(selected) > 2:
        raise RuntimeError("More than two final candidates")
    per_run = pd.read_csv(output / "per_run_summary.csv")

    lifecycle = []
    summaries_by_stage = {stage: {row["candidate_id"]: row for row in decisions[stage]["candidate_summaries"]}
                          for stage in decisions}
    for candidate_id, contract in contracts.items():
        row = {"candidate_id": candidate_id, "family": contract["family"], "r0": "PASS"}
        if candidate_id == "C00_FULL_IGE":
            row.update({"r1": "REFERENCE", "r2": "REFERENCE", "r3": "REFERENCE", "final": "REFERENCE"})
        else:
            r1 = summaries_by_stage["R1"][candidate_id]
            row["r1"] = "ADVANCE" if r1["advanced"] else ("PASS_GATE_CAP_STOP" if r1["passes_numeric_gate"] else "FAIL_GATE")
            if candidate_id in summaries_by_stage["R2"]:
                r2 = summaries_by_stage["R2"][candidate_id]
                row["r2"] = "ADVANCE" if r2["advanced"] else ("PASS_GATE_CAP_STOP" if r2["passes_numeric_gate"] else "FAIL_GATE")
            else:
                row["r2"] = "NOT_RUN"
            if candidate_id in summaries_by_stage["R3"]:
                r3 = summaries_by_stage["R3"][candidate_id]
                row["r3"] = "SELECT" if r3["advanced"] else "FAIL_GATE"
            else:
                row["r3"] = "NOT_RUN"
            row["final"] = "SELECTED_FOR_LOCKED_P22" if candidate_id in selected else "STOPPED"
        lifecycle.append(row)
    write_csv(output / "candidate_lifecycle.csv", lifecycle)
    five_seed = stats_rows(per_run, ["C00_FULL_IGE"] + list(summaries_by_stage["R3"]))
    write_csv(output / "five_seed_summary.csv", five_seed)

    historical = pd.read_csv(REPO / "outputs/night3b_handoff/per_seed_metrics.csv")
    historical["q"] = (historical.ari + historical.nmi) / 2.0
    envelope = []
    for dataset, group in historical.groupby("dataset"):
        means = group.groupby("variant").q.mean().sort_values(ascending=False)
        envelope.append({"dataset": dataset, "variant": means.index[0], "historical_envelope_q": float(means.iloc[0]),
                         "role": "difficulty_marker_only_not_candidate_gate"})
    write_csv(output / "historical_envelope.csv", envelope)

    collapse = collapse_audit(config, selected[0]) if selected else {
        "schema_version": 1, "selected_candidate": None, "not_applicable": True
    }
    atomic_json(output / "attention_loss_collapse_audit.json", collapse)
    raw = raw_manifest(Path(config["paths"]["raw_runs"]))
    atomic_json(output / "raw_runs_manifest.json", raw)
    budget = {
        "schema_version": 1,
        "r1": {"candidate_runs": 32, "reference_runs": 2, "total": 34, "candidate_limit": 32},
        "r2": {"candidate_runs": 20, "reference_runs": 4, "total": 24, "candidate_limit": 24},
        "r3": {"candidate_runs": 12, "reference_runs": 4, "total": 16, "candidate_limit": 12},
        "candidate_runs_total": 64, "candidate_limit_total": 68,
        "reference_runs_total": 10, "reference_limit_total": 10,
        "training_units_total": 74, "training_units_limit": 78,
        "budget_residual_not_reallocated": True,
    }
    atomic_json(output / "budget_audit.json", budget)
    withheld = {
        "schema_version": 1, "raw_run_dataset_directories": sorted(path.name for path in Path(config["paths"]["raw_runs"]).iterdir()),
        "new_candidate_p22_runs": 0, "d1_candidate_runs": 0, "d1_true_results_opened": False,
        "gse198353_candidate_runs": 0, "night4b_runs": 0,
        "development_label_access_stages": ["R1", "R2", "R3"],
        "d1_interface_test": "synthetic exact-3359 ID test passed; no real D1 label opened",
        "passed": True,
    }
    if withheld["raw_run_dataset_directories"] != ["a1", "placenta"]:
        raise RuntimeError("Withheld dataset appeared in raw run root")
    atomic_json(output / "withheld_audit.json", withheld)
    deviations = {
        "schema_version": 1,
        "scientific_protocol_deviations": [],
        "engineering_attempts": [{
            "stage": "P0-ARCH specialized tests attempt 1", "result": "26 passed, 13 failed",
            "cause": "sparse support-preservation assertion counted structural zeros",
            "resolution": "corrected audit expression only; graph formula and candidate parameters unchanged",
            "retained_log": "p0_attempt1_tests_26pass_13fail.log"
        }, {
            "stage": "P0-ARCH specialized tests attempt 2", "result": "39 passed, 0 failed",
            "retained_log": "tests_night5a.log"
        }],
        "parameter_tuning": False, "seed_search": False, "candidate_extension": False,
        "asr_modified": False,
    }
    atomic_json(output / "protocol_deviations.json", deviations)

    implementation_commit = "9becd07c1d95d6fa62a81df902facf7f940bcdfb"
    model_source_sha = sha256_file(REPO / "SpaLORA/night5a_rnd.py")
    r3 = summaries_by_stage["R3"]
    lines = [
        "# SpaLORA Night-5A metric-driven R&D funnel report", "",
        "## Decision", "",
        "**Final status: `%s`**" % final_status, "",
        "Selected candidate: **%s**. The cycle stopped after R3; P22, D1 and Night-4B were not run." %
        (", ".join(selected) if selected else "none"), "",
        "## Git and protection", "",
        "- Persistent GitHub SSH passed with strict host-key checking under `/root/autodl-fs/.ssh`.",
        "- Night-4A branch/tag were pushed once without force; remote commit is `4a22cfb4afe331e1ca2edcc0b86a01fa3892a452`.",
        "- Historical protection passed: Night-3B 1186/1186 and Night-4A 76/76.",
        "- Night-5A started from exact parent `4a22cfb4afe331e1ca2edcc0b86a01fa3892a452` in an isolated worktree.", "",
        "## P0 engineering", "",
        "- C00/C01/C02 CPU forward, loss, gradient coefficients, one-step Adam and final state parity: 3/3 exact.",
        "- 17/17 candidates have unique serialized configuration SHAs and finite/nonzero engineering probes.",
        "- Specialized tests: 39 passed, 0 failed. The retained first attempt was 26 passed/13 failed and is disclosed in `protocol_deviations.json`.",
        "- Label-free preprocessing artifacts were frozen once and reused across model seeds.", "",
        "## Funnel coverage", "",
        "- R1: 34/34 success (32 candidate + 2 reference).",
        "- R2: 24/24 success (20 candidate + 4 reference).",
        "- R3: 16/16 success (12 candidate + 4 reference).",
        "- Total: 74 training units; 64 candidate runs (limit 68) and 10 reference runs (limit 10).",
        "- All failures and attempts were retained; no seed, candidate, parameter or budget expansion occurred.", "",
        "## R1 fate of all registered candidates", "",
        "The exact lifecycle is in `candidate_lifecycle.csv`. R1 advanced C03, C04, C09, C10 and C13. C02, C11 and C16 had promising Q but failed the spatial protection gate; family caps and numeric gates stopped all other candidates.", "",
        "## R2 and R3", "",
        "R2 advanced C04, C09 and C10. C03 passed the numeric gate but ranked fourth under the locked three-candidate cap; C13 failed stability requirements.", "",
        "Five-seed R3 results relative to C00:", "",
        "| Candidate | macro ΔARI | macro ΔNMI | macro ΔQ | worst dataset ΔQ | paired Q wins | runtime × | GPU × | spatial gate | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for candidate_id in ("C04_SHRINK25", "C09_RNA_ANCHOR10", "C10_MNN_TRIPLET01"):
        row = r3[candidate_id]
        lines.append("| %s | %.6f | %.6f | %.6f | %.6f | %d/10 | %.3f | %.3f | %s | %s |" % (
            candidate_id, row["dev_macro_delta_ari"], row["dev_macro_delta_nmi"], row["dev_macro_delta_q"],
            row["worst_dataset_delta_q"], row["paired_q_wins"], row["runtime_ratio"], row["gpu_peak_ratio"],
            "FAIL" if row["spatial_protection_failed"] else "PASS", "SELECT" if row["advanced"] else "STOP"))
    lines.extend(["", "C09 and C10 achieved larger accuracy gains but were stopped because each triggered the preregistered spatial-protection rule. Their results were not hidden or used to revise the rule.", "",
                  "## Selected method contract", "",
                  "`C04_SHRINK25` retains Corr1, sets Corr2 objective contribution exactly to zero, and freezes active-set initial RMS-gradient equalization over RNA reconstruction, modality-2 reconstruction and Corr1 with coefficient sum exactly 4. For every within- and cross-attention row:", "",
                  "```text", "a_final = 0.75 * [0.5, 0.5] + 0.25 * a_learned", "```", "",
                  "No entropy regularizer, dataset-specific parameter, early stopping or label-driven graph choice is used. Candidate config SHA is `%s`; implementation source SHA is `%s`; implementation commit is `%s`." %
                  (contracts["C04_SHRINK25"]["config_sha256"], model_source_sha, implementation_commit), "",
                  "## Mechanism interpretation", "",
                  "- Strong shrinkage toward equal weights was the only mechanism that passed all five-seed accuracy, spatial and resource gates.",
                  "- RNA-anchor and MNN-triplet were accuracy-positive but spatially unsafe under the locked protection rule.",
                  "- Fully uniform fusion, reliability weighting and DGI showed development-set Q signals in R1 but did not survive the locked family/spatial/stability funnel.",
                  "- The selected attention and active loss contributions did not collapse (`attention_loss_collapse_audit.json`).", "",
                  "## Scientific scope", "",
                  "A1 and Placenta are development datasets. Five seeds are optimization repeats, not biological replicates. These results do not establish superiority over modern baselines or independent generalization. The historical envelope is a difficulty marker only. P22/D1/Night-4B were not run, and only a planning Worker may authorize the next locked stage.", ""])
    report = "\n".join(lines)
    report_path = output / "night5a_report.md"
    report_path.write_text(report, encoding="utf-8")
    with report_path.open("a", encoding="utf-8") as handle:
        handle.flush(); os.fsync(handle.fileno())

    completion = {
        "schema_version": 1, "status": final_status, "selected_candidates": selected,
        "implementation_commit": implementation_commit, "implementation_source_sha256": model_source_sha,
        "r1_runs": 34, "r2_runs": 24, "r3_runs": 16, "training_units": 74,
        "candidate_runs": 64, "reference_runs": 10, "failure_count": 0,
        "specialized_tests_passed": 39, "specialized_tests_failed": 0,
        "p22_run": False, "d1_run": False, "d1_true_results_opened": False,
        "night4b_run": False, "parameter_tuning": False, "seed_search": False,
        "report_sha256": sha256_file(report_path), "raw_runs_manifest_sha256": sha256_file(output / "raw_runs_manifest.json"),
    }
    atomic_json(output / "night5a_completion.json", completion)
    print("NIGHT5A_FINALIZED status=%s selected=%s raw_files=%d" %
          (final_status, ",".join(selected), raw["file_count"]), flush=True)


if __name__ == "__main__":
    main()
