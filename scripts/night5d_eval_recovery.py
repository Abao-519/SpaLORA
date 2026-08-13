#!/usr/bin/env python3
"""Deterministic, evaluation-only Night-5D recovery from the frozen metric table."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np

FROZEN_SHA = "f1140157243cbf2c3760e2fd5b5eaa9f28dfd73d246a86759e8ef285577cd11a"
PARENT = "d167fcf790096208ba0148c9ae732dffb7adc8b4"
CANDIDATES = [
    "B00_C00_FULL_IGE", "B01_C04_SHRINK25", "B10_SHRINK25_ANCHOR10",
    "B17_C09_DIFFUSE10", "C09_RNA_ANCHOR10",
]
PRIMARY = [
    ("B01_C04_SHRINK25", "B00_C00_FULL_IGE"),
    ("B10_SHRINK25_ANCHOR10", "B00_C00_FULL_IGE"),
    ("B17_C09_DIFFUSE10", "B00_C00_FULL_IGE"),
]
SECONDARY = [
    ("B10_SHRINK25_ANCHOR10", "B01_C04_SHRINK25"),
    ("B17_C09_DIFFUSE10", "C09_RNA_ANCHOR10"),
]
METRICS = ["ari", "nmi", "q", "neighbor", "moran", "geary", "boundary"]
SOURCE_FIELDS = {
    "ari": "ari", "nmi": "nmi", "q": "q",
    "neighbor": "spatial_neighbor_agreement",
    "moran": "spatial_cluster_moran_mean",
    "geary": "spatial_cluster_geary_mean",
    "boundary": "boundary_disagreement",
}
DELTA_FIELDS = {m: f"delta_{m}" for m in METRICS}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_frozen(rows: list[dict[str, str]], path: Path) -> None:
    keys = [(r["candidate_id"], int(r["seed"])) for r in rows]
    if sha256(path) != FROZEN_SHA or len(rows) != 50 or len(set(keys)) != 50:
        raise ValueError("frozen input SHA/count/primary-key mismatch")
    if sorted(set(r["candidate_id"] for r in rows)) != sorted(CANDIDATES):
        raise ValueError("frozen candidate set mismatch")
    for candidate in CANDIDATES:
        if sorted(int(r["seed"]) for r in rows if r["candidate_id"] == candidate) != list(range(10)):
            raise ValueError(f"seed set mismatch: {candidate}")


def canonicalize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result = []
    for source in rows:
        row = dict(source)
        row["boundary_disagreement_symmetric_union_diagnostic"] = row["boundary_disagreement"]
        row["boundary_disagreement"] = repr(1.0 - float(row["spatial_neighbor_agreement"]))
        result.append(row)
    return result


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError("refusing empty CSV")
    names = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def exact_signflip(deltas: Iterable[float]) -> float:
    d = np.asarray(list(deltas), dtype=float)
    observed = float(d.mean())
    count = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(d)):
        if float((d * np.asarray(signs)).mean()) >= observed - 1e-15:
            count += 1
    return count / float(2 ** len(d))


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def spatial_fail(delta: dict[str, float]) -> bool:
    return ((delta["delta_neighbor"] < -0.03 and delta["delta_moran"] < -0.03) or
            (delta["delta_geary"] > 0.03 and
             (delta["delta_neighbor"] < -0.03 or delta["delta_moran"] < -0.03)))


def validate_stable_delta_payload(payload: dict[str, object]) -> None:
    required = set(DELTA_FIELDS.values())
    if not required.issubset(payload) or any("_mean_mean" in key for key in payload):
        raise ValueError("unstable or former dynamic delta fields")


def paired(rows: list[dict[str, str]], treatment: str, reference: str) -> dict[str, np.ndarray]:
    by_key = {(r["candidate_id"], int(r["seed"])): r for r in rows}
    result = {}
    for metric in METRICS:
        field = SOURCE_FIELDS[metric]
        result[metric] = np.asarray([
            float(by_key[(treatment, seed)][field]) - float(by_key[(reference, seed)][field])
            for seed in range(10)
        ])
    return result


def contrast_record(rows: list[dict[str, str]], pair: tuple[str, str]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    arrays = paired(rows, *pair)
    record: dict[str, object] = {
        "contrast": f"{pair[0]}-{pair[1]}", "treatment": pair[0], "reference": pair[1],
    }
    record.update({DELTA_FIELDS[m]: float(arrays[m].mean()) for m in METRICS})
    record["q_wins"] = int((arrays["q"] > 0).sum())
    record["exact_signflip_p_q"] = exact_signflip(arrays["q"])
    validate_stable_delta_payload(record)
    return record, arrays


def directory_manifest(root: Path) -> tuple[list[dict[str, object]], str]:
    entries = [{"path": p.relative_to(root).as_posix(), "size": p.stat().st_size, "sha256": sha256(p)}
               for p in sorted(root.rglob("*")) if p.is_file()]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return entries, hashlib.sha256(canonical).hexdigest()


def historical_replay(rows: list[dict[str, str]], history_path: Path) -> dict[str, object]:
    history = read_rows(history_path)
    identity_error = max(abs(float(r["boundary_disagreement"]) - (1.0 - float(r["spatial_neighbor_agreement"]))) for r in history)
    h = {(int(r["seed"])): r for r in history if r["dataset"] == "p22" and r["variant"] == "FULL_IGE" and int(r["seed"]) < 5}
    b = {(int(r["seed"])): r for r in rows if r["candidate_id"] == "B00_C00_FULL_IGE" and int(r["seed"]) < 5}
    fields = ["ari", "nmi", "q", "spatial_neighbor_agreement", "spatial_cluster_moran_mean", "spatial_cluster_geary_mean", "boundary_disagreement"]
    errors: dict[str, dict[str, float]] = {}
    for seed in range(5):
        hv = dict(h[seed]); hv["q"] = (float(hv["ari"]) + float(hv["nmi"])) / 2.0
        errors[str(seed)] = {field: abs(float(b[seed][field]) - float(hv[field])) for field in fields}
    maxima = {field: max(errors[str(s)][field] for s in range(5)) for field in fields}
    if identity_error > 1e-12 or max(maxima.values()) > 1e-12:
        raise ValueError("historical canonical replay mismatch")
    return {"night3b_boundary_identity_max_error": identity_error, "per_seed_abs_errors": errors,
            "max_abs_errors": maxima, "tolerance": 1e-12, "passed": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--tests-log", type=Path)
    args = ap.parse_args()
    repo = args.repo.resolve()
    frozen = repo / "outputs/night5d_handoff/p22_per_seed_metrics.csv"
    history = repo / "outputs/night3b_handoff/per_seed_metrics.csv"
    old_root = repo / "outputs/night5d_handoff"
    out = repo / "outputs/night5d_eval_recovery"
    out.mkdir(parents=True, exist_ok=True)
    source = read_rows(frozen)
    validate_frozen(source, frozen)
    old_entries, old_digest = directory_manifest(old_root)
    rows = canonicalize(source)
    canonical_fields = list(rows[0])
    write_csv(out / "canonical_p22_per_seed_metrics.csv", rows, canonical_fields)
    replay = historical_replay(rows, history)
    dump_json(out / "baseline_replay_recovery_audit.json", replay)
    p0 = {
        "status": "P0_RECOVERY_PASSED", "authority_parent_commit": PARENT,
        "authority_parent_tag": "night5d-final-20260814", "frozen_input_sha256": sha256(frozen),
        "row_count": len(rows), "candidate_ids": sorted(CANDIDATES), "seeds": list(range(10)),
        "unique_primary_keys": len({(r["candidate_id"], r["seed"]) for r in rows}),
        "training_units": 0, "diffusion_transforms": 0, "gpu_devices": 0,
        "p22_semantic_label_reads": 0, "old_output_file_count": len(old_entries),
        "old_output_manifest_sha256_at_start": old_digest,
        "canonical_boundary_definition": "1.0-spatial_neighbor_agreement",
    }
    dump_json(out / "p0_recovery_contract.json", p0)

    records: dict[str, list[dict[str, object]]] = {"primary": [], "secondary": []}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for family, pairs in (("primary", PRIMARY), ("secondary", SECONDARY)):
        for pair in pairs:
            record, values = contrast_record(rows, pair)
            records[family].append(record); arrays[str(record["contrast"])] = values
        adjusted = holm([float(r["exact_signflip_p_q"]) for r in records[family]])
        for r, adj in zip(records[family], adjusted): r["holm_adjusted_p_q"] = adj

    rng = np.random.default_rng(20260814)
    boot: dict[str, object] = {"replicates": 100000, "seed": 20260814, "call_order": [], "contrasts": {}}
    for family in ("primary", "secondary"):
        for r in records[family]:
            name = str(r["contrast"]); boot["contrasts"][name] = {}
            for metric in METRICS:
                d = arrays[name][metric]
                indices = rng.integers(0, len(d), size=(100000, len(d)))
                samples = d[indices].mean(axis=1)
                lo, hi = np.percentile(samples, [2.5, 97.5])
                boot["contrasts"][name][metric] = {"low": float(lo), "high": float(hi)}
                boot["call_order"].append(f"{name}:{metric}")
            r["bootstrap_q_ci_low"] = boot["contrasts"][name]["q"]["low"]
            r["bootstrap_q_ci_high"] = boot["contrasts"][name]["q"]["high"]

    spatial = []
    for family in ("primary", "secondary"):
        for r in records[family]:
            failed = spatial_fail(r)  # type: ignore[arg-type]
            spatial.append({"contrast": r["contrast"], "family": family, "delta_neighbor": r["delta_neighbor"],
                            "delta_moran": r["delta_moran"], "delta_geary": r["delta_geary"],
                            "spatial_protection_failed": failed, "spatial_protection_passed": not failed})
    for r in records["primary"]:
        r["material_gain_passed"] = bool(
            r["delta_ari"] > 0 and r["delta_nmi"] > 0 and r["delta_q"] >= 0.01 and r["q_wins"] >= 7
            and r["holm_adjusted_p_q"] < 0.05 and r["bootstrap_q_ci_low"] > 0)
        r["spatial_protection_passed"] = not next(x["spatial_protection_failed"] for x in spatial if x["contrast"] == r["contrast"])
        r["confirmed"] = bool(r["material_gain_passed"] and r["spatial_protection_passed"])
        if r["confirmed"]: r["interpretation"] = "confirmed"
        elif r["delta_q"] > 0 and not r["spatial_protection_passed"]: r["interpretation"] = "positive_mean_inconclusive_spatial_fail"
        elif r["delta_q"] > 0: r["interpretation"] = "positive_mean_but_statistically_inconclusive"
        else: r["interpretation"] = "no_positive_mean_gain"

    reference = {
        "B01_C04_SHRINK25-B00_C00_FULL_IGE": (0.0247043951, 0.0024338477, 0.0135691214, 8, 0.1328125, 0.3662109375, -0.0097155922, 0.0317985888, 0.0002453512, -0.0199816668, 0.0209354388),
        "B10_SHRINK25_ANCHOR10-B00_C00_FULL_IGE": (0.0236542313, -0.0066990058, 0.0084776128, 6, 0.2294921875, 0.3662109375, -0.0112893007, 0.0287919856, -0.0069058558, -0.0235122216, 0.0243409207),
        "B17_C09_DIFFUSE10-B00_C00_FULL_IGE": (0.0274536974, -0.0071950383, 0.0101293295, 7, 0.1220703125, 0.3662109375, -0.0051123793, 0.0245342868, -0.0103183449, -0.0331033199, 0.0336115022),
    }
    reference_errors = {}
    for r in records["primary"]:
        expected = reference[str(r["contrast"])]
        actual = (r["delta_ari"], r["delta_nmi"], r["delta_q"], r["q_wins"], r["exact_signflip_p_q"],
                  r["holm_adjusted_p_q"], r["bootstrap_q_ci_low"], r["bootstrap_q_ci_high"],
                  r["delta_neighbor"], r["delta_moran"], r["delta_geary"])
        errors = [abs(float(a) - float(e)) for a, e in zip(actual, expected)]
        reference_errors[str(r["contrast"])] = errors
        if max(errors) > 5e-10:
            raise ValueError(f"independent planner reference mismatch: {r['contrast']}")

    write_csv(out / "primary_contrasts.csv", records["primary"])
    write_csv(out / "secondary_contrasts.csv", records["secondary"])
    dump_json(out / "exact_signflip_and_holm.json", {
        "alternative": "gain", "permutations": 1024,
        "primary": [{"contrast": r["contrast"], "exact_p": r["exact_signflip_p_q"], "holm_p": r["holm_adjusted_p_q"]} for r in records["primary"]],
        "secondary": [{"contrast": r["contrast"], "exact_p": r["exact_signflip_p_q"], "holm_p": r["holm_adjusted_p_q"]} for r in records["secondary"]],
    })
    dump_json(out / "bootstrap_100000.json", boot)
    dump_json(out / "spatial_protection_decisions.json", {"rule_1": "neighbor<-0.03 and moran<-0.03", "rule_2": "geary>0.03 and (neighbor<-0.03 or moran<-0.03)", "decisions": spatial})
    any_confirmed = any(bool(r["confirmed"]) for r in records["primary"])
    any_positive = any(float(r["delta_q"]) > 0 for r in records["primary"])
    terminal = "P22_CONFIRMATION_SUCCESS" if any_confirmed else ("P22_PARTIAL_OR_MIXED_EVIDENCE" if any_positive else "P22_NO_LOCKED_CANDIDATE_CONFIRMED")
    decision = {"terminal_status": terminal, "confirmed_primary_contrasts": [r["contrast"] for r in records["primary"] if r["confirmed"]],
                "primary_decisions": records["primary"], "training_units": 0, "diffusion_transforms": 0,
                "gpu_devices": 0, "new_p22_label_reads": 0,
                "scope_note": "P22 is cross-dataset confirmation, not a pristine external holdout."}
    dump_json(out / "p22_evaluation_recovery_decision.json", decision)
    dump_json(out / "night5d_tag_force_update_correction.json", {
        "classification": "git_process_deviation_not_scientific_data_deviation",
        "chains": [
            {"from": "28dd6f0e28a2b1d22a9e6facbeeddae54e8d9125", "to": "851edab74027a2ce2607767aafae97576afb1c17", "evidence": "previously_recorded"},
            {"from": "851edab74027a2ce2607767aafae97576afb1c17", "to": "d167fcf790096208ba0148c9ae732dffb7adc8b4", "evidence": "user_reported_and_final_remote_peel_verified"},
        ], "night5c_and_earlier_history_rewritten": False, "scientific_results_changed": False,
        "old_deviation_record_overwritten": False,
    })
    tests_text = args.tests_log.read_text(encoding="utf-8", errors="replace") if args.tests_log and args.tests_log.exists() else "tests pending"
    old_entries_end, old_digest_end = directory_manifest(old_root)
    dump_json(out / "tests_and_invariance_audit.json", {
        "tests_log": tests_text, "recovery_required_tests": 11, "old_output_file_count_start": len(old_entries),
        "old_output_file_count_end": len(old_entries_end), "old_output_manifest_sha256_start": old_digest,
        "old_output_manifest_sha256_end": old_digest_end, "old_outputs_byte_identical": old_digest == old_digest_end,
        "planner_reference_tolerance": 5e-10, "planner_reference_abs_errors": reference_errors,
        "planner_reference_recalculation_passed": True,
        "training_units": 0, "diffusion_transforms": 0, "gpu_devices": 0,
    })
    report = ["# Night-5D evaluation-only recovery report", "", f"Terminal status: `{terminal}`", "",
              "This recovery consumed only the frozen 50-row metric table. It performed 0 training units, 0 diffusion transforms, used 0 GPUs, and did not re-read P22 labels or run artifacts.", "",
              "Canonical boundary is deterministically `1 - spatial_neighbor_agreement`; it is not independent evidence. The frozen symmetric-union boundary is retained only as a diagnostic and never enters inference or protection gates.", "", "## Primary locked contrasts", ""]
    for r in records["primary"]:
        report.append(f"- {r['contrast']}: ΔARI={r['delta_ari']:.10f}, ΔNMI={r['delta_nmi']:.10f}, ΔQ={r['delta_q']:.10f}, wins={r['q_wins']}/10, exact p={r['exact_signflip_p_q']:.10f}, Holm p={r['holm_adjusted_p_q']:.10f}, Q CI=[{r['bootstrap_q_ci_low']:.10f}, {r['bootstrap_q_ci_high']:.10f}], material={r['material_gain_passed']}, spatial={r['spatial_protection_passed']}, interpretation={r['interpretation']}.")
    report += ["", "Positive means are not described as significant gains. B01 and B10 are directionally positive but statistically inconclusive; B17 is directionally positive, statistically inconclusive, and fails the preregistered spatial protection gate.", "", "## Secondary mechanisms", ""]
    for r in records["secondary"]:
        report.append(f"- {r['contrast']}: ΔQ={r['delta_q']:.10f}, Holm p={r['holm_adjusted_p_q']:.10f}, Δneighbor={r['delta_neighbor']:.10f}, ΔMoran={r['delta_moran']:.10f}, ΔGeary={r['delta_geary']:.10f}.")
    report += ["", "P22 participated in earlier Night-3B architecture analysis, so this is not a pristine external test. Night-5 candidate ranking used A1/Placenta only; methods must not be altered after this evaluation.", "", "Original `outputs/night5d_handoff` remains byte-identical and retains its valid `BASELINE_REPLAY_MISMATCH` record for the first attempt.", ""]
    (out / "night5d_evaluation_recovery_report.md").write_text("\n".join(report), encoding="utf-8")
    files = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "delivery_index.json":
            files.append({"path": path.name, "size": path.stat().st_size, "sha256": sha256(path)})
    dump_json(out / "delivery_index.json", {"schema": "non_self_referential_v1", "excluded": ["delivery_index.json"], "files": files})
    print(json.dumps({"terminal_status": terminal, "output_root": str(out), "files": len(files) + 1}, sort_keys=True))


if __name__ == "__main__":
    main()
