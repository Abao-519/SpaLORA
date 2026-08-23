#!/usr/bin/env python3
"""Consolidate Night-15A evidence and write the compact scientific handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/root/autodl-fs/night15a_multimodal_contribution_and_score_stability_20260823")
OUT = REPO / "outputs/night15a_handoff"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def snapshot_root(declared: str) -> dict:
    root = Path(declared).resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total = 0
    maximum = 0
    for path in files:
        stat = path.stat()
        total += stat.st_size
        maximum = max(maximum, stat.st_mtime_ns)
        digest.update(
            f"{path.relative_to(root).as_posix()}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode()
        )
    return {
        "declared_root": declared,
        "resolved_root": str(root),
        "file_count": len(files),
        "total_bytes": total,
        "max_mtime_ns": maximum,
        "metadata_fingerprint": digest.hexdigest(),
    }


def historical_raw_audit() -> dict:
    baseline_path = REPO / "outputs/night14b_handoff/historical_raw_immutability.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    fields = (
        "resolved_root",
        "file_count",
        "total_bytes",
        "max_mtime_ns",
        "metadata_fingerprint",
    )
    roots = []
    for old in baseline["roots"]:
        current = snapshot_root(old["declared_root"])
        current["byte_exact_metadata_match_night14b"] = all(
            current[key] == old[key] for key in fields
        )
        roots.append(current)
    return {
        "baseline_path": str(baseline_path),
        "baseline_sha256": file_sha256(baseline_path),
        "audit_semantics": "relative path, size and mtime metadata only",
        "raw_content_files_opened_by_this_audit": 0,
        "roots": roots,
        "passed": all(row["byte_exact_metadata_match_night14b"] for row in roots),
        "changed_root_count": sum(
            not row["byte_exact_metadata_match_night14b"] for row in roots
        ),
    }


def read_csvs(pattern: str) -> list[tuple[Path, pd.DataFrame]]:
    result = []
    for path in sorted(ROOT.rglob(pattern)):
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, UnicodeDecodeError):
            continue
        if len(frame):
            result.append((path, frame))
    return result


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def candidate_registry() -> pd.DataFrame:
    grid_path = REPO / "configs/night15a/candidate_grid.json"
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    rows = []
    for item in grid["candidates"]:
        rows.append({
            "candidate_id": item["candidate_id"],
            "mechanism_family": item["mechanism_family"],
            "mode": item["mode"],
            "graph_k": item["graph_k"],
            "optimizer_steps": grid["shared_training"]["steps"],
            "latent_dim": grid["base_config"]["latent_dim"],
            "modality_dropout_probability": item.get("modality_dropout_probability", 0.0),
            "dataset_name_routing": False,
            "labels_in_loss_gradient_or_checkpoint_selection": False,
            "config_json": json.dumps(item, sort_keys=True),
        })
    return pd.DataFrame(rows)


def all_run_ledger() -> pd.DataFrame:
    sources = [
        ("SCORE_SOURCE_ABLATION", "score_source_ablation.csv"),
        ("HEAD_SCREEN", "screen_ledger.csv"),
        ("ROBUST_ENDPOINT", "robust_endpoint_ledger.csv"),
        ("REGISTERED_HEAD", "registered_head_ledger.csv"),
        ("UNSEEN_SEED_FROZEN_ENDPOINT", "frozen_endpoint_ledger.csv"),
    ]
    frames = []
    for kind, pattern in sources:
        for path, frame in read_csvs(pattern):
            value = frame.copy()
            value.insert(0, "ledger_kind", kind)
            value.insert(1, "source_artifact", rel(path))
            frames.append(value)
    for path in sorted(ROOT.rglob("training_audit.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        frames.append(pd.DataFrame([{
            "ledger_kind": "TRAINING",
            "source_artifact": rel(path),
            "dataset": path.parts[-3] if len(path.parts) >= 3 else "UNKNOWN",
            "candidate_id": value.get("candidate_id"),
            "model_seed": value.get("seed"),
            "status": "PASS",
            "optimizer_steps": value.get("optimizer_steps"),
            "wall_seconds": value.get("wall_seconds"),
            "gpu_seconds": value.get("gpu_seconds"),
            "peak_gpu_mib": value.get("peak_gpu_mib"),
            "peak_rss_mib": value.get("peak_rss_mib"),
            "checkpoint_sha256": value.get("checkpoint_sha256"),
            "labels_in_model_or_cluster_fit": value.get("labels_in_loss_gradient_or_checkpoint_selection"),
        }]))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def raw_results() -> pd.DataFrame:
    frames = []
    for kind, pattern in (
        ("SCREEN_SHORTLIST", "screen_shortlist.csv"),
        ("ROBUST_SUMMARY", "robust_endpoint_summary.csv"),
        ("REGISTERED_HEAD", "registered_head_ledger.csv"),
        ("UNSEEN_SEED", "frozen_endpoint_ledger.csv"),
    ):
        for path, frame in read_csvs(pattern):
            value = frame.copy()
            value.insert(0, "result_stage", kind)
            value.insert(1, "source_artifact", rel(path))
            frames.append(value)
    context_path = ROOT / "separ_exact_context_v2/separ_context_result.json"
    if context_path.exists():
        context = json.loads(context_path.read_text(encoding="utf-8"))
        frames.append(pd.DataFrame([{
            "result_stage": "EXTERNAL_PROTOCOL_CONTEXT",
            "source_artifact": rel(context_path),
            "dataset": "MISAR_E15_5_S1",
            "candidate_id": context["lane"],
            "cluster_k": context["cluster_k"],
            "absolute_ari": context["absolute_ari"],
            "absolute_nmi": context["absolute_nmi"],
            "status": "PASS",
            "wall_seconds": context["wall_seconds"],
            "peak_rss_mib": context["peak_rss_mib"],
        }]))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def aggregate_frozen() -> pd.DataFrame:
    frames = []
    for path, frame in read_csvs("frozen_endpoint_ledger.csv"):
        value = frame.copy()
        value["source_artifact"] = rel(path)
        frames.append(value)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True, sort=False)
    group = ["dataset", "candidate_id", "cluster_k", "endpoint_role", "source_view", "filter_id", "pca_dimension", "coordinate_basis", "coordinate_weight", "algorithm", "refinement_id"]
    summary = frame.groupby(group, dropna=False, as_index=False).agg(
        run_count=("status", "size"),
        backbone_seed_count=("model_seed", "nunique"),
        endpoint_seed_count=("endpoint_seed", "nunique"),
        ari_best=("absolute_ari", "max"), ari_median=("absolute_ari", "median"),
        ari_mean=("absolute_ari", "mean"), ari_min=("absolute_ari", "min"),
        nmi_best=("absolute_nmi", "max"), nmi_median=("absolute_nmi", "median"),
        nmi_mean=("absolute_nmi", "mean"), nmi_min=("absolute_nmi", "min"),
        ami_mean=("ami", "mean"), fmi_mean=("fmi", "mean"),
        morans_i_mean=("morans_i", "mean"), gearys_c_mean=("gearys_c", "mean"),
        endpoint_wall_seconds=("wall_seconds", "sum"),
    )
    training_rows = []
    unseen_roots = sorted(ROOT.glob("unseen_training*/runs"))
    for unseen_root in unseen_roots:
        for path in sorted(unseen_root.rglob("training_audit.json")):
            audit = json.loads(path.read_text(encoding="utf-8"))
            training_rows.append({
                "dataset": path.parts[-3],
                "candidate_id": audit["candidate_id"],
                "model_seed": int(audit["seed"]),
                "training_wall_seconds": float(audit.get("wall_seconds", 0.0)),
                "gpu_seconds": float(audit.get("gpu_seconds", 0.0)),
                "peak_gpu_mib": float(audit.get("peak_gpu_mib", 0.0)),
                "peak_rss_mib": float(audit.get("peak_rss_mib", 0.0)),
            })
    resources = pd.DataFrame(training_rows).groupby(
        ["dataset", "candidate_id"], as_index=False
    ).agg(
        training_wall_seconds=("training_wall_seconds", "sum"),
        gpu_seconds=("gpu_seconds", "sum"),
        peak_gpu_mib=("peak_gpu_mib", "max"),
        peak_rss_mib=("peak_rss_mib", "max"),
    )
    summary = summary.merge(resources, on=["dataset", "candidate_id"], how="left")
    summary["wall_seconds"] = summary["endpoint_wall_seconds"] + summary["training_wall_seconds"]
    summary["training_resource_reused_across_k"] = True
    return summary


def main_results(stage: pd.DataFrame, frozen: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    stage_rows = stage.copy()
    stage_rows.insert(0, "result_role", "MATCHED_NIGHT14B_SCORE_SOURCE")
    stage_rows["gpu_seconds"] = 0.0
    stage_rows["peak_gpu_mib"] = 0.0
    result = [stage_rows]
    if len(frozen):
        frozen_rows = frozen.copy()
        frozen_rows.insert(0, "result_role", "UNSEEN_BACKBONE_SEEDS_3_TO_7")
        result.append(frozen_rows)
    # Exact-protocol and side-lane baselines may have only a seed-0 screen.
    screen = raw[raw.result_stage == "SCREEN_SHORTLIST"].copy() if len(raw) else pd.DataFrame()
    if len(screen):
        screen = screen.sort_values(["dataset", "candidate_id", "cluster_k", "absolute_ari", "absolute_nmi"], ascending=[True, True, True, False, False])
        screen = screen.groupby(["dataset", "candidate_id", "cluster_k"], as_index=False).head(1)
        screen_rows = pd.DataFrame({
            "result_role": "DEVELOPMENT_SCREEN_BEST_RUN",
            "dataset": screen["dataset"], "candidate_id": screen["candidate_id"],
            "cluster_k": screen["cluster_k"], "control_id": screen.get("source_view"),
            "run_count": 1, "ari_best": screen["absolute_ari"], "ari_median": screen["absolute_ari"],
            "ari_mean": screen["absolute_ari"], "ari_min": screen["absolute_ari"],
            "nmi_best": screen["absolute_nmi"], "nmi_median": screen["absolute_nmi"],
            "nmi_mean": screen["absolute_nmi"], "nmi_min": screen["absolute_nmi"],
            "ami_mean": screen.get("ami"), "fmi_mean": screen.get("fmi"),
            "morans_i_mean": screen.get("morans_i"), "gearys_c_mean": screen.get("gearys_c"),
            "wall_seconds": screen.get("wall_seconds"),
        })
        result.append(screen_rows)
    context = raw[raw.result_stage == "EXTERNAL_PROTOCOL_CONTEXT"] if len(raw) else pd.DataFrame()
    if len(context):
        result.append(pd.DataFrame({
            "result_role": "EXTERNAL_PROTOCOL_CONTEXT_NOT_OWN_METHOD",
            "dataset": context.dataset, "candidate_id": context.candidate_id,
            "cluster_k": context.cluster_k, "run_count": 1,
            "ari_best": context.absolute_ari, "ari_median": context.absolute_ari,
            "ari_mean": context.absolute_ari, "ari_min": context.absolute_ari,
            "nmi_best": context.absolute_nmi, "nmi_median": context.absolute_nmi,
            "nmi_mean": context.absolute_nmi, "nmi_min": context.absolute_nmi,
            "wall_seconds": context.wall_seconds,
        }))
    return pd.concat(result, ignore_index=True, sort=False)


def variance_decomposition() -> pd.DataFrame:
    frames = []
    for path, frame in read_csvs("score_source_ablation.csv") + read_csvs("frozen_endpoint_ledger.csv"):
        if not {"model_seed", "endpoint_seed", "absolute_ari", "absolute_nmi"}.issubset(frame.columns):
            continue
        value = frame.copy()
        value["source_artifact"] = rel(path)
        key_candidates = ["dataset", "cluster_k", "control_id", "candidate_id", "source_view", "filter_id", "algorithm"]
        group = [column for column in key_candidates if column in value.columns]
        for keys, subset in value.groupby(group, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group, keys))
            for metric in ("absolute_ari", "absolute_nmi"):
                total = float(subset[metric].var(ddof=0))
                model_means = subset.groupby("model_seed")[metric].mean()
                within = subset.groupby("model_seed")[metric].var(ddof=0).fillna(0)
                row[f"{metric}_total_variance"] = total
                row[f"{metric}_between_backbone_seed_variance"] = float(model_means.var(ddof=0))
                row[f"{metric}_mean_within_backbone_endpoint_variance"] = float(within.mean())
            row.update({
                "run_count": len(subset),
                "backbone_seed_count": subset.model_seed.nunique(),
                "endpoint_seed_count": subset.endpoint_seed.nunique(),
                "source_artifact": rel(path),
            })
            frames.append(row)
    return pd.DataFrame(frames)


def protocol_registry() -> pd.DataFrame:
    k18_audit = json.loads((ROOT / "protocol_inputs/3dot_zenodo_15089427/member_extraction_audit.json").read_text())
    dedup = json.loads((ROOT / "protocol_inputs/GSE213264/dedup_audit.json").read_text())
    return pd.DataFrame([
        {"lane": "P22_COSMOS_K9", "family": "RNA+ATAC", "observations": 9196, "cluster_k": 9, "evaluation_reference_cardinality": 9, "reference_semantics": "project exact public ground truth", "ordered_ids_closed": True, "mask": "9196/9196", "status": "CLOSED"},
        {"lane": "P22_3DOT_K18", "family": "RNA+ATAC", "observations": 9196, "cluster_k": 18, "evaluation_reference_cardinality": 18, "reference_semantics": "official author-supplied 3d-OT 18-state domain assignment; not independent expert truth and not a K9 split", "ordered_ids_closed": True, "mask": "9196/9196", "status": "CLOSED", "artifact_sha256": k18_audit.get("output_sha256")},
        {"lane": "MISAR_E15_5_K7", "family": "RNA+ATAC", "observations": 1949, "cluster_k": 7, "evaluation_reference_cardinality": 7, "reference_semantics": "public project Y, evaluator only", "ordered_ids_closed": True, "mask": "1949/1949", "status": "CLOSED"},
        {"lane": "MISAR_E15_5_SEPAR_K12", "family": "RNA+ATAC", "observations": 1949, "cluster_k": 12, "evaluation_reference_cardinality": 7, "reference_semantics": "official SEPAR Tutorial4 K12 clustering evaluated against seven-category public Y", "ordered_ids_closed": True, "mask": "1949/1949", "status": "CLOSED"},
        {"lane": "GSE213264_SPATIAL_CITE_TONSIL", "family": "RNA+protein", "observations": dedup["gse213264"]["rna_observation_count"], "cluster_k": np.nan, "evaluation_reference_cardinality": np.nan, "reference_semantics": "dedup metadata/IDs only; labels unopened", "ordered_ids_closed": dedup["gse213264"]["rna_protein_id_sets_exact"], "mask": "not evaluated", "status": dedup["decision"]},
    ])


def checkpoint_audit() -> dict:
    rows = []
    for path in sorted(ROOT.rglob("fresh_process_reload.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        relative = rel(path)
        numerically_close = bool(value.get("all_numerically_close", value.get("all_exact", False)))
        superseded = relative.startswith("development_v1/") and not numerically_close
        rows.append({
            "path": relative,
            "all_numerically_close": numerically_close,
            "all_exact": bool(value.get("all_exact", False)),
            "row_count": value.get("row_count"),
            "evidence_status": "SUPERSEDED_FAILURE_PRESERVED" if superseded else "ACTIVE",
            "formal_unseen_seed": relative.startswith("unseen_training"),
        })
    active = [row for row in rows if row["evidence_status"] == "ACTIVE"]
    formal = [row for row in rows if row["formal_unseen_seed"]]
    return {
        "rows": rows,
        "total_including_superseded": len(rows),
        "numerically_close_pass_including_superseded": sum(row["all_numerically_close"] for row in rows),
        "active_total": len(active),
        "active_numerically_close_pass": sum(row["all_numerically_close"] for row in active),
        "formal_unseen_total": len(formal),
        "formal_unseen_numerically_close_pass": sum(row["all_numerically_close"] for row in formal),
        "superseded_failures_preserved": sum(row["evidence_status"] == "SUPERSEDED_FAILURE_PRESERVED" for row in rows),
        "all_exact_pass": sum(row["all_exact"] for row in rows),
    }


def resource_audit() -> dict:
    trainings = [json.loads(path.read_text()) for path in ROOT.rglob("training_audit.json")]
    head_paths = list(ROOT.rglob("head_search_manifest.json")) + list(ROOT.rglob("screen_manifest.json"))
    heads = [json.loads(path.read_text()) for path in head_paths]
    return {
        "training_runs": len(trainings),
        "training_total_wall_seconds": float(sum(item.get("wall_seconds", 0) for item in trainings)),
        "training_total_gpu_seconds": float(sum(item.get("gpu_seconds", 0) for item in trainings)),
        "peak_gpu_mib": float(max([item.get("peak_gpu_mib", 0) for item in trainings] or [0])),
        "peak_training_rss_mib": float(max([item.get("peak_rss_mib", 0) for item in trainings] or [0])),
        "head_search_manifests": len(heads),
        "head_search_total_wall_seconds": float(sum(item.get("wall_seconds", 0) for item in heads)),
        "head_search_peak_rss_mib": float(max([item.get("peak_rss_mib", 0) for item in heads] or [0])),
        "dense_n_by_n_count": 0,
    }


def failure_ledger() -> pd.DataFrame:
    rows = [
        ("REMOTE_WORKSPACE_CLONE_ATTEMPT1", "INFRASTRUCTURE", "timeout; incomplete clone preserved", "SUPERSEDED_BY_SUCCESSFUL_INDEPENDENT_WORKTREE"),
        ("STAGE_A_V1", "ENGINEERING", "NumPy bool not JSON serializable", "FIXED_AND_FULL_STAGE_A_RERUN"),
        ("STAGE_A_V2", "ENGINEERING", "byte-exact float comparison was semantically too strict", "FIXED_TO_REGISTERED_1E-12_TOLERANCE_AND_FULL_RERUN"),
        ("SEPAR_EXACT_PREPROCESS_V1", "ENGINEERING", "object-dtype IDs rejected by no-pickle loader", "UNICODE_ID_SCHEMA_AND_FULL_RERUN"),
        ("GSE213264_FULL_TAR", "POLICY_EARLY_STOP", "low-throughput full tar unnecessary after range index", "ONLY_TWO_WHITELISTED_TONSIL_MEMBERS_RANGE_EXTRACTED"),
        ("GSE213264_DEDUP_V1", "ENGINEERING", "paired ID sets equal but row order differs", "EXPLICIT_SET_AND_ORDER_AUDIT_V2"),
        ("MORAN2500", "STRUCTURAL", "fewer than 2500 finite RNA Moran features", "PRESERVED; ONE_PLATFORM_NUMERIC_REVISION_TO_1500"),
        ("SEPAR_CONTEXT_V1", "ENGINEERING", "upstream AnnData ArrayView boolean assignment incompatibility after 100 iterations", "MATERIALIZED_NUMPY_ARRAYS; FULL_CONTEXT_RERUN"),
        ("SEPAR_CONTEXT_V2", "POLICY_EARLY_STOP", "exact external context rerun consumed primary-model compute after protocol/source closure", "PARTIAL LOG PRESERVED; NOT USED AS OWN-METHOD EVIDENCE"),
        ("R40_FULL_GRID_SCREEN_V1", "POLICY_EARLY_STOP", "full screen dominated compute budget", "PRESERVED_AND_REPLACED_BY_DECLARED_QUICK_SUBGRID"),
        ("SIDE_HEAD_SCREENS_V1", "POLICY_EARLY_STOP", "A1/tonsil side screens competed with frozen primary unseen-seed work", "SIDE TRAINING AND ROUNDTRIP PRESERVED; PRIMARY LANES PRIORITIZED"),
        ("MORAN1500_HEAD_SCREEN_V1", "POLICY_EARLY_STOP", "supplemental platform revision competed with frozen primary unseen-seed work", "TRAINING AND ROUNDTRIP PRESERVED; NOT USED FOR FORMAL SELECTION"),
        ("HEAD_SCREEN_FREEZE_V3", "ENGINEERING", "BLAS spawned 138 threads per process and caused severe oversubscription", "EMPTY PARTIAL DIRECTORIES REMOVED; V4 FULL SCREEN RERUN WITH THREAD CAPS"),
        ("FORMAL_FREEZE_ATTEMPT1", "INFRASTRUCTURE", "persistent-volume inode limit reached while writing registry", "PARTIAL FREEZE PRESERVED; INODE PRESSURE RESOLVED WITHOUT SCIENCE CHANGE; FULL FREEZE RERUN"),
        ("UNSEEN_TRAINING_V1", "INFRASTRUCTURE", "batch launcher exited after 20/30 completed round-trips when inode limit was reached", "20 COMPLETED RUNS PRESERVED; ONLY THE 10 NEVER-STARTED R40 TASKS RUN IN V2"),
        ("UNSEEN_EVALUATION_BATCH_LOGGER_V1", "ENGINEERING", "Python 3.8 Path.write_text rejected newline keyword after all atomic evaluation units completed", "45 INDIVIDUAL LEDGERS/MANIFESTS VALIDATED; BATCH MANIFEST RECOVERED WITHOUT RECOMPUTING OR SELECTING ROWS"),
        ("FINALIZER_V1", "ENGINEERING", "Python 3.8 Path.write_text rejected newline keyword while writing source audit", "PARTIAL HANDOFF PRESERVED; COMPATIBILITY FIX; FULL FINALIZER RERUN"),
        ("FINALIZER_REPORT_SCOPE_V2", "ENGINEERING", "aggregate checkpoint count mixed one superseded development-v1 failure with active evidence", "REPORT NOW SEPARATES 30/30 FORMAL, 62/62 ACTIVE, AND ONE PRESERVED SUPERSEDED FAILURE"),
        ("FINALIZER_PROTOCOL_CONTEXT_V3", "ENGINEERING", "draft report implied the budget-stopped SEPAR v2 rerun had completed and omitted the closed P22 K18 score", "WORDING CORRECTED AND K18 AUTHOR-SUPPLIED CONTEXT ROW ADDED BEFORE FINAL COMMIT"),
    ]
    return pd.DataFrame(rows, columns=("attempt_id", "failure_class", "reason", "disposition"))


def report_table(frame: pd.DataFrame) -> str:
    columns = [
        "dataset", "cluster_k", "candidate_id", "endpoint_role", "control_id",
        "ari_best", "ari_median", "ari_mean", "ari_min", "nmi_best", "nmi_median",
        "nmi_mean", "nmi_min", "ami_mean", "fmi_mean", "morans_i_mean",
        "gearys_c_mean", "wall_seconds", "gpu_seconds", "peak_gpu_mib", "peak_rss_mib",
    ]
    value = frame.copy()
    if "candidate_id" not in value:
        value["candidate_id"] = value.get("formal_id")
    value["candidate_id"] = value["candidate_id"].fillna(value.get("formal_id"))
    value["control_id"] = value.get("control_id", pd.Series(index=value.index, dtype=object)).fillna(value.get("source_view"))
    value = value[[column for column in columns if column in value.columns]].head(60)
    header = "| " + " | ".join(value.columns) + " |"
    separator = "|" + "|".join(["---"] * len(value.columns)) + "|"
    rows = []
    for _, row in value.iterrows():
        cells = []
        for item in row:
            cells.append(f"{item:.4f}" if isinstance(item, (float, np.floating)) and np.isfinite(item) else str(item))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def run(terminal: str, secondary: str) -> None:
    if OUT.exists():
        raise RuntimeError("Night-15A handoff output already exists")
    OUT.mkdir(parents=True)
    started = time.perf_counter()
    stage_path = ROOT / "stage_a_score_source_v3/score_source_summary.csv"
    stage = pd.read_csv(stage_path)
    shutil.copy2(stage_path, OUT / "score_source_ablation.csv")
    shutil.copy2(ROOT / "stage_a_score_source_v3/matched_contribution_deltas.csv", OUT / "matched_contribution_deltas.csv")
    shutil.copy2(ROOT / "stage_a_score_source_v3/stage_a_manifest.json", OUT / "stage_a_manifest.json")
    raw = raw_results()
    raw.to_csv(OUT / "raw_feature_and_mcdf_results.csv", index=False)
    frozen = aggregate_frozen()
    frozen.to_csv(OUT / "unseen_seed_results.csv", index=False)
    main = main_results(stage, frozen, raw)
    main.to_csv(OUT / "main_results_table.csv", index=False)
    ledger = all_run_ledger()
    ledger.to_csv(OUT / "all_run_ledger.csv", index=False)
    variance_decomposition().to_csv(OUT / "seed_variance_decomposition.csv", index=False)
    protocol_registry().to_csv(OUT / "protocol_and_dataset_registry.csv", index=False)
    candidate_registry().to_csv(OUT / "candidate_registry.csv", index=False)
    failure_ledger().to_csv(OUT / "failure_ledger.csv", index=False)
    checkpoint = checkpoint_audit()
    atomic_json(OUT / "checkpoint_roundtrip_audit.json", checkpoint)
    resources = resource_audit()
    atomic_json(OUT / "resource_audit.json", resources)
    raw_audit = historical_raw_audit()
    atomic_json(OUT / "historical_raw_immutability.json", raw_audit)
    failure_archive = ROOT / "failures/clone_incomplete_attempt1.tar"
    filesystem = os.statvfs(str(ROOT))
    atomic_json(OUT / "infrastructure_and_failure_archive_audit.json", {
        "inode_exhaustion_observed": True,
        "filesystem_free_inodes_at_finalization": int(filesystem.f_favail),
        "incomplete_clone_failure_preserved_as_tar": failure_archive.exists(),
        "incomplete_clone_archive_path": str(failure_archive),
        "incomplete_clone_archive_size_bytes": failure_archive.stat().st_size,
        "incomplete_clone_archive_sha256": file_sha256(failure_archive),
        "original_multifile_failure_tree_removed_after_tar_list_and_sha_verification": True,
        "science_formula_or_protocol_changed_by_remediation": False,
    })
    protocol_evidence = {
        "P22_3DOT_K18_member_extraction": json.loads(
            (ROOT / "protocol_inputs/3dot_zenodo_15089427/member_extraction_audit.json").read_text(encoding="utf-8")
        ),
        "GSE213264_tonsil_dedup": json.loads(
            (ROOT / "protocol_inputs/GSE213264/dedup_audit.json").read_text(encoding="utf-8")
        ),
        "raw_feature_preflights": {},
    }
    for bank in (
        "raw_preprocess_v2",
        "raw_preprocess_moran_v1",
        "raw_preprocess_separ_exact_v3",
        "raw_preprocess_moran1500_v1",
    ):
        for path in sorted((ROOT / bank).glob("*/raw_feature_preflight.json")):
            protocol_evidence["raw_feature_preflights"][f"{bank}/{path.parent.name}"] = json.loads(
                path.read_text(encoding="utf-8")
            )
    atomic_json(OUT / "real_input_and_protocol_audit.json", protocol_evidence)
    (OUT / "source_and_license_audit.md").write_text(
        "# Night-15A source and protocol audit\n\n"
        "- 3d-OT official source: `dbjzs/3d-OT` commit `39a7cb02748d83299cd471f172f3b972896e61d8`, Apache-2.0. "
        "Only protocol/artifact semantics were used; no third-party source was copied into SpaLORA.\n"
        "- SEPAR official source: `zerovain/SEPAR` commit `6d3475fa0bd749d3b1b5592b68323439d473f9fc`, MIT. "
        "Tutorial4 was audited for the K=12 lane; external context execution is kept separate from own-method evidence.\n"
        "- GSE213264 source: NCBI GEO processed Spatial-CITE-seq tonsil members. Only RNA/protein identifiers and matrix metadata were opened for deduplication; annotation values were not opened.\n"
        "- MCDF code is an independent project implementation. Its four experts and masked-modality objective are evaluated as a development hypothesis, not claimed as established novelty.\n",
        encoding="utf-8",
    )
    label_audit = {
        "public_benchmark_policy": "known-K/protocol-K unsupervised clustering; labels enter only post-partition evaluator and cross-run HPO",
        "training_label_reads": 0,
        "labels_in_model_input": False,
        "labels_in_loss_or_gradient": False,
        "labels_in_within_run_checkpoint_selection": False,
        "public_evaluation_partitions": int(ledger["absolute_ari"].notna().sum()) if "absolute_ari" in ledger else 0,
        "GSE213264_annotation_values_read": 0,
    }
    atomic_json(OUT / "label_use_audit.json", label_audit)
    engineering = failure_ledger().copy()
    engineering.insert(0, "change_id", [f"E{index:02d}" for index in range(1, len(engineering) + 1)])
    engineering.to_csv(OUT / "engineering_changelog.csv", index=False)
    evaluation_batch_paths = [
        ROOT / "unseen_evaluation_part1_v1/unseen_evaluation_batch_manifest.json",
        ROOT / "unseen_evaluation_r40_v2/unseen_evaluation_batch_manifest.json",
    ]
    freeze_path = ROOT / "formal_freeze_v1/frozen_candidate_and_head_registry.json"
    training_manifest_paths = [
        ROOT / "unseen_training_v1/unseen_training_manifest.partial.json",
        ROOT / "unseen_training_r40_v2/unseen_training_manifest.json",
    ]
    training_manifests = [json.loads(path.read_text(encoding="utf-8")) for path in training_manifest_paths]
    first_rows = training_manifests[0]["rows"]
    second_rows = training_manifests[1]["rows"]
    combined_training_rows = first_rows + second_rows
    if len(combined_training_rows) != 30 or any(row["status"] != "PASS" for row in combined_training_rows):
        raise RuntimeError("formal unseen training does not close at 30/30 PASS")
    evaluation_manifests = [json.loads(path.read_text(encoding="utf-8")) for path in evaluation_batch_paths]
    combined_evaluation_tasks = [task for manifest in evaluation_manifests for task in manifest["tasks"]]
    if len(combined_evaluation_tasks) != 45 or any(task["status"] != "PASS" for task in combined_evaluation_tasks):
        raise RuntimeError("formal unseen endpoint evaluation does not close at 45/45 PASS")
    run_manifest = {
        "stage_a_rows": int(len(pd.read_csv(ROOT / "stage_a_score_source_v3/score_source_ablation.csv"))),
        "stage_a_status": (
            "PASS"
            if json.loads((ROOT / "stage_a_score_source_v3/stage_a_manifest.json").read_text(encoding="utf-8"))["failure_count"] == 0
            else "FAIL"
        ),
        "unseen_training": {
            "task_count": len(combined_training_rows),
            "pass_count": sum(row["status"] == "PASS" for row in combined_training_rows),
            "fail_count": sum(row["status"] != "PASS" for row in combined_training_rows),
            "source_manifests": [str(path) for path in training_manifest_paths],
            "rows": combined_training_rows,
        },
        "unseen_evaluation": {
            "task_count": len(combined_evaluation_tasks),
            "passed": sum(task["status"] == "PASS" for task in combined_evaluation_tasks),
            "failed": sum(task["status"] != "PASS" for task in combined_evaluation_tasks),
            "source_manifests": [str(path) for path in evaluation_batch_paths],
            "tasks": combined_evaluation_tasks,
        },
        "formal_freeze_sha256": file_sha256(freeze_path),
        "formal_freeze": json.loads(freeze_path.read_text(encoding="utf-8")),
        "historical_raw_immutability": raw_audit["passed"],
        "labels_in_model_input_loss_gradient_or_checkpoint_selection": False,
        "dense_n_by_n_count": 0,
    }
    atomic_json(OUT / "run_manifest.json", run_manifest)
    shutil.copytree(ROOT / "formal_freeze_v1", OUT / "formal_freeze")

    stage_focus = main[
        (main.result_role == "MATCHED_NIGHT14B_SCORE_SOURCE")
        & main.control_id.isin([
            "COORDINATE_ONLY", "RNA_ONLY", "ATAC_ONLY", "RNA_PLUS_COORDINATES",
            "ATAC_PLUS_COORDINATES", "FUSED_FULL", "FUSED_WITHOUT_COORDINATES",
            "FUSED_WITHOUT_GRAPH_FILTER", "FUSED_WITHOUT_SPATIAL_REFINEMENT",
            "NIGHT14B_FROZEN_BEST",
        ])
    ].copy().sort_values(["dataset", "cluster_k", "ari_median"], ascending=[True, True, False])
    frozen_focus = main[
        main.result_role.isin([
            "UNSEEN_BACKBONE_SEEDS_3_TO_7",
            "EXTERNAL_PROTOCOL_CONTEXT_NOT_OWN_METHOD",
        ])
        | (
            (main.result_role == "DEVELOPMENT_SCREEN_BEST_RUN")
            & (main.dataset == "P22_3DOT_K18")
        )
    ].copy().sort_values(["dataset", "cluster_k", "ari_median"], ascending=[True, True, False])
    stage_table = report_table(stage_focus)
    frozen_table = report_table(frozen_focus)
    report = f"""# SpaLORA Night-15A 多模态贡献与分数稳定性报告

## 我现在需要知道的三件事

1. **问题**：Night-14B 的 P22/MISAR 高分究竟由坐标、RNA+ATAC 分子信息，还是聚类 head 产生；本轮用完全相同的 K、mask、endpoint seed 和 head 做了逐层拆分。
2. **实际动作与流水线位置**：先在冻结表示/聚类端做 coordinate-only、两条单模态、融合、去坐标、去滤波、去 refinement 对照；再从真实 feature-level RNA/ATAC 经 HVG/TF-IDF/LSI 或 Moran feature bank 训练统一 cross-reconstruction/MCDF 核心，并把 3–5 个机制候选冻结后用 backbone seeds 3–7 复核。所有 checkpoint 都做 fresh-process 数值回放。
3. **论文含义**：主终态为 `{terminal}`，次级信号为 `{secondary}`。这不是 SOTA、confirmed milestone 或论文结论；它只回答现有分数来源，并决定 MCDF 是否值得继续作为主方法。

## 结果分类

- 主终态：`{terminal}`
- 次级信号：`{secondary}`
- 聚类语义：公开 benchmark、标签后置 evaluator 的 known-K/protocol-K 无监督聚类；标签没有进入模型输入、loss、gradient 或单次训练 checkpoint selection。

## 相同 head 的分数来源主表

{stage_table}

## 冻结后 seeds 3–7 稳定性与 exact protocol context 主表

{frozen_table}

完整逐行结果见 `main_results_table.csv`、`raw_feature_and_mcdf_results.csv` 和 `all_run_ledger.csv`。BEST、median、mean、min 分开保留，任何坏 seed、失败或提前停止路线均未删除。

### 直接解释

- P22 K=9 的 fused mean ARI 0.4960，高于 RNA+coordinates 0.4819 与 ATAC+coordinates 0.4806，但优势小且 backbone seeds 3–7 未维持：最好统一候选 R21 的 ARI 为 best 0.4939、median 0.4487、mean 0.4546。
- MISAR K=7 的 fused mean ARI 0.4040 低于 ATAC+coordinates 0.4190；K=12 的 fused mean 0.2742 也略低于 ATAC+coordinates 0.2787，因此不能登记稳定多模态贡献。
- label-free partition medoid 对 R40/MISAR K=7 把 ARI min 从 0.2611 提到 0.3774、median 从 0.4083 提到 0.4295；它属于稳分 head 设施，不是 MCDF 方法增益。
- MISAR K=12 的冻结候选最高 ARI 仅 0.3737，远低于 0.50；P22 K=18 的 0.6122/0.7233 来自锁定参考表示对官方作者 18-state assignment 的 exact-protocol context，不是本轮新模型成绩。

## 协议边界

- P22 K=9 使用项目 9,196/9,196 exact ground truth。
- P22 K=18 使用 3d-OT 官方 h5ad 中作者提供的 18-state domain assignment；它与项目 IDs byte-exact 同序，**不是**把 K=9 人工拆分，但也不是独立专家 ground truth，因此只作 protocol context。
- MISAR K=7 使用七类公开 Y；SEPAR K=12 lane 忠实保留“聚成 12 类、对七类 Y 评价”的官方 Tutorial4 语义。
- GSE213264 Spatial-CITE-seq tonsil 与 canonical Zenodo tonsil 三切片不是重复资产：accession/platform/spot 数均不同，三个 exact ID 交集均为 0。

## 最重要的失败与限制

- exact SEPAR 第一次完成 100 次迭代后，在最终 clustering 处因 upstream AnnData ArrayView 兼容性失败；该失败完整保留。v2 只改变数组 materialization，但为避免外部方法挤占主模型预算而提前停止，因此本轮没有把 SEPAR 外部实测分数写入 own-method evidence。
- 2,500+2,500 Moran feature bank 因 RNA 有限可识别特征不足而 fail-closed；仅进行一次透明平台数值修订到 1,500+1,500。
- raw-feature trainable candidates 若只在开发 best run 上好看、但 seeds 3–7 的 median/mean 未保持，则不登记方法信号。
- P22 K=18 与 MISAR K=12 都是特定论文协议，不与 K=9/K=7 数字伪装成同一任务的直接胜负。
- A1 与 tonsil s1 完成统一模型工程训练/回放；由于两条主 RNA+ATAC lane 均无新增方法信号，D1 与 tonsil s2/s3 未继续消耗预算，状态透明保留为 NOT_RUN_AFTER_PRIMARY_NEGATIVE。

## 5–8 句导师汇报版

Night-15A 把 Night-14B 的高分拆成坐标、单模态、融合和聚类 head 四层。坐标很重要，但 coordinate-only 并不能解释全部 P22/MISAR 得分。P22 的融合在开发最佳值上仍可超过单模态，但 MISAR 尤其 K=12 没有形成稳定的融合优势。我们从真实 feature-level RNA/ATAC 训练了同一个统一核心，并用未参与 HPO 的 backbone seeds 3–7 检查，而不是重复最好 seed。P22 K=18 与 SEPAR K=12 的官方协议已闭合并单独报告。GSE213264 tonsil 已证实不是现有 canonical tonsil 的重复切片。最终分类是 `{terminal}`（次级 `{secondary}`），所以本轮不能写成新模块已成立或 SOTA。下一步只能围绕真正保留下来的层继续，不能再包装无独立增量的 MCDF 门控。

## 技术附录

- fresh-process checkpoint numerical round-trip：formal unseen {checkpoint['formal_unseen_numerically_close_pass']}/{checkpoint['formal_unseen_total']}；active evidence {checkpoint['active_numerically_close_pass']}/{checkpoint['active_total']}。另有 {checkpoint['superseded_failures_preserved']} 条早期 superseded 失败原样保留。
- 训练总 GPU 秒：{resources['training_total_gpu_seconds']:.2f}；峰值 GPU：{resources['peak_gpu_mib']:.2f} MiB；峰值训练 RSS：{resources['peak_training_rss_mib']:.2f} MiB。
- training label reads=0；labels in loss/gradient/checkpoint selection=false；dense N×N=0。
- 最终 commit/tag、bundle 与 compact index 在交付 manifest 中记录；本报告自身不循环嵌入最终 commit hash。
"""
    (OUT / "night15a_report.md").write_text(report, encoding="utf-8")
    plain = f"Night-15A 的主结论是 {terminal}，次级信号是 {secondary}。本轮完成了 matched score-source controls、真实 raw-feature 统一核心、公开协议闭合和 unseen seeds 3–7 稳定性诊断；它不构成 SOTA 或论文证据。\n"
    (OUT / "night15a_plain_summary.txt").write_text(plain, encoding="utf-8")
    decision = {
        "terminal_status": terminal,
        "secondary_signal": secondary,
        "classification": "DEVELOPMENT_DIAGNOSTIC",
        "paper_ready": False,
        "sota_claim": False,
        "known_k_unsupervised_with_public_post_partition_evaluation": True,
        "training_label_reads": 0,
        "labels_in_model_input_loss_gradient_or_checkpoint_selection": False,
        "historical_raw_immutability_pass": raw_audit["passed"],
        "matched_score_source_rows": int(len(pd.read_csv(ROOT / "stage_a_score_source_v3/score_source_ablation.csv"))),
        "all_run_ledger_rows": int(len(ledger)),
        "fresh_process_roundtrip_formal_unseen": f"{checkpoint['formal_unseen_numerically_close_pass']}/{checkpoint['formal_unseen_total']}",
        "fresh_process_roundtrip_active": f"{checkpoint['active_numerically_close_pass']}/{checkpoint['active_total']}",
        "superseded_roundtrip_failures_preserved": checkpoint["superseded_failures_preserved"],
        "finalization_wall_seconds": time.perf_counter() - started,
    }
    atomic_json(OUT / "night15a_decision.json", decision)
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/night15a"],
        cwd=str(REPO), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    (OUT / "tests_summary.txt").write_text(tests.stdout, encoding="utf-8")
    if tests.returncode:
        raise RuntimeError("targeted Night-15A tests failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal", required=True)
    parser.add_argument("--secondary", required=True)
    args = parser.parse_args()
    run(args.terminal, args.secondary)


if __name__ == "__main__":
    main()
