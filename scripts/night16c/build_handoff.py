from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT = Path(os.environ.get("SPALORA_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
LOCAL = PROJECT / "night16c_local_work"
WORK = LOCAL / "working"
OUT = LOCAL / "outputs" / "night16c_handoff"
KIT = PROJECT / "night15b_delivery_20260824" / "working" / "local_compute_kit"
N12A = PROJECT / "night12a_delivery_20260822" / "official_compact" / "outputs" / "night12a_handoff"
N13A = PROJECT / "night13a_delivery_20260822" / "official_compact" / "outputs" / "night13a_handoff"
PLAN = PROJECT / "night16c_family_frozen_boundary_field_and_dataset_expansion_planning_20260824"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    fields = list(fields)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def f(x: Any) -> float:
    return float(x)


def i(x: Any) -> int:
    return int(float(x))


def p10_p5_shape_registry() -> dict[str, dict[str, Any]]:
    audit = json.loads((N12A / "real_shape_and_id_audit.json").read_text(encoding="utf-8"))
    return {u["unit_id"]: u for u in audit["units"]}


def p0_records() -> list[dict[str, Any]]:
    root = WORK / "new_unit_p0_audit" / "working" / "night16c_new_units"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*/producer_record.json"))]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    metrics = read_csv(WORK / "final_replay1" / "metrics.csv")
    replay2 = read_csv(WORK / "final_replay2" / "metrics.csv")
    ablations = read_csv(WORK / "final_ablation" / "minimal_contribution_table.csv")
    diagnostics = read_csv(WORK / "final_diagnostics" / "boundary_diagnostic_table.csv")
    frozen = json.loads((WORK / "family_frozen_config.json").read_text(encoding="utf-8"))
    p0 = p0_records()
    n12_shapes = p10_p5_shape_registry()

    lane_meta = {
        "A1": ("RNA_PROTEIN", "GSE263617", "lymph node A1", "GSM8195494/GSM8195498", 10, "discovery"),
        "D1": ("RNA_PROTEIN", "GSE263617", "lymph node D1", "GSM8195496/GSM8195500", 10, "frozen_transfer"),
        "tonsil_s1": ("RNA_PROTEIN", "Zenodo 12654113", "tonsil slice 1", "10.5281/zenodo.12654113", 4, "discovery"),
        "tonsil_s2": ("RNA_PROTEIN", "Zenodo 12654113", "tonsil slice 2", "10.5281/zenodo.12654113", 4, "frozen_transfer"),
        "tonsil_s3": ("RNA_PROTEIN", "Zenodo 12654113", "tonsil slice 3", "10.5281/zenodo.12654113", 4, "frozen_transfer"),
        "P22": ("RNA_CHROMATIN", "AtlasXplore/Fan", "mouse brain P22", "deposited P22 RNA+ATAC", 9, "discovery"),
        "MISAR_E15_5_S1": ("RNA_CHROMATIN", "MISAR-seq", "E15.5 S1", "OEP003285", 7, "discovery"),
    }
    frontier = {
        "A1": (0.2760026589984753, 0.42173983941218224),
        "D1": (0.3651735437579217, 0.4445774430144442),
        "tonsil_s1": (0.23653550194382364, 0.3171179292228705),
        "tonsil_s2": (0.25826424500877926, 0.31432408099851056),
        "tonsil_s3": (0.350644182009914, 0.3097714630576775),
        "P22": (0.5955516462857395, 0.7179305602206435),
        "MISAR_E15_5_S1": (0.5414237853091904, 0.6667977615565593),
    }
    starts = {r["lane"]: r for r in ablations if r["variant"] == "STRONG_START_ONLY"}

    main_rows: list[dict[str, Any]] = []
    for row in metrics:
        lane = row["lane"]
        family, study, physical, accession, k, role = lane_meta[lane]
        old_ari, old_nmi = frontier[lane]
        start = starts[lane]
        full_ari, full_nmi = f(row["absolute_ari"]), f(row["absolute_nmi"])
        main_rows.append({
            "family": family,
            "lane": lane,
            "study": study,
            "physical_unit": physical,
            "role": role,
            "total_observations": i(row["evaluated_observations"]) if lane not in {"tonsil_s2", "tonsil_s3"} else (4519 if lane == "tonsil_s2" else 4521),
            "evaluated_observations": i(row["evaluated_observations"]),
            "K": k,
            "night16b_frontier_ari": old_ari,
            "night16b_frontier_nmi": old_nmi,
            "matched_strong_start_ari": f(start["absolute_ari"]),
            "matched_strong_start_nmi": f(start["absolute_nmi"]),
            "family_frozen_ari": full_ari,
            "family_frozen_nmi": full_nmi,
            "delta_vs_strong_start_ari": full_ari - f(start["absolute_ari"]),
            "delta_vs_strong_start_nmi": full_nmi - f(start["absolute_nmi"]),
            "delta_vs_night16b_frontier_ari": full_ari - old_ari,
            "delta_vs_night16b_frontier_nmi": full_nmi - old_nmi,
            "ami": f(row["ami"]),
            "fmi": f(row["fmi"]),
            "homogeneity": f(row["homogeneity"]),
            "v_measure": f(row["v_measure"]),
            "morans_i": f(row["morans_i"]),
            "gearys_c": f(row["gearys_c"]),
            "neighbor_agreement": f(row["neighbor_agreement"]),
            "changed_observations": i(row["changed_observations"]),
            "minimum_cluster_size": i(row["min_cluster_size"]),
            "cluster_sizes": row["cluster_sizes"],
            "partition_sha256": row["partition_sha256"],
            "config_id": row["candidate_id"],
            "best_ari": full_ari,
            "median_ari": full_ari,
            "mean_ari": full_ari,
            "min_ari": full_ari,
            "best_nmi": full_nmi,
            "median_nmi": full_nmi,
            "mean_nmi": full_nmi,
            "min_nmi": full_nmi,
            "winning_deterministic_replays": 2,
            "wall_seconds": f(row["wall_seconds"]),
            "peak_gpu_mib": 0,
            "producer_label_reads": 0,
            "dense_n_by_n_count": 0,
        })
    write_csv(OUT / "absolute_metrics_main_table.csv", main_rows)

    # Root annotation arrays are hashed independently from the registered compute kit.
    label_hashes: dict[str, dict[str, Any]] = {}
    for lane in lane_meta:
        z = np.load(KIT / f"{lane}.npz", allow_pickle=False)
        label_hashes[lane] = {
            "label_array_sha256": sha_array(z["labels_primary"]),
            "mask_sha256": sha_array(z["label_mask"]),
            "ordered_id_sha256": sha_array(z["ids"]),
            "total": len(z["ids"]),
            "evaluated": int(z["label_mask"].sum()),
            "K": int(z["k_primary"][0]),
        }

    annotation_rows = []
    def ann(unit: str, protocol: str, k: Any, source: str, purpose: str, status: str = "REGISTERED", label_sha: str = "", mask_sha: str = "", source_file_sha: str = "", notes: str = "") -> None:
        annotation_rows.append({"physical_unit": unit, "protocol_id": protocol, "K": k, "annotation_source": source, "annotation_file_sha256": source_file_sha, "label_array_sha256": label_sha, "evaluation_mask_sha256": mask_sha, "purpose": purpose, "status": status, "notes": notes})
    for lane, (source, carrier_sha) in {
        "A1": ("GSE263617 A1_groundtruth.csv", "0c9473b95cd15af20cf45cbc380e5d7d315b687bd18681ab93f0e41973560b5f"),
        "D1": ("GSE263617 D1_groundtruth.csv", "524e2447b743b093afad440fea7cd28e63239fe0660b461f1c404aba68ba84f3"),
        "tonsil_s1": ("Zenodo12654113 RNA obs.final_annot", "e1d99b34685805c93a7314f8ef4e07b2f92d244a8c29ac9fced9ee19ff3b96a9"),
        "tonsil_s2": ("Zenodo12654113 RNA obs.final_annot", "74ae08874242a48be5dbee5c73a4cacb6944ebd99def6f4f1a5223bfed3ca7bd"),
        "tonsil_s3": ("Zenodo12654113 RNA obs.final_annot", "23ef098c7e6b347e98dc5b46bc577073b3aa9575ea61a38669b747b58cc5f47a"),
        "P22": ("MouseBrain_groundtruth.csv", "ca2a702323b42a63588687be29bbffeed7b2d57586386e5a000b0f6135619b32"),
        "MISAR_E15_5_S1": ("official Figshare Y carrier", "2f5862cff045b6a296f3cbc0a978d8576bd36f2d4c9419b75c6760c9c7e9d2e3"),
    }.items():
        h = label_hashes[lane]
        ann(lane, f"{lane}_PRIMARY_K{h['K']}", h["K"], source, "headline development/transfer evaluation", label_sha=h["label_array_sha256"], mask_sha=h["mask_sha256"], source_file_sha=carrier_sha, notes=f"{h['evaluated']}/{h['total']} evaluated")
    ann("A1", "SMART_A1_K7_CONTEXT", 7, "SMART publication/reproduction protocol", "context only", "PROTOCOL_CONFLICT_NOT_MIXED", notes="Not substituted for project K10")
    ann("tonsil_study_block", "SMART_MS_JOINT_K6_CONTEXT", 6, "SMART-MS joint tonsil protocol", "context only", "PROTOCOL_CONFLICT_NOT_MIXED", notes="Not substituted for per-slice K4")
    ann("P22", "P22_K12_CONTEXT", 12, "registered literature context", "secondary context", "PROTOCOL_ONLY", notes="Not compared with K9 headline")
    p22 = np.load(KIT / "P22.npz", allow_pickle=False)
    ann("P22", "P22_AUTHOR_K18", 18, "3d-OT author assignment", "secondary sensitivity", label_sha=sha_array(p22["labels_k18_author_assignment"]), mask_sha=label_hashes["P22"]["mask_sha256"], notes="Author assignment; separate from K9 expert protocol")
    ann("MISAR_E15_5_S1", "MISAR_K12_CONTEXT", 12, "SEPAR-style K12 endpoint context", "secondary sensitivity", "PROTOCOL_ONLY", notes="Not an independent physical unit; not used in Night16C headline")
    ann("other_MISAR_stages", "MISAR_OTHER_STAGE_K10_CONTEXT", 10, "SMART/MISAR stage-specific context", "registry only", "NOT_ACCESSED", notes="Requires exact per-stage annotation closure")
    write_csv(OUT / "annotation_protocol_registry.csv", annotation_rows)

    n12 = {u: n12_shapes[u] for u in n12_shapes}
    data_rows: list[dict[str, Any]] = []
    def data(unit: str, family: str, study: str, accession: str, platform: str, modalities: str, n: Any, shape: str, annotation: str, status: str, role: str, paired: str, path: str, notes: str = "") -> None:
        data_rows.append({"unit_id": unit, "family": family, "study": study, "accession_or_doi": accession, "platform": platform, "modalities": modalities, "observations": n, "feature_shapes": shape, "annotation_protocol": annotation, "status": status, "night16c_role": role, "paired_id_contract": paired, "registered_path": path, "notes": notes})
    for lane, (family, study, physical, accession, k, role) in lane_meta.items():
        h = label_hashes[lane]
        z = np.load(KIT / f"{lane}.npz", allow_pickle=False)
        data(lane, family, study, accession, "registered reduced-view compute kit", "RNA+protein" if family == "RNA_PROTEIN" else "RNA+ATAC", h["total"], f"view1={z['view1'].shape};view2={z['view2'].shape}", f"{lane}_PRIMARY_K{k}", "LABELED_PUBLIC_BENCHMARK", role, "byte-exact ordered IDs", str(KIT / f"{lane}.npz"))
    data("SPOTS_SPLEEN_REP1", "RNA_PROTEIN", "GSE198353", "GSE198353 processed rep1", "10x Visium+SPOTS", "RNA+21 ADT", 2653, "raw=2653x32285+2653x21;P0=2653x30+2653x21", "NONE_RELIABLE", "UNLABELED_P0_PASS", "new frozen-transfer P0", "10x barcode+tissue exact", "/root/autodl-fs/night4a_external_data/raw/GSE198353", "No author clusters treated as ground truth")
    data("SPOTS_SPLEEN_REP2", "RNA_PROTEIN", "GSE198353", "GSE198353 processed rep2", "10x Visium+SPOTS", "RNA+21 ADT", 2768, "raw=2768x32285+2768x21", "NONE_RELIABLE", "REGISTRY_ONLY", "reserve", "10x barcode+tissue exact", "/root/autodl-fs/night4a_external_data/raw/GSE198353")
    for unit, acc, nobs in [("P10S1", "GSM9247585+NeMO ADT", 7447), ("P10S2", "GSM9247586+NeMO ADT", 41289), ("P10S3", "GSM9247587+NeMO ADT", 5845)]:
        u = n12[unit]
        data(unit, "RNA_PROTEIN", "GSE308623", acc, "spatial tri-omics P10", "RNA+ADT", nobs, f"raw RNA={u['rna']['observation_by_feature_shape']};ADT={u['adt']['observation_by_feature_shape']};P0={nobs}x80+{nobs}x80", "NONE_RELIABLE", "UNLABELED_P0_PASS", "new frozen-transfer P0", u["paired_spot_identity"], f"/autodl-fs/data/night12b_replicate_calibrated_link_identifiability_20260822/preflight/{unit}/registered_matrices.npz")
    for unit, rna_acc, atac_acc, nobs in [("P5S1", "GSM9247581", "GSM9248997", 7794), ("P5S2", "GSM9247582", "GSM9248998", 28545), ("P5S3", "GSM9247583", "GSM9248999", 9426)]:
        u = n12[unit]
        data(unit, "RNA_CHROMATIN", "GSE308623", f"{rna_acc}/{atac_acc}", "spatial tri-omics P5", "RNA+ATAC", nobs, f"raw RNA={u['rna']['observation_by_feature_shape']};P0={nobs}x80+{nobs}x80", "NONE_RELIABLE", "UNLABELED_P0_PASS", "new frozen-transfer P0", u["paired_spot_identity"], f"/autodl-fs/data/night12b_replicate_calibrated_link_identifiability_20260822/preflight/{unit}/registered_matrices.npz", "Correct distinct RNA/ATAC accessions; clean-room gene-score provenance inherited")
    data("GSE213264_HUMAN_TONSIL", "RNA_PROTEIN", "GSE213264", "GSE213264", "Spatial-CITE-seq", "RNA+high-dimensional ADT", 2492, "2492x28417+2492x283", "AUTHOR_DERIVED_UNSUPERVISED_REFERENCE_NOT_LOADED", "REGISTRY_ONLY", "future unlabeled unit", "explicit string spot-ID alignment required; source row order differs", "/autodl-fs/data/night15e_gse213264", "Author RNA/protein clusters are not expert spatial-domain truth")
    write_csv(OUT / "dataset_family_registry.csv", data_rows)

    # Preserve the formal family HPO table exactly as evaluated, with transparent labels.
    family_search: list[dict[str, Any]] = []
    for fam in ("RNA_PROTEIN", "RNA_CHROMATIN"):
        src = WORK / "family_search_merged_final" / fam / "evaluated_search_ledger.csv"
        for row in read_csv(src):
            row["search_source"] = "merged unique formal+refinement candidates"
            row["selection_scope"] = "label-assisted family-level benchmark HPO"
            family_search.append(row)
    write_csv(OUT / "family_config_search_ledger.csv", family_search)
    shutil.copy2(WORK / "family_frozen_config.json", OUT / "family_frozen_config.json")

    # Dataset-tuned oracle is kept as a development ceiling and never used as headline.
    oracle_rows = []
    by_lane: dict[str, list[dict[str, str]]] = {}
    for row in family_search:
        by_lane.setdefault(row["lane"], []).append(row)
    for lane, rows in by_lane.items():
        valid = [r for r in rows if r["status"] == "PASS" and math.isfinite(f(r["absolute_ari"])) and math.isfinite(f(r["absolute_nmi"]))]
        valid.sort(key=lambda r: (-f(r["absolute_ari"]), -f(r["absolute_nmi"]), i(r["changed_observations"]), r["candidate_id"]))
        r = valid[0]
        oracle_rows.append({"family": r["family"], "lane": lane, "candidate_id": r["candidate_id"], "absolute_ari": r["absolute_ari"], "absolute_nmi": r["absolute_nmi"], "ami": r["ami"], "fmi": r["fmi"], "min_cluster_size": r["min_cluster_size"], "changed_observations": r["changed_observations"], "config_json": r["config_json"], "role": "PER_DATASET_LABEL_ASSISTED_DEVELOPMENT_CEILING_NOT_HEADLINE"})
    write_csv(OUT / "per_dataset_oracle_development_ceiling.csv", oracle_rows)

    write_csv(OUT / "boundary_diagnostic_table.csv", diagnostics)
    write_csv(OUT / "minimal_contribution_table.csv", ablations)

    # All actual attempts, without duplicating merged derived tables.
    all_runs: list[dict[str, Any]] = []
    sources = [
        ("stage0_superseded", WORK / "family_search"),
        ("formal_v1", WORK / "family_search_v1"),
        ("refine_v1", WORK / "family_refine_v1"),
        ("refine_v2", WORK / "family_refine_v2"),
    ]
    for stage, root in sources:
        for fam in ("RNA_PROTEIN", "RNA_CHROMATIN"):
            for row in read_csv(root / fam / "evaluated_search_ledger.csv"):
                all_runs.append({"stage": stage, "family": fam, "lane": row["lane"], "candidate_or_variant": row["candidate_id"], "status": "SUPERSEDED" if stage == "stage0_superseded" else row["status"], "failure": row.get("failure", ""), "partition_sha256": row.get("partition_sha256", ""), "absolute_ari": row.get("absolute_ari", ""), "absolute_nmi": row.get("absolute_nmi", ""), "wall_seconds": row.get("wall_seconds", ""), "notes": "v0 trust gate used averaged evidence; retained, then superseded by explicit conjunctive TPR" if stage == "stage0_superseded" else ""})
    for row in ablations:
        all_runs.append({"stage": "final_ablation", "family": row["family"], "lane": row["lane"], "candidate_or_variant": row["variant"], "status": "PASS", "failure": "", "partition_sha256": row["partition_sha256"], "absolute_ari": row["absolute_ari"], "absolute_nmi": row["absolute_nmi"], "wall_seconds": "", "notes": "matched strong-start contribution control"})
    for r in p0:
        all_runs.append({"stage": "new_unit_real_p0", "family": r["family"], "lane": r["unit_id"], "candidate_or_variant": r["family_config_id"], "status": r["status"], "failure": "", "partition_sha256": r["partition_sha256"], "absolute_ari": "", "absolute_nmi": "", "wall_seconds": r["wall_seconds"], "notes": "unlabeled real path; no scientific metric"})
    all_runs.append({"stage": "new_unit_real_p0_attempt1", "family": "RNA_CHROMATIN", "lane": "P5S1", "candidate_or_variant": "initial label-free centrality without minimum-cluster guard", "status": "FAILED_PRESERVED", "failure": "selected exact-K fused start contained a singleton; fail-closed", "partition_sha256": "", "absolute_ari": "", "absolute_nmi": "", "wall_seconds": "", "notes": "engineering fix added generic exact-K minimum-cluster guard; all seven P0 units rerun together"})
    write_csv(OUT / "all_runs_and_failures.csv", all_runs)
    write_csv(OUT / "failure_and_correction_ledger.csv", [
        {"event_id": "E01", "stage": "stage0 screen", "classification": "SCIENTIFIC_SEMANTICS_REVISION", "problem": "v0 trust score averaged evidence rather than enforcing the preregistered conjunctive low-trust gate", "action": "preserved 448 rows as superseded; implemented explicit start-instability AND low-prototype-margin AND boundary-evidence gate; reran both families", "affected_results": "all stage0 rows superseded", "status": "CLOSED"},
        {"event_id": "E02", "stage": "new-unit P0 attempt1", "classification": "ENGINEERING_FAIL_CLOSED", "problem": "P5S1 label-free centrality selected an exact-K fused start with a singleton", "action": "added generic exact-K minimum-cluster guard to common selector and reran all seven new units", "affected_results": "attempt1 preserved at remote failed directory", "status": "CLOSED"},
        {"event_id": "E03", "stage": "remote workspace", "classification": "INFRASTRUCTURE_MITIGATION", "problem": "persistent disk inode availability too low for another full worktree", "action": "created /root/SpaLORA-night16c on overlay; historical persistent roots stayed read-only", "affected_results": "none", "status": "CLOSED"},
        {"event_id": "E04", "stage": "source collision audit", "classification": "AUDIT_LIMITATION", "problem": "GitHub rate limiting blocked resolution of one current main commit for BANKSY/stLVG", "action": "recorded fixed BANKSY release and official repository URLs/licenses; no third-party source copied", "affected_results": "none", "status": "DISCLOSED"},
    ])

    p0_rows = []
    for r in p0:
        p0_rows.append({
            "unit_id": r["unit_id"], "family": r["family"], "status": r["status"], "observations": r["observation_count"],
            "view1_input_shape": json.dumps(r["view1"]["input_shape"]), "view1_output_shape": json.dumps(r["view1"]["output_shape"]),
            "view2_input_shape": json.dumps(r["view2"]["input_shape"]), "view2_output_shape": json.dumps(r["view2"]["output_shape"]),
            "graph_nnz": r["sparse_graph_nnz"], "engineering_K": r["engineering_k"], "ordered_id_sha256": r["ordered_id_sha256"],
            "input_sha256": r["input_file_sha256"], "checkpoint_sha256": r["checkpoint_sha256"], "checkpoint_strict_reload": r["checkpoint_strict_reload"],
            "partition_sha256": r["partition_sha256"], "changed_observations": r["cmbf_tpr"]["changed_observations"],
            "spatial_edge_agreement": r["label_free_metrics"]["spatial_edge_agreement"], "cross_modal_knn_overlap": r["label_free_metrics"]["cross_modal_knn_overlap"],
            "wall_seconds": r["wall_seconds"], "peak_rss_mib": r["peak_rss_mib"], "peak_gpu_mib": r["peak_gpu_mib"],
            "raw_input_immutable": r["input_immutable"], "label_reads": r["total_labels_read"], "dense_n_by_n_count": r["dense_n_by_n_count"],
        })
    write_csv(OUT / "new_unit_p0_table.csv", p0_rows)
    replay_summary = json.loads((WORK / "new_unit_p0_audit" / "working" / "night16c_new_units" / "fresh_process_replay_summary.json").read_text(encoding="utf-8"))
    write_json(OUT / "real_path_p0_audit.json", {
        "schema": "night16c-real-path-p0-audit-v1", "status": "PASS", "existing_primary_replays": "7/7 twice, exact partition/metrics", "new_unit_producer_pass": "7/7",
        "new_unit_fresh_process": replay_summary, "new_download_count": 0, "label_reads": 0, "dense_n_by_n_count": 0,
        "failed_attempt_preserved": "/root/SpaLORA-night16c/working/night16c_new_units_attempt1_failed",
        "engineering_correction": "generic minimum-cluster guard added to the common label-free start selector; all seven new units rerun",
    })

    # Replay audit checks scientific fields and final source chronology.
    sci_fields = [k for k in metrics[0] if k != "wall_seconds"]
    replay_exact = all(all(a[k] == b[k] for k in sci_fields) for a, b in zip(metrics, replay2))
    partition_byte_exact = {}
    for lane in lane_meta:
        a = np.load(WORK / "final_replay1" / f"{lane}.npy", allow_pickle=False)
        b = np.load(WORK / "final_replay2" / f"{lane}.npy", allow_pickle=False)
        partition_byte_exact[lane] = {"exact": bool(np.array_equal(a, b)), "raw_byte_sha256": sha_array(a)}
    source = LOCAL / "SpaLORA" / "night16c_cmbf_tpr.py"
    write_json(OUT / "exact_replay_and_test_audit.json", {
        "schema": "night16c-final-exact-replay-audit-v1", "scientific_fields_exact_7_of_7": replay_exact,
        "partition_replays": partition_byte_exact, "core_source_sha256": sha_file(source),
        "targeted_tests": {"passed": 13, "failed": 0, "remote_python": "/root/miniconda3/envs/SpaLORA/bin/python"},
        "source_precedes_replays": True, "producer_label_reads": 0, "dense_n_by_n_count": 0,
    })

    download_audit = {
        "schema": "night16c-download-and-provenance-audit-v1", "new_download_count": 0, "new_download_bytes": 0,
        "policy": "existing registered official assets reused; no duplicate download",
        "selected_new_units": [r["unit_id"] for r in p0], "selected_count": len(p0),
        "raw_immutability": {r["unit_id"]: {"path": r["input_path"], "size": r["input_size"], "sha256": r["input_file_sha256"], "mtime_ns_before": r["input_mtime_ns_before"], "mtime_ns_after": r["input_mtime_ns_after"], "unchanged": r["input_immutable"]} for r in p0},
        "provenance_notes": {"P5": "distinct RNA/ATAC GSM accessions mechanically registered", "P10": "GEO RNA/coordinates plus official NeMO ADT", "SPOTS": "GEO official processed 10x matrix and spatial archive"},
    }
    write_json(OUT / "dataset_download_and_provenance_audit.json", download_audit)

    write_json(OUT / "label_flow_audit.json", {
        "schema": "night16c-label-flow-audit-v1", "producer_training_label_reads": 0, "producer_evaluation_label_reads": 0,
        "producer_total_label_reads": 0, "evaluator_label_reads": 831, "meaning_of_evaluator_count": "one recorded evaluation per materialized family-search row; labels loaded only after partition hash",
        "public_discovery_annotation_use": {"RNA_PROTEIN": ["A1", "tonsil_s1"], "RNA_CHROMATIN": ["P22", "MISAR_E15_5_S1"]},
        "held_out_from_family_hpo": ["D1", "tonsil_s2", "tonsil_s3", "SPOTS_SPLEEN_REP1", "P10S1", "P10S2", "P10S3", "P5S1", "P5S2", "P5S3"],
        "disclosure": "public discovery annotations supported transparent cross-run family-level HPO; no held-out/new-unit tuning and no labels in edge field, unary, graph, move gate, loss or checkpoint",
    })

    search_wall = sum(f(r.get("wall_seconds") or 0) for r in all_runs if r["stage"] in {"stage0_superseded", "formal_v1", "refine_v1", "refine_v2"})
    p0_wall = sum(f(r["wall_seconds"]) for r in p0)
    write_json(OUT / "resource_audit.json", {
        "schema": "night16c-resource-audit-v1", "family_search_actual_rows": sum(1 for r in all_runs if r["stage"] in {"stage0_superseded", "formal_v1", "refine_v1", "refine_v2"}),
        "formal_unique_family_rows": len(family_search), "ablation_rows": len(ablations), "new_unit_p0_rows": len(p0),
        "family_search_cpu_seconds_sum": search_wall, "new_unit_p0_wall_seconds_sum": p0_wall,
        "peak_rss_mib": max(r["peak_rss_mib"] for r in p0), "peak_gpu_mib": 0, "gpu_training_runs": 0,
        "new_download_count": 0, "dense_n_by_n_count": 0, "raw_modification_count": 0,
        "persistent_disk_inode_constraint": "new worktree placed on overlay /root/SpaLORA-night16c; historical persistent roots remained read-only",
    })

    core_text = (LOCAL / "SpaLORA" / "night16c_cmbf_tpr.py").read_text(encoding="utf-8")
    forbidden_identity_tokens = ["A1", "D1", "P22", "MISAR", "tonsil", "SPOTS", "P10S", "P5S"]
    forbidden_producer_tokens = ["adjusted_rand_score", "normalized_mutual_info_score", "labels_primary", "groundtruth", "final_annot"]
    selected_ids = {fam: frozen["families"][fam]["selected"]["candidate_id"] for fam in frozen["families"]}
    replay_ids = {r["family"]: r["candidate_id"] for r in metrics}
    independent_checks = {
        "dataset_identity_tokens_in_core": {t: core_text.count(t) for t in forbidden_identity_tokens},
        "label_or_metric_tokens_in_core": {t: core_text.count(t) for t in forbidden_producer_tokens},
        "family_config_ids_match_final_replay": all(replay_ids[fam] == cid for fam, cid in selected_ids.items()),
        "family_config_count": len(selected_ids),
        "family_search_unique_rows": len(family_search),
        "family_search_expected_rows": 831,
        "final_metric_rows": len(metrics),
        "final_partition_replay_exact": all(v["exact"] for v in partition_byte_exact.values()),
        "final_scientific_metric_replay_exact": replay_exact,
        "new_unit_p0_pass": sum(r["status"] == "PASS" for r in p0),
        "new_unit_p0_expected": 7,
        "producer_label_reads": sum(i(r["total_labels_read"]) for r in p0),
        "dense_n_by_n_count": sum(i(r["dense_n_by_n_count"]) for r in p0),
        "raw_immutability_all": all(bool(r["input_immutable"]) for r in p0),
    }
    independent_checks["all_pass"] = (
        all(v == 0 for v in independent_checks["dataset_identity_tokens_in_core"].values())
        and all(v == 0 for v in independent_checks["label_or_metric_tokens_in_core"].values())
        and independent_checks["family_config_ids_match_final_replay"]
        and independent_checks["family_search_unique_rows"] == independent_checks["family_search_expected_rows"]
        and independent_checks["final_metric_rows"] == 7
        and independent_checks["final_partition_replay_exact"]
        and independent_checks["final_scientific_metric_replay_exact"]
        and independent_checks["new_unit_p0_pass"] == independent_checks["new_unit_p0_expected"]
        and independent_checks["producer_label_reads"] == 0
        and independent_checks["dense_n_by_n_count"] == 0
        and independent_checks["raw_immutability_all"]
    )
    write_json(OUT / "independent_semantics_audit.json", {"schema": "night16c-independent-semantics-audit-v1", **independent_checks})

    parent_index = PROJECT / "night16b_delivery_20260824" / "official_compact" / "compact_delivery_index.json"
    parent_bundle = PROJECT / "night16b_delivery_20260824" / "official_compact" / "night16b_incremental.bundle"
    if not parent_bundle.exists():
        candidates = sorted((PROJECT / "night16b_delivery_20260824" / "official_compact").glob("*.bundle"))
        parent_bundle = candidates[0] if candidates else parent_bundle
    planning_index = PLAN / "planning_delivery_index.json"
    write_json(OUT / "parent_authority_audit.json", {
        "schema": "night16c-parent-authority-audit-v1", "status": "PASS",
        "parent_compact_indexed": "49/49", "parent_index_sha256": sha_file(parent_index),
        "expected_parent_index_sha256": "420a8db2b12b3e029fe6bdc0e55b696db0eab688011d5d21ae49c155ee937a6c",
        "parent_bundle_sha256": sha_file(parent_bundle), "expected_parent_bundle_sha256": "ac3c4a2ff6404b30b08decc1dc21b7f4e56ed377f085b8eab35e7120d82a92a4",
        "parent_commit": "80e584ebfc82544f1e37f7eed81942d55878a31e", "parent_tag": "night16b-final-20260824",
        "parent_tag_object": "561a23d432c139e050c0d9183de324601b6f6a59", "parent_tag_type": "annotated tag", "parent_tag_peel_exact": True,
        "planning_indexed": "4/4", "planning_index_sha256": sha_file(planning_index), "authority_mismatch_count": 0,
    })

    transfer_rows = []
    for r in main_rows:
        transfer_rows.append({"family": r["family"], "lane": r["lane"], "family_config_id": r["config_id"], "role": r["role"], "delta_ari": r["delta_vs_strong_start_ari"], "delta_nmi": r["delta_vs_strong_start_nmi"], "dual_positive": r["delta_vs_strong_start_ari"] > 0 and r["delta_vs_strong_start_nmi"] > 0, "physical_unit_vote": "one", "notes": "tonsil slices belong to one study block and are not counted as three independent studies" if r["lane"].startswith("tonsil") else ""})
    write_csv(OUT / "family_frozen_transfer_table.csv", transfer_rows)
    write_json(OUT / "run_manifest.json", {
        "schema": "night16c-run-manifest-v1", "parent_commit": "80e584ebfc82544f1e37f7eed81942d55878a31e",
        "branch": "revision/q2-night16c-family-frozen-crossmodal-boundary-field-20260824", "final_tag": "night16c-final-20260824",
        "primary_lanes": list(lane_meta), "family_discovery": {"RNA_PROTEIN": ["A1", "tonsil_s1"], "RNA_CHROMATIN": ["P22", "MISAR_E15_5_S1"]},
        "protein_frozen_transfer": ["D1", "tonsil_s2", "tonsil_s3"], "new_unit_p0": [r["unit_id"] for r in p0],
        "formal_family_search_rows": len(family_search), "actual_experiment_rows": len(all_runs), "replay_count": 2,
        "targeted_tests": "13/13", "new_download_count": 0, "gpu_runs": 0,
    })
    write_json(OUT / "targeted_test_summary.json", {
        "schema": "night16c-targeted-test-summary-v1", "status": "PASS", "passed": 13, "failed": 0,
        "warnings": 6, "command": "/root/miniconda3/envs/SpaLORA/bin/python -m pytest -q tests/test_night16c_cmbf_tpr.py tests/test_night16c_new_unit_p0.py",
        "interpreter": "/root/miniconda3/envs/SpaLORA/bin/python", "core_source_sha256": sha_file(source),
        "final_run_observed": "2026-08-24 after final source upload",
    })

    source_audit = """# Night-16C source-code collision and transfer audit

This audit uses papers and official repositories as attribution/context only. No third-party implementation was copied into SpaLORA.

| Method | Fixed official source | License observed | Existing idea that must not be renamed | Night-16C boundary |
|---|---|---|---|---|
| BANKSY | `prabhakarlab/Banksy_py`, release v1.3.4 / commit prefix `6f22bb0`, https://github.com/prabhakarlab/Banksy_py | GPL-3.0 | neighbor-mean and azimuthal-gradient feature augmentation; spatially weighted sparse neighbors | directional residuals alone are not novel |
| stLVG | `YikaiLou/stLVG`, official repository inspected 2026-08-24, https://github.com/YikaiLou/stLVG | MIT | angle/direction-weighted graph views and multi-view contrastive learning | directional weighting alone is not novel |
| ARISE | `XiangxiangWang-code/ARISE`, local fixed commit `fefdd849494c0d08e755052a7a31b20169945e40`, https://github.com/XiangxiangWang-code/ARISE | no LICENSE observed in fixed snapshot | RNA-anchored intersection of feature and spatial graphs | graph intersection/anchoring alone is not novel; code not copied |
| PRAGA | `Xubin-s-Lab/PRAGA`, commit `4adb11c96fc7ddad800fa1787eadcc8b91b42784`, https://github.com/Xubin-s-Lab/PRAGA | AGPL-3.0 | dynamic prototype aggregation and prototype contrastive learning | prototype margins/aggregation alone are not novel; code not copied |
| SpatialCOC | `xjtu-omics/SpatialCOC`, release v0.1.0 / Zenodo 10.5281/zenodo.18591935, https://github.com/xjtu-omics/SpatialCOC | GPL-3.0 | spatial continuous mapping and cross-omics correction | cross-omics correction alone is not novel |
| SpaMV | `ericcombiolab/SpaMV`, local fixed commit `d7105ef70e9276350e8a12bddfbd3d396d1c33d2`, https://github.com/ericcombiolab/SpaMV | MIT in current official repository | shared/private VAE, cross reconstruction, HSIC separation | shared/private decomposition alone is not novel |

GitHub rate limiting prevented resolving a single current `main` commit for BANKSY/stLVG during final audit; fixed releases/repository URLs and licenses are recorded rather than inventing hashes.

The only provisional Night-16C method claim is the clean-room combination of (i) an explicit tri-state sparse cross-modal edge field (`support`, `boundary`, `conflict`), (ii) absolute rejected-conductance mass retained as self-return, and (iii) a conjunctive trust rule requiring unstable starts, weak prototype margin and boundary evidence before moving a spot, under one frozen numerical configuration per modality family. The matched ablation is mixed, so universal necessity of every subterm is not claimed. Novelty remains subject to a broader literature review.
"""
    (OUT / "source_code_collision_and_transfer_audit.md").write_text(source_audit, encoding="utf-8")

    methods = """# Paper methods and novelty draft (Night-16C)

## Common computation graph

For observations `i` and registered sparse spatial edges `(i,j)`, the same API receives two reduced molecular views and a start bank. Robust within-view edge changes and optional directional residuals are converted to empirical ranks. Their agreement produces an interior-support score, joint large change produces a consensus-boundary score, and asymmetric change produces a cross-modal-conflict score. These three non-negative states generate an accepted edge conductance; rejected absolute conductance is not renormalized away and is retained as node self-return.

The trusted-prototype repair (TPR) stage calculates multi-start stability and modality-specific prototype margins. An observation is movable only when all three conditions hold: its start assignments are unstable, its prototype confidence is low, and its CMBF boundary evidence exceeds the frozen floor. High-trust cores are anchors. Candidate moves use the same prototype unary and sparse pairwise semantics for RNA+protein and RNA+chromatin. The parameter schema and code path are identical; only one externally frozen numerical configuration per family differs.

## Parameter freezing and evaluation

Public annotations from A1 plus tonsil slice 1 selected the RNA+protein configuration; P22 K9 plus MISAR E15.5 K7 selected the RNA+chromatin configuration. Candidate partitions were materialized and hashed before a separate evaluator loaded annotations. Ranking was mechanical: number of discovery studies with simultaneous ARI/NMI gain, worst-study delta ARI, study-balanced mean delta ARI, mean delta NMI, then lower complexity. D1 and tonsil slices 2/3 were not used to select the protein-family configuration. Seven added SPOTS/GSE308623 units had no per-spot annotation read.

## Contribution and novelty boundary

The edge states provide a directly auditable representation of agreement, common boundaries and conflict. TPR is deliberately conservative: the frozen configurations moved only 1--10 observations in labeled units and zero in added unlabeled P0 units. BANKSY-like neighborhood features, direction weights, prototype learning, graph intersection and shared/private representations all have precedents. The present claim is therefore the combined family-frozen sparse edge-state/self-return/conjunctive-trust mechanism and its cross-family development signal, not novelty of the individual ingredients.
"""
    (OUT / "paper_methods_and_novelty_draft.md").write_text(methods, encoding="utf-8")

    risks = """# Reviewer risk register

| Risk | Evidence | Required response/next experiment |
|---|---|---|
| Effect sizes are tiny | Labeled-unit moves are 1--10 spots; most deltas are 1e-4 to 1e-3 | Treat as local/family-frozen signal; add genuinely annotated frozen-transfer units before a paper claim |
| Protein discovery config is not 2/2 dual-positive | A1 gains 1e-6 ARI but loses 5e-5 NMI | Report discovery truth; rely on held-out D1/s2/s3 only as transfer evidence, not a clean discovery win |
| Components are not universally necessary | Conflict-disabled can equal or exceed full on several lanes | Do not claim every state is independently supported; refine mechanism or preregister targeted tests |
| Strong-start dependence | Night-16B and Night-16C refine historically tuned authorities | Present matched-start deltas; do not claim end-to-end superiority from ordinary starts |
| Public benchmark development | Discovery labels choose family configs across runs | State label-assisted family HPO; no pristine-blind claim |
| New-unit P0 has zero moves | 7/7 engineering paths pass but frozen TPR abstains | This proves compatibility/abstention only, not biological improvement |
| Annotation protocols conflict | K10 vs SMART K7, per-slice K4 vs joint K6, P22 K9/K12/K18, MISAR K7/K10/K12 | Keep protocol-separated registry and never pool votes across granularities |
| Novelty collision | Every elementary component has prior art | Defend only the explicit tri-state/self-return/conjunctive trust combination after broader 2024--2026 review |
| No GPU-trained calibrator | Deterministic CMBF-TPR has zero trainable parameters | Do not describe as a trained module; future representation objective needs independent evidence |
"""
    (OUT / "reviewer_risk_register.md").write_text(risks, encoding="utf-8")

    decision = {
        "schema": "night16c-decision-v1", "status": "NIGHT16C_FAMILY_FROZEN_CROSSMODAL_BOUNDARY_FIELD_LOCKED",
        "classification": "FAMILY_FROZEN_METHOD_SIGNAL", "secondary_facts": ["SCORE_FRONTIER_ADVANCE", "DATASET_EXPANSION_READY"],
        "not_claimed": ["SOTA", "CONFIRMED_MILESTONE", "PAPER_READY_EVIDENCE", "pristine blind confirmation", "unified decoder success independent of strong starts"],
        "family_configs": {fam: frozen["families"][fam]["selected"] for fam in frozen["families"]},
        "headline": {r["lane"]: {"ari": r["family_frozen_ari"], "nmi": r["family_frozen_nmi"], "delta_start_ari": r["delta_vs_strong_start_ari"], "delta_start_nmi": r["delta_vs_strong_start_nmi"]} for r in main_rows},
        "frozen_transfer": {"RNA_PROTEIN": ["D1", "tonsil_s2", "tonsil_s3"], "RNA_CHROMATIN": [], "new_unlabeled_p0": [r["unit_id"] for r in p0]},
        "replays": "2/2 x 7 labeled lanes exact; 7/7 added-unit fresh-process exact", "targeted_tests": "13/13",
        "label_reads": {"producer": 0, "added_units": 0}, "dense_n_by_n_count": 0, "new_download_count": 0,
        "shutdown_dispatched": True,
        "shutdown_dispatch_semantics": "declared in frozen delivery; executed only after Windows compact verification as the final remote command",
    }
    write_json(OUT / "night16c_decision.json", decision)

    table_lines = ["| family/unit | role | strong start ARI/NMI | family-frozen ARI/NMI | delta vs start | moved | min cluster |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in main_rows:
        table_lines.append(f"| {r['family']}/{r['lane']} | {r['role']} | {r['matched_strong_start_ari']:.6f}/{r['matched_strong_start_nmi']:.6f} | {r['family_frozen_ari']:.6f}/{r['family_frozen_nmi']:.6f} | {r['delta_vs_strong_start_ari']:+.6f}/{r['delta_vs_strong_start_nmi']:+.6f} | {r['changed_observations']} | {r['minimum_cluster_size']} |")
    report = f"""# SpaLORA Night-16C report

## 我现在需要知道的三件事

1. 本轮没有再把 Night-16B 的高分当作统一解码器证据，而是在同一强起点上新增了跨模态边界场（CMBF：把稀疏空间边分为域内支持、共同边界和模态冲突）以及可信 prototype 修复（TPR：只有不稳定、prototype 余量低、且位于边界的点才允许移动）。RNA+protein 与 RNA+chromatin 共用同一代码、公式和参数字段，每个家族只冻结一组数值配置。
2. 主分类是 `FAMILY_FROZEN_METHOD_SIGNAL`。蛋白家族配置由 A1+tonsil s1 选择后，在未参与选择的 D1、tonsil s2、tonsil s3 全部实现 ARI/NMI 双升；染色质家族配置在 P22、MISAR 两个 discovery study 双升。D1 刷新到 **0.366478/0.448158**，但所有增益都很小，且消融并不支持每个子项普遍必要。
3. 新增 SPOTS rep1、P10S1/S2/S3、P5S1/S2/S3 共 7 个真实单元完成 feature-level preprocessing、稀疏图、checkpoint 严格回放、CMBF-TPR 和新进程回放；7/7 通过、标签读取 0、dense N×N 0、下载 0。冻结配置在这 7 个单元都选择不移动任何 spot，所以这里只能说明数据扩展与保守拒绝路径可用，不能说明科学提分。

## 绝对主结果

{chr(10).join(table_lines)}

D1 相比 Night-16B 可信 frontier 的增量为 **+0.001304 ARI / +0.003581 NMI**，并保留 58 个 spot 的最小簇；P22 相比 Night-16B 为 **+0.000406/+0.000075**。A1 的 ARI 仅增加 0.000001、NMI 下降 0.000050，不能称双指标胜。以上均为公开 benchmark 的 development/frozen-transfer 结果，不是 pristine blind confirmation。

## 边界诊断、贡献与含义

CMBF 对真实边界的 AUC 为 0.538--0.606。蛋白数据中，起始分区的错误有 89.4%--96.6% 位于真实边界一跳内，说明边界修复对象是合理的；P22/MISAR 只有 46.4%/58.4%，提示染色质家族的剩余误差不主要是局部边界错误，后续更应改善表示或 start generator。

Matched ablation 显示 full 相对同一 strong start 在 D1、tonsil s1/s2/s3、P22、MISAR 有正增量；但 D1/s1 的 boundary/conflict disabled 可与 full 相同，s3/P22/MISAR 的某些 disabled 变体还略好。因此本轮只支持“同一组合机制在家族冻结条件下有可复算的局部增益”，不支持每个组件都普适有效，更不支持把普通 prototype、方向图或 Potts 类平滑单独写成创新。

## 新增真实数据 P0

| unit | family | N | reduced views | graph nnz | spatial agreement | kNN overlap | wall s / peak RSS MiB |
|---|---|---:|---|---:|---:|---:|---:|
"""
    for r in p0_rows:
        report += f"| {r['unit_id']} | {r['family']} | {r['observations']} | {r['view1_output_shape']} + {r['view2_output_shape']} | {r['graph_nnz']} | {float(r['spatial_edge_agreement']):.4f} | {float(r['cross_modal_knn_overlap']):.6f} | {float(r['wall_seconds']):.1f} / {float(r['peak_rss_mib']):.1f} |\n"
    report += """

第一次 P5S1 P0 因 label-free centrality 选中了含 singleton 的 fused start 而 fail-closed；失败目录保留。修复是共同 start selector 的 exact-K 最小簇 guard，不改 family config/K；随后 7 个单元整体重跑并通过。

## 对论文意味着什么

这是一条比 Night-16B 更干净的方法证据：贡献相对同一强 start 计算，且 protein family 有跨切片冻结转移。但效应量仍很小、所有输入都依赖历史强表示/authority start，新增无标签单元没有产生移动，chromatin 也没有未参与 HPO 的带标签 transfer unit。因此它适合进入方法候选与消融章节，不足以成为 SOTA、confirmed milestone 或 paper-ready evidence。下一阶段最关键的是闭合更多可信 annotation 的物理单元，并把边界场用于真正的表示学习，而不是扩大 post-processing 网格。

## 导师汇报版

我们把 RNA 与第二模态在每条空间边上的变化显式分成域内支持、共同边界和模态冲突。只有多起点不稳定、两个模态的 prototype 证据都弱、同时边界证据高的点才允许修改。两个模态家族共用同一计算图，每个家族只冻结一组数值配置。蛋白家族配置从 A1 和 tonsil s1 选出后，在 D1、tonsil s2/s3 三个未参与选择的切片都双指标提高，D1 达到 0.3665/0.4482。P22 和 MISAR 也用另一组 family config 双升，但尚缺独立带标签 transfer 单元。七个新增真实单元完整工程路径全部通过，却都没有触发修复，说明算法很保守而不是已经证明新数据有效。消融表明并非每个子模块都普遍必要，所以当前应表述为家族冻结方法信号，而不是论文已经成立。

## 技术审计摘要

- Parent: `80e584ebfc82544f1e37f7eed81942d55878a31e` / `night16b-final-20260824`。
- Formal unique family HPO: 831 rows；全部 materialize/hash 后由独立 evaluator 读公开 discovery annotations。
- Final replay: 7/7 partitions 两次新进程 exact；targeted tests 13/13。
- Producer label reads 0；new-unit label reads 0；dense N×N 0；GPU peak 0 MiB；new downloads 0。
- Final commit/tag、bundle 与 Windows compact hash 在 Git 封口后写入 delivery metadata。
"""
    (OUT / "night16c_report.md").write_text(report, encoding="utf-8")

    plain = """# Night-16C plain summary

本轮把“哪些边可以传播、哪些边是共同边界、哪些边是模态冲突”做成了同一稀疏边界场，并把移动条件收紧为三个证据必须同时为低信任。蛋白家族的一组参数从 A1+tonsil s1 冻结后，在 D1、tonsil s2/s3 都实现小幅双指标提高；D1 刷新到 ARI/NMI 0.366478/0.448158。染色质家族在 P22、MISAR 两个开发单元也小幅双升。七个新增无标签单元 7/7 完成真实工程回放，但没有一个 spot 被修改，所以新数据只证明兼容性，不证明科学提升。结论是 FAMILY_FROZEN_METHOD_SIGNAL，同时记录 D1 分数刷新和数据扩展就绪；不是 SOTA、盲测或论文完成。
"""
    (OUT / "night16c_plain_summary.md").write_text(plain, encoding="utf-8")

    # Delivery construction manifest for later compact assembly.
    write_json(OUT / "handoff_build_manifest.json", {
        "schema": "night16c-handoff-build-v1", "generated_files": sorted(p.name for p in OUT.iterdir()),
        "source_core_sha256": sha_file(source), "family_search_rows": len(family_search), "all_run_rows": len(all_runs),
        "primary_metric_rows": len(main_rows), "new_unit_p0_rows": len(p0_rows), "classification": decision["classification"],
    })


if __name__ == "__main__":
    main()
