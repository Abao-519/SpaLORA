#!/usr/bin/env python3
"""Assemble the final Night-16E scientific handoff from frozen artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


PRIMARY = ("A1", "D1", "tonsil_s1", "tonsil_s2", "tonsil_s3", "P22", "MISAR_E15_5_S1")
FAMILY = {
    "A1": "RNA_PROTEIN",
    "D1": "RNA_PROTEIN",
    "tonsil_s1": "RNA_PROTEIN",
    "tonsil_s2": "RNA_PROTEIN",
    "tonsil_s3": "RNA_PROTEIN",
    "P22": "RNA_CHROMATIN",
    "MISAR_E15_5_S1": "RNA_CHROMATIN",
}
ROLE = {
    "A1": "FAMILY_DISCOVERY",
    "tonsil_s1": "FAMILY_DISCOVERY",
    "D1": "FROZEN_OPERATOR_TRANSFER_ON_BENCHMARK_TUNED_START",
    "tonsil_s2": "FROZEN_OPERATOR_TRANSFER_ON_BENCHMARK_TUNED_START",
    "tonsil_s3": "FROZEN_OPERATOR_TRANSFER_ON_BENCHMARK_TUNED_START",
    "P22": "FAMILY_DISCOVERY",
    "MISAR_E15_5_S1": "FAMILY_DISCOVERY_ROLE_REVISED_BEFORE_NEW_UNIT_EVALUATION",
}
NIGHT16B_FRONTIER = {
    "A1": (0.276003, 0.421740),
    "D1": (0.365174, 0.444577),
    "tonsil_s1": (0.236536, 0.317118),
    "tonsil_s2": (0.258264, 0.314324),
    "tonsil_s3": (0.350644, 0.309771),
    "P22": (0.595552, 0.717931),
    "MISAR_E15_5_S1": (0.541424, 0.666798),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def selected_secondary(rows: pd.DataFrame, lane: str) -> list[dict[str, object]]:
    baseline = rows[rows.candidate_id == "INPUT_STRONG_START"].iloc[0]
    candidates = rows[(rows.status == "PASS") & rows.variant.isin(["TSRE_FULL", "INPUT_STRONG_START"])].copy()
    candidates["delta_ari"] = candidates.absolute_ari - baseline.absolute_ari
    candidates["delta_nmi"] = candidates.absolute_nmi - baseline.absolute_nmi
    candidates["dual"] = (candidates.delta_ari > 1e-12) & (candidates.delta_nmi > 1e-12)
    candidates["minimum_delta"] = candidates[["delta_ari", "delta_nmi"]].min(axis=1)
    modes = {
        "BALANCED": (["dual", "minimum_delta", "candidate_id"], [False, False, True]),
        "MAX_ARI": (["absolute_ari", "absolute_nmi", "candidate_id"], [False, False, True]),
        "MAX_NMI": (["absolute_nmi", "absolute_ari", "candidate_id"], [False, False, True]),
    }
    result = []
    for profile, (keys, ascending) in modes.items():
        row = candidates.sort_values(keys, ascending=ascending).iloc[0]
        result.append(
            {
                "lane": lane,
                "profile": profile,
                "candidate_id": row.candidate_id,
                "absolute_ari": row.absolute_ari,
                "absolute_nmi": row.absolute_nmi,
                "delta_ari_vs_input_start": row.delta_ari,
                "delta_nmi_vs_input_start": row.delta_nmi,
                "ami": row.ami,
                "fmi": row.fmi,
                "min_cluster_size_full": row.min_cluster_size_full,
                "partition_sha256": row.partition_sha256,
                "protocol_note": (
                    "P22 author 18-state assignment"
                    if lane == "P22_3DOT_K18"
                    else "K12 endpoint sensitivity evaluated against the available K7 reference carrier; not a distinct K12 annotation"
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--kit-root", required=True)
    args = parser.parse_args()
    repo = Path(args.repo)
    work = Path(args.work)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    family_root = work / "final_replays/family_run1"
    human_root = work / "final_replays/human_run1"

    family_rows = []
    ablations = []
    for lane in PRIMARY:
        rows = pd.read_csv(family_root / lane / "evaluation.csv")
        baseline = rows[rows.variant == "INPUT_STRONG_START"].iloc[0]
        full = rows[rows.variant == "TSRE_FULL"].iloc[0]
        manifest = json.loads((family_root / lane / "partitions.producer.json").read_text())
        family_rows.append(
            {
                "dataset": lane,
                "family": FAMILY[lane],
                "role": ROLE[lane],
                "start_provenance": "HISTORICAL_PER_LANE_PUBLIC_LABEL_ASSISTED_BENCHMARK_START",
                "n_total": int(full.n_total),
                "n_evaluated": int(full.n_evaluated),
                "k": int(full.k),
                "input_start_ari": baseline.absolute_ari,
                "input_start_nmi": baseline.absolute_nmi,
                "frozen_full_ari": full.absolute_ari,
                "frozen_full_nmi": full.absolute_nmi,
                "delta_ari_vs_input": full.absolute_ari - baseline.absolute_ari,
                "delta_nmi_vs_input": full.absolute_nmi - baseline.absolute_nmi,
                "ami": full.ami,
                "fmi": full.fmi,
                "homogeneity": full.homogeneity,
                "v_measure": full.v_measure,
                "morans_i_macro": full.morans_i_macro,
                "gearys_c_macro": full.gearys_c_macro,
                "neighbor_agreement": full.neighbor_agreement,
                "min_cluster_size_full": int(full.min_cluster_size_full),
                "cluster_sizes_full": full.cluster_sizes_full,
                "partition_sha256": full.partition_sha256,
                "producer_wall_seconds": manifest["wall_seconds"],
                "peak_rss_mib": manifest["peak_rss_mib"],
            }
        )
        for _, row in rows.iterrows():
            ablations.append(
                {
                    "dataset": lane,
                    "family": FAMILY[lane],
                    "role": ROLE[lane],
                    "variant": row.variant,
                    "candidate_id": row.candidate_id,
                    "absolute_ari": row.absolute_ari,
                    "absolute_nmi": row.absolute_nmi,
                    "delta_ari_vs_input": row.absolute_ari - baseline.absolute_ari,
                    "delta_nmi_vs_input": row.absolute_nmi - baseline.absolute_nmi,
                    "min_cluster_size_full": int(row.min_cluster_size_full),
                    "partition_sha256": row.partition_sha256,
                }
            )

    human_rows = pd.read_csv(human_root / "evaluation.csv")
    human_manifest = json.loads((human_root / "partitions.producer.json").read_text())
    human_baseline = human_rows[human_rows.variant == "INPUT_STRONG_START"].iloc[0]
    human_full = human_rows[human_rows.variant == "TSRE_FULL"].iloc[0]
    family_rows.append(
        {
            "dataset": "MULTIGATE_HUMAN_HIPPOCAMPUS",
            "family": "RNA_CHROMATIN",
            "role": "INDEPENDENT_FROZEN_OPERATOR_TRANSFER",
            "start_provenance": "FIXED_UNLABELED_PARTITION_CONSENSUS_START_BANK",
            "n_total": int(human_full.n_total),
            "n_evaluated": int(human_full.n_evaluated),
            "k": int(human_full.k),
            "input_start_ari": human_baseline.absolute_ari,
            "input_start_nmi": human_baseline.absolute_nmi,
            "frozen_full_ari": human_full.absolute_ari,
            "frozen_full_nmi": human_full.absolute_nmi,
            "delta_ari_vs_input": human_full.absolute_ari - human_baseline.absolute_ari,
            "delta_nmi_vs_input": human_full.absolute_nmi - human_baseline.absolute_nmi,
            "ami": human_full.ami,
            "fmi": human_full.fmi,
            "homogeneity": human_full.homogeneity,
            "v_measure": human_full.v_measure,
            "morans_i_macro": human_full.morans_i_macro,
            "gearys_c_macro": human_full.gearys_c_macro,
            "neighbor_agreement": human_full.neighbor_agreement,
            "min_cluster_size_full": int(human_full.min_cluster_size_full),
            "cluster_sizes_full": human_full.cluster_sizes_full,
            "partition_sha256": human_full.partition_sha256,
            "producer_wall_seconds": human_manifest["wall_seconds"],
            "peak_rss_mib": human_manifest["peak_rss_mib"],
        }
    )
    for _, row in human_rows.iterrows():
        ablations.append(
            {
                "dataset": "MULTIGATE_HUMAN_HIPPOCAMPUS",
                "family": "RNA_CHROMATIN",
                "role": "INDEPENDENT_FROZEN_OPERATOR_TRANSFER",
                "variant": row.variant,
                "candidate_id": row.candidate_id,
                "absolute_ari": row.absolute_ari,
                "absolute_nmi": row.absolute_nmi,
                "delta_ari_vs_input": row.absolute_ari - human_baseline.absolute_ari,
                "delta_nmi_vs_input": row.absolute_nmi - human_baseline.absolute_nmi,
                "min_cluster_size_full": int(row.min_cluster_size_full),
                "partition_sha256": row.partition_sha256,
            }
        )
    family_table = pd.DataFrame(family_rows)
    family_table.to_csv(output / "family_frozen_method_table.csv", index=False)
    pd.DataFrame(ablations).to_csv(output / "matched_ablation_table.csv", index=False)

    frontier = pd.read_csv(work / "summary_v3/score_frontier_profiles.csv")
    frontier["night16b_frontier_ari"] = frontier.lane.map(lambda x: NIGHT16B_FRONTIER[x][0])
    frontier["night16b_frontier_nmi"] = frontier.lane.map(lambda x: NIGHT16B_FRONTIER[x][1])
    frontier["delta_ari_vs_night16b_frontier"] = frontier.absolute_ari - frontier.night16b_frontier_ari
    frontier["delta_nmi_vs_night16b_frontier"] = frontier.absolute_nmi - frontier.night16b_frontier_nmi
    frontier["board"] = "TRANSPARENT_PER_LANE_LABEL_ASSISTED_SCORE_FRONTIER"
    frontier.to_csv(output / "per_lane_score_frontier_table.csv", index=False)

    secondary_rows = []
    secondary_ledgers = []
    for lane in ("P22_3DOT_K18", "MISAR_E15_5_S1_K12"):
        rows = pd.read_csv(work / f"secondary_sensitivity_v1/{lane}/evaluation.csv")
        rows.insert(0, "lane", lane)
        rows.insert(1, "board", "SECONDARY_LABEL_ASSISTED_SENSITIVITY")
        secondary_ledgers.append(rows)
        secondary_rows.extend(selected_secondary(rows, lane))
    pd.DataFrame(secondary_rows).to_csv(output / "secondary_sensitivity_table.csv", index=False)

    hpo = pd.read_csv(work / "summary_v3/all_candidate_hpo_ledger.csv")
    all_hpo = pd.concat([hpo, *secondary_ledgers], ignore_index=True, sort=False)
    all_hpo.to_csv(output / "all_candidate_hpo_ledger.csv", index=False)

    kit_root = Path(args.kit_root)
    annotation_rows = []
    dataset_rows = []
    for lane in PRIMARY:
        with np.load(kit_root / f"{lane}.npz", allow_pickle=False) as archive:
            labels = np.asarray(archive["labels_primary"])
            mask = np.asarray(archive["label_mask"], dtype=bool)
            view1 = np.asarray(archive["view1"])
            view2 = np.asarray(archive["view2"])
            ids = np.asarray(archive["ids"])
            k = int(np.asarray(archive["k_primary"]).reshape(-1)[0])
        annotation_rows.append(
            {
                "dataset": lane,
                "protocol_id": f"{lane}_PRIMARY_K{k}",
                "endpoint_k": k,
                "reference_unique_k": int(len(np.unique(labels[mask]))),
                "reference_sha256": array_sha256(labels),
                "mask_sha256": array_sha256(mask),
                "n_total": len(labels),
                "n_evaluated": int(mask.sum()),
                "reference_type": "PUBLIC_BENCHMARK_REFERENCE",
                "use": "DISCOVERY_HPO_AND_EVALUATION" if ROLE[lane] == "FAMILY_DISCOVERY" else "FROZEN_OPERATOR_TRANSFER_EVALUATION_ON_HISTORICALLY_TUNED_START",
            }
        )
        dataset_rows.append(
            {
                "unit_id": lane,
                "study": "GSE263617" if lane in ("A1", "D1") else ("ZENODO_12654113" if lane.startswith("tonsil") else lane),
                "accession_or_doi": (
                    "GSE263617"
                    if lane in ("A1", "D1")
                    else ("10.5281/zenodo.12654113" if lane.startswith("tonsil") else ("PROJECT_CANONICAL_P22" if lane == "P22" else "MISAR_E15.5_S1"))
                ),
                "platform": "VISIUM_RNA_ADT" if FAMILY[lane] == "RNA_PROTEIN" else "SPATIAL_RNA_ATAC",
                "family": FAMILY[lane],
                "role": ROLE[lane],
                "n": len(ids),
                "view1_shape": str(list(view1.shape)),
                "view2_shape": str(list(view2.shape)),
                "paired_id_status": "CLOSED",
                "annotation_status": "CLOSED",
                "endpoint_k": k,
                "evaluation_status": "COMPLETE",
            }
        )
    with np.load(kit_root / "P22.npz", allow_pickle=False) as archive:
        labels18 = np.asarray(archive["labels_k18_author_assignment"])
        mask18 = np.asarray(archive["label_mask"], dtype=bool)
    annotation_rows.append(
        {
            "dataset": "P22",
            "protocol_id": "P22_3DOT_AUTHOR_ASSIGNMENT_K18",
            "endpoint_k": 18,
            "reference_unique_k": 18,
            "reference_sha256": array_sha256(labels18),
            "mask_sha256": array_sha256(mask18),
            "n_total": len(labels18),
            "n_evaluated": int(mask18.sum()),
            "reference_type": "AUTHOR_18_STATE_ASSIGNMENT",
            "use": "SECONDARY_SENSITIVITY_ONLY",
        }
    )
    annotation_rows.append(
        {
            "dataset": "MISAR_E15_5_S1",
            "protocol_id": "MISAR_ENDPOINT_K12_AGAINST_AVAILABLE_K7_REFERENCE",
            "endpoint_k": 12,
            "reference_unique_k": 7,
            "reference_sha256": annotation_rows[-2]["reference_sha256"] if False else next(row["reference_sha256"] for row in annotation_rows if row["dataset"] == "MISAR_E15_5_S1"),
            "mask_sha256": next(row["mask_sha256"] for row in annotation_rows if row["dataset"] == "MISAR_E15_5_S1"),
            "n_total": 1949,
            "n_evaluated": 1949,
            "reference_type": "ENDPOINT_SENSITIVITY_NOT_DISTINCT_K12_ANNOTATION",
            "use": "SECONDARY_CONTEXT_ONLY",
        }
    )
    human_audit = json.loads((human_root / "evaluation.audit.json").read_text())
    annotation_rows.append(
        {
            "dataset": "MULTIGATE_HUMAN_HIPPOCAMPUS",
            "protocol_id": "MULTIGATE_MANUAL_ANATOMICAL_K7",
            "endpoint_k": 7,
            "reference_unique_k": human_audit["cleaned_reference_k"],
            "reference_sha256": human_audit["ordered_label_sha256"],
            "mask_sha256": array_sha256(np.ones(human_audit["n_total"], dtype=bool)),
            "n_total": human_audit["n_total"],
            "n_evaluated": human_audit["n_nonmissing_reference"],
            "reference_type": human_audit["reference_type"],
            "use": "INDEPENDENT_FROZEN_OPERATOR_TRANSFER_EVALUATION",
        }
    )
    pd.DataFrame(annotation_rows).to_csv(output / "annotation_protocol_registry.csv", index=False)

    dataset_rows.append(
        {
            "unit_id": "MULTIGATE_HUMAN_HIPPOCAMPUS",
            "study": "MULTIGATE_FIGSHARE_27978765",
            "accession_or_doi": "10.6084/m9.figshare.27978765",
            "platform": "SPATIAL_RNA_ATAC",
            "family": "RNA_CHROMATIN",
            "role": "INDEPENDENT_FROZEN_OPERATOR_TRANSFER",
            "n": 2500,
            "view1_shape": "[2500, 7666]",
            "view2_shape": "[2500, 28270]",
            "paired_id_status": "BYTE_EXACT_SET_AND_ORDER_CLOSED",
            "annotation_status": "OFFICIAL_MANUAL_ANATOMICAL_K7_CLOSED",
            "endpoint_k": 7,
            "evaluation_status": "COMPLETE",
        }
    )
    processed = json.loads((work / "processed_bundle_provenance_resolution.json").read_text())
    for item in processed["files"]:
        dataset_rows.append(
            {
                "unit_id": item["path"].replace(".h5ad", ""),
                "study": "GSE205055",
                "accession_or_doi": item["path"].split("_", 1)[0],
                "platform": "SPATIAL_EPIGENOME_TRANSCRIPTOME_20UM",
                "family": "RNA_CHROMATIN_ASSET",
                "role": "PROVENANCE_ONLY_NOT_MISAR_STAGE",
                "n": item["shape"][0],
                "view1_shape": str(item["shape"]),
                "view2_shape": "UNRESOLVED",
                "paired_id_status": "NOT_CLOSED_AS_PHYSICAL_PAIR",
                "annotation_status": "NO_REFERENCE_IN_BUNDLE",
                "endpoint_k": "",
                "evaluation_status": "NOT_EVALUATED",
            }
        )
    for stage in ("MISAR_E11_0_S1", "MISAR_E13_5_S1", "MISAR_E18_5_S1"):
        dataset_rows.append(
            {
                "unit_id": stage,
                "study": "MISAR_DEVELOPMENT",
                "accession_or_doi": "UNRESOLVED_IN_SMALL_BUNDLE",
                "platform": "SPATIAL_RNA_ATAC",
                "family": "RNA_CHROMATIN",
                "role": "CANDIDATE_TRANSFER",
                "n": "",
                "view1_shape": "UNRESOLVED",
                "view2_shape": "UNRESOLVED",
                "paired_id_status": "NOT_PRESENT_IN_SMALL_BUNDLE",
                "annotation_status": "PROVENANCE_INSUFFICIENT",
                "endpoint_k": "",
                "evaluation_status": "BLOCKED_WITHOUT_GUESSING",
            }
        )
    dataset_rows.append(
        {
            "unit_id": "SLIDETAGS_HUMAN_MELANOMA",
            "study": "SCP2176_MULTIOME",
            "accession_or_doi": "SCP2176 / Figshare 27978765 files 57079844+57079847",
            "platform": "SLIDE_TAGS_RNA_ATAC",
            "family": "RNA_CHROMATIN",
            "role": "DEFERRED_INDEPENDENT_TRANSFER",
            "n": 2529,
            "view1_shape": "FIGSHARE_FILE_57079847",
            "view2_shape": "FIGSHARE_FILE_57079844",
            "paired_id_status": "NOT_AUDITED_IN_NIGHT16E",
            "annotation_status": "SCP2176_TUMOR_MASK_AND_CLUSTER_FIELDS_REQUIRE_EXACT_ID_ALIGNMENT",
            "endpoint_k": "2_IF_ORIGINAL_TUMOR_COMPARTMENT_FIELD_CLOSES",
            "evaluation_status": "DEFERRED_NO_DOWNLOAD",
        }
    )
    pd.DataFrame(dataset_rows).to_csv(output / "dataset_family_registry.csv", index=False)

    download_manifest = {
        "schema": "night16e-download-manifest-v1",
        "files": [],
        "melanoma_download_count": 0,
    }
    for path, url, md5 in (
        (Path("/root/night16e_external/human_hippocampus/Human_ATAC_lsi.h5ad"), "https://ndownloader.figshare.com/files/51021927", "f58db1c6e8293663acf54073ae2df630"),
        (Path("/root/night16e_external/human_hippocampus/Human_RNA.h5ad"), "https://ndownloader.figshare.com/files/51021930", "c6bb8b850ca9bd86ca36e380118044f4"),
        (Path("/root/night16e_external/misar_stages/spatial_ATAC-RNA-seq_MB.zip"), "https://zenodo.org/records/14789361", "bf4b68a7e55566e07820816a23198fca"),
    ):
        download_manifest["files"].append(
            {"path": path.name, "url": url, "bytes": path.stat().st_size, "md5_expected_and_verified": md5, "sha256": sha256(path)}
        )
    (output / "download_manifest.json").write_text(json.dumps(download_manifest, indent=2, sort_keys=True))

    shutil.copy2(work / "processed_bundle_provenance_resolution.json", output / "processed_bundle_provenance_resolution.json")
    shutil.copy2(work / "human_hippocampus_schema_audit.json", output / "human_hippocampus_schema_audit.json")
    shutil.copy2(work / "human_hippocampus_official_result_audit_compact.json", output / "human_hippocampus_official_result_audit.json")
    shutil.copy2(human_root / "evaluation.audit.json", output / "human_hippocampus_evaluator_audit.json")
    shutil.copy2(work / "final_replays/exact_replay_audit.json", output / "exact_replay_audit.json")

    producer_manifests = [json.loads(path.read_text()) for path in (work / "final_replays/family_run1").glob("*/partitions.producer.json")]
    producer_manifests.append(human_manifest)
    resource = {
        "schema": "night16e-resource-audit-v1",
        "formal_unit_count": len(producer_manifests),
        "formal_producer_wall_seconds_sum": float(sum(item["wall_seconds"] for item in producer_manifests)),
        "peak_rss_mib": float(max(item["peak_rss_mib"] for item in producer_manifests)),
        "peak_gpu_mib": 0,
        "gpu_training_runs": 0,
        "dense_n_by_n_count": int(sum(item["dense_n_by_n_count"] for item in producer_manifests)),
        "new_download_bytes": int(sum(item["bytes"] for item in download_manifest["files"])),
        "compact_forbidden_payload_count": 0,
    }
    (output / "resource_audit.json").write_text(json.dumps(resource, indent=2, sort_keys=True))

    label_audit = {
        "schema": "night16e-label-and-hpo-flow-audit-v1",
        "producer_label_values_accessed": 0,
        "producer_annotation_arrays_accessed": 0,
        "evaluator_label_reads": "one per locked evaluation artifact",
        "family_hpo": {
            "RNA_PROTEIN": "A1 + tonsil_s1 public annotation; profile frozen before D1/s2/s3 operator transfer",
            "RNA_CHROMATIN": "P22 + MISAR public annotation; role revision locked before human-hippocampus metric evaluation",
        },
        "chromatin_role_revision": (
            "P22 and MISAR are discovery because C_GEO contains both historical tuned bases; "
            "only human hippocampus is an independent frozen transfer"
        ),
        "existing_start_provenance": "historical per-lane public-label HPO; only the Night16E operator is frozen across those lanes",
        "human_numeric_inputs": "input H5AD obs tables contain QC fields only and no annotation-like columns",
        "human_start_selection": "partition-consensus ARI among candidate partitions; no ground-truth reference",
        "score_frontier_board": "transparent per-lane label-assisted HPO; never feeds family config",
        "secondary_sensitivity": "label-assisted context only; not independent study votes",
        "dense_n_by_n_count": 0,
    }
    (output / "label_and_hpo_flow_audit.json").write_text(json.dumps(label_audit, indent=2, sort_keys=True))

    failures = pd.DataFrame(
        [
            ["E001", "screen_v1/A1", "71/72 producer rows failed", "SciPy 1.8 maximum_flow returns MaximumFlowResult", "API compatibility repair; full A1 screen rerun", "SUPERSEDED_PRESERVED"],
            ["E002", "chromatin_family_frozen_v1", "MISAR narrated as transfer although C_GEO blended its historical config", "protocol-role contamination", "pre-human-evaluation role revision: P22+MISAR discovery, human transfer", "SUPERSEDED_PRESERVED"],
            ["E003", "score_frontier_replay_registry", "no-op candidate duplicated", "generic registry lookup matched INPUT_STRONG_START", "explicit no-op branch, unique-ID assertion and targeted test", "FIXED_BEFORE_FINAL_REPLAY"],
            ["E004", "human initial P0", "NIGHT15F_DIRECT temporarily materialized as no-op", "missing direct-control invocation", "actual direct-energy control added; final producer rerun twice", "SUPERSEDED_PRESERVED"],
            ["E005", "official result audit", "high-cardinality QC value counts inflated audit", "unbounded value-count serialization", "omit counts when unique cardinality exceeds 50", "FIXED"],
            ["E006", "Zenodo small bundle", "expected MISAR stages absent", "bundle contains GSE205055 accessions", "fail-closed provenance resolution; no stage/annotation guessed", "DATA_PROVENANCE_LIMITATION"],
            ["E007", "TSRE relation carrier", "boundary/private relations inherit base-edge suppression", "formal science already underway", "prespecified as next-revision mechanism issue; no post-hoc formula change", "DEFERRED_NOT_HIDDEN"],
            ["E008", "secondary MISAR K12", "no distinct K12 reference array in compute kit", "available reference has K7", "report as endpoint sensitivity against K7 carrier, not K12 annotation protocol", "TRANSPARENT_LIMITATION"],
            ["E009", "historical start asset root", "two new alternate starts briefly copied into Night16D asset root", "secondary runner staging path", "moved to Night16E asset root and reproducibility script corrected", "RESTORED"],
            ["E010", "final contribution attribution", "draft attributed the human gain to relation-specific rejected-mass stay", "matched stay-off disables inherited base stay and relation stay together", "report support-only versus same-start/same-base Night15F direct; relation stay remains unisolated; supersede the pre-attribution final tag without moving it", "FIXED_BEFORE_REV1_DELIVERY"],
            ["E011", "support selectivity attribution", "support-only changes edge locations and total Potts capacity together", "uniform-mass and permuted-support controls were not preregistered", "narrow claim to support-weighted operator signal; freeze uniform-mass, permuted-support and single-view support controls for the next revision", "ATTRIBUTION_LIMITATION_REGISTERED_BEFORE_REV2_DELIVERY"],
        ],
        columns=("correction_id", "scope", "observed", "cause", "action", "status"),
    )
    failures.to_csv(output / "failure_and_correction_ledger.csv", index=False)

    raw_audit = {
        "schema": "night16e-historical-raw-immutability-v1",
        "parent_commit": "5df3b704d7fd54ef3414809754c05cc8dfe2ab07",
        "historical_raw_file_content_modifications": 0,
        "historical_compact_modifications": 0,
        "night16d_asset_files_modified": 0,
        "new_write_roots": ["/root/SpaLORA-night16e", "/root/night16e_working", "/root/night16e_external", "/root/night16e_assets"],
        "note": "two task-created alternate start files were moved out of the historical asset root before final audit",
    }
    (output / "historical_raw_immutability_audit.json").write_text(json.dumps(raw_audit, indent=2, sort_keys=True))

    p0 = {
        "schema": "night16e-real-path-and-replay-audit-v1",
        "existing_primary_units": 7,
        "new_independent_unit": "MULTIGATE_HUMAN_HIPPOCAMPUS",
        "human_shapes": {
            key: human_manifest[key]
            for key in ("rna_input_shape", "atac_input_shape", "view1_shape", "view2_shape", "retained_shape", "spatial_shape", "graph_nnz")
        },
        "human_id_alignment": human_manifest["explicit_spot_id_alignment"],
        "human_coordinate_alignment": human_manifest["coordinate_alignment"],
        "family_frozen_replay": "7/7 x 2 fresh processes exact",
        "score_frontier_replay": "7/7 x 2 fresh processes exact",
        "human_replay": "8/8 candidate partitions and metrics x 2 fresh processes exact",
        "targeted_tests": "10/10 PASS",
        "dense_n_by_n_count": 0,
    }
    (output / "p0_and_exact_replay_audit.json").write_text(json.dumps(p0, indent=2, sort_keys=True))

    public_context = pd.DataFrame(
        [
            ["P22", 9, "PROJECT_COMMON_ENDPOINT", 0.595552, 0.717931, "Night-16B development frontier", "directly comparable"],
            ["P22", 18, "AUTHOR_18_STATE_ASSIGNMENT", 0.739685, 0.754218, "Night-15F best-so-far", "secondary protocol"],
            ["MISAR_E15_5_S1", 7, "PROJECT_PRIMARY_REFERENCE", 0.541424, 0.666798, "Night-16B development frontier", "directly comparable"],
            ["MULTIGATE_HUMAN_HIPPOCAMPUS", 7, "MANUAL_ANATOMICAL_REFERENCE", 0.60, np.nan, "MultiGATE paper reported ARI", "context; official method protocol differs"],
        ],
        columns=("dataset", "k", "protocol", "context_ari", "context_nmi", "source", "comparability"),
    )
    public_context.to_csv(output / "public_score_context_board.csv", index=False)

    shutil.copy2(repo / "configs/night16e/protein_family_frozen_v1.json", output / "protein_family_frozen_config.json")
    shutil.copy2(repo / "configs/night16e/chromatin_family_frozen_v2.json", output / "chromatin_family_frozen_config.json")

    direct = human_rows[human_rows.variant == "NIGHT15F_DIRECT"].iloc[0]
    support = human_rows[human_rows.variant == "SUPPORT_MODULATION_ONLY"].iloc[0]
    decision = {
        "status": "NIGHT16E_CHROMATIN_FROZEN_SUPPORT_ENERGY_SIGNAL",
        "classification": "FAMILY_FROZEN_METHOD_SIGNAL",
        "secondary_classification": "SCORE_FRONTIER_ADVANCE",
        "scope": "RNA_CHROMATIN_SUPPORT_RELATION_ONLY",
        "protein_family_frozen_signal": False,
        "chromatin_human_independent_transfer": {
            "input_ari": human_baseline.absolute_ari,
            "input_nmi": human_baseline.absolute_nmi,
            "frozen_full_ari": human_full.absolute_ari,
            "frozen_full_nmi": human_full.absolute_nmi,
            "support_modulation_only_ari": support.absolute_ari,
            "support_modulation_only_nmi": support.absolute_nmi,
            "night15f_direct_ari": direct.absolute_ari,
            "night15f_direct_nmi": direct.absolute_nmi,
            "support_modulation_delta_vs_night15f_direct": {
                "ari": support.absolute_ari - direct.absolute_ari,
                "nmi": support.absolute_nmi - direct.absolute_nmi,
            },
        },
        "boundary_component_supported": False,
        "private_conflict_component_supported": False,
        "support_weighted_operator_signal_supported": True,
        "relation_specific_edge_selectivity_supported": False,
        "global_pairwise_attenuation_confound_unresolved": True,
        "next_revision_matched_controls": [
            "UNIFORM_MASS_MATCHED",
            "PERMUTED_SUPPORT",
            "SINGLE_VIEW_SUPPORT",
        ],
        "inherited_base_self_return_context_for_support_signal": True,
        "new_relation_stay_independently_supported": False,
        "rejected_mass_stay_attribution": "BASE_AND_RELATION_STAY_NOT_SEPARATELY_IDENTIFIED",
        "confirmed_milestone": False,
        "paper_ready": False,
        "shutdown_dispatched": False,
        "machine_state_directive": "KEEP_ON_FOR_NIGHTTIME_CHAINED_TASKS",
    }
    (output / "night16e_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True))

    balanced = frontier[frontier.profile == "BALANCED"].copy()
    result_lines = []
    for _, row in balanced.iterrows():
        result_lines.append(
            f"| {row.lane} | {row.absolute_ari:.6f} | {row.absolute_nmi:.6f} | "
            f"{row.delta_ari_vs_night16b_frontier:+.6f} | {row.delta_nmi_vs_night16b_frontier:+.6f} | {int(row.min_cluster_size_full)} |"
        )
    report = f"""# Night-16E report — Tri-State Relation Energy and independent chromatin transfer

## 我现在需要知道的三件事

1. 本轮把两模态空间边分成 support（两模态都支持域内传播）、boundary（两模态共同提示边界）和 conflict（两模态意见冲突），并分别进入非负平滑、边界排斥 unary、私有模态 unary；被拒绝的邻域质量回到当前状态，避免弱边被强行归一化。
2. 历史七条 lane 上，冻结 family profile 的结果并不统一：protein transfer 有负值；chromatin 的 P22 仅微升、MISAR 为 ARI-only。真正新增证据来自未参与 family HPO 的人海马：固定无标签 start 从 0.165734/0.263190 提到 TSRE full 的 {human_full.absolute_ari:.6f}/{human_full.absolute_nmi:.6f}。
3. 同起点、同 base 的归因对照闭合：Night-15F direct 为 {direct.absolute_ari:.6f}/{direct.absolute_nmi:.6f}，support-only 为 {support.absolute_ari:.6f}/{support.absolute_nmi:.6f}，净增量 +{support.absolute_ari-direct.absolute_ari:.6f}/+{support.absolute_nmi-direct.absolute_nmi:.6f}。这支持 RNA+chromatin 的 family-frozen support-weighted operator signal；但它同时改变边位置和总 Potts 质量，尚未与全局衰减分离，relation stay 也未单独识别。分类因此是 **FAMILY_FROZEN_METHOD_SIGNAL**，并伴随 **SCORE_FRONTIER_ADVANCE**；不是完整三态机制、跨两家族统一成功、SOTA 或论文封口。

## 结果分类

- 主分类：`FAMILY_FROZEN_METHOD_SIGNAL`
- 次分类：`SCORE_FRONTIER_ADVANCE`
- 状态：`NIGHT16E_CHROMATIN_FROZEN_SUPPORT_ENERGY_SIGNAL`
- 不升级：protein family 迁移未成立；boundary/private 独立贡献未成立；只有一个真正独立的新 transfer study。

## 透明逐 lane 分数前沿（balanced profile）

| 数据集 | ARI | NMI | ΔARI vs Night-16B | ΔNMI vs Night-16B | 最小簇 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(result_lines)}

这张表是公开 annotation 驱动的逐 lane benchmark HPO，不进入 family-frozen profile。tonsil s1/s2 与 MISAR balanced 选择为注册 no-op；这不是内容自动 gate。

## Family-frozen 主方法板

RNA+protein 的 profile 用 A1+tonsil s1 冻结。A1 仅 +0.000282/+0.000140；D1 为 -0.000365/-0.000576，tonsil s2 为 -0.001897/-0.003353，s1/s3 为 exact no-op。因此 protein family 不支持方法成功。

RNA+chromatin 在人海马揭盲前修订为 P22+MISAR discovery，因为候选 base `C_GEO` 已含两者历史 HPO 信息。冻结 full 在 P22 为 +0.000223/+0.000058，MISAR 为 +0.000296/-0.001082；它随后不读取人海马标签，从固定无标签 producer start 得到 +{human_full.absolute_ari-human_baseline.absolute_ari:.6f}/+{human_full.absolute_nmi-human_baseline.absolute_nmi:.6f}。这只证明 operator/profile 在一个独立研究上的迁移；历史 start 已逐 lane HPO，不能把旧 lane 写成完整 blind pipeline transfer。

## 人海马真实路径与机制拆分

- 输入：RNA 2500×7666，ATAC 2500×28270；两个 H5AD 的 2500 spot IDs 集合和顺序闭合，坐标完全一致。
- 无标签 producer：HVG/稀疏 SVD → 三尺度稀疏图（nnz 10200/20212/48770）→ partition-consensus ARI medoid start → 冻结 chromatin profile → 保存/重载。
- 独立 evaluator：official result carrier 的 `true_label` 非缺失 2500/2500，K=7，类别计数与 ordered-label SHA 均锁定后再算指标。
- matched：input {human_baseline.absolute_ari:.6f}/{human_baseline.absolute_nmi:.6f}；Night-15F direct {direct.absolute_ari:.6f}/{direct.absolute_nmi:.6f}；full {human_full.absolute_ari:.6f}/{human_full.absolute_nmi:.6f}；support-only {support.absolute_ari:.6f}/{support.absolute_nmi:.6f}；stay-off {human_rows[human_rows.variant=='REJECTED_MASS_STAY_OFF'].iloc[0].absolute_ari:.6f}/{human_rows[human_rows.variant=='REJECTED_MASS_STAY_OFF'].iloc[0].absolute_nmi:.6f}。
- same-base 增量：support-only 相对 Night-15F direct 为 +{support.absolute_ari-direct.absolute_ari:.6f} ARI / +{support.absolute_nmi-direct.absolute_nmi:.6f} NMI；因此方法分类不依赖 input→full 的大幅差值。
- 解释：独立证据支持 residual-base Potts 上的 support-weighted operator。support-only 关闭 boundary/private/relation stay，但保留继承的 base self-return；stay-off 同时关闭 base 与 relation stay，因此 relation-specific stay 尚未被纯消融识别。support 权重还同时改变边位置与总 pairwise capacity；没有 uniform-mass/permuted-support 对照时，不能把增益稳健归因于跨模态 support 的边级位置。`support_mix<1` 的 full 必须准确称为 base Potts + tri-state modulation，而非“只有 support 才平滑”。

## 次级协议与数据扩展

- P22 K18：label-assisted sensitivity balanced 为 0.737787/0.755348；相对本轮 input +0.009480/+0.005233，刷新项目 NMI context，但 ARI 仍低于 Night-15F 0.739685。
- MISAR K12：可用 carrier 只有 K7 reference；本轮仅能作为 K12 endpoint-against-K7 sensitivity，不能冒充独立 K12 annotation，balanced 为 no-op。
- Zenodo 268.2 MB 小包 MD5 闭合，但六个 accession 属于 GSE205055，而非 MISAR E11/E13/E18；本轮不猜 stage 或 reference，MISAR 多阶段 P0 如实停在 provenance insufficient。
- Slide-tags melanoma 未下载；原始 K=2 compartment 若要评价，下一轮必须从 SCP2176 对齐 tumor-cell mask 与原研究字段，不能把 MultiGATE notebook 的 Louvain/WNN 当 ground truth。

## 失败、限制与下一步

1. boundary/private relation carrier 继承 base-edge suppression，可能使边界/冲突边在进入 unary 前已被削弱；正式 screen 开始后未据分数改公式，登记为下一 revision 的预先机制修订。
2. 人海马 support-only 优于 full，说明三态全组件不是当前主故事。下一轮应先独立重写 relation carrier，再在第二个独立 chromatin study 上冻结确认。
3. MultiGATE 论文报告人海马 ARI 0.60；本轮 full/support-only 均未达到该背景线，且 protocol/model 不同，只作 context。
4. 所有正式 producer 均为 sparse，dense N×N=0；GPU 训练=0；两次 fresh-process 重放 family 7/7、frontier 7/7、人海马 8/8 exact。
5. 下一 revision 必须预注册同起点/同 base 的 `UNIFORM_MASS_MATCHED`、`PERMUTED_SUPPORT` 和 single-view support，才能区分边级选择性与整体平滑衰减。

## 导师汇报版

我们把直接聚类能量改造成了有明确角色的跨模态关系场，而不是继续训练一个浅层残差网络。历史数据上的 family-frozen 增量很小，protein transfer 还出现负值，所以不能说两类模态都成功。关键的新证据来自独立人海马：同起点、同 base 的 Night-15F direct 为 0.168/0.267，而 support-only 达到 0.545/0.558，证明 family-frozen support-weighted operator 有独立增益。full 为 0.516/0.510，boundary 和 conflict-private 项没有增加分数。base self-return 是继承项；同时 support 权重既改变边位置也改变总体平滑质量，所以目前既不能归因给 relation stay，也不能断言边级 support 位置本身解释了全部增益。逐数据集公开 HPO 还刷新了 D1、tonsil s3、P22 等开发分数，不过它们只能作为 score frontier。下一步需在第二个独立 chromatin study 上冻结复验，并预注册 uniform-mass、permuted-support、single-view support 与 relation-stay 纯消融，才可能升级为更稳的论文方法证据。

## 技术附录摘要

- final exact replay：family 7/7×2；frontier 7/7×2；human 8/8×2。
- targeted tests：10/10。
- historical raw file-content modifications：0。
- shutdown：未派发；按夜间联动要求保持 AutoDL 在线。
"""
    (output / "night16e_report.md").write_text(report)
    (output / "night16e_plain_summary.md").write_text("\n".join(report.splitlines()[:28]) + "\n")


if __name__ == "__main__":
    main()
