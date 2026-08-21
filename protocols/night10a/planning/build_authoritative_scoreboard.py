from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

METRICS = (
    "ari",
    "nmi",
    "q",
    "ami",
    "fmi",
    "homogeneity",
    "completeness",
    "v_measure",
    "neighbor_agreement",
    "moran_i",
    "geary_c",
    "boundary_disagreement",
)

FIELDS = (
    "data_family",
    "dataset",
    "method",
    "evidence_role",
    "seed_count",
    "seed_scope",
    *METRICS,
    "runtime_seconds",
    "peak_gpu_mib",
    "source_file",
    "notes",
)


def read_csv(relative: str) -> list[dict[str, str]]:
    path = ROOT / relative
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def mean(rows: list[dict[str, str]], column: str) -> float | None:
    values = [number(row.get(column)) for row in rows]
    present = [value for value in values if value is not None]
    return fmean(present) if present else None


def blank_row(**values: object) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in FIELDS}
    row.update(values)
    return row


rows_out: list[dict[str, object]] = []

# Night-7A is the authoritative four-dataset C00/C01 census.
n7a_rel = (
    "night7a_handoff_20260818/official_compact/handoff/"
    "four_dataset_summary.csv"
)
n7a = read_csv(n7a_rel)
for candidate, role, datasets in (
    ("C00_G04_H05_CONFIRMED", "retained_balanced_reference", {"a1", "d1", "tonsil", "p22"}),
    ("C01_G00_H05", "diagnostic_accuracy_frontier", {"d1", "p22"}),
):
    for source in n7a:
        if source["candidate_id"] != candidate or source["dataset"] not in datasets:
            continue
        rows_out.append(
            blank_row(
                data_family=(
                    "RNA+protein"
                    if source["dataset"] in {"a1", "d1", "tonsil"}
                    else "RNA+ATAC"
                ),
                dataset=source["dataset"],
                method=candidate,
                evidence_role=role,
                seed_count=int(source["successful_cells"]),
                seed_scope="Night-7A registered seeds",
                ari=number(source["mean_ari"]),
                nmi=number(source["mean_nmi"]),
                q=number(source["mean_q"]),
                neighbor_agreement=number(source["mean_neighbor_agreement"]),
                moran_i=number(source["mean_moran_i"]),
                geary_c=number(source["mean_geary_c"]),
                boundary_disagreement=number(source["mean_boundary_disagreement"]),
                source_file=n7a_rel,
                notes=(
                    "C01 is retained only as a dataset accuracy frontier; "
                    "it is not the unified final candidate."
                    if candidate == "C01_G00_H05"
                    else "Confirmed in Night-6D and preserved by Night-7A."
                ),
            )
        )

# Night-7B preserves the ten-seed P22 score frontier and exposes seed heterogeneity.
n7b_rel = (
    "night7b_handoff_20260818/official_compact/handoff/"
    "R2_final_summary_vs_C00.csv"
)
n7b = read_csv(n7b_rel)
for candidate_prefix, role in (
    ("R02__", "ten_seed_p22_accuracy_frontier"),
    ("R08__", "ten_seed_d1_p22_tradeoff_frontier"),
):
    source = next(
        row
        for row in n7b
        if row["config_id"].startswith(candidate_prefix)
        and row["config_id"].endswith("H01")
    )
    for dataset in ("a1", "d1", "tonsil", "p22"):
        rows_out.append(
            blank_row(
                data_family="RNA+protein" if dataset != "p22" else "RNA+ATAC",
                dataset=dataset,
                method=source["config_id"],
                evidence_role=role,
                seed_count=10 if dataset in {"d1", "p22"} else 5,
                seed_scope="Night-7B registered seeds",
                ari=number(source[f"{dataset}_mean_ari"]),
                nmi=number(source[f"{dataset}_mean_nmi"]),
                q=number(source[f"{dataset}_mean_q"]),
                source_file=n7b_rel,
                notes="Specialist/trade-off result; not a unified winner.",
            )
        )

# Night-9B five-seed family reference and hierarchical development clue.
n9b_rel = (
    "night9b_handoff_20260821/official_compact/outputs/night9b/"
    "r1_r2_per_seed_metrics.csv"
)
n9b = read_csv(n9b_rel)
for candidate, method_name, role in (
    ("REFERENCE", "FAMILY_REFERENCE_C00_OR_F00", "five_seed_family_reference"),
    ("N02_HIER_ONLY", "N02_HIER_ONLY", "five_seed_p22_development_frontier"),
):
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for source in n9b:
        if source["candidate_id"] == candidate:
            grouped[source["dataset"]].append(source)
    for dataset, group in sorted(grouped.items()):
        rows_out.append(
            blank_row(
                data_family="RNA+protein" if dataset == "a1" else "RNA+ATAC",
                dataset=dataset,
                method=method_name,
                evidence_role=role,
                seed_count=len(group),
                seed_scope="Night-9B seeds 0-4",
                ari=mean(group, "ari"),
                nmi=mean(group, "nmi"),
                q=mean(group, "q"),
                neighbor_agreement=mean(group, "neighbor_agreement"),
                moran_i=mean(group, "moran_i"),
                geary_c=mean(group, "geary_c"),
                boundary_disagreement=mean(group, "boundary_disagreement"),
                runtime_seconds=mean(group, "total_runtime_seconds"),
                source_file=n9b_rel,
                notes=(
                    "N02 improves P22 but failed the A1 protection gate; development clue only."
                    if candidate == "N02_HIER_ONLY"
                    else "C00 for A1 and F00/R02 for P22."
                ),
            )
        )

# Same-protocol COSMOS P22 calibration.
cosmos_rel = (
    "night9b_handoff_20260821/official_compact/outputs/night9b/"
    "cosmos_p22_per_seed_metrics.csv"
)
cosmos = read_csv(cosmos_rel)
cosmos_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
for source in cosmos:
    cosmos_groups[source["lane"]].append(source)
for lane, group in sorted(cosmos_groups.items()):
    rows_out.append(
        blank_row(
            data_family="RNA+ATAC",
            dataset="p22",
            method=lane,
            evidence_role=(
                "same_protocol_family_reference"
                if lane == "F00_R02_REFERENCE"
                else "same_protocol_external_baseline"
            ),
            seed_count=len(group),
            seed_scope="Night-9B seeds 0-4",
            ari=mean(group, "ari"),
            nmi=mean(group, "nmi"),
            q=mean(group, "q"),
            neighbor_agreement=mean(group, "neighbor_agreement"),
            moran_i=mean(group, "moran_i"),
            geary_c=mean(group, "geary_c"),
            boundary_disagreement=mean(group, "boundary_disagreement"),
            source_file=cosmos_rel,
            notes=(
                "Five-seed F00/R02 reference included in the COSMOS calibration."
                if lane == "F00_R02_REFERENCE"
                else "Fair same-input/K/seed protocol; not a paper-headline comparison."
            ),
        )
    )

# MISAR external evaluation has the broadest metric panel available locally.
n8b_rel = (
    "night8b_cardinality_safe_eval_handoff_20260820/official_compact/"
    "outputs/night8b_cardinality_safe_eval/cardinality_safe_20row_metrics.csv"
)
n8b = read_csv(n8b_rel)
misar_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
for source in n8b:
    misar_groups[source["method"]].append(source)
for method, group in sorted(misar_groups.items()):
    values = {metric: mean(group, metric) for metric in METRICS}
    rows_out.append(
        blank_row(
            data_family="RNA+ATAC",
            dataset="misar_e15_5_s1",
            method=method,
            evidence_role=(
                "post_lock_external_accuracy_reference"
                if method == "HR_F00"
                else "post_lock_external_baseline"
            ),
            seed_count=len(group),
            seed_scope="Night-8B seeds 0-9",
            **values,
            runtime_seconds=mean(group, "end_to_end_seconds"),
            peak_gpu_mib=mean(group, "peak_gpu_mib"),
            source_file=n8b_rel,
            notes=(
                "Accuracy/spatial improvement confirmed; runtime gate failed."
                if method == "HR_F00"
                else "Unified-head comparator."
            ),
        )
    )

scoreboard_path = OUT / "authoritative_scoreboard_20260821.csv"
with scoreboard_path.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows_out)

coverage_rows = [
    ("ARI", "label agreement", "yes", "yes", "clusters + labels", "retain primary"),
    ("NMI", "label agreement", "yes", "yes", "clusters + labels", "retain primary"),
    ("AMI", "label agreement", "no", "yes", "clusters + labels", "backfill all core results"),
    ("FMI", "label agreement", "no", "yes", "clusters + labels", "backfill all core results"),
    ("MI", "label agreement", "no", "no", "clusters + labels", "backfill all results"),
    ("V-measure", "label agreement", "no", "yes", "clusters + labels", "backfill all core results"),
    ("Homogeneity", "label agreement", "no", "yes", "clusters + labels", "backfill all core results"),
    ("Completeness", "label agreement", "no", "yes", "clusters + labels", "backfill all core results"),
    ("Silhouette", "embedding geometry", "no", "no", "embedding + predicted clusters", "compute with fixed metric and scaling"),
    ("Davies-Bouldin", "embedding geometry", "no", "no", "embedding + predicted clusters", "compute as secondary metric"),
    ("Calinski-Harabasz", "embedding geometry", "no", "no", "embedding + predicted clusters", "compute as secondary metric"),
    ("Neighbor agreement", "spatial continuity", "yes", "yes", "clusters + spatial graph", "retain"),
    ("Moran's I", "spatial continuity", "yes", "yes", "clusters + coordinates", "retain"),
    ("Geary's C", "spatial continuity", "yes", "yes", "clusters + coordinates", "retain"),
    ("Boundary disagreement", "spatial continuity", "yes", "yes", "clusters + spatial graph", "retain"),
    ("CHAOS/PAS", "spatial continuity", "no", "no", "clusters + coordinates", "add after definition parity test"),
    ("FOSCTTM", "cross-modal alignment", "no", "no", "paired modality embeddings", "backfill where private views exist"),
    ("Cross-modal retrieval", "cross-modal alignment", "no", "no", "paired modality embeddings", "add Recall@K and median rank"),
    ("Marker coherence", "biological validity", "no", "no", "raw modalities + clusters", "defer until candidate freeze"),
    ("Peak-gene coherence", "biological validity", "no", "no", "RNA/ATAC + clusters", "defer until candidate freeze"),
]
coverage_path = OUT / "metric_coverage_matrix_20260821.csv"
with coverage_path.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        ("metric", "category", "core_four_available", "misar_available", "requires", "next_action")
    )
    writer.writerows(coverage_rows)

manifest = {
    "schema": "spalora.night10a.local_scoreboard.v1",
    "row_count": len(rows_out),
    "scoreboard": scoreboard_path.name,
    "metric_coverage": coverage_path.name,
    "source_files": sorted({str(row["source_file"]) for row in rows_out}),
    "interpretation_constraints": [
        "Do not compare means across different seed scopes as if they were paired.",
        "N02 is a P22 development frontier, not a cross-platform locked winner.",
        "Night-7B R02 ten-seed results and Night-9B five-seed F00 results are different summaries of overlapping but non-identical seed sets.",
        "Missing current clusters/embeddings must be read from AutoDL persistent raw roots; do not retrain merely to backfill metrics.",
    ],
}
with (OUT / "scoreboard_manifest_20260821.json").open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")

print(json.dumps(manifest, ensure_ascii=False, indent=2))
