#!/usr/bin/env python3
"""Explore additional generic heads on molecular+morphology+coordinate blocks.

The same numeric construction is available to every lane with an image.
Dataset labels are evaluated only after each complete partition and are used
for public-benchmark cross-run HPO.  No label enters a representation or head.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15e_continuous_reliability_energy import standardize  # noqa: E402
from scripts.night15g.night15g_optional_view_search import (  # noqa: E402
    array_sha256,
    evaluate,
    feature_variants,
    load_lane,
    reduce_feature,
    selection_key,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--morphology", type=Path, required=True)
    parser.add_argument("--night15f-registry", type=Path, required=True)
    parser.add_argument("--lane", default="D1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "partitions").mkdir(exist_ok=True)

    data, graph, _, retained, authority, labels, mask, k, morph = load_lane(args, args.lane)
    authority_metrics = evaluate(labels, mask, authority, graph)
    variants = feature_variants(morph)
    retained_block = reduce_feature(retained, min(64, retained.shape[1]))
    retained_block = retained_block / math.sqrt(max(retained_block.shape[1], 1))
    coordinates = standardize(np.asarray(data["coordinates"], dtype=np.float32))
    coordinates = coordinates / math.sqrt(max(coordinates.shape[1], 1))

    rows: list[dict] = []
    best: dict[str, tuple[dict, np.ndarray]] = {}

    def consider(row, partition):
        keys = {
            "balanced": selection_key,
            "max_ari": lambda value: (float(value["absolute_ari"]), float(value["absolute_nmi"])),
            "max_nmi": lambda value: (float(value["absolute_nmi"]), float(value["absolute_ari"])),
        }
        for profile, key in keys.items():
            if profile not in best or key(row) > key(best[profile][0]):
                best[profile] = (row, partition.copy())

    selected_variants = [
        name
        for name in variants
        if name in {
            "HANDCRAFTED_D32", "HANDCRAFTED_D64",
            "RESNET18_D32", "RESNET18_D64",
            "COMBINED_D32", "COMBINED_D64",
        }
    ]
    for variant_name in selected_variants:
        morphology = variants[variant_name]
        morphology = morphology / math.sqrt(max(morphology.shape[1], 1))
        for morphology_weight in (0.10, 0.20, 0.30, 0.50, 1.00):
            for coordinate_weight in (0.0, 0.03, 0.10, 0.30):
                direct = np.concatenate(
                    (
                        retained_block,
                        morphology_weight * morphology,
                        coordinate_weight * coordinates,
                    ),
                    axis=1,
                ).astype(np.float32)
                pca = PCA(
                    n_components=min(64, direct.shape[1], direct.shape[0] - 1),
                    svd_solver="full",
                ).fit_transform(direct).astype(np.float32)
                for representation_name, representation in (("DIRECT", direct), ("PCA64", pca)):
                    specifications = []
                    for seed in range(5):
                        specifications.extend(
                            [
                                (
                                    "KMEANS",
                                    seed,
                                    lambda seed=seed: KMeans(
                                        n_clusters=k,
                                        random_state=seed,
                                        n_init=30,
                                    ).fit_predict(representation),
                                ),
                                (
                                    "BISECTING_KMEANS",
                                    seed,
                                    lambda seed=seed: BisectingKMeans(
                                        n_clusters=k,
                                        random_state=seed,
                                        n_init=10,
                                    ).fit_predict(representation),
                                ),
                            ]
                        )
                    for covariance in ("diag", "tied", "spherical", "full"):
                        for seed in range(3):
                            specifications.append(
                                (
                                    f"GMM_{covariance.upper()}",
                                    seed,
                                    lambda covariance=covariance, seed=seed: GaussianMixture(
                                        n_components=k,
                                        covariance_type=covariance,
                                        random_state=seed,
                                        n_init=3,
                                        max_iter=400,
                                        reg_covar=1e-5,
                                    ).fit_predict(representation),
                                )
                            )
                    for algorithm, seed, runner in specifications:
                        started = time.perf_counter()
                        identifier = (
                            f"{variant_name}__MW{morphology_weight:g}__CW{coordinate_weight:g}"
                            f"__{representation_name}__{algorithm}__S{seed}"
                        )
                        try:
                            partition = np.asarray(runner(), dtype=np.int32)
                            metrics = evaluate(labels, mask, partition, graph)
                            sizes = np.bincount(partition, minlength=k)
                            row = {
                                "lane": args.lane,
                                "config_id": identifier,
                                "feature_variant": variant_name,
                                "morphology_weight": morphology_weight,
                                "coordinate_weight": coordinate_weight,
                                "representation": representation_name,
                                "algorithm": algorithm,
                                "seed": seed,
                                "status": "PASS",
                                "failure": "",
                                "partition_sha256": array_sha256(partition),
                                "absolute_ari": metrics["absolute_ari"],
                                "absolute_nmi": metrics["absolute_nmi"],
                                "delta_ari": metrics["absolute_ari"] - authority_metrics["absolute_ari"],
                                "delta_nmi": metrics["absolute_nmi"] - authority_metrics["absolute_nmi"],
                                "min_cluster_size": int(sizes.min()),
                                "cluster_sizes": json.dumps(sizes.astype(int).tolist(), separators=(",", ":")),
                                "wall_seconds": time.perf_counter() - started,
                                **{key: value for key, value in metrics.items() if key not in {"absolute_ari", "absolute_nmi"}},
                            }
                            rows.append(row)
                            consider(row, partition)
                        except Exception as error:
                            rows.append(
                                {
                                    "lane": args.lane,
                                    "config_id": identifier,
                                    "feature_variant": variant_name,
                                    "morphology_weight": morphology_weight,
                                    "coordinate_weight": coordinate_weight,
                                    "representation": representation_name,
                                    "algorithm": algorithm,
                                    "seed": seed,
                                    "status": "FAILED",
                                    "failure": repr(error),
                                    "wall_seconds": time.perf_counter() - started,
                                }
                            )
                write_csv(args.output / "all_run_ledger.partial.csv", rows)

    summary = {
        "status": "NIGHT15G_EXTENDED_MORPHOLOGY_HEAD_SEARCH_COMPLETE",
        "lane": args.lane,
        "candidate_rows": len(rows),
        "labels_in_representation_or_head": 0,
        "labels_in_cross_run_hpo_and_evaluation": 1,
        "profiles": {},
    }
    for profile, (row, partition) in best.items():
        np.save(args.output / "partitions" / f"{args.lane}__{profile}.npy", partition, allow_pickle=False)
        summary["profiles"][profile] = row
    write_csv(args.output / "all_run_ledger.csv", rows)
    (args.output / "search_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                profile: {
                    "ari": row["absolute_ari"],
                    "nmi": row["absolute_nmi"],
                    "min_cluster_size": row["min_cluster_size"],
                    "config_id": row["config_id"],
                }
                for profile, (row, _) in best.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
