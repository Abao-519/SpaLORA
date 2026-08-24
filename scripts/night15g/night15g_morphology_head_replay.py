#!/usr/bin/env python3
"""Freeze and exactly replay selected Night-15G morphology-head profiles.

This is a deterministic replay of rows already present in the completed head
search ledger.  Public benchmark labels are used only to choose between whole
partitions after clustering; they never enter the representation or clusterer.
Alongside unconstrained score profiles, the script records profiles whose
smallest cluster contains at least one percent of observations.  This keeps a
scientifically interpretable alternative without hiding the unconstrained run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

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


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            if row.get("status") != "PASS":
                rows.append(row)
                continue
            for key in (
                "absolute_ari",
                "absolute_nmi",
                "delta_ari",
                "delta_nmi",
                "min_cluster_size",
                "morphology_weight",
                "coordinate_weight",
                "seed",
            ):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def choose_profiles(rows: list[dict[str, object]], minimum_cluster: int) -> dict[str, dict[str, object]]:
    passed = [row for row in rows if row.get("status") == "PASS"]
    nonmicro = [row for row in passed if float(row["min_cluster_size"]) >= minimum_cluster]
    if not passed or not nonmicro:
        raise RuntimeError("No eligible rows for one or more replay profiles")
    return {
        "balanced": max(passed, key=selection_key),
        "max_ari": max(passed, key=lambda row: (float(row["absolute_ari"]), float(row["absolute_nmi"]))),
        "max_nmi": max(passed, key=lambda row: (float(row["absolute_nmi"]), float(row["absolute_ari"]))),
        "nonmicro_balanced": max(nonmicro, key=selection_key),
        "nonmicro_max_ari": max(
            nonmicro,
            key=lambda row: (float(row["absolute_ari"]), float(row["absolute_nmi"])),
        ),
        "nonmicro_max_nmi": max(
            nonmicro,
            key=lambda row: (float(row["absolute_nmi"]), float(row["absolute_ari"])),
        ),
    }


def replay_partition(
    row: dict[str, object],
    retained_block: np.ndarray,
    variants: dict[str, np.ndarray],
    coordinates: np.ndarray,
    k: int,
    *,
    morphology_mode: str = "FULL",
    coordinate_enabled: bool = True,
) -> np.ndarray:
    morphology = variants[str(row["feature_variant"])]
    morphology = morphology / math.sqrt(max(morphology.shape[1], 1))
    if morphology_mode == "MISSING":
        morphology = np.zeros_like(morphology)
    elif morphology_mode == "PERMUTED":
        order = np.random.default_rng(150701).permutation(morphology.shape[0])
        morphology = morphology[order]
    elif morphology_mode != "FULL":
        raise ValueError(f"Unsupported morphology mode: {morphology_mode}")
    coordinate_weight = float(row["coordinate_weight"]) if coordinate_enabled else 0.0
    direct = np.concatenate(
        (
            retained_block,
            float(row["morphology_weight"]) * morphology,
            coordinate_weight * coordinates,
        ),
        axis=1,
    ).astype(np.float32)
    representation_name = str(row["representation"])
    if representation_name == "DIRECT":
        representation = direct
    elif representation_name == "PCA64":
        representation = PCA(
            n_components=min(64, direct.shape[1], direct.shape[0] - 1),
            svd_solver="full",
        ).fit_transform(direct).astype(np.float32)
    else:
        raise ValueError(f"Unsupported representation: {representation_name}")

    seed = int(float(row["seed"]))
    algorithm = str(row["algorithm"])
    if algorithm == "KMEANS":
        result = KMeans(n_clusters=k, random_state=seed, n_init=30).fit_predict(representation)
    elif algorithm == "BISECTING_KMEANS":
        result = BisectingKMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(representation)
    elif algorithm.startswith("GMM_"):
        covariance = algorithm.removeprefix("GMM_").lower()
        result = GaussianMixture(
            n_components=k,
            covariance_type=covariance,
            random_state=seed,
            n_init=3,
            max_iter=400,
            reg_covar=1e-5,
        ).fit_predict(representation)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return np.asarray(result, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--morphology", type=Path, required=True)
    parser.add_argument("--night15f-registry", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--lane", default="D1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "partitions").mkdir(exist_ok=True)

    data, graph, _, retained, _, labels, mask, k, morph = load_lane(args, args.lane)
    minimum_cluster = max(5, math.ceil(len(labels) * 0.01))
    rows = read_rows(args.ledger)
    profiles = choose_profiles(rows, minimum_cluster)

    variants = feature_variants(morph)
    retained_block = reduce_feature(retained, min(64, retained.shape[1]))
    retained_block = retained_block / math.sqrt(max(retained_block.shape[1], 1))
    coordinates = standardize(np.asarray(data["coordinates"], dtype=np.float32))
    coordinates = coordinates / math.sqrt(max(coordinates.shape[1], 1))

    replayed: dict[str, object] = {}
    for profile, row in profiles.items():
        partition = replay_partition(row, retained_block, variants, coordinates, k)
        replay_hash = array_sha256(partition)
        expected_hash = str(row["partition_sha256"])
        if replay_hash != expected_hash:
            raise RuntimeError(
                f"{profile} partition mismatch: expected {expected_hash}, observed {replay_hash}"
            )
        metrics = evaluate(labels, mask, partition, graph)
        if not np.isclose(metrics["absolute_ari"], float(row["absolute_ari"]), rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{profile} ARI replay mismatch")
        if not np.isclose(metrics["absolute_nmi"], float(row["absolute_nmi"]), rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{profile} NMI replay mismatch")
        destination = args.output / "partitions" / f"{args.lane}__{profile}.npy"
        np.save(destination, partition, allow_pickle=False)
        replayed[profile] = {
            "source_config_id": row["config_id"],
            "feature_variant": row["feature_variant"],
            "morphology_weight": row["morphology_weight"],
            "coordinate_weight": row["coordinate_weight"],
            "representation": row["representation"],
            "algorithm": row["algorithm"],
            "seed": int(float(row["seed"])),
            "absolute_ari": metrics["absolute_ari"],
            "absolute_nmi": metrics["absolute_nmi"],
            "ami": metrics["ami"],
            "fmi": metrics["fmi"],
            "morans_i": metrics["morans_i"],
            "gearys_c": metrics["gearys_c"],
            "min_cluster_size": int(np.bincount(partition, minlength=k).min()),
            "partition_sha256": replay_hash,
            "byte_exact_source_replay": True,
        }

    control_source = profiles["nonmicro_balanced"]
    matched_controls: dict[str, object] = {}
    for control, morphology_mode, coordinate_enabled in (
        ("FULL", "FULL", True),
        ("MISSING_MORPHOLOGY", "MISSING", True),
        ("PERMUTED_MORPHOLOGY", "PERMUTED", True),
        ("COORDINATE_OFF", "FULL", False),
        ("MOLECULAR_ONLY", "MISSING", False),
    ):
        partition = replay_partition(
            control_source,
            retained_block,
            variants,
            coordinates,
            k,
            morphology_mode=morphology_mode,
            coordinate_enabled=coordinate_enabled,
        )
        metrics = evaluate(labels, mask, partition, graph)
        control_hash = array_sha256(partition)
        np.save(
            args.output / "partitions" / f"{args.lane}__nonmicro_balanced__{control.lower()}.npy",
            partition,
            allow_pickle=False,
        )
        matched_controls[control] = {
            "absolute_ari": metrics["absolute_ari"],
            "absolute_nmi": metrics["absolute_nmi"],
            "ami": metrics["ami"],
            "fmi": metrics["fmi"],
            "morans_i": metrics["morans_i"],
            "gearys_c": metrics["gearys_c"],
            "min_cluster_size": int(np.bincount(partition, minlength=k).min()),
            "partition_sha256": control_hash,
        }

    summary = {
        "status": "NIGHT15G_MORPHOLOGY_HEAD_REPLAY_COMPLETE",
        "lane": args.lane,
        "source_ledger": str(args.ledger.resolve()),
        "source_rows": len(rows),
        "nonmicro_minimum_cluster_size": minimum_cluster,
        "labels_in_representation_or_head": 0,
        "labels_in_cross_run_profile_selection_and_evaluation": 1,
        "profiles": replayed,
        "matched_controls_for_nonmicro_balanced": matched_controls,
    }
    (args.output / "replay_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
