#!/usr/bin/env python3
"""Replay and freeze every useful Night-15G score profile.

Search labels choose a public-benchmark profile between completed runs only.
They never enter feature construction, an energy, or a clustering call.  This
runner independently reconstructs balanced, max-ARI and max-NMI partitions and
requires an exact partition hash match before it writes a frozen artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15g_optional_morphology_energy import (  # noqa: E402
    OptionalMorphologyConfig,
    optional_morphology_expansion,
    prepare_optional_morphology_evidence,
)
from scripts.night15g.night15g_optional_view_search import (  # noqa: E402
    array_sha256,
    evaluate,
    feature_variants,
    load_lane,
    parse_base,
    reduce_feature,
    selection_key,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_optional(payload: str) -> OptionalMorphologyConfig:
    value = json.loads(payload)
    base = parse_base(value.pop("base"))
    return OptionalMorphologyConfig(base=base, **value)


def passing(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status") == "PASS"]


def select_profiles(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    valid = passing(rows)
    return {
        "balanced": max(valid, key=selection_key),
        "max_ari": max(valid, key=lambda row: float(row["absolute_ari"])),
        "max_nmi": max(valid, key=lambda row: float(row["absolute_nmi"])),
    }


def direct_partition(
    selected: dict[str, str],
    retained: np.ndarray,
    variants: dict[str, np.ndarray],
    k: int,
) -> np.ndarray:
    feature = variants[selected["feature_variant"]]
    payload = json.loads(selected["config_json"])
    seed = int(payload["seed"])
    algorithm = selected["algorithm"]
    if selected["stage"] == "DIRECT_CONTROL":
        if algorithm.endswith("KMEANS"):
            return KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(feature)
        return GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=seed,
            n_init=3,
            max_iter=300,
        ).fit_predict(feature)
    retained_block = reduce_feature(retained, min(64, retained.shape[1]))
    retained_equalized = retained_block / math.sqrt(max(retained_block.shape[1], 1))
    morphology_equalized = feature / math.sqrt(max(feature.shape[1], 1))
    direct = np.concatenate(
        (retained_equalized, float(payload["weight"]) * morphology_equalized),
        axis=1,
    ).astype(np.float32)
    if algorithm.startswith("BLOCK_WEIGHTED_PCA"):
        direct = PCA(
            n_components=min(64, direct.shape[1], direct.shape[0] - 1),
            svd_solver="full",
        ).fit_transform(direct).astype(np.float32)
    return KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(direct)


def reconstruct(
    selected: dict[str, str],
    retained: np.ndarray,
    variants: dict[str, np.ndarray],
    initial: np.ndarray,
    k: int,
    graphs,
    molecular_views,
):
    stage = selected["stage"]
    if stage == "AUTHORITY":
        return initial.copy(), {}, None, None
    if stage in ("DIRECT_CONTROL", "DIRECT_CONTROL_REV2"):
        return direct_partition(selected, retained, variants, k).astype(np.int32), {}, None, None
    if stage != "OPTIONAL_ENERGY_SEARCH":
        raise ValueError(f"unsupported selected stage {stage!r}")
    feature = variants[selected["feature_variant"]]
    evidence = prepare_optional_morphology_evidence(
        graphs,
        retained,
        molecular_views,
        [feature],
        [np.ones(len(feature), dtype=np.float32)],
        optional_dim=feature.shape[1],
        optional_edge_dim=min(24, feature.shape[1]),
    )
    config = parse_optional(selected["config_json"])
    partition, diagnostics = optional_morphology_expansion(initial, k, evidence, config)
    return partition, diagnostics, evidence, config


def assert_selected(selected, partition, metrics) -> None:
    observed_hash = array_sha256(partition)
    if observed_hash != selected["partition_sha256"]:
        raise RuntimeError(
            "selected partition did not replay: "
            f"observed={observed_hash}, expected={selected['partition_sha256']}"
        )
    for key in ("absolute_ari", "absolute_nmi"):
        if abs(float(metrics[key]) - float(selected[key])) > 1e-12:
            raise RuntimeError(
                f"selected {key} did not replay: observed={metrics[key]!r}, "
                f"expected={selected[key]!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--morphology", type=Path, required=True)
    parser.add_argument("--night15f-registry", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", nargs="+", default=["A1", "D1", "tonsil_s3"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "partitions").mkdir(exist_ok=True)

    registry = {
        "status": "NIGHT15G_EXACT_PROFILE_REPLAY",
        "labels_in_features_or_clustering": 0,
        "labels_in_cross_run_profile_selection": 1,
        "lanes": {},
    }
    replay_rows: list[dict] = []
    ablation_rows: list[dict] = []
    for lane_index, lane in enumerate(args.lanes):
        rows = read_rows(args.search_root / f"{lane}_all_run_ledger.csv")
        profiles = select_profiles(rows)
        data, graph, graphs, retained, initial, labels, mask, k, morph = load_lane(args, lane)
        variants = feature_variants(morph)
        molecular_views = [np.asarray(data["view1"]), np.asarray(data["view2"])]
        lane_registry = {}
        for profile, selected in profiles.items():
            started = time.perf_counter()
            partition, diagnostics, evidence, config = reconstruct(
                selected,
                retained,
                variants,
                initial,
                k,
                graphs,
                molecular_views,
            )
            metrics = evaluate(labels, mask, partition, graph)
            assert_selected(selected, partition, metrics)
            path = args.output / "partitions" / f"{lane}__{profile}.npy"
            np.save(path, partition.astype(np.int32), allow_pickle=False)
            lane_registry[profile] = {
                "stage": selected["stage"],
                "algorithm": selected["algorithm"],
                "config_id": selected["config_id"],
                "feature_variant": selected["feature_variant"],
                "partition_sha256": array_sha256(partition),
                "absolute_ari": metrics["absolute_ari"],
                "absolute_nmi": metrics["absolute_nmi"],
            }
            replay_rows.append(
                {
                    "lane": lane,
                    "profile": profile,
                    **lane_registry[profile],
                    "wall_seconds": time.perf_counter() - started,
                    **diagnostics,
                }
            )
            if profile != "balanced" or evidence is None or config is None:
                continue
            rng = np.random.default_rng(20260824 + lane_index)
            feature = variants[selected["feature_variant"]]
            permuted = feature[rng.permutation(len(feature))]
            permuted_evidence = prepare_optional_morphology_evidence(
                graphs,
                retained,
                molecular_views,
                [permuted],
                [np.ones(len(feature), dtype=np.float32)],
                optional_dim=feature.shape[1],
                optional_edge_dim=min(24, feature.shape[1]),
            )
            zero_evidence = prepare_optional_morphology_evidence(
                graphs,
                retained,
                molecular_views,
                [feature],
                [np.zeros(len(feature), dtype=np.float32)],
                optional_dim=feature.shape[1],
                optional_edge_dim=min(24, feature.shape[1]),
            )
            ablations = [
                ("FULL", config, evidence),
                ("RELIABILITY_CONSTANT", replace(config, reliability_mix=0.0), evidence),
                ("UNARY_ONLY", replace(config, morphology_edge_weight=0.0), evidence),
                ("EDGE_ONLY", replace(config, morphology_unary_weight=0.0), evidence),
                ("CONSISTENCY_OFF", replace(config, reliability_consistency_weight=0.0), evidence),
                ("CONFLICT_OFF", replace(config, reliability_conflict_weight=0.0), evidence),
                ("MARGIN_OFF", replace(config, reliability_margin_weight=0.0), evidence),
                ("PERMUTED_MORPHOLOGY", config, permuted_evidence),
                ("MISSING_VIEW", config, zero_evidence),
            ]
            for ablation, ablation_config, ablation_evidence in ablations:
                part, diag = optional_morphology_expansion(initial, k, ablation_evidence, ablation_config)
                score = evaluate(labels, mask, part, graph)
                if ablation == "MISSING_VIEW" and not np.array_equal(part, initial):
                    raise RuntimeError(f"{lane}: missing morphology did not return authority exactly")
                ablation_rows.append(
                    {
                        "lane": lane,
                        "ablation": ablation,
                        "partition_sha256": array_sha256(part),
                        **score,
                        **diag,
                    }
                )
        registry["lanes"][lane] = lane_registry
    write_csv(args.output / "profile_replay.csv", replay_rows)
    if ablation_rows:
        write_csv(args.output / "matched_ablation.csv", ablation_rows)
    (args.output / "frozen_profile_registry.json").write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
