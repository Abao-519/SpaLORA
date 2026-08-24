#!/usr/bin/env python3
"""Freeze and ablate selected Night-15G optional-morphology candidates.

The search summary supplies public-benchmark HPO choices.  Reference labels are
used only by ``evaluate`` after a complete partition has been produced.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import numpy as np

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
)


def parse_optional(payload: str) -> OptionalMorphologyConfig:
    value = json.loads(payload)
    base = parse_base(value.pop("base"))
    return OptionalMorphologyConfig(base=base, **value)


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
    parser.add_argument("--search-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "partitions").mkdir(exist_ok=True)
    search = json.loads(args.search_summary.read_text(encoding="utf-8"))
    rows: list[dict] = []
    registry = {
        "status": "NIGHT15G_FROZEN_OPTIONAL_MORPHOLOGY_CANDIDATES",
        "labels_in_model_or_energy": 0,
        "lanes": {},
    }
    for lane_index, lane in enumerate(("A1", "D1", "tonsil_s3")):
        data, graph, graphs, retained, initial, labels, mask, k, morph = load_lane(args, lane)
        selected = search[lane]["balanced"]
        config = parse_optional(selected["config_json"])
        feature = feature_variants(morph)[selected["feature_variant"]]
        presence = np.ones(len(feature), dtype=np.float32)
        evidence = prepare_optional_morphology_evidence(
            graphs,
            retained,
            [np.asarray(data["view1"]), np.asarray(data["view2"])],
            [feature],
            [presence],
            optional_dim=feature.shape[1],
            optional_edge_dim=min(24, feature.shape[1]),
        )
        rng = np.random.default_rng(20260824 + lane_index)
        permuted = feature[rng.permutation(len(feature))]
        permuted_evidence = prepare_optional_morphology_evidence(
            graphs,
            retained,
            [np.asarray(data["view1"]), np.asarray(data["view2"])],
            [permuted],
            [presence],
            optional_dim=permuted.shape[1],
            optional_edge_dim=min(24, permuted.shape[1]),
        )
        zero_evidence = prepare_optional_morphology_evidence(
            graphs,
            retained,
            [np.asarray(data["view1"]), np.asarray(data["view2"])],
            [feature],
            [np.zeros(len(feature), dtype=np.float32)],
            optional_dim=feature.shape[1],
            optional_edge_dim=min(24, feature.shape[1]),
        )
        variants = [
            ("FULL", config, evidence),
            ("RELIABILITY_CONSTANT", replace(config, reliability_mix=0.0), evidence),
            ("UNARY_ONLY", replace(config, morphology_edge_weight=0.0), evidence),
            ("EDGE_ONLY", replace(config, morphology_unary_weight=0.0), evidence),
            (
                "CONSISTENCY_OFF",
                replace(config, reliability_consistency_weight=0.0),
                evidence,
            ),
            ("CONFLICT_OFF", replace(config, reliability_conflict_weight=0.0), evidence),
            ("MARGIN_OFF", replace(config, reliability_margin_weight=0.0), evidence),
            ("PERMUTED_MORPHOLOGY", config, permuted_evidence),
            ("MISSING_VIEW", config, zero_evidence),
        ]
        full_partition = None
        full_metrics = None
        for ablation, candidate_config, candidate_evidence in variants:
            started = time.perf_counter()
            partition, diagnostics = optional_morphology_expansion(
                initial,
                k,
                candidate_evidence,
                candidate_config,
            )
            metrics = evaluate(labels, mask, partition, graph)
            row = {
                "lane": lane,
                "ablation": ablation,
                "feature_variant": selected["feature_variant"],
                "config_id": selected["config_id"],
                "partition_sha256": array_sha256(partition),
                "authority_partition_sha256": array_sha256(initial),
                "absolute_ari": metrics["absolute_ari"],
                "absolute_nmi": metrics["absolute_nmi"],
                "delta_ari": metrics["absolute_ari"] - float(selected["night15f_ari"]),
                "delta_nmi": metrics["absolute_nmi"] - float(selected["night15f_nmi"]),
                "ami": metrics["ami"],
                "fmi": metrics["fmi"],
                "morans_i": metrics["morans_i"],
                "gearys_c": metrics["gearys_c"],
                "wall_seconds": time.perf_counter() - started,
                **diagnostics,
            }
            rows.append(row)
            if ablation == "FULL":
                full_partition = partition
                full_metrics = metrics
                if abs(metrics["absolute_ari"] - float(selected["absolute_ari"])) > 1e-12:
                    raise RuntimeError(
                        f"{lane}: frozen ARI does not replay selected search row: "
                        f"observed={metrics['absolute_ari']!r}, expected={float(selected['absolute_ari'])!r}, "
                        f"observed_partition={array_sha256(partition)}, "
                        f"expected_partition={selected['partition_sha256']}"
                    )
                if abs(metrics["absolute_nmi"] - float(selected["absolute_nmi"])) > 1e-12:
                    raise RuntimeError(f"{lane}: frozen NMI does not replay selected search row")
            if ablation == "MISSING_VIEW" and not np.array_equal(partition, initial):
                raise RuntimeError(f"{lane}: missing optional view did not return exact authority")
        assert full_partition is not None and full_metrics is not None
        np.save(args.output / "partitions" / f"{lane}.npy", full_partition, allow_pickle=False)
        registry["lanes"][lane] = {
            "feature_variant": selected["feature_variant"],
            "config_id": selected["config_id"],
            "config": asdict(config),
            "partition_sha256": array_sha256(full_partition),
            "absolute_ari": full_metrics["absolute_ari"],
            "absolute_nmi": full_metrics["absolute_nmi"],
            "source_search_summary": str(args.search_summary),
        }
    write_csv(args.output / "matched_ablation.csv", rows)
    (args.output / "frozen_registry.json").write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
