#!/usr/bin/env python3
"""Refine a morphology-separated partition with the unified Night-15F energy.

This is a score-development arena.  Public labels are consumed only after a
complete partition exists, for cross-run configuration selection and metrics.
The sparse energy itself sees numeric views, sparse graphs, K and a starting
partition.  Failed configurations remain in the ledger.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15f_multiscale_expansion import (  # noqa: E402
    ExpansionEnergyConfig,
    continuous_multiscale_expansion,
    prepare_expansion_evidence,
)
from scripts.night15f.night15f_solver_search import (  # noqa: E402
    designed_configs,
    random_configs,
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
from scripts.night15g.night15g_profile_replay import (  # noqa: E402
    direct_partition,
    read_rows,
    select_profiles,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fused_representations(selected, retained, variants) -> dict[str, np.ndarray]:
    payload = json.loads(selected["config_json"])
    feature = variants[selected["feature_variant"]]
    retained_block = reduce_feature(retained, min(64, retained.shape[1]))
    direct = np.concatenate(
        (
            retained_block / math.sqrt(max(retained_block.shape[1], 1)),
            float(payload["weight"]) * feature / math.sqrt(max(feature.shape[1], 1)),
        ),
        axis=1,
    ).astype(np.float32)
    reduced = PCA(
        n_components=min(64, direct.shape[1], direct.shape[0] - 1),
        svd_solver="full",
    ).fit_transform(direct).astype(np.float32)
    return {
        "ORIGINAL_RETAINED": retained,
        "FUSED_DIRECT": direct,
        "FUSED_PCA64": reduced,
    }


def profile_key(row: dict) -> tuple:
    return selection_key(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--morphology", type=Path, required=True)
    parser.add_argument("--night15f-registry", type=Path, required=True)
    parser.add_argument("--search-ledger", type=Path, required=True)
    parser.add_argument("--balanced-partition", type=Path)
    parser.add_argument("--lane", default="D1")
    parser.add_argument("--random-configs", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "partitions").mkdir(exist_ok=True)

    data, graph, graphs, retained, authority, labels, mask, k, morph = load_lane(args, args.lane)
    variants = feature_variants(morph)
    selected = select_profiles(read_rows(args.search_ledger))["max_ari"]
    if selected["stage"] not in ("DIRECT_CONTROL", "DIRECT_CONTROL_REV2"):
        raise RuntimeError("seeded refinement requires a direct morphology max-ARI start")
    direct = direct_partition(selected, retained, variants, k).astype(np.int32)
    if array_sha256(direct) != selected["partition_sha256"]:
        raise RuntimeError("direct start did not replay exactly")
    starts = {"NIGHT15F_AUTHORITY": authority, "MORPHOLOGY_MAX_ARI": direct}
    if args.balanced_partition is not None:
        balanced = np.load(args.balanced_partition, allow_pickle=False).astype(np.int32)
        if len(balanced) != len(authority) or len(np.unique(balanced)) != k:
            raise RuntimeError("balanced morphology start failed shape/cardinality contract")
        starts["MORPHOLOGY_BALANCED"] = balanced

    evidence_bank = {}
    for name, representation in fused_representations(selected, retained, variants).items():
        evidence_bank[name] = prepare_expansion_evidence(
            graphs,
            representation,
            np.asarray(data["view1"]),
            np.asarray(data["view2"]),
        )

    registry = json.loads(args.night15f_registry.read_text(encoding="utf-8"))
    anchor = parse_base(registry["lanes"][args.lane]["config"])
    configs: dict[str, ExpansionEnergyConfig] = {"NIGHT15F_ANCHOR": anchor}
    configs.update(designed_configs(anchor.local))
    configs.update(
        {
            f"SEEDED_{key}": value
            for key, value in random_configs(
                anchor.local,
                args.random_configs,
                20260824 + 1507,
                anchor=anchor,
            ).items()
        }
    )

    authority_metrics = evaluate(labels, mask, authority, graph)
    all_rows: list[dict] = []
    best_partitions: dict[str, tuple[dict, np.ndarray]] = {}

    def consider(row: dict, partition: np.ndarray) -> None:
        candidates = {
            "balanced": profile_key,
            "max_ari": lambda value: (float(value["absolute_ari"]), float(value["absolute_nmi"])),
            "max_nmi": lambda value: (float(value["absolute_nmi"]), float(value["absolute_ari"])),
            "composite": lambda value: (
                float(value["absolute_ari"]) + 0.35 * float(value["absolute_nmi"]),
            ),
            "nonmicro_balanced": lambda value: (
                int(float(value["min_cluster_size"]) >= max(5, math.ceil(len(partition) * 0.001))),
                *profile_key(value),
            ),
        }
        for profile, key in candidates.items():
            previous = best_partitions.get(profile)
            if previous is None or key(row) > key(previous[0]):
                best_partitions[profile] = (row, partition.copy())

    for start_name, initial in starts.items():
        for evidence_name, evidence in evidence_bank.items():
            for config_id, config in configs.items():
                started = time.perf_counter()
                try:
                    partition, diagnostics = continuous_multiscale_expansion(initial, k, evidence, config)
                    metrics = evaluate(labels, mask, partition, graph)
                    row = {
                        "lane": args.lane,
                        "start": start_name,
                        "evidence": evidence_name,
                        "config_id": config_id,
                        "config_json": json.dumps(asdict(config), sort_keys=True, separators=(",", ":")),
                        "status": "PASS",
                        "failure": "",
                        "partition_sha256": array_sha256(partition),
                        "initial_partition_sha256": array_sha256(initial),
                        "absolute_ari": metrics["absolute_ari"],
                        "absolute_nmi": metrics["absolute_nmi"],
                        "delta_ari": metrics["absolute_ari"] - authority_metrics["absolute_ari"],
                        "delta_nmi": metrics["absolute_nmi"] - authority_metrics["absolute_nmi"],
                        "wall_seconds": time.perf_counter() - started,
                        **{key: value for key, value in metrics.items() if key not in {"absolute_ari", "absolute_nmi"}},
                        **diagnostics,
                    }
                    all_rows.append(row)
                    if int(diagnostics["observed_cardinality"]) == k:
                        consider(row, partition)
                except Exception as error:
                    all_rows.append(
                        {
                            "lane": args.lane,
                            "start": start_name,
                            "evidence": evidence_name,
                            "config_id": config_id,
                            "config_json": json.dumps(asdict(config), sort_keys=True, separators=(",", ":")),
                            "status": "FAILED",
                            "failure": repr(error),
                            "wall_seconds": time.perf_counter() - started,
                        }
                    )
            write_csv(args.output / "all_run_ledger.partial.csv", all_rows)

    summary = {
        "status": "NIGHT15G_MORPHOLOGY_SEEDED_REFINEMENT_COMPLETE",
        "lane": args.lane,
        "authority": authority_metrics,
        "source_direct_profile": selected,
        "candidate_rows": len(all_rows),
        "labels_in_energy": 0,
        "labels_in_cross_run_hpo_and_evaluation": 1,
        "dense_n_by_n_count": 0,
        "profiles": {},
    }
    for profile, (row, partition) in best_partitions.items():
        np.save(args.output / "partitions" / f"{args.lane}__{profile}.npy", partition, allow_pickle=False)
        summary["profiles"][profile] = row
    write_csv(args.output / "all_run_ledger.csv", all_rows)
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
                    "start": row["start"],
                    "evidence": row["evidence"],
                }
                for profile, (row, _) in best_partitions.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
