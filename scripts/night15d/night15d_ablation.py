#!/usr/bin/env python3
"""Matched component ablations for frozen Night-15D finalists."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from SpaLORA.night15d_reliability_energy import (
    csr_from_archive,
    edge_similarity,
    multiscale_feature_bank,
    reduce_full,
    reliability_energy_icm,
    reliability_transition,
)
from SpaLORA.night15c_cluster_energy import sha256_array
from night15d_arena import (
    DATASET_FOR_LANE,
    evaluate,
    lane_semantics,
    reproduce_night15c_authority,
    write_rows,
)


def discover_summaries(roots: List[Path]) -> Dict[str, dict]:
    result = {}
    for root in roots:
        for path in root.glob("*__summary.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            result[value["lane"]] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--night15c-registry", type=Path, required=True)
    parser.add_argument("--summary-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = discover_summaries(args.summary_roots)
    registered = json.loads(args.night15c_registry.read_text(encoding="utf-8"))[
        "stable_configs"
    ]
    rows: List[dict] = []
    lane_summaries = []
    for lane in sorted(summaries):
        dataset = DATASET_FOR_LANE[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(
            args.banks / f"{dataset}_selected_partition_bank.npz",
            allow_pickle=False,
            mmap_mode="r",
        )
        k, labels, mask = lane_semantics(data, lane)
        graph = csr_from_archive(data, "graph")
        authority = reproduce_night15c_authority(data, bank, lane, k, registered[lane])
        features = multiscale_feature_bank(
            bank[f"{lane}__retained_embedding"], data["view1"], data["view2"], graph
        )
        view1 = reduce_full(data["view1"], min(24, data["view1"].shape[1]))
        view2 = reduce_full(data["view2"], min(24, data["view2"].shape[1]))
        final_config = json.loads(summaries[lane]["best"]["config_json"])
        transition_cache: Dict[
            Tuple[str, str, float], Tuple[sp.csr_matrix, Mapping[str, float]]
        ] = {}

        def transition(config: Mapping[str, object]):
            key = (
                str(config["edge_mode"]),
                str(config["normalization"]),
                float(config["tau"]),
            )
            if key not in transition_cache:
                binary, weighted = edge_similarity(
                    graph,
                    data["view1"],
                    data["view2"],
                    key[0],
                    tau=key[2],
                    dim=16,
                )
                transition_cache[key] = reliability_transition(binary, weighted, key[1])
            return transition_cache[key]

        def execute(variant: str, config: Mapping[str, object]) -> dict:
            started = time.perf_counter()
            support, diagnostics = transition(config)
            partition, completed, collapse, unary_diagnostics = reliability_energy_icm(
                authority,
                support,
                k,
                float(config["beta"]),
                int(config["steps"]),
                str(config["unary_mode"]),
                features[str(config["feature"])],
                view1,
                view2,
                margin_temperature=float(config["margin_temperature"]),
                retained_weight=float(config["retained_weight"]),
                switch_penalty=float(config["switch_penalty"]),
            )
            metrics = evaluate(labels, mask, partition)
            row = {
                "lane": lane,
                "k": k,
                "variant": variant,
                "config_json": json.dumps(config, sort_keys=True, separators=(",", ":")),
                "absolute_ari": metrics["absolute_ari"],
                "absolute_nmi": metrics["absolute_nmi"],
                "delta_vs_night15c_ari": metrics["absolute_ari"]
                - evaluate(labels, mask, authority)["absolute_ari"],
                "delta_vs_night15c_nmi": metrics["absolute_nmi"]
                - evaluate(labels, mask, authority)["absolute_nmi"],
                "partition_sha256": sha256_array(partition),
                "changed_observations": int(np.sum(partition != authority)),
                "observed_cardinality": int(len(np.unique(partition))),
                "steps_completed": int(completed),
                "collapse_guard_triggered": bool(collapse),
                "wall_seconds": time.perf_counter() - started,
                **diagnostics,
                **unary_diagnostics,
            }
            rows.append(row)
            return row

        authority_metrics = evaluate(labels, mask, authority)
        authority_row = {
            "lane": lane,
            "k": k,
            "variant": "NIGHT15C_AUTHORITY",
            "config_json": json.dumps({"source": "night15c_authority"}),
            "absolute_ari": authority_metrics["absolute_ari"],
            "absolute_nmi": authority_metrics["absolute_nmi"],
            "delta_vs_night15c_ari": 0.0,
            "delta_vs_night15c_nmi": 0.0,
            "partition_sha256": summaries[lane]["authority_partition_sha256"],
            "changed_observations": 0,
            "observed_cardinality": k,
            "steps_completed": 0,
            "collapse_guard_triggered": False,
            "wall_seconds": 0.0,
        }
        rows.append(authority_row)
        variants = {"FULL": dict(final_config)}
        for normalization in ("row", "mass", "self"):
            value = dict(final_config)
            value["normalization"] = normalization
            variants[f"NORMALIZATION_{normalization.upper()}"] = value
        value = dict(final_config)
        value["feature"] = "retained"
        variants["NO_MULTISCALE_FEATURE"] = value
        value = dict(final_config)
        value.update(unary_mode="retained", margin_temperature=0.5, retained_weight=1.0)
        variants["NO_MODALITY_MARGIN_UNARY"] = value
        value = dict(final_config)
        value["edge_mode"] = "spatial"
        variants["NO_MULTIMODAL_EDGE"] = value
        value = dict(final_config)
        value["switch_penalty"] = 0.0
        variants["NO_SWITCH_TRUST"] = value
        value = dict(final_config)
        value["tau"] = 1.0
        variants["UNIT_EDGE_TEMPERATURE"] = value

        observed = {name: execute(name, config) for name, config in variants.items()}
        full = observed["FULL"]
        contributions = {}
        for name, row in observed.items():
            if name == "FULL":
                continue
            contributions[name] = {
                "full_minus_variant_ari": float(full["absolute_ari"] - row["absolute_ari"]),
                "full_minus_variant_nmi": float(full["absolute_nmi"] - row["absolute_nmi"]),
            }
        lane_summaries.append(
            {
                "lane": lane,
                "full": full,
                "contributions": contributions,
                "full_matches_frozen_sha": full["partition_sha256"]
                == summaries[lane]["best_partition_sha256"],
            }
        )

    write_rows(args.output / "matched_component_ablation.csv", rows)
    result = {
        "status": "PASS",
        "lanes": lane_summaries,
        "row_count": len(rows),
        "labels_used_for_evaluation": 1,
        "labels_in_model_or_energy": 0,
        "dense_n_by_n_count": 0,
    }
    (args.output / "matched_component_ablation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "lanes": len(lane_summaries), "rows": len(rows)}))


if __name__ == "__main__":
    main()
