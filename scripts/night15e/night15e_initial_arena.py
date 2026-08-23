#!/usr/bin/env python3
"""Score-ceiling-only multi-initial arena for Night-15E.

This arena is explicitly public-label development HPO.  It is not used for
the incremental deployable-policy audit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from night15e_continuous_search import (
    DATASET_FOR_LANE,
    NIGHT15D_AUTHORITY,
    baseline_row,
    csr_from_archive,
    evaluate,
    global_config_registry,
    jitter_registry,
    lane_semantics,
    load_lane,
    prepare_continuous_evidence,
    run_one,
    selection_key,
    sha256_array,
    write_rows,
)


def unique(values, k):
    seen = set()
    output = []
    for name, value in values:
        value = np.asarray(value, dtype=np.int32)
        digest = sha256_array(value)
        if len(np.unique(value)) == int(k) and digest not in seen:
            seen.add(digest)
            output.append((name, value))
    return output


def sources(data, bank, lane, k, authority):
    values = [("night15d_authority", authority)]
    values.extend((f"selected_teacher_{i}", part) for i, part in enumerate(bank[f"{lane}__teacher_partitions"]))
    values.append(("selected_consensus", bank[f"{lane}__consensus"]))
    values.append(("selected_medoid", bank[f"{lane}__medoid"]))
    key = "teacher_partitions_k12" if lane == "MISAR_E15_5_S1_K12" else "teacher_partitions"
    if key in data.files:
        values.extend((f"kit_teacher_{i}", part) for i, part in enumerate(data[key]))
    return unique(values, k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", default="A1,MISAR_E15_5_S1,tonsil_s3")
    parser.add_argument("--configs", type=int, default=64)
    parser.add_argument("--refine", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config_registry = global_config_registry(args.configs, seed=20260825)
    all_rows = []
    summaries = {}
    for lane_index, lane in enumerate([value for value in args.lanes.split(",") if value]):
        dataset = DATASET_FOR_LANE[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(args.banks / f"{dataset}_selected_partition_bank.npz", allow_pickle=False, mmap_mode="r")
        graph = csr_from_archive(data, "graph")
        k, labels, mask = lane_semantics(data, lane)
        authority = np.load(args.partitions / f"{lane}.npy", allow_pickle=False).astype(np.int32)
        retained = np.asarray(bank[f"{lane}__retained_embedding"], dtype=np.float32)
        evidence = prepare_continuous_evidence(graph, retained, data["view1"], data["view2"])
        base = baseline_row(lane, graph, labels, mask, k, authority)
        all_rows.append(base)
        initial_registry = []
        candidate_rows = []
        candidate_partitions = {}
        for source_name, initial in sources(data, bank, lane, k, authority):
            initial_metrics = evaluate(labels, mask, initial, graph)
            initial_registry.append(
                {
                    "source": source_name,
                    "partition_sha256": sha256_array(initial),
                    "absolute_ari": initial_metrics["absolute_ari"],
                    "absolute_nmi": initial_metrics["absolute_nmi"],
                }
            )
            for config_id, config in config_registry.items():
                identifier = f"{lane}__{source_name}__{config_id}"
                row, partition = run_one(
                    lane, graph, labels, mask, k, initial, evidence, identifier, config, "SCORE_CEILING_MULTI_INITIAL"
                )
                row["initial_source"] = source_name
                all_rows.append(row)
                candidate_rows.append(row)
                if row.get("status") == "PASS":
                    candidate_partitions[identifier] = partition
        anchor_row = max(candidate_rows, key=selection_key)
        anchor_config = json.loads(anchor_row["config_json"])
        from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig

        anchor = ContinuousEnergyConfig(**anchor_config)
        anchor_source = str(anchor_row["initial_source"])
        initial_map = {name: value for name, value in sources(data, bank, lane, k, authority)}
        anchor_initial = initial_map[anchor_source]
        for config_id, config in jitter_registry(anchor, args.refine, 20260825 + lane_index * 4099).items():
            identifier = f"{lane}__{anchor_source}__J{config_id}"
            row, partition = run_one(
                lane, graph, labels, mask, k, anchor_initial, evidence, identifier, config, "SCORE_CEILING_MULTI_INITIAL_REFINEMENT"
            )
            row["initial_source"] = anchor_source
            all_rows.append(row)
            candidate_rows.append(row)
            if row.get("status") == "PASS":
                candidate_partitions[identifier] = partition
        best = max([base] + candidate_rows, key=selection_key)
        if best["config_id"] == "NIGHT15D_AUTHORITY":
            best_partition = authority
        else:
            best_partition = candidate_partitions[str(best["config_id"])]
        (args.output / "partitions").mkdir(exist_ok=True)
        np.save(args.output / "partitions" / f"{lane}.npy", best_partition, allow_pickle=False)
        summaries[lane] = {
            "best": best,
            "initial_registry": initial_registry,
            "partition_sha256": sha256_array(best_partition),
            "score_ceiling_only": True,
            "public_labels_used_for_initial_and_config_hpo": 1,
            "eligible_for_incremental_deployable_policy": False,
        }
        write_rows(args.output / "all_run_ledger.partial.csv", all_rows)
        print(
            json.dumps(
                {
                    "lane": lane,
                    "best_ari": best["absolute_ari"],
                    "best_nmi": best["absolute_nmi"],
                    "delta_ari": best["delta_ari"],
                    "delta_nmi": best["delta_nmi"],
                    "initial": best.get("initial_source", "night15d_authority"),
                }
            ),
            flush=True,
        )
    write_rows(args.output / "all_run_ledger.csv", all_rows)
    result = {
        "status": "PASS",
        "lanes": summaries,
        "run_rows": len(all_rows),
        "labels_used_for_public_development_hpo": 1,
        "labels_in_model_or_energy": 0,
        "dataset_name_reads_in_model_core": 0,
        "dense_n_by_n_count": 0,
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
