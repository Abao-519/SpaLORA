#!/usr/bin/env python3
"""Five-row, same-initialization contribution paths for each primary lane.

The upstream Night-15F energy is regenerated from its registered Night-15E
start.  Each variant then receives the identical Night-16B generic-repair
chain.  This makes reliability/self-return contrasts genuinely matched while
also revealing when the tuned headline depends on a later Night-15G/16A start
that is not recovered by this clean common path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score

from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig, csr_from_archive
from SpaLORA.night15f_multiscale_expansion import ExpansionEnergyConfig, continuous_multiscale_expansion, prepare_expansion_evidence
from SpaLORA.night16b_unified_structured_decoder import RepairConfig, generic_repair, partition_sha256
from scripts.night16b.night16b_candidate_pipeline import (
    LANE_DATASET,
    _feature_bank,
    encode_partition,
    graph_from_archive,
    lane_k,
    lane_reference,
    moran_geary,
)


PRIMARY = ["A1", "D1", "tonsil_s1", "tonsil_s2", "tonsil_s3", "P22", "MISAR_E15_5_S1"]


def producer(args: argparse.Namespace) -> None:
    energy_registry = json.loads(args.energy_registry.read_text(encoding="utf-8"))
    decoder_registry = json.loads(args.decoder_registry.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    partition_dir = args.output / "partitions"
    partition_dir.mkdir(exist_ok=True)
    rows = []
    for lane in PRIMARY:
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(args.banks / f"{dataset}_selected_partition_bank.npz", allow_pickle=False)
        graph = graph_from_archive(data)
        graphs = (csr_from_archive(data, "operator4"), graph, csr_from_archive(data, "operator18"))
        retained = np.asarray(bank[f"{lane}__retained_embedding"], dtype=np.float32)
        evidence = prepare_expansion_evidence(graphs, retained, data["view1"], data["view2"])
        feature_bank, _ = _feature_bank(data, bank, lane, args.morphology)
        initial = np.load(args.initials / f"{lane}.npy", allow_pickle=False)
        k = lane_k(data, lane)
        raw = energy_registry["lanes"][lane]["config"]
        energy = ExpansionEnergyConfig(**{**raw, "local": ContinuousEnergyConfig(**raw["local"])})
        repair_chain = [RepairConfig(**item) for item in decoder_registry["lanes"][lane]["replay_config_chain"]]

        def apply_repair(partition: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
            diagnostics = []
            for config in repair_chain:
                feature = feature_bank.get(config.feature_mode, feature_bank["retained"])
                partition, detail = generic_repair(partition, feature, graph, k, config)
                diagnostics.append({"config": asdict(config), **detail})
            return partition, diagnostics

        variants = []
        variants.append(("START_PARTITION", initial.copy(), {"energy": "disabled", "repair": "disabled"}))
        full_energy, full_detail = continuous_multiscale_expansion(initial, k, evidence, energy)
        full, repair_detail = apply_repair(full_energy.copy())
        variants.append(("FULL_COMMON_PATH", full, {"energy": full_detail, "repair": repair_detail}))
        without_reliability, detail = continuous_multiscale_expansion(initial, k, evidence, replace(energy, pairwise_beta=0.0))
        without_reliability, repair_detail = apply_repair(without_reliability)
        variants.append(("CROSS_MODAL_RELIABILITY_DISABLED", without_reliability, {"energy": detail, "repair": repair_detail, "scope": "pairwise reliability channel disabled"}))
        without_return, detail = continuous_multiscale_expansion(initial, k, evidence, replace(energy, self_return_strength=0.0))
        without_return, repair_detail = apply_repair(without_return)
        variants.append(("SELF_RETURN_DISABLED", without_return, {"energy": detail, "repair": repair_detail}))
        variants.append(("GENERIC_REPAIR_DISABLED", full_energy, {"energy": full_detail, "repair": "disabled"}))
        payload = {}
        for variant, partition, diagnostics in variants:
            digest = partition_sha256(partition)
            payload[variant] = np.asarray(partition, dtype=np.int32)
            rows.append(
                {
                    "lane": lane,
                    "variant": variant,
                    "n": len(partition),
                    "k": k,
                    "initial_partition_sha256": partition_sha256(initial),
                    "partition_sha256": digest,
                    "cluster_sizes": json.dumps(np.bincount(partition, minlength=k).astype(int).tolist(), separators=(",", ":")),
                    "diagnostics": diagnostics,
                    "labels_opened": 0,
                    "dense_n_by_n_count": 0,
                }
            )
        np.savez_compressed(partition_dir / f"{lane}.npz", **payload)
    (args.output / "contribution_producer.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(rows),
                "same_initialization_within_lane": True,
                "labels_opened": 0,
                "rows_detail": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def evaluator(args: argparse.Namespace) -> None:
    producer_data = json.loads((args.input / "contribution_producer.json").read_text(encoding="utf-8"))
    rows = []
    for lane in PRIMARY:
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        labels, mask = lane_reference(data, lane)
        truth = encode_partition(labels[mask])
        graph = graph_from_archive(data)
        archive = np.load(args.input / "partitions" / f"{lane}.npz", allow_pickle=False)
        for item in [row for row in producer_data["rows_detail"] if row["lane"] == lane]:
            partition = np.asarray(archive[item["variant"]], dtype=np.int32)
            if partition_sha256(partition) != item["partition_sha256"]:
                raise RuntimeError(f"{lane}/{item['variant']}: partition mismatch")
            moran, geary = moran_geary(partition, graph)
            rows.append(
                {
                    "lane": lane,
                    "variant": item["variant"],
                    "absolute_ari": float(adjusted_rand_score(truth, partition[mask])),
                    "absolute_nmi": float(normalized_mutual_info_score(truth, partition[mask])),
                    "ami": float(adjusted_mutual_info_score(truth, partition[mask])),
                    "fmi": float(fowlkes_mallows_score(truth, partition[mask])),
                    "morans_i": moran,
                    "gearys_c": geary,
                    "min_cluster_size": int(np.bincount(partition).min()),
                    "cluster_sizes": item["cluster_sizes"],
                    "partition_sha256": item["partition_sha256"],
                }
            )
    frame = pd.DataFrame(rows)
    full = frame[frame.variant == "FULL_COMMON_PATH"].set_index("lane")
    for metric in ("absolute_ari", "absolute_nmi"):
        frame[f"delta_{metric}_vs_full"] = frame.apply(
            lambda row: float(row[metric] - full.loc[row.lane, metric]), axis=1
        )
    frame.to_csv(args.output, index=False)
    (args.output.parent / f"{args.output.stem}_audit.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(frame),
                "labels_in_producer": 0,
                "public_reference_array_open_events_in_evaluator": len(PRIMARY),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("produce")
    p.add_argument("--energy-registry", type=Path, required=True)
    p.add_argument("--decoder-registry", type=Path, required=True)
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--banks", type=Path, required=True)
    p.add_argument("--initials", type=Path, required=True)
    p.add_argument("--morphology", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--kit", type=Path, required=True)
    e.add_argument("--input", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    producer(args) if args.action == "produce" else evaluator(args)


if __name__ == "__main__":
    main()
