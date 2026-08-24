#!/usr/bin/env python3
"""Post-lock public-reference evaluator for generic start ensembles."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scripts.night16a.night16a_calibrated_arena import LANE_DATASET, lane_semantics


def encode(value):
    return np.unique(np.asarray(value, dtype=str), return_inverse=True)[1]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ensemble", type=Path, required=True); p.add_argument("--kit", type=Path, required=True); p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    registry = pd.read_csv(args.ensemble / "generic_ensemble_registry.csv")
    archive = np.load(args.ensemble / "generic_ensemble_partitions.npz", allow_pickle=False)
    output = []
    for lane, group in registry.groupby("lane"):
        data = np.load(args.kit / f"{LANE_DATASET[lane]}.npz", allow_pickle=False, mmap_mode="r")
        _, labels, mask = lane_semantics(data, lane)
        truth = encode(labels[mask])
        for _, row in group.iterrows():
            partition = archive[row.partition_key]
            item = row.to_dict(); item.update({"absolute_ari": adjusted_rand_score(truth, partition[mask]), "absolute_nmi": normalized_mutual_info_score(truth, partition[mask]), "min_cluster_size": int(np.bincount(partition).min())}); output.append(item)
    pd.DataFrame(output).to_csv(args.output, index=False)


if __name__ == "__main__": main()
