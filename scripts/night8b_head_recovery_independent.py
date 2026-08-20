#!/usr/bin/env python3
"""Independent post-label recomputation from a locked evaluation snapshot."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score, completeness_score,
    fowlkes_mallows_score, homogeneity_score, normalized_mutual_info_score,
    v_measure_score,
)

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from SpaLORA.night1_evaluation import _mean_cluster_moran
from SpaLORA.night3b_metrics import mean_one_vs_rest_geary


def metric(truth: np.ndarray, prediction: np.ndarray, graph: sp.csr_matrix) -> dict:
    row, col = graph.nonzero()
    ari = float(adjusted_rand_score(truth, prediction))
    nmi = float(normalized_mutual_info_score(truth, prediction))
    neighbor = float(np.mean(prediction[row] == prediction[col]))
    geary, _ = mean_one_vs_rest_geary(prediction, graph)
    return {
        "ari": ari, "nmi": nmi, "q": (ari + nmi) / 2.0,
        "ami": float(adjusted_mutual_info_score(truth, prediction)),
        "fmi": float(fowlkes_mallows_score(truth, prediction)),
        "homogeneity": float(homogeneity_score(truth, prediction)),
        "completeness": float(completeness_score(truth, prediction)),
        "v_measure": float(v_measure_score(truth, prediction)),
        "neighbor_agreement": neighbor,
        "moran_i": float(_mean_cluster_moran(prediction, graph)),
        "geary_c": float(geary), "boundary_disagreement": 1.0 - neighbor,
    }


def signflip(values: np.ndarray) -> dict:
    observed = float(values.mean())
    means = np.asarray([np.mean(values * np.asarray(signs, dtype=np.float64))
                        for signs in itertools.product((-1.0, 1.0), repeat=10)])
    tail = int(np.sum(means >= observed - 1e-15))
    return {"observed_mean": observed, "enumerations": 1024,
            "tail_count": tail, "p_one_sided": tail / 1024.0}


def bootstrap(values: np.ndarray) -> dict:
    rng = np.random.default_rng(20260820)
    indices = rng.integers(0, 10, size=(100000, 10))
    means = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {"replicates": 100000, "seed": 20260820,
            "mean": float(values.mean()), "ci_lower": float(lower),
            "ci_upper": float(upper)}


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    snap = np.load(args.snapshot, allow_pickle=False)
    graph = sp.csr_matrix((snap["graph_data"], snap["graph_indices"],
                           snap["graph_indptr"]), shape=tuple(snap["graph_shape"]))
    truth = snap["truth"]
    rows = []
    for method in ("HR_U00", "HR_F00"):
        for seed in range(10):
            pred = snap[f"prediction_{method}_{seed}"]
            rows.append({"method": method, "seed": seed,
                         **metric(truth, pred, graph)})
    paired = []
    for seed in range(10):
        u = next(row for row in rows if row["method"] == "HR_U00" and row["seed"] == seed)
        f = next(row for row in rows if row["method"] == "HR_F00" and row["seed"] == seed)
        paired.append({"seed": seed, **{
            f"delta_{key}": f[key] - u[key] for key in
            ("ari", "nmi", "q", "neighbor_agreement", "moran_i",
             "geary_c", "boundary_disagreement")}})
    q = np.asarray([row["delta_q"] for row in paired], dtype=np.float64)
    statistics = {
        "mean_delta_ari": float(np.mean([row["delta_ari"] for row in paired])),
        "mean_delta_nmi": float(np.mean([row["delta_nmi"] for row in paired])),
        "mean_delta_q": float(q.mean()), "std_delta_q": float(q.std(ddof=1)),
        "q_wins": int(np.sum(q > 0.0)), "exact_sign_flip": signflip(q),
        "bootstrap_delta_q": bootstrap(q),
    }
    spatial = {
        "mean_delta_neighbor": float(np.mean([row["delta_neighbor_agreement"] for row in paired])),
        "mean_delta_moran": float(np.mean([row["delta_moran_i"] for row in paired])),
        "mean_delta_geary": float(np.mean([row["delta_geary_c"] for row in paired])),
        "mean_delta_boundary": float(np.mean([row["delta_boundary_disagreement"] for row in paired])),
    }
    atomic_json(Path(args.output), {"schema_version": 1, "rows": rows,
                                    "paired": paired, "statistics": statistics,
                                    "spatial": spatial})


if __name__ == "__main__":
    main()
