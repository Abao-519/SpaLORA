#!/usr/bin/env python3
"""Independent post-label recomputation from the locked per-seed metric table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night6d_pipeline import atomic_json

OUT = REPO / "outputs/night6d_handoff"
G00 = "G00_SP18_F20_CORR_UNION"
G04 = "G04_SP10_F10_EUC_UNION"
H00 = "H00_FUSED_PCA20_MCLUST_EEE"
H05 = "H05_EQUAL3_AFFINITY_SPECTRAL"


def paired(frame: pd.DataFrame, dataset: str, graph: str, head: str) -> pd.DataFrame:
    ref = frame[(frame.dataset == dataset) & (frame.graph_id == G00) &
                (frame.head_id == H00)].sort_values("seed")
    other = frame[(frame.dataset == dataset) & (frame.graph_id == graph) &
                  (frame.head_id == head)].sort_values("seed")
    if len(ref) != 10 or len(other) != 10 or not np.array_equal(ref.seed, other.seed):
        raise RuntimeError("paired metric table is incomplete")
    result = pd.DataFrame({"seed": ref.seed.to_numpy(dtype=int)})
    for source, target in (("ari", "delta_ari"), ("nmi", "delta_nmi"), ("q", "delta_q"),
                           ("neighbor_agreement", "delta_neighbor"), ("moran_i", "delta_moran"),
                           ("geary_c", "delta_geary"),
                           ("boundary_disagreement", "delta_boundary")):
        result[target] = other[source].to_numpy(dtype=float) - ref[source].to_numpy(dtype=float)
    return result


def exact(values: np.ndarray) -> float:
    observed = float(np.mean(values))
    tail = 0
    for mask in range(1024):
        signs = np.asarray([1.0 if mask & (1 << bit) else -1.0 for bit in range(10)])
        tail += float(np.mean(values * signs)) >= observed - 1e-15
    return tail / 1024.0


def holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda k: (values[k], k))
    result = {}
    running = 0.0
    for index, key in enumerate(ordered):
        running = max(running, (len(ordered) - index) * values[key])
        result[key] = min(1.0, running)
    return result


def bootstrap(values: pd.DataFrame) -> dict:
    rng = np.random.default_rng(20260817)
    draw = rng.integers(0, 10, size=(100000, 10))
    result = {}
    for key in ("delta_ari", "delta_nmi", "delta_q"):
        means = values[key].to_numpy()[draw].mean(axis=1)
        low, high = np.percentile(means, (2.5, 97.5))
        result[key] = (float(low), float(high))
    return result


def close(actual: float, expected: float, name: str, atol: float = 1e-12) -> None:
    if not np.isclose(actual, expected, atol=atol, rtol=0):
        raise RuntimeError(f"independent recomputation mismatch {name}: {actual} != {expected}")


def main() -> None:
    frame = pd.read_csv(OUT / "d1_p22_per_seed_metrics.csv")
    primary = json.loads((OUT / "primary_confirmatory_tests.json").read_text())
    secondary = json.loads((OUT / "secondary_factorial_tests.json").read_text())
    spatial = json.loads((OUT / "spatial_protection.json").read_text())
    decision = json.loads((OUT / "night6d_decision.json").read_text())
    if len(frame) != 80 or frame.duplicated(["dataset", "graph_id", "head_id", "seed"]).any():
        raise RuntimeError("locked per-seed table is not exactly 80 unique rows")
    raw = {}
    checks = 0
    for dataset in ("d1", "p22"):
        delta = paired(frame, dataset, G04, H05)
        official = primary["datasets"][dataset]
        for name in ("ari", "nmi", "q"):
            close(float(delta[f"delta_{name}"].mean()), official[f"mean_delta_{name}"],
                  f"{dataset} mean delta {name}")
            checks += 1
        p = exact(delta.delta_q.to_numpy())
        close(p, official["exact_sign_flip"]["raw_p"], f"{dataset} exact p")
        raw[dataset] = p
        checks += 1
        boot = bootstrap(delta)
        for name in ("delta_ari", "delta_nmi", "delta_q"):
            close(boot[name][0], official["bootstrap"][name]["ci_lower"], f"{dataset} {name} lower")
            close(boot[name][1], official["bootstrap"][name]["ci_upper"], f"{dataset} {name} upper")
            checks += 2
        means = {name: float(delta[name].mean()) for name in
                 ("delta_neighbor", "delta_moran", "delta_geary", "delta_boundary")}
        for name, value in means.items():
            close(value, spatial["datasets"][dataset][name], f"{dataset} spatial {name}")
            checks += 1
        failed = ((means["delta_neighbor"] < -.03 and means["delta_moran"] < -.03) or
                  (means["delta_geary"] > .03 and
                   (means["delta_neighbor"] < -.03 or means["delta_moran"] < -.03)))
        if failed != spatial["datasets"][dataset]["spatial_gate_failed"]:
            raise RuntimeError("independent spatial-gate decision mismatch")
        checks += 1
    adjusted = holm(raw)
    for dataset in ("d1", "p22"):
        close(adjusted[dataset], primary["datasets"][dataset]["holm_adjusted_p"],
              f"{dataset} primary Holm")
        checks += 1

    definitions = {"head_only": (G00, H05), "graph_only": (G04, H00)}
    secondary_raw = {}
    secondary_delta = {}
    for dataset in ("d1", "p22"):
        for name, (graph, head) in definitions.items():
            value = paired(frame, dataset, graph, head)
            key = f"{dataset}:{name}"
            secondary_delta[key] = value
            secondary_raw[key] = exact(value.delta_q.to_numpy())
        interaction = paired(frame, dataset, G04, H05)
        graph = secondary_delta[f"{dataset}:graph_only"]
        head = secondary_delta[f"{dataset}:head_only"]
        for column in [x for x in interaction.columns if x.startswith("delta_")]:
            interaction[column] = interaction[column] - graph[column] - head[column]
        key = f"{dataset}:interaction"
        secondary_delta[key] = interaction
        secondary_raw[key] = exact(interaction.delta_q.to_numpy())
    secondary_holm = holm(secondary_raw)
    for key, delta in secondary_delta.items():
        official = secondary["contrasts"][key]
        close(float(delta.delta_q.mean()), official["mean_delta_q"], f"{key} mean delta Q")
        close(secondary_raw[key], official["exact_sign_flip"]["raw_p"], f"{key} exact p")
        close(secondary_holm[key], official["holm_adjusted_p"], f"{key} secondary Holm")
        checks += 3
    if decision["terminal_status"] != "NIGHT6D_D1_P22_BALANCED_CONFIRMED":
        raise RuntimeError("terminal status differs from independently recomputed gates")
    atomic_json(OUT / "independent_statistical_recompute_audit.json", {
        "status": "PASS", "source": "locked per-seed metric table only",
        "ground_truth_reopened": False, "training_or_transform_reopened": False,
        "metric_rows": 80, "unique_primary_keys": 80, "numeric_checks": checks,
        "primary_exact_enumerations_per_dataset": 1024,
        "secondary_exact_family_size": 6, "bootstrap_resamples_per_contrast": 100000,
        "bootstrap_seed": 20260817, "absolute_tolerance": 1e-12,
        "recomputed_terminal_status": decision["terminal_status"],
    })
    print(json.dumps({"status": "PASS", "numeric_checks": checks,
                      "terminal_status": decision["terminal_status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
