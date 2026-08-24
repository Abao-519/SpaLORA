#!/usr/bin/env python3
"""Cross-study meta-selector probe using label-free candidate descriptors.

For each held-out study, the fitted selector sees only candidate descriptors
for the held-out data.  Public benchmark metrics are confined to training
studies and the post-lock evaluator.  This probe compares model classes; the
winning class is not yet the frozen Night-16A policy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge


STUDY = {
    "A1": "LYMPH_NODE",
    "D1": "LYMPH_NODE",
    "tonsil_s1": "TONSIL",
    "tonsil_s2": "TONSIL",
    "tonsil_s3": "TONSIL",
    "P22": "P22",
    "P22_3DOT_K18": "P22",
    "MISAR_E15_5_S1": "MISAR",
    "MISAR_E15_5_S1_K12": "MISAR",
}

FAMILY = {
    "A1": "RNA_PROTEIN",
    "D1": "RNA_PROTEIN",
    "tonsil_s1": "RNA_PROTEIN",
    "tonsil_s2": "RNA_PROTEIN",
    "tonsil_s3": "RNA_PROTEIN",
    "P22": "RNA_ATAC",
    "P22_3DOT_K18": "RNA_ATAC",
    "MISAR_E15_5_S1": "RNA_ATAC",
    "MISAR_E15_5_S1_K12": "RNA_ATAC",
}

CURRENT_BEST = {
    "A1": (0.2760026589984753, 0.42173983941218224),
    "D1": (0.28887600833487515, 0.41376160498731795),
    "tonsil_s1": (0.23622005623915696, 0.3168843080420388),
    "tonsil_s2": (0.2575602926237894, 0.31187103018718415),
    "tonsil_s3": (0.349880567011789, 0.3092665192529602),
    "P22": (0.5939121542899773, 0.7142428513607495),
    "P22_3DOT_K18": (0.7396852743147485, 0.7542182135683022),
    "MISAR_E15_5_S1": (0.5414237853091904, 0.6667977615565593),
    "MISAR_E15_5_S1_K12": (0.45317629302756757, 0.5986289047196983),
}


FEATURES = [
    "graph_same_fraction",
    "partition_centrality_median_ari",
    "normalized_cluster_entropy",
    "min_cluster_relative_to_equal",
    "separation__retained",
    "separation__view1",
    "separation__view2",
    "margin__retained",
    "margin__view1",
    "margin__view2",
    "separation__morph_handcrafted",
    "separation__morph_resnet",
    "separation__molecular_morph_coord",
    "margin__morph_handcrafted",
    "margin__morph_resnet",
    "margin__molecular_morph_coord",
]


def robust_within_lane(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    blocks = []
    for lane, group in frame.groupby("lane", sort=False):
        values = group[columns].astype(float).fillna(0.0).to_numpy()
        median = np.nanmedian(values, axis=0)
        mad = np.nanmedian(np.abs(values - median), axis=0)
        scale = np.maximum(1.4826 * mad, np.nanstd(values, axis=0))
        scale = np.maximum(scale, 1e-8)
        transformed = np.clip((values - median) / scale, -6.0, 6.0)
        blocks.append((group.index.to_numpy(), transformed))
    output = np.zeros((len(frame), len(columns)), dtype=np.float64)
    for indices, values in blocks:
        output[indices] = values
    return output


def target_within_lane(frame: pd.DataFrame) -> np.ndarray:
    raw = frame["absolute_ari"].astype(float).to_numpy() + 0.35 * frame["absolute_nmi"].astype(float).to_numpy()
    target = np.zeros(len(frame), dtype=np.float64)
    for _, group in frame.groupby("lane", sort=False):
        indices = group.index.to_numpy()
        values = raw[indices]
        low, high = float(np.min(values)), float(np.max(values))
        target[indices] = (values - low) / max(high - low, 1e-8)
    return target


def model_factories():
    factories = {}
    for alpha in (0.1, 1.0, 10.0, 100.0):
        factories[f"ridge_a{alpha:g}"] = lambda alpha=alpha: Ridge(alpha=alpha)
    for depth in (2, 3, 4, 5):
        for leaf in (5, 10, 20):
            factories[f"rf_d{depth}_l{leaf}"] = lambda depth=depth, leaf=leaf: RandomForestRegressor(
                n_estimators=300,
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=20260824,
                n_jobs=1,
            )
            factories[f"extra_d{depth}_l{leaf}"] = lambda depth=depth, leaf=leaf: ExtraTreesRegressor(
                n_estimators=300,
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=20260824,
                n_jobs=1,
            )
    return factories


def source_features(frame: pd.DataFrame) -> np.ndarray:
    names = frame["candidate"].astype(str)
    classes = ["authority::", "night15g::", "bank::", "teacher::", "kmeans::", "gmm_"]
    return np.column_stack([names.str.startswith(prefix).astype(float).to_numpy() for prefix in classes])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="candidate-name prefix excluded from the eligible deployment bank",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.ledger)
    if args.exclude_prefix:
        eligible = np.ones(len(frame), dtype=bool)
        for prefix in args.exclude_prefix:
            eligible &= ~frame["candidate"].astype(str).str.startswith(prefix).to_numpy()
        frame = frame.loc[eligible].copy().reset_index(drop=True)
    for column in FEATURES:
        if column not in frame:
            frame[column] = 0.0
    # A universal structural safeguard: a candidate with less than 2% of the
    # equal-size N/K target in any cluster is not eligible for the deployable
    # profile.  It remains in the source ledger and oracle analysis.
    frame = frame[frame["min_cluster_relative_to_equal"].astype(float) >= 0.02].copy().reset_index(drop=True)
    frame["study"] = frame["lane"].map(STUDY)
    frame["family"] = frame["lane"].map(FAMILY)
    numeric = robust_within_lane(frame, FEATURES)
    categorical = source_features(frame)
    x = np.concatenate((numeric, categorical), axis=1)
    y = target_within_lane(frame)
    factories = model_factories()
    rows = []
    predictions = {}
    for scope in ("GLOBAL", "FAMILY"):
        for model_name, factory in factories.items():
            for held_out in sorted(frame["study"].unique()):
                test = frame["study"].to_numpy() == held_out
                if scope == "GLOBAL":
                    train = ~test
                else:
                    held_family = frame.loc[test, "family"].iloc[0]
                    train = (~test) & (frame["family"].to_numpy() == held_family)
                if not np.any(train):
                    continue
                model = factory()
                model.fit(x[train], y[train])
                predicted = model.predict(x[test])
                test_indices = np.flatnonzero(test)
                for lane in frame.loc[test, "lane"].unique():
                    lane_indices = test_indices[frame.loc[test_indices, "lane"].to_numpy() == lane]
                    selected_index = int(lane_indices[np.argmax(predicted[frame.loc[test, "lane"].to_numpy() == lane])])
                    selected = frame.loc[selected_index]
                    current = CURRENT_BEST[lane]
                    row = {
                        "scope": scope,
                        "model": model_name,
                        "held_out_study": held_out,
                        "lane": lane,
                        "selected_candidate": selected["candidate"],
                        "partition_sha256": selected["partition_sha256"],
                        "absolute_ari": float(selected["absolute_ari"]),
                        "absolute_nmi": float(selected["absolute_nmi"]),
                        "delta_ari_vs_current": float(selected["absolute_ari"]) - current[0],
                        "delta_nmi_vs_current": float(selected["absolute_nmi"]) - current[1],
                        "min_cluster_size": int(selected["min_cluster_size"]),
                        "predicted_score": float(np.max(predicted[frame.loc[test, "lane"].to_numpy() == lane])),
                    }
                    rows.append(row)
                    predictions[(scope, model_name, held_out, lane)] = row
    output = pd.DataFrame(rows)
    output.to_csv(args.output / "meta_selector_loo_ledger.csv", index=False)
    summaries = []
    for (scope, model_name), group in output.groupby(["scope", "model"]):
        summaries.append(
            {
                "scope": scope,
                "model": model_name,
                "lanes": len(group),
                "mean_ari": float(group["absolute_ari"].mean()),
                "mean_nmi": float(group["absolute_nmi"].mean()),
                "mean_delta_ari_vs_current": float(group["delta_ari_vs_current"].mean()),
                "mean_delta_nmi_vs_current": float(group["delta_nmi_vs_current"].mean()),
                "dual_nonnegative_lanes": int(((group["delta_ari_vs_current"] >= -1e-12) & (group["delta_nmi_vs_current"] >= -1e-12)).sum()),
                "mean_selector_target": float((group["absolute_ari"] + 0.35 * group["absolute_nmi"]).mean()),
            }
        )
    summary_frame = pd.DataFrame(summaries).sort_values(
        ["dual_nonnegative_lanes", "mean_selector_target"], ascending=False
    )
    summary_frame.to_csv(args.output / "meta_selector_model_summary.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "rows": len(output),
                "candidate_rows": len(frame),
                "excluded_candidate_prefixes": args.exclude_prefix,
                "feature_names": FEATURES + ["source::authority", "source::night15g", "source::bank", "source::teacher", "source::kmeans", "source::gmm"],
                "top_models": summary_frame.head(10).to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
