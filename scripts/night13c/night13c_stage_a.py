#!/usr/bin/env python3
"""Night-13C Stage A: endpoint, ablation and strong-backbone robustness."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/night13b"))

import night13b_run as n13b  # noqa: E402
from SpaLORA.night13c_core import (  # noqa: E402
    aligned_partition_change, centroid_margin, consensus_medoid,
    deterministic_residual_variants, fixed_beta_residual,
)

ROOT = Path("/root/autodl-fs/night13c_endpoint_robustness_trainable_core_20260823")
OUT = ROOT / "stage_a"
ENDPOINT_SEEDS = tuple(range(30))
DATASETS = tuple(n13b.DATASETS)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp), str(path))


def reorder(ids: Sequence[str], target: Sequence[str], z: np.ndarray) -> np.ndarray:
    ids = list(map(str, ids))
    target = list(map(str, target))
    if ids == target:
        return np.asarray(z, dtype=np.float32)
    if len(ids) != len(set(ids)) or set(ids) != set(target):
        raise RuntimeError("strong-backbone observation identity mismatch")
    lookup = {value: index for index, value in enumerate(ids)}
    return np.asarray(z, dtype=np.float32)[[lookup[value] for value in target]]


def run_endpoint_seeds(dataset: str, variant: str, embedding: np.ndarray,
                       payload: Mapping[str, object], n_init: int,
                       endpoint_seeds: Sequence[int] = ENDPOINT_SEEDS) -> Tuple[List[dict], dict, np.ndarray]:
    rows: List[dict] = []
    partitions: List[np.ndarray] = []
    endpoint_seeds = tuple(map(int, endpoint_seeds))
    for endpoint_seed in endpoint_seeds:
        started = time.perf_counter()
        model = KMeans(n_clusters=int(payload["k"]), random_state=endpoint_seed,
                       n_init=n_init)
        partition = model.fit_predict(embedding).astype(np.int64)
        partitions.append(partition)
        metrics = n13b.partition_metrics(payload["labels"], payload["label_mask"],
                                          partition, payload["metric_graph"])
        rows.append({
            "dataset": dataset, "variant": variant,
            "deterministic_embedding_sha256": n13b.array_sha256(embedding),
            "training_seed": "NOT_APPLICABLE", "endpoint_seed": endpoint_seed,
            "k": int(payload["k"]), "total_observations": int(len(payload["ids"])),
            "evaluated_observations": int(np.sum(payload["label_mask"])),
            "ordered_id_sha256": n13b.ordered_id_sha256(payload["ids"]),
            "partition_sha256": n13b.array_sha256(partition),
            "inertia": float(model.inertia_),
            "centroid_margin": centroid_margin(model.transform(embedding), partition),
            "change_rate_vs_endpoint_seed0": 0.0 if endpoint_seed == 0 else
                aligned_partition_change(partitions[0], partition),
            "wall_seconds": float(time.perf_counter() - started),
            **metrics,
        })
    consensus, medoid_seed, mean_agreement = consensus_medoid(partitions)
    consensus_metrics = n13b.partition_metrics(payload["labels"], payload["label_mask"],
                                                consensus, payload["metric_graph"])
    summary = {
        "dataset": dataset, "variant": variant,
        "deterministic_embedding_sha256": n13b.array_sha256(embedding),
        "endpoint_seed_count": len(endpoint_seeds), "n_init": n_init,
        "ari_mean": float(np.mean([x["absolute_ari"] for x in rows])),
        "ari_sd": float(np.std([x["absolute_ari"] for x in rows], ddof=1)),
        "nmi_mean": float(np.mean([x["absolute_nmi"] for x in rows])),
        "nmi_sd": float(np.std([x["absolute_nmi"] for x in rows], ddof=1)),
        "ami_mean": float(np.mean([x["ami"] for x in rows])),
        "fmi_mean": float(np.mean([x["fmi"] for x in rows])),
        "inertia_mean": float(np.mean([x["inertia"] for x in rows])),
        "centroid_margin_mean": float(np.mean([x["centroid_margin"] for x in rows])),
        "partition_change_rate_mean": float(np.mean([x["change_rate_vs_endpoint_seed0"] for x in rows])),
        "consensus_method": "PAIRWISE_ARI_MEDOID",
        "consensus_medoid_endpoint_seed": int(medoid_seed),
        "consensus_mean_pairwise_ari": mean_agreement,
        "consensus_partition_sha256": n13b.array_sha256(consensus),
        **{"consensus_" + key: value for key, value in consensus_metrics.items()},
    }
    return rows, summary, consensus


def pair_to_identity(rows: pd.DataFrame, summaries: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    identity = rows[rows.variant == "IDENTITY"][
        ["dataset", "endpoint_seed", "absolute_ari", "absolute_nmi"]
    ].rename(columns={"absolute_ari": "identity_ari", "absolute_nmi": "identity_nmi"})
    merged = rows.merge(identity, on=["dataset", "endpoint_seed"], how="left")
    merged["delta_ari"] = merged.absolute_ari - merged.identity_ari
    merged["delta_nmi"] = merged.absolute_nmi - merged.identity_nmi
    merged["win_both"] = (merged.delta_ari > 0) & (merged.delta_nmi > 0)
    identity_s = summaries[summaries.variant == "IDENTITY"][
        ["dataset", "ari_mean", "nmi_mean", "consensus_absolute_ari", "consensus_absolute_nmi"]
    ].rename(columns={"ari_mean": "identity_ari_mean", "nmi_mean": "identity_nmi_mean",
                      "consensus_absolute_ari": "identity_consensus_ari",
                      "consensus_absolute_nmi": "identity_consensus_nmi"})
    merged_s = summaries.merge(identity_s, on="dataset", how="left")
    merged_s["consensus_delta_ari"] = (merged_s.consensus_absolute_ari -
                                         merged_s.identity_consensus_ari)
    merged_s["consensus_delta_nmi"] = (merged_s.consensus_absolute_nmi -
                                         merged_s.identity_consensus_nmi)
    paired = merged.groupby(["dataset", "variant"], as_index=False).agg(
        delta_ari_mean=("delta_ari", "mean"),
        delta_nmi_mean=("delta_nmi", "mean"),
        paired_endpoint_win_both_rate=("win_both", "mean"),
    )
    merged_s = merged_s.merge(paired, on=["dataset", "variant"], how="left")
    return merged, merged_s


def strong_backbones(payloads: Mapping[str, dict]) -> List[dict]:
    values: List[dict] = []
    c00 = [
        ("A1", Path("/root/autodl-fs/night6c_raw_runs_20260817/r1/G04_SP10_F10_EUC_UNION/a1/seed_0/attempt_001")),
        ("D1", Path("/root/autodl-fs/night6d_raw_runs_20260817/G04_SP10_F10_EUC_UNION/d1/seed_0/attempt_001")),
        ("tonsil_s1", Path("/root/autodl-fs/night6c_raw_runs_20260817/r1/G04_SP10_F10_EUC_UNION/tonsil/seed_0/attempt_001")),
    ]
    for dataset, run in c00:
        archive = np.load(run / "views.npz", allow_pickle=False)
        ids = pd.read_csv(run / "observation_ids.csv").iloc[:, 0].astype(str).tolist()
        values.append({"dataset": dataset, "backbone": "C00_G04_MODEL_SEED0",
                       "embedding": reorder(ids, payloads[dataset]["ids"], archive["SpaLORA_fused"]),
                       "source": run / "views.npz"})
    f00 = Path("/root/autodl-fs/night7b_score_rnd_20260818/adapter_stage/R1/formal/R02/u020/attempt_001/worker/embedding.npy")
    f00_ids = [x for x in Path("/root/autodl-fs/night7b_score_rnd_20260818/source/u020/observation_ids.txt").read_text(encoding="utf-8").splitlines()]
    values.append({"dataset": "P22", "backbone": "F00_R02_MODEL_SEED0",
                   "embedding": reorder(f00_ids, payloads["P22"]["ids"], np.load(f00, allow_pickle=False)),
                   "source": f00})
    n02run = Path("/root/autodl-fs/night9b_racf_20260820/r1/r1-u009/attempt_001")
    config = json.loads((n02run / "resolved_config.json").read_text(encoding="utf-8"))
    n02ids = pd.read_csv(config["observation_ids_path"]).iloc[:, 0].astype(str).tolist()
    values.append({"dataset": "P22", "backbone": "N02_HIER_MODEL_SEED0",
                   "embedding": reorder(n02ids, payloads["P22"]["ids"],
                                        np.load(n02run / "views.npz", allow_pickle=False)["SpaLORA_fused"]),
                   "source": n02run / "views.npz"})
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-init", type=int, default=20)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    all_rows: List[dict] = []
    all_summaries: List[dict] = []
    payloads: Dict[str, dict] = {}
    diagnostics: List[dict] = []
    for dataset in DATASETS:
        payload = n13b.base_payload(dataset)
        payloads[dataset] = payload
        variants = deterministic_residual_variants(payload["embedding"],
                                                   payload["operators"][0],
                                                   payload["operators"][1])
        diagnostics.append({"dataset": dataset, "discrepancy": variants.discrepancy,
                            "gate": variants.gate, "b10_beta": variants.beta,
                            "embedding_shape": list(variants.identity.shape),
                            "embedding_sha256": n13b.array_sha256(variants.identity)})
        primary = {
            "IDENTITY": variants.identity,
            "B10_MIXTURE": variants.b10_mixture,
        }
        ablations = {"P4_ONLY": variants.p4_only, "P18_ONLY": variants.p18_only}
        fixed = [0.04, 0.16, 0.32]
        if dataset == "MISAR_E15_5_S1":
            fixed = [x / 100.0 for x in range(0, 9)]
        for beta in fixed:
            ablations[f"FIXED_BETA_{beta:.2f}"] = fixed_beta_residual(
                variants.identity, variants.multiscale_direction, beta)
        for name, embedding in primary.items():
            rows, summary, consensus = run_endpoint_seeds(dataset, name, embedding,
                                                           payload, args.n_init)
            all_rows.extend(rows)
            all_summaries.append(summary)
            np.save(OUT / f"consensus_{dataset}_{name}.npy", consensus, allow_pickle=False)
        # Ablations use ten registered endpoint seeds; the decision-driving
        # identity/B10 comparison retains all 30 as required by the contract.
        for name, embedding in ablations.items():
            rows, summary, consensus = run_endpoint_seeds(
                dataset, name, embedding, payload, args.n_init, range(10))
            all_rows.extend(rows)
            all_summaries.append(summary)
            np.save(OUT / f"consensus_{dataset}_{name}.npy", consensus, allow_pickle=False)
    rows_df, summary_df = pair_to_identity(pd.DataFrame(all_rows),
                                           pd.DataFrame(all_summaries))
    rows_df.to_csv(OUT / "b10_endpoint_robustness_rows.csv", index=False)
    summary_df.to_csv(OUT / "b10_endpoint_robustness_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "b10_embedding_diagnostics.csv", index=False)

    strong_rows: List[dict] = []
    strong_summaries: List[dict] = []
    strong_audit: List[dict] = []
    strong_items = strong_backbones(payloads)
    for item in strong_items:
        dataset = item["dataset"]
        payload = payloads[dataset]
        source = Path(item["source"])
        variants = deterministic_residual_variants(item["embedding"],
                                                   payload["operators"][0],
                                                   payload["operators"][1])
        for suffix, embedding in (("IDENTITY", variants.identity),
                                  ("B10_MIXTURE", variants.b10_mixture)):
            name = f"{item['backbone']}__{suffix}"
            rows, summary, _ = run_endpoint_seeds(dataset, name, embedding,
                                                  payload, args.n_init)
            strong_rows.extend(rows)
            strong_summaries.append(summary)
        strong_audit.append({"dataset": dataset, "backbone": item["backbone"],
                             "source": str(source), "size": source.stat().st_size,
                             "sha256": file_sha256(source),
                             "ordered_id_sha256": n13b.ordered_id_sha256(payload["ids"])})
    strong_df = pd.DataFrame(strong_rows)
    for backbone in sorted(set(x["backbone"] for x in strong_items)):
        baseline = strong_df[strong_df.variant == backbone + "__IDENTITY"][
            ["dataset", "endpoint_seed", "absolute_ari", "absolute_nmi"]
        ].rename(columns={"absolute_ari": "identity_ari", "absolute_nmi": "identity_nmi"})
        mask = strong_df.variant == backbone + "__B10_MIXTURE"
        target = strong_df.loc[mask].merge(baseline, on=["dataset", "endpoint_seed"], how="left")
        strong_df.loc[mask, "delta_ari"] = (target.absolute_ari - target.identity_ari).to_numpy()
        strong_df.loc[mask, "delta_nmi"] = (target.absolute_nmi - target.identity_nmi).to_numpy()
    strong_df.to_csv(OUT / "strong_backbone_residual_rows.csv", index=False)
    pd.DataFrame(strong_summaries).to_csv(OUT / "strong_backbone_residual_summary.csv", index=False)
    pd.DataFrame(strong_audit).to_csv(OUT / "strong_backbone_artifact_audit.csv", index=False)

    native = pd.DataFrame([
        {"dataset": "A1", "method": "C00_H05_NATIVE", "ari": .2692, "nmi": .4087,
         "semantic_lane": "NATIVE_FULL_PIPELINE_CONTEXT"},
        {"dataset": "D1", "method": "C00_H05_NATIVE", "ari": .2412, "nmi": .3777,
         "semantic_lane": "NATIVE_FULL_PIPELINE_CONTEXT"},
        {"dataset": "P22", "method": "F00_NATIVE", "ari": .4677, "nmi": .6334,
         "semantic_lane": "NATIVE_FULL_PIPELINE_CONTEXT"},
        {"dataset": "P22", "method": "N02_NATIVE", "ari": .5063, "nmi": .6562,
         "semantic_lane": "NATIVE_FULL_PIPELINE_CONTEXT"},
    ])
    native.to_csv(OUT / "native_full_pipeline_context.csv", index=False)
    summary_df.assign(semantic_lane="COMMON_HEAD_ROBUSTNESS").to_csv(
        OUT / "common_head_robustness.csv", index=False)

    lead = summary_df[(summary_df.variant == "B10_MIXTURE") &
                      (summary_df.dataset.isin(["A1", "P22"]))]
    gates = []
    for _, row in lead.iterrows():
        gates.append({"dataset": row.dataset,
                      "majority_endpoint_seeds_win_both": bool(row.paired_endpoint_win_both_rate > .5),
                      "consensus_delta_ari_positive": bool(row.consensus_delta_ari > 0),
                      "consensus_delta_nmi_positive": bool(row.consensus_delta_nmi > 0),
                      "paired_endpoint_win_both_rate": float(row.paired_endpoint_win_both_rate),
                      "consensus_delta_ari": float(row.consensus_delta_ari),
                      "consensus_delta_nmi": float(row.consensus_delta_nmi)})
    robust = all(x["majority_endpoint_seeds_win_both"] and
                 x["consensus_delta_ari_positive"] and x["consensus_delta_nmi_positive"]
                 for x in gates)
    decision = {
        "schema": "spalora.night13c.stage_a.v1",
        "stage_a_state": "B10_GRAPH_HEAD_ENDPOINT_ROBUST" if robust else
                         "B10_ENDPOINT_ARTIFACT_OR_FRAGILE",
        "b10_paper_role": "AUXILIARY_HEAD_OR_ABLATION_ONLY",
        "threshold_centric_search_allowed": bool(robust),
        "stage_b_non_b10_trainable_search_allowed": True,
        "endpoint_seeds": list(ENDPOINT_SEEDS), "n_init": args.n_init,
        "gates": gates, "deterministic_embeddings_registered_once": True,
        "dense_n_by_n_count": 0, "new_download_count": 0,
        "wall_seconds": float(time.perf_counter() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }
    atomic_json(OUT / "stage_a_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
