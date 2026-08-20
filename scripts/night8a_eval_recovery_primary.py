#!/usr/bin/env python3
"""Primary evaluation over the frozen Night-8A recovery view."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
    calinski_harabasz_score, completeness_score, davies_bouldin_score,
    fowlkes_mallows_score, homogeneity_score, normalized_mutual_info_score,
    silhouette_score, v_measure_score)

REPO = Path(os.environ.get("NIGHT8A_RECOVERY_REPO", "/root/autodl-fs/SpaLORA-night8a-eval-recovery"))
OUT = REPO / "outputs/night8a_eval_recovery"
RAW = Path("/root/autodl-fs/night8a_eval_recovery_20260820")
LABEL_ROOT = Path("/root/autodl-fs/night7a_consensus_20260818/evaluation_label_snapshots")
COORDS = {"a1": Path("/root/autodl-fs/night6c_cache_20260817/base/a1/coordinates.npy"),
          "tonsil": Path("/root/autodl-fs/night6c_cache_20260817/base/tonsil/coordinates.npy"),
          "d1": Path("/root/autodl-fs/night6d_cache_20260817/base/d1/coordinates.npy"),
          "p22": Path("/root/autodl-fs/night3af_p0d_builds_20260810/process_a/p22/coordinates.npy")}

import sys
sys.path.insert(0, str(REPO))
from SpaLORA.night1_evaluation import _mean_cluster_moran
from SpaLORA.night3b_metrics import mean_one_vs_rest_geary, symmetric_knn_adjacency


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def array_sha(x: np.ndarray) -> str:
    value = np.ascontiguousarray(x)
    h = hashlib.sha256(); h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def order_sha(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(ids.astype(str).tolist()).encode()).hexdigest()


def no_diag(a: sp.spmatrix) -> sp.csr_matrix:
    b = a.tocsr().astype(np.float64); b.setdiag(0); b.eliminate_zeros(); b.sum_duplicates(); b.sort_indices(); return b


def spatial(pred: np.ndarray, adjacency: sp.spmatrix) -> dict[str, float]:
    a = no_diag(adjacency); rows, cols = a.nonzero()
    neighbor = float(np.mean(pred[rows] == pred[cols])); geary, _ = mean_one_vs_rest_geary(pred, a)
    return {"neighbor_agreement": neighbor, "moran_i": float(_mean_cluster_moran(pred, a)),
            "geary_c": float(geary), "boundary_disagreement": 1.0 - neighbor}


def modularity(pred: np.ndarray, adjacency: sp.spmatrix) -> float:
    a = no_diag(adjacency); degree = np.asarray(a.sum(1)).ravel(); total = float(degree.sum())
    if total <= 0: return float("nan")
    result = 0.0
    for group in np.unique(pred):
        ix = np.flatnonzero(pred == group)
        result += float(a[ix][:, ix].sum()) / total - (float(degree[ix].sum()) / total) ** 2
    return float(result)


def embedding(path: Path) -> np.ndarray:
    if path.suffix == ".npy": return np.load(path, allow_pickle=False)
    z = np.load(path, allow_pickle=False)
    for key in ("SpaLORA_fused", "embedding", "fused"):
        if key in z: return z[key]
    raise RuntimeError(f"no fused embedding in {path}")


def descriptive(x: np.ndarray, pred: np.ndarray, adjacency: sp.spmatrix) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64); unique = np.unique(pred)
    if len(unique) < 2:
        return {k: float("nan") for k in ("silhouette", "davies_bouldin", "calinski_harabasz", "graph_connectivity", "modularity")}
    return {"silhouette": float(silhouette_score(x, pred, sample_size=min(len(x), 2000), random_state=20260820)),
            "davies_bouldin": float(davies_bouldin_score(x, pred)),
            "calinski_harabasz": float(calinski_harabasz_score(x, pred)),
            "graph_connectivity": spatial(pred, adjacency)["neighbor_agreement"],
            "modularity": modularity(pred, adjacency)}


def load_prediction(row: dict) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(row["cluster_path"])
    ids = table["observation_id"].astype(str).to_numpy()
    if str(row["partition_path"]).endswith(".npy"):
        pred = np.load(row["partition_path"], allow_pickle=False).astype(np.int64)
    else:
        pred = table["cluster"].to_numpy(dtype=np.int64)
    if len(ids) != len(pred): raise RuntimeError("cluster/id size mismatch")
    return ids, pred


def main() -> None:
    view = json.loads((OUT / "evaluation_view_manifest.json").read_text())
    rule = json.loads((OUT / "recovery_rule_lock.json").read_text())
    if view["status"] != "LOCKED_PRE_LABEL" or rule["status"] != "LOCKED_PRE_LABEL":
        raise RuntimeError("pre-label view/rule not locked")
    labels = {}; label_audit = {}
    for dataset in ("a1", "tonsil", "d1", "p22"):
        path = LABEL_ROOT / f"{dataset}_labels_locked.npz"; z = np.load(path, allow_pickle=False)
        ids = z["observation_id"].astype(str); y = z["label"].astype(str)
        labels[dataset] = (ids, y)
        label_audit[dataset] = {"path": str(path), "file_sha256": file_sha(path), "rows": len(ids),
                                "K": len(np.unique(y)), "label_vector_sha256": array_sha(y),
                                "authorized_role": "night8a_eval_recovery_primary", "used_for_training_or_transform": False}
    cache = RAW / "adjacency"; cache.mkdir(parents=True, exist_ok=True)
    adjacency = {}
    for dataset, path in COORDS.items():
        target = cache / f"{dataset}_k18.npz"
        if not target.exists(): sp.save_npz(target, symmetric_knn_adjacency(np.load(path, allow_pickle=False), 18), compressed=True)
        adjacency[dataset] = sp.load_npz(target).tocsr()
    rows = []; desc_cache = {}
    for row in view["rows"]:
        base = {k: row[k] for k in ("stage", "config_id", "dataset", "family", "seed", "unit_id", "role")}
        if row["status"] == "INELIGIBLE_UPSTREAM_FAILURE":
            rows.append({**base, "evaluation_status": row["status"]}); continue
        ids, pred = load_prediction(row); label_ids, y = labels[row["dataset"]]
        if not np.array_equal(ids, label_ids): raise RuntimeError(f"observation order mismatch {base}")
        if len(np.unique(pred)) != len(np.unique(y)): raise RuntimeError(f"locked K mismatch {base}")
        ari = float(adjusted_rand_score(y, pred)); nmi = float(normalized_mutual_info_score(y, pred)); q = .5 * (ari + nmi)
        secondary = {"ami": float(adjusted_mutual_info_score(y, pred)), "fmi": float(fowlkes_mallows_score(y, pred)),
                     "homogeneity": float(homogeneity_score(y, pred)), "completeness": float(completeness_score(y, pred)),
                     "v_measure": float(v_measure_score(y, pred))}
        sm = spatial(pred, adjacency[row["dataset"]])
        key = (row["embedding_sha256"], row["cluster_sha256"], row["dataset"])
        if key not in desc_cache: desc_cache[key] = descriptive(embedding(Path(row["embedding_path"])), pred, adjacency[row["dataset"]])
        rows.append({**base, "evaluation_status": "SUCCESS", "ari": ari, "nmi": nmi, "q": q, **secondary, **sm,
                     **desc_cache[key], "cluster_file_sha256": row["cluster_sha256"],
                     "observation_order_sha256": order_sha(ids), "cluster_vector_sha256": array_sha(pred)})
    frame = pd.DataFrame(rows)
    RAW.mkdir(parents=True, exist_ok=True)
    frame[frame.stage == "R1"].to_csv(RAW / "primary_r1.csv", index=False)
    frame[frame.stage == "R2"].to_csv(RAW / "primary_r2.csv", index=False)
    manifest = {"schema_version": "night8a-eval-recovery-primary-v1", "status": "PASS",
                "success_rows": int((frame.evaluation_status == "SUCCESS").sum()),
                "failure_rows": int((frame.evaluation_status != "SUCCESS").sum()), "label_audit": label_audit,
                "r1_sha256": file_sha(RAW / "primary_r1.csv"), "r2_sha256": file_sha(RAW / "primary_r2.csv"),
                "training": 0, "transform": 0, "external_benchmark": 0, "gpu_used": False}
    (RAW / "primary_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__": main()
