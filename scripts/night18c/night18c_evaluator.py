#!/usr/bin/env python3
"""Independent post-lock evaluator for Night-18C Stage A."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score

from SpaLORA.night18c_rsp_gtd import sha256_array


def csr(archive: np.lib.npyio.NpzFile, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix((archive[f"{prefix}__data"], archive[f"{prefix}__indices"], archive[f"{prefix}__indptr"]),
                         shape=tuple(int(x) for x in archive[f"{prefix}__shape"]))


def spatial_metrics(partition: np.ndarray, graph: sp.csr_matrix) -> dict[str, float]:
    graph = sp.csr_matrix(graph, dtype=np.float64).maximum(sp.csr_matrix(graph, dtype=np.float64).T).tocsr()
    graph.setdiag(0); graph.eliminate_zeros()
    rows = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr)); cols = graph.indices
    weight_sum = max(float(graph.data.sum()), 1e-12)
    agreement = float(np.sum(graph.data * (partition[rows] == partition[cols])) / weight_sum)
    degree = np.asarray(graph.sum(1)).ravel(); n = len(partition)
    moran, geary = [], []
    for group in np.unique(partition):
        x = (partition == group).astype(np.float64); centered = x - x.mean(); denom = max(float(np.sum(centered ** 2)), 1e-12)
        moran.append(float(n / weight_sum * np.sum(graph.data * centered[rows] * centered[cols]) / denom))
        geary.append(float((n - 1) / (2 * weight_sum) * np.sum(graph.data * (x[rows] - x[cols]) ** 2) / denom))
    return {"neighbor_agreement": agreement, "moran_macro_indicator": float(np.mean(moran)), "geary_macro_indicator": float(np.mean(geary))}


def load_truth(args: argparse.Namespace, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, str]:
    if args.authority_kind == "kit":
        with np.load(args.authority, allow_pickle=False) as authority:
            authority_ids = authority["ids"].astype("U")
            if not np.array_equal(authority_ids, ids):
                raise ValueError("kit ordered-ID mismatch")
            labels = authority["labels_primary"].astype("U")
            mask = authority["label_mask"].astype(bool)
        return labels, mask, sha256_array(labels), "KIT_LABELS_PRIMARY"
    reference = ad.read_h5ad(args.authority)
    reference_ids = np.asarray(reference.obs_names.astype(str))
    if set(reference_ids) != set(ids):
        raise ValueError("human authority ID-set mismatch")
    lookup = {identifier: index for index, identifier in enumerate(reference_ids)}
    order = np.asarray([lookup[identifier] for identifier in ids], dtype=np.int64)
    if args.label_column not in reference.obs:
        raise KeyError(args.label_column)
    series = reference.obs.iloc[order][args.label_column]
    mask = np.asarray(series.notna(), dtype=bool)
    labels = np.asarray(
        [str(value) if valid else "__MISSING__" for value, valid in zip(series.to_numpy(), mask)],
        dtype="U",
    )
    return labels, mask, sha256_array(labels), "OFFICIAL_MANUAL_ANATOMICAL_REFERENCE"


def run(args: argparse.Namespace) -> None:
    artifact_path, manifest_path = Path(args.artifact), Path(args.manifest)
    with np.load(artifact_path, allow_pickle=False) as artifact:
        ids = artifact["ids"].astype("U")
        partitions = artifact["partitions"].astype(np.int32)
        candidate_ids = artifact["candidate_ids"].astype("U")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["artifact_sha256"] != file_sha(artifact_path):
        raise RuntimeError("artifact file hash mismatch")
    if not np.array_equal(candidate_ids, np.asarray([x["candidate_id"] for x in manifest["records"]]).astype("U")):
        raise RuntimeError("candidate order mismatch")
    labels, mask, label_sha, reference_type = load_truth(args, ids)
    if int(mask.sum()) == 0 or len(np.unique(labels[mask])) != int(args.k):
        raise RuntimeError("evaluation authority K/mask mismatch")
    with np.load(args.carrier, allow_pickle=False) as carrier:
        if not np.array_equal(carrier["ids"].astype("U"), ids):
            raise ValueError("carrier ordered-ID mismatch")
        graph = csr(carrier, "graph0")
    rows = []
    for index, record in enumerate(manifest["records"]):
        partition = partitions[index]
        if sha256_array(partition) != record["partition_sha256"]:
            raise RuntimeError("partition hash mismatch")
        sizes_full = np.bincount(partition, minlength=args.k)
        sizes_eval = np.bincount(partition[mask], minlength=args.k)
        truth, predicted = labels[mask], partition[mask]
        rows.append({
            "lane": manifest["lane"], "candidate_id": record["candidate_id"], "arm": record["arm"], "config_id": record["config_id"],
            "partition_sha256": record["partition_sha256"], "ari": adjusted_rand_score(truth, predicted),
            "nmi": normalized_mutual_info_score(truth, predicted), "ami": adjusted_mutual_info_score(truth, predicted),
            "fmi": fowlkes_mallows_score(truth, predicted), "n_total": len(ids), "n_eval": int(mask.sum()), "k": args.k,
            "min_cluster_size_full": int(sizes_full.min()), "cluster_sizes_full": json.dumps(sizes_full.tolist()),
            "min_cluster_size_eval": int(sizes_eval.min()), "cluster_sizes_eval": json.dumps(sizes_eval.tolist()),
            "wall_seconds": manifest["wall_seconds"], "peak_rss_mib": manifest["peak_rss_mib"],
            "gpu_time_seconds": manifest["gpu_time_seconds"], "peak_gpu_mib": manifest["peak_gpu_mib"],
            **spatial_metrics(partition, graph),
        })
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps({
        "status": "PASS", "rows": len(rows), "evaluator_label_reads": 1, "producer_label_reads": 0,
        "reference_type": reference_type, "authority_labels_sha256": label_sha, "n_evaluated": int(mask.sum()),
        "authority_k": int(len(np.unique(labels[mask]))),
    }, indent=2, sort_keys=True), encoding="utf-8")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--carrier", required=True); parser.add_argument("--authority", required=True)
    parser.add_argument("--authority-kind", choices=("kit", "human_h5ad"), required=True)
    parser.add_argument("--label-column", default="true_label"); parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", required=True); run(parser.parse_args())


if __name__ == "__main__": main()
