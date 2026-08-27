"""Independent label-opening evaluator for locked Night-22A geometry banks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from SpaLORA.night22a_geometry import array_sha


def graph_from(archive, prefix):
    return sp.csr_matrix(
        (archive[prefix + "__data"], archive[prefix + "__indices"], archive[prefix + "__indptr"]),
        shape=tuple(int(x) for x in archive[prefix + "__shape"]),
    )


def load_truth(authority_path, ids, reference_h5ad=None):
    with np.load(authority_path, allow_pickle=False) as archive:
        authority_ids = np.asarray(archive["ids"])
        if reference_h5ad:
            graph = graph_from(archive, "graph0")
        else:
            labels = np.asarray(archive["labels_primary"])
            mask = np.asarray(archive["label_mask"], dtype=bool)
            graph = graph_from(archive, "operator4")
    if not np.array_equal(ids, authority_ids):
        raise RuntimeError("authority ordered IDs mismatch")
    if reference_h5ad:
        import anndata as ad

        reference = ad.read_h5ad(reference_h5ad)
        reference_ids = np.asarray(reference.obs_names.astype(str), dtype=ids.dtype)
        if not np.array_equal(ids, reference_ids):
            raise RuntimeError("reference ordered IDs mismatch")
        raw = reference.obs["cell_type"]
        mask = np.asarray(raw.notna(), dtype=bool)
        labels = np.asarray(raw.astype(str))
    return labels, mask, graph


def categorical_spatial(partition, graph):
    graph = sp.csr_matrix(graph, dtype=np.float64).maximum(sp.csr_matrix(graph, dtype=np.float64).T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    upper = sp.triu(graph, k=1).tocoo()
    agreement = float(np.mean(partition[upper.row] == partition[upper.col])) if upper.nnz else float("nan")
    total = float(graph.sum())
    moran, geary = [], []
    for label in np.unique(partition):
        value = (partition == label).astype(float)
        centered = value - value.mean()
        denominator = float(np.sum(centered**2))
        if denominator <= 0 or total <= 0 or upper.data.sum() <= 0:
            continue
        moran.append(len(value) / total * float(centered @ (graph @ centered)) / denominator)
        geary.append(
            (len(value) - 1)
            / (2 * float(upper.data.sum()))
            * float(np.sum(upper.data * np.square(value[upper.row] - value[upper.col])))
            / denominator
        )
    return agreement, float(np.mean(moran)), float(np.mean(geary))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--reference-h5ad")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bank_path = Path(args.bank)
    manifest = json.loads(bank_path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(bank_path, allow_pickle=False) as archive:
        ids = np.asarray(archive["ids"])
        candidate_ids = np.asarray(archive["candidate_ids"])
        partitions = np.asarray(archive["partitions"])
    if len(candidate_ids) != len(set(candidate_ids.tolist())):
        raise RuntimeError("duplicate candidate ID")
    for name, partition in zip(candidate_ids.tolist(), partitions):
        if manifest["candidate_partition_sha256"].get(name) != array_sha(partition):
            raise RuntimeError("candidate partition SHA mismatch")
    labels, mask, graph = load_truth(args.authority, ids, args.reference_h5ad)
    rows = []
    for name, partition in zip(candidate_ids.tolist(), partitions):
        truth, predicted = labels[mask], partition[mask]
        full_sizes = np.unique(partition, return_counts=True)[1]
        eval_sizes = np.unique(predicted, return_counts=True)[1]
        agreement, moran, geary = categorical_spatial(partition, graph)
        rows.append(
            {
                "lane": manifest["lane"],
                "representation_source": manifest["representation_source"],
                "candidate_id": name,
                "candidate_partition_sha256": array_sha(partition),
                "ari": adjusted_rand_score(truth, predicted),
                "nmi": normalized_mutual_info_score(truth, predicted),
                "ami": adjusted_mutual_info_score(truth, predicted),
                "fmi": fowlkes_mallows_score(truth, predicted),
                "homogeneity": homogeneity_score(truth, predicted),
                "v_measure": v_measure_score(truth, predicted),
                "n_total": len(partition),
                "n_eval": int(mask.sum()),
                "observed_k_full": int(np.unique(partition).size),
                "min_cluster_size_full": int(full_sizes.min()),
                "cluster_sizes_full": "|".join(map(str, sorted(full_sizes.tolist()))),
                "min_cluster_size_eval": int(eval_sizes.min()),
                "cluster_sizes_eval": "|".join(map(str, sorted(eval_sizes.tolist()))),
                "neighbor_agreement": agreement,
                "moran_indicator_macro": moran,
                "geary_indicator_macro": geary,
                "label_access": "independent_evaluator_after_bank_lock",
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "night22a-geometry-evaluation-v1",
                "bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                "candidate_count": len(rows),
                "ground_truth_loaded_after_bank_lock": True,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
