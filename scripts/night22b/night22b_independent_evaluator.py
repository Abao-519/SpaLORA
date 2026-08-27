"""Open labels only after a Night-22B partition/checkpoint bank is locked."""
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


def file_sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def ordered_id_sha(ids: np.ndarray) -> str:
    return hashlib.sha256(b"\0".join(str(item).encode("utf-8") for item in ids)).hexdigest()


def load_graph(archive, prefix: str = "graph0") -> sp.csr_matrix:
    graph = sp.csr_matrix(
        (archive[prefix + "__data"], archive[prefix + "__indices"], archive[prefix + "__indptr"]),
        shape=tuple(int(x) for x in archive[prefix + "__shape"]),
        dtype=np.float64,
    )
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def load_reference(spec: dict, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    path = Path(spec["reference"])
    kind = spec["reference_kind"]
    ids_text = np.asarray(ids).astype(str)
    if kind == "npz":
        with np.load(path, allow_pickle=False) as authority:
            authority_ids = np.asarray(authority["ids"]).astype(str)
            labels = np.asarray(authority[spec["reference_label_key"]]).astype(str)
            mask = np.asarray(authority[spec["reference_mask_key"]], dtype=bool)
        if not np.array_equal(ids_text, authority_ids):
            raise RuntimeError("NPZ reference ordered IDs mismatch")
    elif kind == "h5ad":
        import anndata as ad

        reference = ad.read_h5ad(path)
        authority_ids = np.asarray(reference.obs_names.astype(str))
        if not np.array_equal(ids_text, authority_ids):
            raise RuntimeError("H5AD reference ordered IDs mismatch")
        raw = reference.obs[spec["reference_label_key"]]
        mask = np.asarray(raw.notna(), dtype=bool)
        labels = np.asarray(raw.astype(str))
    elif kind == "tsv":
        mapping = {}
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                key = row[spec["reference_id_key"]]
                if key in mapping:
                    raise RuntimeError("duplicate ID in TSV reference")
                mapping[key] = row[spec["reference_label_key"]]
        missing = [item for item in ids_text.tolist() if item not in mapping]
        if missing:
            raise RuntimeError(f"TSV reference missing IDs: {missing[:3]}")
        labels = np.asarray([mapping[item] for item in ids_text.tolist()])
        mask = np.asarray([bool(value) for value in labels], dtype=bool)
    else:
        raise RuntimeError(f"unsupported reference kind: {kind}")
    clean = labels[mask]
    if not len(clean) or any(str(value).strip().lower() in {"", "nan", "none"} for value in clean):
        raise RuntimeError("missing/invalid labels remain in evaluation mask")
    unique, counts = np.unique(clean, return_counts=True)
    if len(unique) != int(spec["k"]):
        raise RuntimeError(f"reference K mismatch: expected {spec['k']}, observed {len(unique)}")
    audit = {
        "reference_path": str(path),
        "reference_sha256": file_sha(path),
        "reference_kind": kind,
        "n_reference_rows": int(len(labels)),
        "n_evaluated": int(mask.sum()),
        "reference_k": int(len(unique)),
        "reference_category_counts": {str(k): int(v) for k, v in zip(unique, counts)},
        "ordered_label_sha256": array_sha(labels.astype("U")),
    }
    return labels, mask, audit


def categorical_spatial(partition: np.ndarray, graph: sp.csr_matrix) -> tuple[float, float, float]:
    upper = sp.triu(graph, k=1).tocoo()
    agreement = float(np.mean(partition[upper.row] == partition[upper.col]))
    total = float(graph.sum())
    moran, geary = [], []
    for label in np.unique(partition):
        value = (partition == label).astype(np.float64)
        centered = value - value.mean()
        denominator = float(np.sum(centered**2))
        if denominator <= 0:
            continue
        moran.append(len(value) / total * float(centered @ (graph @ centered)) / denominator)
        geary.append(
            (len(value) - 1)
            / (2.0 * float(upper.data.sum()))
            * float(np.sum(upper.data * np.square(value[upper.row] - value[upper.col])))
            / denominator
        )
    return agreement, float(np.mean(moran)), float(np.mean(geary))


def internal_edge_support(partition: np.ndarray, graph: sp.csr_matrix, k: int) -> list[int]:
    upper = sp.triu(graph, k=1).tocoo()
    counts = []
    for label in range(int(k)):
        counts.append(int(np.sum((partition[upper.row] == label) & (partition[upper.col] == label))))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bank_path = Path(args.bank)
    manifest = json.loads(bank_path.with_suffix(".json").read_text(encoding="utf-8"))
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    spec = contract["lanes"][args.lane]
    if manifest["lane"] != args.lane or manifest["labels_read"] != 0:
        raise RuntimeError("producer manifest lane/label firewall mismatch")
    if file_sha(bank_path) != manifest["partition_bank_sha256"]:
        raise RuntimeError("partition bank file SHA mismatch")
    checkpoint_path = bank_path.with_suffix(".pt")
    if file_sha(checkpoint_path) != manifest["checkpoint_bank_sha256"]:
        raise RuntimeError("checkpoint bank file SHA mismatch")

    with np.load(bank_path, allow_pickle=False) as bank:
        ids = np.asarray(bank["ids"])
        candidate_ids = np.asarray(bank["candidate_ids"]).astype(str)
        partitions = np.asarray(bank["partitions"], dtype=np.int32)
    if len(candidate_ids) != len(set(candidate_ids.tolist())):
        raise RuntimeError("duplicate candidate ID")
    if len(partitions) != len(candidate_ids):
        raise RuntimeError("candidate/partition length mismatch")
    for name, partition in zip(candidate_ids.tolist(), partitions):
        if manifest["candidate_partition_sha256"].get(name) != array_sha(partition):
            raise RuntimeError(f"candidate partition SHA mismatch: {name}")
    carrier_path = Path(args.carrier)
    if file_sha(carrier_path) != spec["carrier_sha256"]:
        raise RuntimeError("carrier SHA differs from frozen contract")
    with np.load(carrier_path, allow_pickle=False) as carrier:
        carrier_ids = np.asarray(carrier["ids"])
        graph = load_graph(carrier, "graph0")
    if not np.array_equal(ids, carrier_ids) or ordered_id_sha(ids) != spec["ordered_ids_sha256"]:
        raise RuntimeError("bank/carrier ordered-ID authority mismatch")

    # This is the first reference-label read in the evaluator process.
    labels, mask, reference_audit = load_reference(spec, ids)
    start = partitions[0]
    rows = []
    for candidate_id, partition in zip(candidate_ids.tolist(), partitions):
        truth = labels[mask]
        predicted = partition[mask]
        full_counts = np.bincount(partition, minlength=spec["k"])
        eval_counts = np.unique(predicted, return_counts=True)[1]
        support = internal_edge_support(partition, graph, spec["k"])
        agreement, moran, geary = categorical_spatial(partition, graph)
        rows.append(
            {
                "lane": args.lane,
                "role": spec["role"],
                "start_candidate": manifest["start_candidate"],
                "candidate_id": candidate_id,
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
                "min_cluster_size_full": int(full_counts.min()),
                "cluster_sizes_full": "|".join(map(str, sorted(full_counts.tolist()))),
                "min_cluster_size_eval": int(eval_counts.min()),
                "cluster_sizes_eval": "|".join(map(str, sorted(eval_counts.tolist()))),
                "internal_smallest_scale_edge_counts": "|".join(map(str, support)),
                "all_clusters_have_internal_smallest_scale_edge": all(value > 0 for value in support),
                "changed_observations_vs_start": int(np.sum(partition != start)),
                "neighbor_agreement": agreement,
                "moran_indicator_macro": moran,
                "geary_indicator_macro": geary,
                "label_access": "independent_evaluator_after_partition_and_checkpoint_lock",
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "schema": "night22b-independent-evaluation-v1",
        "lane": args.lane,
        "bank_sha256": file_sha(bank_path),
        "checkpoint_sha256": file_sha(checkpoint_path),
        "candidate_count": len(rows),
        "partitions_and_checkpoints_locked_before_reference_read": True,
        "producer_label_reads": 0,
        "evaluator_label_reads": 1,
        **reference_audit,
    }
    output.with_suffix(".json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
