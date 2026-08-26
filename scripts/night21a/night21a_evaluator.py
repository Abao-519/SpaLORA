#!/usr/bin/env python3
"""Independent label-opening evaluator for locked Night-21A partitions."""

from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import anndata as ad
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score

from SpaLORA.night21a_amcf import sha256_array


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", required=True); parser.add_argument("--manifest", required=True)
    group = parser.add_mutually_exclusive_group(required=True); group.add_argument("--authority-npz"); group.add_argument("--authority-h5ad")
    parser.add_argument("--label-column", default="cell_type"); parser.add_argument("--output", required=True); args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest["status"] != "PARTITION_LOCKED_BEFORE_EVALUATION" or manifest["annotation_keys_accessed"] != 0:
        raise RuntimeError("producer label firewall failed")
    with np.load(args.artifact, allow_pickle=False) as artifact:
        ids = artifact["ids"].astype("U"); partition = artifact["partition"].astype(np.int32)
    if sha256_array(partition) != manifest["partition_sha256"]: raise RuntimeError("partition SHA mismatch")
    if args.authority_npz:
        with np.load(args.authority_npz, allow_pickle=False) as authority:
            aid = authority["ids"].astype("U"); labels = authority["labels_primary"].astype("U"); mask = authority["label_mask"].astype(bool)
    else:
        reference = ad.read_h5ad(args.authority_h5ad); aid = np.asarray(reference.obs_names.astype(str), dtype="U")
        labels = np.asarray(reference.obs[args.label_column].astype(str), dtype="U"); mask = np.ones(len(labels), dtype=bool)
    if not np.array_equal(ids, aid): raise RuntimeError("authority ordered ID mismatch")
    truth, predicted = labels[mask], partition[mask]
    sizes = np.bincount(partition, minlength=manifest["k"])
    row = {"lane": manifest["lane"], "family": manifest["family"], "arm": manifest["arm"], "profile": manifest["profile"],
           "training_seed": manifest["training_seed"], "endpoint_seed": manifest["endpoint_seed"], "n_total": len(ids), "n_eval": int(mask.sum()), "k": manifest["k"],
           "ari": adjusted_rand_score(truth, predicted), "nmi": normalized_mutual_info_score(truth, predicted),
           "ami": adjusted_mutual_info_score(truth, predicted), "fmi": fowlkes_mallows_score(truth, predicted),
           "min_cluster_size": int(sizes.min()), "cluster_sizes": json.dumps([int(x) for x in sizes]),
           "representation_sha256": manifest["representation_sha256"], "partition_sha256": manifest["partition_sha256"],
           "wall_seconds": manifest["wall_seconds"], "gpu_peak_mib": manifest["gpu_peak_mib"], "peak_rss_mib": manifest["peak_rss_mib"]}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    output.with_suffix(".json").write_text(json.dumps({"status": "PASS", "label_reads": 1, "ordered_labels_sha256": sha256_array(labels), "mask_sha256": sha256_array(mask)}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__": main()

