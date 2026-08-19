#!/usr/bin/env python3
"""Run one fixed label-free Night-8A family endpoint."""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night6c_pipeline import self_tuning_affinity, spectral, sparse_sha  # noqa: E402
from SpaLORA.night7b_adaptive import row_sparse_strict, sym_zero  # noqa: E402
from SpaLORA.night8a_mfspc import array_sha, file_sha, select_family  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--training-output", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    config = json.loads(args.config.read_text()); family = select_family(config["assay_metadata"])
    training = json.loads((args.training_output / "training_manifest.json").read_text())
    reload_audit = json.loads((args.training_output / "reload_audit.json").read_text())
    if training["status"] != "SUCCESS_PRE_LABEL" or reload_audit["status"] != "PASS":
        raise RuntimeError("training/checkpoint round trip not locked")
    args.output.mkdir(parents=True, exist_ok=False); start = time.perf_counter()
    z = np.load(args.training_output / "embeddings.npz", allow_pickle=False)
    worker = json.loads(Path(config["worker_input"]).read_text())
    ids = [x.strip() for x in Path(worker["observation_ids"]).read_text().splitlines() if x.strip()]
    views = [z["emb_latent_omics1"], z["emb_latent_omics2"], z["SpaLORA_fused"]]
    if family == "RNA_PROTEIN":
        pieces = [self_tuning_affinity(v, 10, ids) for v in views]
        affinity = (pieces[0] + pieces[1] + pieces[2]) * (1.0 / 3.0)
        endpoint = "C00_G04_H05_EQUAL3_AFFINITY_SPECTRAL"
    elif family == "RNA_EPIGENOME":
        c06_path = Path(worker["g04_views"]).parent / "c06_affinity.npz"
        if not c06_path.exists():
            raise RuntimeError("locked C06 affinity missing")
        learned = self_tuning_affinity(z["SpaLORA_fused"], 10, ids)
        c06 = sp.load_npz(c06_path)
        affinity = sym_zero(row_sparse_strict(learned) * .5 + row_sparse_strict(c06) * .5)
        endpoint = "R02_E1_ADAPTER_C06_MEAN_H01_SPECTRAL"
    else:
        raise RuntimeError("unknown family")
    affinity = affinity.tocsr(); affinity.sum_duplicates(); affinity.eliminate_zeros(); affinity.sort_indices()
    labels = spectral(affinity, int(config["K"]))
    replay = spectral(affinity, int(config["K"]))
    if not np.array_equal(labels, replay):
        raise RuntimeError("cluster exact replay mismatch")
    if len(np.unique(labels)) != int(config["K"]):
        raise RuntimeError("cluster K mismatch")
    affinity_path = args.output / "affinity.npz"; sp.save_npz(affinity_path, affinity, compressed=True)
    clusters_path = args.output / "clusters.csv"
    pd.DataFrame({"observation_id": ids, "cluster": labels.astype(int)}).to_csv(clusters_path, index=False)
    placeholder = args.output / "metrics_placeholder.json"
    atomic_json(placeholder, {"status": "SEALED_PRE_LABEL", "label_access": False,
                              "ari": None, "nmi": None, "q": None})
    manifest = {
        "schema_version": "night8a-transform-manifest-v1", "status": "SUCCESS_PRE_LABEL",
        "stage": config["stage"], "config_id": config["config_id"], "unit_id": config["unit_id"],
        "seed": int(config["seed"]), "family": family, "endpoint": endpoint,
        "label_access": False, "fallback": False, "scientific_retry": 0,
        "cpu_thread_limits": {k: os.environ.get(k) for k in
                              ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "affinity_shape": list(affinity.shape), "affinity_nnz": int(affinity.nnz),
        "affinity_canonical_sha256": sparse_sha(affinity),
        "affinity_file_sha256": file_sha(affinity_path),
        "cluster_canonical_sha256": array_sha(np.asarray(labels, dtype=np.int64)),
        "cluster_file_sha256": file_sha(clusters_path), "cluster_exact_replay": True,
        "training_manifest_sha256": file_sha(args.training_output / "training_manifest.json"),
        "reload_audit_sha256": file_sha(args.training_output / "reload_audit.json"),
        "metrics_placeholder_sha256": file_sha(placeholder),
        "runtime_seconds": time.perf_counter() - start,
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_json(args.output / "transform_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "config_id": config["config_id"],
                      "unit_id": config["unit_id"], "runtime_seconds": manifest["runtime_seconds"]},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
