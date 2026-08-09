#!/usr/bin/env python3
"""Build one independent Night-3A-F deterministic preprocessing candidate."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.night3af_cache import save_cache, sha256_file
from SpaLORA.night3af_protocol import ScientificWindow, atomic_json, preprocessing_config
from SpaLORA.night3ar_protocol import integrity_read, record_integrity_reads


CONFIG_PATH = REPO / "configs/night3af_deterministic_pca.json"


def numpy_state_equal(first, second):
    return first[0] == second[0] and np.array_equal(first[1], second[1]) and first[2:] == second[2:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", required=True, choices=("a", "b"))
    parser.add_argument("--external-seed", required=True, type=int)
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    build_root = Path(config["paths"]["p0d_temp_root"]) / ("process_" + args.process)
    if build_root.exists() and any(build_root.iterdir()):
        raise RuntimeError("Refusing to overwrite existing P0D process build: %s" % build_root)
    build_root.mkdir(parents=True, exist_ok=True)

    reads = []
    for dataset, cfg in config["datasets"].items():
        reads.append(integrity_read(Path(cfg["rna"]), cfg["rna_sha256"], "%s RNA input" % dataset))
        reads.append(integrity_read(Path(cfg["modality2"]), cfg["modality2_sha256"], "%s modality2 input" % dataset))
        if str(cfg["ground_truth"]).startswith("/"):
            reads.append(integrity_read(Path(cfg["ground_truth"]), cfg["ground_truth_sha256"],
                                        "%s ground-truth integrity only" % dataset))
    record_integrity_reads(output, "p0d_process_" + args.process + "_prewindow", reads)
    if not all(row["match"] for row in reads):
        raise RuntimeError("Raw input SHA mismatch")

    random.seed(args.external_seed)
    np.random.seed(args.external_seed)
    torch.manual_seed(args.external_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.external_seed)
    window = ScientificWindow(config, output, "p0d_process_" + args.process).install()
    dataset_rows = {}
    try:
        from SpaLORA.night1_pipeline import prepare_corrected

        pre_cfg = preprocessing_config(config)
        for dataset, cfg in config["datasets"].items():
            python_before = copy.deepcopy(random.getstate())
            numpy_before = copy.deepcopy(np.random.get_state())
            torch_before = torch.get_rng_state().clone()
            cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
            prepared = prepare_corrected(dataset, cfg, pre_cfg, "corrected_unweighted")
            python_after = random.getstate()
            numpy_after = np.random.get_state()
            torch_after = torch.get_rng_state()
            cuda_after = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            rng = {
                "python_unchanged": python_before == python_after,
                "numpy_unchanged": numpy_state_equal(numpy_before, numpy_after),
                "torch_cpu_unchanged": torch.equal(torch_before, torch_after),
                "torch_cuda_unchanged": len(cuda_before) == len(cuda_after) and all(
                    torch.equal(a, b) for a, b in zip(cuda_before, cuda_after)
                ),
            }
            if not all(rng.values()):
                raise RuntimeError("Preprocessing changed global RNG state: %s %r" % (dataset, rng))
            if prepared.data["rna_pca_metadata"]["svd_solver_resolved"] != "randomized":
                raise RuntimeError("Deterministic RNA PCA did not resolve to randomized solver")
            manifest = save_cache(build_root / dataset, dataset, prepared, config["preprocessing"])
            dataset_rows[dataset] = {
                "cache_manifest_sha256": sha256_file(build_root / dataset / "manifest.json"),
                "canonical_cache_content_sha256": manifest["canonical_cache_content_sha256"],
                "canonical_model_input_sha256": manifest["canonical_model_input_sha256"],
                "file_sha256": {name: row["sha256"] for name, row in manifest["files"].items()},
                "rng": rng,
            }
        firewall = window.close(passed=True)
    except Exception:
        window.close(passed=False)
        raise
    if not firewall["passed"]:
        raise RuntimeError("P0D build label firewall failed")
    payload = {
        "schema_version": 1, "stage": "P0D independent process " + args.process.upper(),
        "process": args.process.upper(), "external_rng_seed": args.external_seed,
        "pca_svd_solver": config["preprocessing"]["pca_svd_solver"],
        "pca_random_state": config["preprocessing"]["pca_random_state"],
        "datasets": dataset_rows, "scientific_window": firewall,
        "semantic_label_values_read": False,
        "versions": {
            "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
            "sklearn": __import__("sklearn").__version__, "scanpy": __import__("scanpy").__version__,
            "scipy": __import__("scipy").__version__,
        },
    }
    atomic_json(output / ("p0d_process_%s.json" % args.process), payload)
    print("P0D_PROCESS_%s_COMPLETE %s" % (args.process.upper(), json.dumps(
        {key: row["canonical_model_input_sha256"] for key, row in dataset_rows.items()}, sort_keys=True
    )), flush=True)


if __name__ == "__main__":
    main()
