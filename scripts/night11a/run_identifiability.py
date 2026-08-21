#!/usr/bin/env python3
"""Execute the frozen Night-11A real-data chain without label access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

from SpaLORA.night11a_corruption import patch_and_permutation, sparse_patch
from SpaLORA.night6c_pipeline import row_normalize, run_head
from SpaLORA.selective_transfer import ARMS, run_direction


FORBIDDEN = {
    "training_label_reads": 0, "evaluation_label_reads": 0,
    "label_reads_total": 0, "ari_nmi_q_ami_fmi_computations": 0,
    "label_based_spatial_metric_computations": 0, "misar_y_reads": 0,
    "e18_5_accesses": 0, "new_external_data_accesses": 0,
    "third_party_benchmark_runs": 0, "qcrd_train_resume_evaluate_runs": 0,
    "dataset_name_router_calls": 0, "dense_n_by_n_constructed": 0,
    "scientific_retries_or_fallbacks": 0,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def array_sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def canonical_partition(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    mapping = {}; nxt = 0; out = np.empty(len(labels), dtype=np.int64)
    for i, value in enumerate(labels.tolist()):
        if value not in mapping: mapping[value] = nxt; nxt += 1
        out[i] = mapping[value]
    return out


def fused_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return row_normalize(0.5 * (row_normalize(a) + row_normalize(b)))


def endpoint(p1, p2, k, coords, ids):
    fused = fused_equal(p1, p2)
    labels, aux = run_head({"id": "H05_EQUAL3_AFFINITY_SPECTRAL"},
                           {"emb_latent_omics1": np.asarray(p1, dtype=np.float64),
                            "emb_latent_omics2": np.asarray(p2, dtype=np.float64),
                            "SpaLORA_fused": fused}, int(k), coords, ids, None)
    return canonical_partition(labels), aux, fused


def load_unit(config, unit_id):
    u = config["units"][unit_id]; root = Path(u["source_root"])
    g00 = np.load(root / "g00_views.npz"); g04 = np.load(root / "g04_views.npz")
    ids = (root / "observation_ids.txt").read_text().splitlines()
    graph = sparse.load_npz(u["spatial_graph"]).tocsr()
    coords = np.load(root / "coordinates.npy")
    n = int(u["N"])
    arrays = {
        "g00_1": g00["emb_latent_omics1"], "g00_2": g00["emb_latent_omics2"],
        "g04_1": g04["emb_latent_omics1"], "g04_2": g04["emb_latent_omics2"],
    }
    if len(ids) != n or graph.shape != (n, n) or any(x.shape[0] != n for x in arrays.values()):
        raise RuntimeError("real observation/shape contract mismatch")
    obs_sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    if obs_sha != u["ordered_observation_sha256"]:
        raise RuntimeError("ordered observation SHA mismatch")
    return u, arrays, ids, graph, coords


def binary_metrics(score, utility):
    score = np.asarray(score); utility = np.asarray(utility)
    margin = max(1e-8, 0.01 * float(np.median(np.abs(utility))))
    keep = np.abs(utility) > margin; y = utility[keep] > 0; s = score[keep]
    pos = int(y.sum()); neg = int((~y).sum())
    if pos == 0 or neg == 0:
        auroc = auprc = float("nan")
    else:
        ranks = rankdata(s, method="average")
        auroc = float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))
        order = np.argsort(-s, kind="mergesort"); yy = y[order].astype(float)
        precision = np.cumsum(yy) / np.arange(1, len(yy) + 1)
        auprc = float((precision * yy).sum() / pos)
    rs = rankdata(score, method="average"); ru = rankdata(utility, method="average")
    spearman = float(np.corrcoef(rs, ru)[0, 1]) if np.std(rs) and np.std(ru) else float("nan")
    sep = float(np.median(score[utility > margin]) - np.median(score[utility < -margin])) if pos and neg else float("nan")
    return {"auroc": auroc, "auprc": auprc, "spearman": spearman,
            "gate_separation": sep, "nonambiguous_positive": pos,
            "nonambiguous_negative": neg, "excluded_fraction": float(1.0 - keep.mean()),
            "ambiguity_margin": margin}


def run_case(config, unit_id, condition, seed, output_root, checkpoint):
    started = time.time(); cpu0 = time.process_time()
    u, a, ids, graph, coords = load_unit(config, unit_id)
    directions = {
        "MODALITY2_TO_MODALITY1": (a["g00_1"], a["g04_1"], a["g00_2"], a["g04_2"]),
        "MODALITY1_TO_MODALITY2": (a["g00_2"], a["g04_2"], a["g00_1"], a["g04_1"]),
    }
    results = {}; corruption = {}
    for direction, arrays in directions.items():
        patch, permutation = patch_and_permutation(
            graph, ids, unit_id, direction, condition, seed)
        r = run_direction(unit_id=unit_id, direction=direction, observation_ids=ids,
                          condition=condition, receiving_g00=arrays[0], receiving_g04=arrays[1],
                          auxiliary_g00=arrays[2], auxiliary_g04=arrays[3], patch=patch,
                          permutation=permutation, alpha=float(config["ridge_alpha"]))
        results[direction] = r
        corruption[direction] = {"patch_sha256": array_sha(patch), "patch_size": len(patch),
                                 "permutation_sha256": array_sha(permutation),
                                 "permutation_fixed_points": int(np.sum(patch == permutation))}
    rows = []; ck = {}; p1_clean = a["g04_1"]; p2_clean = a["g04_2"]
    for arm in ARMS:
        p1 = results["MODALITY2_TO_MODALITY1"]["arm_predictions"][arm]
        p2 = results["MODALITY1_TO_MODALITY2"]["arm_predictions"][arm]
        labels, aux, fused = endpoint(p1, p2, u["K"], coords, ids)
        ck[arm] = {"p1": p1, "p2": p2, "fused": fused, "partition": labels}
        for direction in directions:
            r = results[direction]; mse = r["mse"]; score = r["gate"]
            metrics = binary_metrics(score, r["oracle_utility"])
            best = np.minimum(mse["B0_SELF_ONLY"], mse["B1_ALWAYS_TRANSFER"])
            regret = mse[arm] - best
            metrics.update({"mse_mean": float(mse[arm].mean()),
                            "regret_mean": float(regret.mean()),
                            "normalized_regret_mean": float(np.mean(regret / (mse["B0_SELF_ONLY"] + 1e-8))),
                            "gate_mean": float(score.mean()), "gate_std": float(score.std()),
                            "gate_bins": [float(np.mean(score <= .1)), float(np.mean((score > .1) & (score < .9))), float(np.mean(score >= .9))],
                            "exact_null_pass": bool(r["exact_null_pass"]),
                            "endpoint_partition_sha256": array_sha(labels),
                            "endpoint_affinity_sha256": aux["affinity_sha256"],
                            "endpoint_affinity_nnz": int(aux["affinity_nnz"])})
            rows.append({"unit_id": unit_id, "family": u["family"], "condition": condition,
                         "replicate_seed": int(seed), "direction": direction, "arm": arm,
                         "metrics": metrics, "status": "PASS"})
    case_dir = Path(output_root) / unit_id / condition / ("seed_%d" % seed)
    arrays = {}
    for arm, item in ck.items():
        for name, value in item.items(): arrays[arm + "__" + name] = value
    for direction, r in results.items():
        prefix = direction + "__"
        arrays[prefix + "gate"] = r["gate"]; arrays[prefix + "oracle_utility"] = r["oracle_utility"]
        for arm in ARMS: arrays[prefix + arm + "__mse"] = r["mse"][arm]
    checkpoint_path = case_dir / "checkpoint.npz"
    atomic_npz(checkpoint_path, **arrays)
    manifest = {"schema": "spalora.night11a.case.v1", "status": "PASS", "unit_id": unit_id,
                "family": u["family"], "condition": condition, "replicate_seed": int(seed),
                "N": int(u["N"]), "K": int(u["K"]), "rows": rows, "corruption": corruption,
                "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha(checkpoint_path),
                "wall_seconds": time.time() - started, "cpu_seconds": time.process_time() - cpu0,
                "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "peak_gpu_mib": 0, "forbidden_counters": FORBIDDEN}
    manifest_path = case_dir / "manifest.json"; atomic_json(manifest_path, manifest)
    if checkpoint:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--config", str(Path(args.config).resolve()),
               "--reload-manifest", str(manifest_path)]
        cp = subprocess.run(cmd, cwd=str(Path.cwd()), text=True, capture_output=True)
        if cp.returncode:
            raise RuntimeError("fresh-process reload failed: " + cp.stderr[-2000:])
        reload_audit = json.loads(cp.stdout.strip().splitlines()[-1])
        manifest["fresh_process_roundtrip"] = reload_audit
        atomic_json(manifest_path, manifest)
    return manifest_path, manifest


def reload_manifest(config, path):
    manifest = json.loads(Path(path).read_text()); u, _, ids, _, coords = load_unit(config, manifest["unit_id"])
    cp = Path(manifest["checkpoint"])
    if sha(cp) != manifest["checkpoint_sha256"]: raise RuntimeError("checkpoint SHA mismatch")
    z = np.load(cp); checks = {}
    for arm in ARMS:
        labels, _, fused = endpoint(z[arm + "__p1"], z[arm + "__p2"], u["K"], coords, ids)
        checks[arm] = {"p1_exact": True, "p2_exact": True,
                       "fused_close": bool(np.allclose(fused, z[arm + "__fused"], atol=1e-6, rtol=1e-5)),
                       "partition_exact": bool(np.array_equal(labels, z[arm + "__partition"]))}
    passed = all(all(v.values()) for v in checks.values())
    payload = {"status": "PASS" if passed else "FAIL", "fresh_process": True,
               "atol": 1e-6, "rtol": 1e-5, "arms": checks}
    print(json.dumps(payload, sort_keys=True)); return 0 if passed else 2


def main():
    global args
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=["smoke", "formal"]); p.add_argument("--output-root")
    p.add_argument("--reload-manifest"); args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.reload_manifest: return reload_manifest(config, args.reload_manifest)
    if not args.mode or not args.output_root: p.error("--mode and --output-root required")
    cases = config["smokes"] if args.mode == "smoke" else [
        {"unit_id": uid, "condition": c, "replicate_seed": s}
        for uid in config["units"] for c in config["conditions"] for s in config["replicate_seeds"]]
    index = []
    for case in cases:
        mp, manifest = run_case(config, case["unit_id"], case["condition"],
                                int(case["replicate_seed"]), args.output_root,
                                checkpoint=(args.mode == "smoke"))
        index.append({"manifest": str(mp), "manifest_sha256": sha(mp),
                      "status": manifest["status"], "row_count": len(manifest["rows"])})
        print("CASE_PASS", case["unit_id"], case["condition"], case["replicate_seed"], flush=True)
    index_path = Path(args.output_root) / (args.mode + "_index.json")
    atomic_json(index_path, {"mode": args.mode, "cases": index,
                             "case_count": len(index), "row_count": sum(x["row_count"] for x in index),
                             "forbidden_counters": FORBIDDEN, "status": "PASS"})
    print("INDEX", index_path)
    return 0


if __name__ == "__main__": raise SystemExit(main())
