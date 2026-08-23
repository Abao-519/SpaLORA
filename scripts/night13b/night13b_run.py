#!/usr/bin/env python3
"""Night-13B unified model, corrected anchors, search and formal runner."""
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
from typing import Dict, List, Mapping, Sequence, Tuple

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score,
    homogeneity_score, normalized_mutual_info_score, v_measure_score,
)
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from SpaLORA.night13a_runner import (  # noqa: E402
    load_h5ad_pair, mean_binary_moran, simple_embedding, sparse_spatial_graph,
)
from SpaLORA.night13b_unified import (  # noqa: E402
    ContentAdaptiveGraphResidual, array_sha256, canonical_config_sha256,
    gradient_probe_loss, mean_binary_geary, model_state_sha256,
    ordered_id_sha256, row_stochastic, scipy_to_torch,
)

ROOT = Path("/root/autodl-fs/night13b_unified_model_performance_sprint_20260823")
N13A = Path("/root/autodl-fs/night13a_benchmark_expansion_20260822")
N8B = Path("/root/autodl-fs/night8b_raw_runs_20260820")
SEED = 0


DATASETS: Dict[str, dict] = {
    "A1": {
        "adapter": "protein", "k": 10, "phase": "discovery",
        "rna": "/root/autodl-fs/Human lymph node/A1/humanlymphnode_rna.h5ad",
        "other": "/root/autodl-fs/Human lymph node/A1/humanlymphnode_adt.h5ad",
        "label": "/root/autodl-fs/Human lymph node/A1/A1_groundtruth.csv",
        "label_kind": "csv_strip_slice",
    },
    "D1": {
        "adapter": "protein", "k": 10, "phase": "confirmation",
        "rna": "/root/autodl-fs/Human lymph node/D1/humanlymphnode_rna.h5ad",
        "other": "/root/autodl-fs/Human lymph node/D1/humanlymphnode_adt.h5ad",
        "label": "/root/autodl-fs/Human lymph node/D1/D1_groundtruth.csv",
        "label_kind": "csv_exact",
    },
    "tonsil_s1": {
        "adapter": "protein", "k": 4, "phase": "discovery",
        "rna": str(N13A / "data/canonical_tonsil/s1_adata_rna.h5ad"),
        "other": str(N13A / "data/canonical_tonsil/s1_adata_adt.h5ad"),
        "label_kind": "obs_final_annot",
    },
    "tonsil_s2": {
        "adapter": "protein", "k": 4, "phase": "confirmation",
        "rna": str(N13A / "data/canonical_tonsil/s2_adata_rna.h5ad"),
        "other": str(N13A / "data/canonical_tonsil/s2_adata_adt.h5ad"),
        "label_kind": "obs_final_annot",
    },
    "tonsil_s3": {
        "adapter": "protein", "k": 4, "phase": "confirmation",
        "rna": str(N13A / "data/canonical_tonsil/s3_adata_rna.h5ad"),
        "other": str(N13A / "data/canonical_tonsil/s3_adata_adt.h5ad"),
        "label_kind": "obs_final_annot",
    },
    "P22": {
        "adapter": "atac", "k": 9, "phase": "discovery",
        "rna": "/root/autodl-fs/P22 mouse brain coronal section/mousebrain_rna.h5ad",
        "other": "/root/autodl-fs/P22 mouse brain coronal section/mousebrain_atac.h5ad",
        "label": "/root/autodl-fs/P22 mouse brain coronal section/MouseBrain_groundtruth.csv",
        "label_kind": "csv_exact_registered",
    },
    "MISAR_E15_5_S1": {
        "adapter": "atac", "k": 7, "phase": "confirmation",
        "label_kind": "misar_carrier",
    },
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_registry() -> List[dict]:
    rows = [
        ("B00_IDENTITY", .66, 40.0, 0.0),
        ("B01_T62_S20_M50", .62, 20.0, .50),
        ("B02_T62_S30_M60", .62, 30.0, .60),
        ("B03_T62_S40_M70", .62, 40.0, .70),
        ("B04_T64_S20_M50", .64, 20.0, .50),
        ("B05_T64_S30_M60", .64, 30.0, .60),
        ("B06_T64_S40_M70", .64, 40.0, .70),
        ("B07_T66_S20_M50", .66, 20.0, .50),
        ("B08_T66_S30_M60", .66, 30.0, .60),
        ("B09_T66_S30_M70", .66, 30.0, .70),
        ("B10_T66_S40_M70", .66, 40.0, .70),
        ("B11_T68_S20_M50", .68, 20.0, .50),
        ("B12_T68_S30_M60", .68, 30.0, .60),
        ("B13_T68_S40_M70", .68, 40.0, .70),
        ("B14_T70_S40_M70", .70, 40.0, .70),
    ]
    result = []
    for identifier, threshold, slope, maximum in rows:
        config = {
            "threshold": threshold, "slope": slope, "max_residual": maximum,
            "fine_k": 4, "broad_k": 18,
        }
        result.append({
            "candidate_id": identifier,
            "formula": "g=sigmoid(s*(mean[1-cos(z,P4z)]-t)); beta=m*g; "
                       "z_out=(1-beta)z+beta*((1-g)P4z+g*P18z)",
            "source": "clean-room project synthesis of sparse graph residual and continuous fusion",
            "config": config, "config_sha256": canonical_config_sha256(config),
            "parameter_count": 1, "label_in_loss": False,
            "dataset_name_routing": False,
        })
    return result


def _labels_for_payload(name: str, spec: Mapping[str, object],
                        ids: Sequence[str], rna: ad.AnnData = None) -> Tuple[np.ndarray, np.ndarray]:
    kind = str(spec["label_kind"])
    if kind.startswith("csv"):
        table = pd.read_csv(str(spec["label"]))
        mapping = dict(zip(table["Barcode"].astype(str), table["manual-anno"]))
        if kind == "csv_strip_slice":
            values = [mapping[str(item)[3:]] for item in ids]
        else:
            values = [mapping[str(item)] for item in ids]
        mask = np.ones(len(ids), dtype=bool)
    elif kind == "obs_final_annot":
        if rna is None:
            rna = ad.read_h5ad(str(spec["rna"]), backed="r")
        series = rna.obs["final_annot"]
        values = [series.loc[str(item)] for item in ids]
        mask = pd.notna(pd.Series(values)).to_numpy()
    else:
        raise ValueError("unsupported label kind for h5ad payload")
    return np.asarray(values, dtype=object), mask


def _load_misar() -> dict:
    base = N8B / "cache/base"
    ids = pd.read_csv(base / "observation_ids.tsv", sep="\t").iloc[:, 0].astype(str).to_numpy()
    coordinates = np.load(base / "coordinates.npy", allow_pickle=False).astype(np.float64)
    rna = np.load(base / "rna_pca_scores.npy", allow_pickle=False)[:, :30]
    atac = np.load(base / "features_omics2.npy", allow_pickle=False)[:, :50]
    embedding = PCA(n_components=64, random_state=SEED).fit_transform(
        np.column_stack((StandardScaler().fit_transform(rna),
                         StandardScaler().fit_transform(atac)))
    ).astype(np.float32)
    mapping = pd.read_csv(REPO / "outputs/night8b_handoff/prelabel_observation_mapping.csv")
    if mapping["observation_id"].astype(str).tolist() != ids.tolist():
        raise RuntimeError("MISAR authority observation order mismatch")
    carrier = N8B / "annotation_carrier/MISAR_seq_mouse_E15_brain_ATAC_data.h5"
    if file_sha256(carrier) != "2f5862cff045b6a296f3cbc0a978d8576bd36f2d4c9419b75c6760c9c7e9d2e3":
        raise RuntimeError("MISAR label carrier SHA mismatch")
    with h5py.File(str(carrier), "r") as handle:
        raw = np.asarray(handle["Y"]).reshape(-1)
    decoded = np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in raw])
    labels = decoded[mapping["carrier_row"].to_numpy(np.int64)]
    return {
        "ids": ids, "coordinates": coordinates, "embedding": embedding,
        "view1": rna.astype(np.float32), "view2": atac.astype(np.float32),
        "view1_shape": [len(ids), 30], "view2_shape": [len(ids), 50],
        "raw_shapes": [[1949, int(np.load(base / "features_omics1.npy", mmap_mode="r").shape[1])],
                       [1949, 50]],
        "labels": labels, "label_mask": np.ones(len(ids), dtype=bool),
        "source_sha256": {"cache_manifest": file_sha256(base / "manifest.json"),
                          "label_carrier": file_sha256(carrier)},
    }


def load_dataset(name: str) -> dict:
    spec = DATASETS[name]
    if name == "MISAR_E15_5_S1":
        return _load_misar()
    observation_ids = None
    if spec["label_kind"] == "csv_exact_registered":
        observation_ids = pd.read_csv(str(spec["label"]))["Barcode"].astype(str).tolist()
    payload = load_h5ad_pair(Path(str(spec["rna"])), Path(str(spec["other"])),
                             str(spec["adapter"]), observation_ids=observation_ids)
    rna = ad.read_h5ad(str(spec["rna"]), backed="r")
    labels, mask = _labels_for_payload(name, spec, payload["ids"], rna)
    embedding = simple_embedding(payload)
    return {
        **payload, "embedding": embedding, "labels": labels, "label_mask": mask,
        "view1_shape": list(np.asarray(payload["view1"]).shape),
        "view2_shape": list(np.asarray(payload["view2"]).shape),
    }


def operators(coordinates: np.ndarray, ids: Sequence[str]) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    return tuple(row_stochastic(sparse_spatial_graph(coordinates, ids, k=k))
                 for k in (4, 18))


def partition_metrics(labels: Sequence[object], mask: np.ndarray,
                      partition: np.ndarray, graph: sp.spmatrix) -> dict:
    mask = np.asarray(mask, dtype=bool)
    encoded, _ = pd.factorize(pd.Series(np.asarray(labels, dtype=object)[mask]), sort=True)
    observed = np.asarray(partition, dtype=np.int64)[mask]
    if np.any(encoded < 0):
        raise ValueError("evaluator mask retained a missing label")
    return {
        "absolute_ari": float(adjusted_rand_score(encoded, observed)),
        "absolute_nmi": float(normalized_mutual_info_score(encoded, observed)),
        "ami": float(adjusted_mutual_info_score(encoded, observed)),
        "fmi": float(fowlkes_mallows_score(encoded, observed)),
        "homogeneity": float(homogeneity_score(encoded, observed)),
        "v_measure": float(v_measure_score(encoded, observed)),
        "morans_i": float(mean_binary_moran(partition, graph)),
        "gearys_c": float(mean_binary_geary(partition, graph)),
    }


def run_model(payload: Mapping[str, object], config: Mapping[str, object],
              seed: int, gradient_probe: bool = False) -> Tuple[np.ndarray, dict, dict]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if "operators" in payload:
        fine, broad = payload["operators"]
    else:
        fine, broad = operators(np.asarray(payload["coordinates"]), payload["ids"])
    model = ContentAdaptiveGraphResidual(config).to(device)
    embedding = torch.as_tensor(np.asarray(payload["embedding"]), dtype=torch.float32,
                                device=device)
    gpu_seconds = None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    output = model(embedding, scipy_to_torch(fine, device), scipy_to_torch(broad, device))
    if device.type == "cuda":
        end_event.record()
        torch.cuda.synchronize(device)
        gpu_seconds = float(start_event.elapsed_time(end_event) / 1000.0)
    gradient = None
    if gradient_probe:
        loss = gradient_probe_loss(output, embedding)
        loss.backward()
        value = model.threshold_offset.grad
        gradient = {"loss": float(loss.detach().cpu()),
                    "threshold_offset_gradient": float(value.detach().cpu()),
                    "finite_nonzero": bool(torch.isfinite(value) and value.abs() > 0)}
    fused = output["fused"].detach().cpu().numpy().astype(np.float32)
    # The Night-13A common evaluator is a fixed seed-0 endpoint. Scientific run
    # seeds belong to a trainable model, not to the evaluator. This deterministic
    # layer therefore repeats exactly across the registered seed rows.
    partition = KMeans(n_clusters=int(payload["k"]), random_state=SEED,
                       n_init=20).fit_predict(fused).astype(np.int64)
    diagnostics = {
        "discrepancy": float(output["discrepancy"].detach().cpu()),
        "gate": float(output["gate"].detach().cpu()),
        "beta": float(output["beta"].detach().cpu()),
        "embedding_sha256": array_sha256(fused),
        "partition_sha256": array_sha256(partition),
        "model_state_sha256": model_state_sha256(model.state_dict()),
        "gpu_forward_seconds": gpu_seconds,
        "gradient_probe": gradient,
        "device": str(device),
    }
    return fused, partition, {"diagnostics": diagnostics, "model": model,
                              "fine": fine, "broad": broad}


def base_payload(name: str) -> dict:
    payload = load_dataset(name)
    payload["k"] = int(DATASETS[name]["k"])
    payload["operators"] = operators(np.asarray(payload["coordinates"]), payload["ids"])
    payload["metric_graph"] = sparse_spatial_graph(payload["coordinates"], payload["ids"], k=6)
    return payload


def resource_snapshot(start: float, gpu_seconds: float = None) -> dict:
    gpu = float(torch.cuda.max_memory_allocated() / 1048576.0) if torch.cuda.is_available() else 0.0
    return {"wall_seconds": float(time.perf_counter() - start),
            "gpu_seconds": gpu_seconds, "peak_gpu_mib": gpu,
            "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)}


def run_anchor(name: str, output: Path) -> dict:
    start = time.perf_counter()
    payload = base_payload(name)
    graph = payload["metric_graph"]
    partition = KMeans(int(payload["k"]), random_state=0, n_init=20).fit_predict(payload["embedding"])
    metrics = partition_metrics(payload["labels"], payload["label_mask"], partition, graph)
    row = {
        "dataset": name, "method": "simple_standardized_concatenation_corrected",
        "seed": 0, "endpoint": "COMMON_KMEANS", "k": int(payload["k"]),
        "total_observations": int(len(payload["ids"])),
        "evaluated_observations": int(np.sum(payload["label_mask"])),
        "ordered_id_sha256": ordered_id_sha256(payload["ids"]),
        "embedding_sha256": array_sha256(payload["embedding"]),
        "partition_sha256": array_sha256(partition),
        "processed_shapes": [payload["view1_shape"], payload["view2_shape"]],
        "raw_shapes": payload["raw_shapes"], "source_sha256": payload["source_sha256"],
        "status": "PASS", "dense_n_by_n_count": 0, **metrics, **resource_snapshot(start),
    }
    atomic_json(output, row)
    return row


def run_one(name: str, candidate: Mapping[str, object], seed: int,
            phase: str, payload: Mapping[str, object] = None) -> dict:
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    payload = base_payload(name) if payload is None else payload
    fused, partition, aux = run_model(payload, candidate["config"], seed)
    graph = payload["metric_graph"]
    metrics = partition_metrics(payload["labels"], payload["label_mask"], partition, graph)
    return {
        "dataset": name, "phase": phase, "method": "NIGHT13B_UNIFIED_RESIDUAL",
        "candidate_id": candidate["candidate_id"], "config_sha256": candidate["config_sha256"],
        "seed": int(seed), "endpoint": "COMMON_KMEANS", "k": int(payload["k"]),
        "endpoint_seed": SEED,
        "total_observations": int(len(payload["ids"])),
        "evaluated_observations": int(np.sum(payload["label_mask"])),
        "ordered_id_sha256": ordered_id_sha256(payload["ids"]),
        "embedding_sha256": aux["diagnostics"]["embedding_sha256"],
        "partition_sha256": aux["diagnostics"]["partition_sha256"],
        "processed_shapes": [payload["view1_shape"], payload["view2_shape"]],
        "raw_shapes": payload["raw_shapes"], "status": "PASS",
        "discrepancy": aux["diagnostics"]["discrepancy"],
        "gate": aux["diagnostics"]["gate"], "beta": aux["diagnostics"]["beta"],
        "parameter_count": 1, "label_in_loss": False,
        "within_run_label_selection": False, "dataset_name_routing": False,
        "dense_n_by_n_count": 0, **metrics,
        **resource_snapshot(start, aux["diagnostics"]["gpu_forward_seconds"]),
    }


def p0_one(name: str, output: Path) -> dict:
    started = time.perf_counter()
    candidate = [x for x in candidate_registry() if x["candidate_id"] == "B10_T66_S40_M70"][0]
    payload = base_payload(name)
    fused, partition, aux = run_model(payload, candidate["config"], seed=0, gradient_probe=True)
    run = output / name
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "checkpoint.pt"
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in aux["model"].state_dict().items()},
                "config": candidate["config"], "config_sha256": candidate["config_sha256"]}, checkpoint)
    np.savez_compressed(run / "roundtrip_input.npz", embedding=payload["embedding"],
                        ids=np.asarray(payload["ids"], dtype=str),
                        coordinates=payload["coordinates"], expected=fused,
                        expected_partition=partition, k=np.asarray([payload["k"]], dtype=np.int64))
    sp.save_npz(run / "fine.npz", aux["fine"], compressed=True)
    sp.save_npz(run / "broad.npz", aux["broad"], compressed=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "reload",
                    "--run-dir", str(run)], cwd=str(REPO), check=True)
    reload_audit = json.loads((run / "fresh_process_reload.json").read_text(encoding="utf-8"))
    row = {
        "dataset": name, "candidate_id": candidate["candidate_id"],
        "raw_shapes": payload["raw_shapes"],
        "processed_shapes": [payload["view1_shape"], payload["view2_shape"]],
        "embedding_shape": list(payload["embedding"].shape),
        "total_observations": int(len(payload["ids"])),
        "evaluated_observations": int(np.sum(payload["label_mask"])),
        "ordered_id_sha256": ordered_id_sha256(payload["ids"]),
        "gradient_probe": aux["diagnostics"]["gradient_probe"],
        "checkpoint_sha256": file_sha256(checkpoint),
        "fresh_process_reload": reload_audit,
        "status": "PASS" if (aux["diagnostics"]["gradient_probe"]["finite_nonzero"]
                              and reload_audit["status"] == "PASS") else "FAIL",
        "label_in_forward_or_loss": False, "dense_n_by_n_count": 0,
        **resource_snapshot(started, aux["diagnostics"]["gpu_forward_seconds"]),
    }
    atomic_json(run / "p0.json", row)
    return row


def reload_run(run: Path) -> None:
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu")
    if canonical_config_sha256(checkpoint["config"]) != checkpoint["config_sha256"]:
        raise RuntimeError("checkpoint config SHA mismatch")
    values = np.load(run / "roundtrip_input.npz", allow_pickle=False)
    model = ContentAdaptiveGraphResidual(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    embedding = torch.as_tensor(values["embedding"], dtype=torch.float32)
    output = model(embedding, scipy_to_torch(sp.load_npz(run / "fine.npz"), torch.device("cpu")),
                   scipy_to_torch(sp.load_npz(run / "broad.npz"), torch.device("cpu")))
    fused = output["fused"].detach().numpy().astype(np.float32)
    partition = KMeans(int(values["k"][0]), random_state=0, n_init=20).fit_predict(fused).astype(np.int64)
    maximum = float(np.max(np.abs(fused - values["expected"])))
    audit = {
        "fresh_process": True,
        "embedding_exact": bool(np.array_equal(fused, values["expected"])),
        "embedding_max_abs": maximum,
        "embedding_numerical_roundtrip": bool(maximum <= 1e-6),
        "numerical_tolerance": 1e-6,
        "partition_exact": bool(np.array_equal(partition, values["expected_partition"])),
        "state_sha256": model_state_sha256(model.state_dict()),
    }
    audit["status"] = "PASS" if (audit["embedding_numerical_roundtrip"]
                                      and audit["partition_exact"]) else "FAIL"
    atomic_json(run / "fresh_process_reload.json", audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["anchors", "p0", "search", "formal", "reload"])
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    if args.mode == "reload":
        reload_run(Path(args.run_dir))
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    registry = candidate_registry()
    atomic_json(ROOT / "manifests/candidate_registry.json", registry)
    if args.mode == "anchors":
        rows = [run_anchor(name, ROOT / "anchors" / (name + ".json")) for name in DATASETS]
        pd.DataFrame(rows).to_csv(ROOT / "anchors/corrected_simple_anchors.csv", index=False)
    elif args.mode == "p0":
        rows = [p0_one(name, ROOT / "p0") for name in ("A1", "P22")]
        atomic_json(ROOT / "p0/real_p0_summary.json", {"passed": sum(x["status"] == "PASS" for x in rows),
                                                        "expected": 2, "rows": rows})
    elif args.mode == "search":
        (ROOT / "search").mkdir(parents=True, exist_ok=True)
        rows = []
        payloads = {name: base_payload(name) for name in ("A1", "tonsil_s1", "P22")}
        for candidate in registry:
            for name in ("A1", "tonsil_s1", "P22"):
                rows.append(run_one(name, candidate, 0, "discovery", payloads[name]))
        pd.DataFrame(rows).to_csv(ROOT / "search/search_results.csv", index=False)
    elif args.mode == "formal":
        (ROOT / "formal").mkdir(parents=True, exist_ok=True)
        candidate = [x for x in registry if x["candidate_id"] == "B10_T66_S40_M70"][0]
        rows = []
        for name in DATASETS:
            payload = base_payload(name)
            for seed in range(5):
                rows.append(run_one(name, candidate, seed, str(DATASETS[name]["phase"]), payload))
        pd.DataFrame(rows).to_csv(ROOT / "formal/formal_metrics.csv", index=False)


if __name__ == "__main__":
    main()
