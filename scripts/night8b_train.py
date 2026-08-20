#!/usr/bin/env python3
"""CUDA training, reload and fixed transforms for Night-8B."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night3a_ige import model_state_sha256
from SpaLORA.night3af_cache import sha256_file
from SpaLORA.night5a_rnd import Night5ATrainer
from SpaLORA.night6c_pipeline import array_sha, atomic_json, forward_model, sparse_sha
from SpaLORA.night7a_consensus import atomic_sparse, canonical_partition
from SpaLORA.night7b_adaptive import VIEWS, run_partition
from SpaLORA.night8b_pipeline import (
    BASE_C04, CACHE, G00, G04, GRAPH_CONTRACTS, MISAR_CFG, OUT, RAW, RUNS,
    adapter_endpoint, base_affinities, canonical_json_sha, load_graph,
    observation_sha, resource_row, rng_snapshot, state_sha,
)


def atomic_npz(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **values)
    os.replace(tmp, path)


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp); os.replace(tmp, path)


def save_clusters(path: Path, ids, labels) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame({"observation_id": ids, "cluster": labels}).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _train_one_graph(seed: int, graph_id: str, target: Path, epochs: int) -> dict:
    prepared, data, graph_manifest = load_graph(graph_id)
    cfg = dict(MISAR_CFG); cfg["epochs"] = int(epochs)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    trainer = Night5ATrainer(data, cfg, BASE_C04, seed, device, {}, 1e-12)
    before_rng = rng_snapshot()
    original_adam = torch.optim.Adam; captured = {}
    def capture_adam(*args, **kwargs):
        optimizer = original_adam(*args, **kwargs); captured["optimizer"] = optimizer; return optimizer
    torch.optim.Adam = capture_adam
    started = time.perf_counter()
    try:
        result = trainer.train()
    finally:
        torch.optim.Adam = original_adam
    if "optimizer" not in captured:
        raise RuntimeError("base optimizer state was not captured")
    views = forward_model(result.model, data, device)
    target.mkdir(parents=True, exist_ok=False)
    view_path = target / "views.npz"; atomic_npz(view_path, views)
    pd.DataFrame(result.logs).to_csv(target / "loss_curve.csv", index=False)
    config = {
        "assay_family": "RNA_EPIGENOME", "dataset_identity_used": False,
        "candidate": BASE_C04, "dataset_config": cfg,
        "graph_contract": GRAPH_CONTRACTS[graph_id], "seed": int(seed),
        "optimizer": "Adam", "learning_rate": 1e-4, "weight_decay": 0.0,
        "scheduler": None, "checkpoint_policy": "fixed_final_epoch",
    }
    checkpoint = {
        "model_state_dict": {k: v.detach().cpu() for k, v in result.model.state_dict().items()},
        "optimizer_state_dict": captured["optimizer"].state_dict(),
        "canonical_config": config, "rng_before": before_rng,
        "rng_after": rng_snapshot(), "final_tensor_state_sha256": model_state_sha256(result.model),
        "view_array_sha256": {k: array_sha(v) for k, v in views.items()},
    }
    ckpt = target / "model_final.pt"; atomic_torch(ckpt, checkpoint)
    manifest = {
        "status": "smoke_invalid_for_science" if epochs != 1600 else "success",
        "seed": int(seed), "graph_id": graph_id, "epochs": int(epochs),
        "scientific_training": epochs == 1600, "label_access": False,
        "retry": False, "fallback": False, "cuda_used": True,
        "gpu_model": torch.cuda.get_device_name(0),
        "canonical_config": config, "canonical_config_sha256": canonical_json_sha(config),
        "checkpoint_path": str(ckpt), "checkpoint_sha256": sha256_file(ckpt),
        "final_tensor_state_sha256": checkpoint["final_tensor_state_sha256"],
        "views_path": str(view_path), "views_sha256": sha256_file(view_path),
        "view_array_sha256": checkpoint["view_array_sha256"],
        "base_cache_manifest_sha256": sha256_file(CACHE / "base/manifest.json"),
        "graph_cache_manifest_sha256": sha256_file(CACHE / "graphs" / graph_id / "manifest.json"),
        "ordered_observation_sha256": observation_sha(prepared.obs_names.astype(str)),
        "coefficients": result.coefficients, "initial_state_sha256": result.initial_state_sha256,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        **resource_row(started),
    }
    atomic_json(target / "training_manifest.json", manifest)
    return manifest


def base(seed: int, root: Path, epochs: int) -> None:
    target = root / f"seed_{seed}" / "attempt_001"
    if target.exists(): raise RuntimeError(f"refusing to overwrite base cell {target}")
    started = time.perf_counter(); rows = []
    for graph_id in (G00, G04):
        rows.append(_train_one_graph(seed, graph_id, target / graph_id, epochs))
    aggregate = {
        "status": "smoke_invalid_for_science" if epochs != 1600 else "success",
        "scientific_unit_count": 0 if epochs != 1600 else 1,
        "paired_graph_submodels": 2, "seed": seed, "epochs": epochs,
        "label_access": False, "cuda_used": True, "retry": False,
        "submodels": rows, "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_mib": max(row["peak_gpu_mib"] for row in rows),
    }
    atomic_json(target / "base_unit_manifest.json", aggregate)
    print(json.dumps({"event":"base_complete","seed":seed,"status":aggregate["status"]}))


def reload_base(seed: int, root: Path) -> None:
    target = root / f"seed_{seed}" / "attempt_001"; results = []
    for graph_id in (G00, G04):
        sub = target / graph_id
        manifest = json.loads((sub / "training_manifest.json").read_text())
        checkpoint = torch.load(sub / "model_final.pt", map_location="cuda")
        prepared, data, _ = load_graph(graph_id)
        cfg = checkpoint["canonical_config"]["dataset_config"]
        trainer = Night5ATrainer(data, cfg, BASE_C04, seed, torch.device("cuda"), {}, 1e-12)
        model = trainer.new_model(); model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval(); views = forward_model(model, data, torch.device("cuda"))
        with np.load(sub / "views.npz", allow_pickle=False) as saved:
            errors = {k: float(np.max(np.abs(views[k] - saved[k]))) for k in views}
            close = {k: bool(np.allclose(views[k], saved[k], atol=1e-6, rtol=1e-5))
                     for k in views}
        state_exact = model_state_sha256(model) == manifest["final_tensor_state_sha256"]
        results.append({"graph_id":graph_id,"view_allclose":all(close.values()),
                        "per_view_allclose":close,"atol":1e-6,"rtol":1e-5,"max_abs":errors,
                        "state_exact":state_exact,"fresh_process":True})
    audit = {"status":"PASS" if all(x["view_allclose"] and x["state_exact"] for x in results) else "FAIL",
             "seed":seed,"results":results,"label_access":False,"fresh_process":True}
    atomic_json(target / "base_checkpoint_reload_audit.json", audit)
    if audit["status"] != "PASS": raise RuntimeError("base checkpoint reload mismatch")


def prepare_adapter(seed: int, base_root: Path, adapter_root: Path) -> tuple[Path, Path]:
    base_dir = base_root / f"seed_{seed}" / "attempt_001"
    audit = json.loads((base_dir / "base_checkpoint_reload_audit.json").read_text())
    if audit["status"] != "PASS": raise RuntimeError("base reload gate not passed")
    prepared, _, _ = load_graph(G00); ids = prepared.obs_names.astype(str).tolist()
    with np.load(base_dir / G00 / "views.npz", allow_pickle=False) as z:
        v00 = {k: np.asarray(z[k]) for k in VIEWS}
    with np.load(base_dir / G04 / "views.npz", allow_pickle=False) as z:
        v04 = {k: np.asarray(z[k]) for k in VIEWS}
    base_values, c06, _ = base_affinities(v00, v04, ids, prepared.coordinates)
    unit = adapter_root / "inputs" / f"seed_{seed}"; unit.mkdir(parents=True, exist_ok=False)
    for name, value in (("s00.npz", base_values["S_G00"]), ("s04.npz", base_values["S_G04"]),
                        ("c06_affinity.npz", c06)):
        atomic_sparse(unit / name, value)
    with (unit / "observation_ids.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(ids) + "\n")
    worker = {
        "unit_id": f"opaque_seed_{seed:02d}", "K": 12, "observation_count": len(ids),
        "ordered_observation_sha256": observation_sha(ids),
        "g00_views": str(base_dir / G00 / "views.npz"),
        "g04_views": str(base_dir / G04 / "views.npz"),
        "observation_ids": str(unit / "observation_ids.txt"),
        "s00": str(unit / "s00.npz"), "s04": str(unit / "s04.npz"),
        "pseudo_partition": "", "pseudo_affinity": "",
    }
    atomic_json(unit / "worker_input.json", worker)
    config = {"recipe_id":"R02","losses":["RECON","MNN"],"fusion":"equal",
              "seed":seed,"epochs":160}
    config_path = adapter_root / "configs" / f"seed_{seed}.json"; atomic_json(config_path, config)
    return unit, config_path


def adapter(seed: int, base_root: Path, adapter_root: Path, smoke: bool) -> None:
    unit, config = prepare_adapter(seed, base_root, adapter_root)
    output = adapter_root / "formal" / f"seed_{seed}" / "attempt_001" / "worker"
    command = [sys.executable, str(REPO / "scripts/night7b_train.py"), "train",
               "--unit-dir", str(unit), "--config", str(config), "--output", str(output)]
    if smoke: command.append("--smoke")
    started=time.perf_counter(); completed=subprocess.run(command, cwd=REPO)
    if completed.returncode: raise RuntimeError("R02 adapter training process failed")
    subprocess.run([sys.executable, str(REPO / "scripts/night7b_train.py"), "reload",
                    "--unit-dir", str(unit), "--config", str(config), "--output", str(output)],
                   cwd=REPO, check=True)
    manifest=json.loads((output/"training_manifest.json").read_text())
    wrapper={"status":manifest["status"],"seed":seed,"scientific_training":not smoke,
             "label_access":False,"retry":False,"fallback":False,
             "worker_manifest_sha256":sha256_file(output/"training_manifest.json"),
             "reload_audit_sha256":sha256_file(output/"reload_forward_audit.json"),
             "runtime_seconds":time.perf_counter()-started,"worker":manifest}
    atomic_json(output.parent/"adapter_unit_manifest.json",wrapper)


def transform(seed: int, method: str, base_root: Path, adapter_root: Path,
              transform_root: Path) -> None:
    started=time.perf_counter(); prepared,_,_=load_graph(G00)
    ids=prepared.obs_names.astype(str).tolist(); unit=adapter_root/"inputs"/f"seed_{seed}"
    s04=sp.load_npz(unit/"s04.npz"); c06=sp.load_npz(unit/"c06_affinity.npz")
    if method == "U00": affinity=s04; head="H05"
    elif method == "F00":
        embedding=np.load(adapter_root/"formal"/f"seed_{seed}"/"attempt_001/worker/embedding.npy",allow_pickle=False)
        affinity=adapter_endpoint(embedding,c06,ids); head="H01"
    else: raise KeyError(method)
    labels,part=run_partition(head,affinity,12,[])
    target=transform_root/method/f"seed_{seed}"; target.mkdir(parents=True,exist_ok=False)
    atomic_sparse(target/"affinity.npz",affinity); save_clusters(target/"clusters.csv",ids,labels)
    manifest={"status":"success","method":method,"seed":seed,"K":12,"head_id":head,
              "method_full_id":"U00_UNIVERSAL_C00" if method=="U00" else "F00_FAMILY_R02",
              "label_access":False,
              "retry":False,"fallback":False,"partition":part,
              "canonical_affinity_sha256":sparse_sha(affinity),
              "affinity_file_sha256":sha256_file(target/"affinity.npz"),
              "canonical_partition_sha256":array_sha(canonical_partition(labels)),
              "clusters_file_sha256":sha256_file(target/"clusters.csv"),
              **resource_row(started)}
    atomic_json(target/"transform_manifest.json",manifest)


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="mode",required=True)
    b=sub.add_parser("base"); b.add_argument("--seed",type=int,required=True); b.add_argument("--root",type=Path,required=True); b.add_argument("--epochs",type=int,default=1600)
    r=sub.add_parser("reload-base"); r.add_argument("--seed",type=int,required=True); r.add_argument("--root",type=Path,required=True)
    a=sub.add_parser("adapter"); a.add_argument("--seed",type=int,required=True); a.add_argument("--base-root",type=Path,required=True); a.add_argument("--adapter-root",type=Path,required=True); a.add_argument("--smoke",action="store_true")
    t=sub.add_parser("transform"); t.add_argument("--seed",type=int,required=True); t.add_argument("--method",choices=("U00","F00"),required=True); t.add_argument("--base-root",type=Path,required=True); t.add_argument("--adapter-root",type=Path,required=True); t.add_argument("--transform-root",type=Path,required=True)
    args=ap.parse_args()
    if args.mode=="base": base(args.seed,args.root,args.epochs)
    elif args.mode=="reload-base": reload_base(args.seed,args.root)
    elif args.mode=="adapter": adapter(args.seed,args.base_root,args.adapter_root,args.smoke)
    else: transform(args.seed,args.method,args.base_root,args.adapter_root,args.transform_root)


if __name__ == "__main__": main()
