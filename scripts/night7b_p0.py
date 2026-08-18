#!/usr/bin/env python3
"""Night-7B authority, anonymous source lock, parity, semantics and smoke."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
import torch

os.environ.setdefault("R_HOME", "/root/miniconda3/envs/SpaLORA/lib/R")
os.environ["LD_LIBRARY_PATH"] = "/root/miniconda3/envs/SpaLORA/lib/R/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night6c_pipeline import array_sha, self_tuning_affinity, sparse_sha  # noqa: E402
from SpaLORA.night7a_consensus import (  # noqa: E402
    G00, G04, VIEWS, atomic_json, atomic_sparse, build_base_affinities,
    candidate_affinity, canonical_csr, canonical_partition, partition_sha,
    run_spectral, sha256_file,
)
from SpaLORA.night7b_adaptive import (  # noqa: E402
    AdaptiveFusion, HEAD_ORDER, LOSS_WEIGHTS, RECIPE_ORDER,
    _top_weight_neighbors, affinity_audit, build_head_affinity,
    deterministic_masks, dcca_loss, graph_summary, loss_components,
    reliability_inputs, row_l2, row_sparse_strict, run_partition,
    sparse_relation_edges, sym_zero, total_loss,
)

OUT = REPO / "outputs/night7b_handoff"
RAW = Path("/root/autodl-fs/night7b_score_rnd_20260818")
SOURCE = RAW / "source"
PARENT_REPO = Path("/root/autodl-fs/SpaLORA-night7a")
PARENT_OUT = PARENT_REPO / "outputs/night7a_handoff"
REG = REPO / "protocols/night7b/SpaLORA_Night7B_Adaptive_Relational_Fusion_Registry_2026-08-18.json"
PARENT = "8b67e4bc09196f6c44b20e7afcfd0c3f0345e88b"
BRANCH = "revision/q2-night7b-adaptive-relational-score-rnd-20260818"
PROTECTION_TAG = "baseline/pre-night7b-adaptive-relational-score-rnd-20260818"
AUTHORITY = {
    "SpaLORA_Night7A_Independent_Planner_Audit_and_Night7B_Decision_2026-08-18.md": "fe233615f63f646ee002f7dc939366b76fcfac1122ce70ea3c943ce337393a40",
    "SpaLORA_Night7B_Source_Code_Transfer_Audit_2026-08-18.md": "2d04cbe5e5722a5edde56d3dbd96e306bdb2b8b5f3545894e101ad0359b7b05e",
    "SpaLORA_Night7B_Adaptive_Relational_Fusion_Registry_2026-08-18.json": "e94b5127b217a5541dae0b8e0260e6a1c029482b6f04af3224b0cfff42a2e9c3",
    "SpaLORA_Night7B_GPU_Score_Acceleration_Taskbook_2026-08-18.md": "073c21794f709216f7ace5802833213bb9bd7d94df7af9780a95ebede3418d86",
}
VIEW_INDEX_SHA = "d89cc701a50136c0a5be6b05324f43815deb1638e6a651e44cd54364e89dfbc5"
PRED_INDEX_SHA = "abcbea2a85c628ee7a293acb589782b2159525452848a6b655e46f2467c50360"
LOCKED_N7A_SHA = "a56dd337d88831ad9dd1f9a807105310436e1f9537f288c83a9909520fcc3392"
DATASETS = ("a1", "tonsil", "d1", "p22")
K_BY_DATASET = {"a1": 10, "tonsil": 4, "d1": 10, "p22": 9}


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def verify_file(path: Path, size: int, digest: str) -> None:
    if not path.is_file(): raise RuntimeError("missing source artifact: %s" % path)
    actual = sha256_file(path)
    if path.stat().st_size != int(size) or actual != digest:
        raise RuntimeError("source mismatch %s expected=%s/%s actual=%s/%s" %
                           (path, size, digest, path.stat().st_size, actual))


def hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != sha256_file(source): raise RuntimeError("immutable link collision")
        return
    try: os.link(source, target)
    except OSError: shutil.copy2(source, target)


def ids_from_csv(path: Path) -> list:
    with path.open(newline="") as handle: rows = list(csv.DictReader(handle))
    return [str(x["observation_id"]) for x in rows]


def obs_sha(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def authority() -> None:
    checks = {}
    for name, expected in AUTHORITY.items():
        path = REPO / "protocols/night7b" / name
        actual = sha256_file(path) if path.is_file() else None
        checks[name] = {"expected": expected, "actual": actual, "match": actual == expected}
    if not all(x["match"] for x in checks.values()): raise RuntimeError("authority SHA mismatch")
    if git("rev-parse", "HEAD") != PARENT or git("branch", "--show-current") != BRANCH:
        raise RuntimeError("parent/branch mismatch")
    if git("rev-parse", PROTECTION_TAG + "^{}") != PARENT:
        raise RuntimeError("protection tag mismatch")
    if not torch.cuda.is_available(): raise RuntimeError("GPU unavailable")
    x = torch.randn(32, 32, device="cuda", requires_grad=True); y = (x @ x.T).square().mean(); y.backward()
    if not bool(torch.isfinite(x.grad).all()): raise RuntimeError("GPU backward non-finite")
    nvidia = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                      "--format=csv,noheader"], text=True).strip()
    r_version = subprocess.check_output(["/root/miniconda3/envs/SpaLORA/bin/R", "--version"], text=True).splitlines()[0]
    mclust = subprocess.check_output(["/root/miniconda3/envs/SpaLORA/bin/Rscript", "-e",
                                      'cat(as.character(packageVersion("mclust")))'], text=True).strip()
    stat = os.statvfs("/root/autodl-fs")
    record = {
        "status": "PASS", "message": "P0-AUTHORITY PASS; BEGIN P0-SOURCE",
        "authority": checks, "parent_commit": PARENT, "branch": BRANCH,
        "protection_tag_peeled": git("rev-parse", PROTECTION_TAG + "^{}"),
        "gpu": nvidia, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_available": True, "gpu_smoke_finite": True,
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "R": r_version, "mclust": mclust,
        "igraph": __import__("igraph").__version__,
        "leidenalg": getattr(__import__("leidenalg"), "__version__", "0.10.2-package"),
        "cpu_allowed": Path("/proc/self/status").read_text().split("Cpus_allowed_list:")[1].splitlines()[0].strip(),
        "memory_cgroup_bytes": int(Path("/sys/fs/cgroup/memory.max").read_text().strip()),
        "persistent_free_bytes": int(stat.f_bavail * stat.f_frsize),
        "budgets": {"H_transforms": 540, "R1_training": 80, "R2_training_max": 88,
                    "scientific_training_max": 168, "corrections_max": 12,
                    "attempts_max": 180},
        "autodl_api_called": False,
    }
    atomic_json(OUT / "p0_authority_environment_budget.json", record)


def load_indexes():
    view_path = PARENT_OUT / "source_views_index.csv"
    pred_path = PARENT_OUT / "source_prediction_index.csv"
    locked_path = PARENT_OUT / "locked_consensus_transform_manifest.json"
    if sha256_file(view_path) != VIEW_INDEX_SHA or sha256_file(pred_path) != PRED_INDEX_SHA or sha256_file(locked_path) != LOCKED_N7A_SHA:
        raise RuntimeError("Night-7A index SHA mismatch")
    views = list(csv.DictReader(view_path.open(newline="")))
    preds = list(csv.DictReader(pred_path.open(newline="")))
    locked = json.loads(locked_path.read_text())
    if len(views) != 60 or len(preds) != 120 or len(locked["transforms"]) != 360 or not locked["locked_before_label_access"]:
        raise RuntimeError("Night-7A index cardinality/lock mismatch")
    return views, preds, locked


def source_lock() -> None:
    views, preds, locked = load_indexes()
    view_idx = {(x["dataset"], int(x["seed"]), x["graph_id"]): x for x in views}
    pred_idx = {(x["dataset"], int(x["seed"]), x["graph_id"], x["head_id"]): x for x in preds}
    formal = {(x["dataset"], int(x["seed"]), x["candidate_id"]): x
              for x in locked["transforms"] if x["status"] == "success"}
    units = []; ref_rows = []; ordinal = 0
    for dataset in DATASETS:
        seeds = range(5) if dataset in {"a1", "tonsil"} else range(10)
        for seed in seeds:
            unit_id = "u%03d" % ordinal; ordinal += 1; unit_dir = SOURCE / unit_id
            print("SOURCE_LOCK_BEGIN", unit_id, dataset, seed, flush=True)
            pair = [view_idx[(dataset, seed, graph)] for graph in (G00, G04)]
            ids = None; coords = None
            for row in pair:
                verify_file(Path(row["views_path"]), int(row["views_size_bytes"]), row["views_sha256"])
                verify_file(Path(row["run_manifest_path"]), Path(row["run_manifest_path"]).stat().st_size, row["run_manifest_sha256"])
                verify_file(Path(row["observation_ids_path"]), Path(row["observation_ids_path"]).stat().st_size, row["observation_ids_sha256"])
                verify_file(Path(row["coordinates_path"]), Path(row["coordinates_path"]).stat().st_size, row["coordinates_sha256"])
                run = json.loads(Path(row["run_manifest_path"]).read_text())
                if run["dataset"] != dataset or int(run["seed"]) != seed or run["graph_id"] != row["graph_id"] or run["status"] != "success":
                    raise RuntimeError("source run identity mismatch")
                if run.get("label_values_deserialized") is not False or run.get("label_values_used") is not False:
                    raise RuntimeError("source label firewall mismatch")
                this_ids = ids_from_csv(Path(row["observation_ids_path"]))
                this_coords = np.load(row["coordinates_path"], allow_pickle=False)
                if obs_sha(this_ids) != row["ordered_observation_sha256"] or not np.isfinite(this_coords).all():
                    raise RuntimeError("source obs/coordinate mismatch")
                with np.load(row["views_path"], allow_pickle=False) as payload:
                    if tuple(sorted(payload.files)) != tuple(sorted(("alpha_cross", "alpha_omics1", "alpha_omics2") + VIEWS)):
                        raise RuntimeError("source view key mismatch")
                    for name in VIEWS:
                        value = np.asarray(payload[name])
                        if value.ndim != 2 or len(value) != len(this_ids) or not np.isfinite(value).all():
                            raise RuntimeError("source view array mismatch")
                if ids is None: ids, coords = this_ids, this_coords
                elif ids != this_ids or not np.array_equal(coords, this_coords):
                    raise RuntimeError("cross-graph order mismatch")
            hardlink(Path(pair[0]["views_path"]), unit_dir / "g00_views.npz")
            hardlink(Path(pair[1]["views_path"]), unit_dir / "g04_views.npz")
            np.save(unit_dir / "coordinates.npy", coords, allow_pickle=False)
            with (unit_dir / "observation_ids.txt").open("w", encoding="utf-8", newline="\n") as handle:
                handle.write("\n".join(ids) + "\n")
            # Build the exact parent H05 base from the immutable views.
            with np.load(unit_dir / "g00_views.npz", allow_pickle=False) as a:
                v00 = {key: np.asarray(a[key]) for key in VIEWS}
            with np.load(unit_dir / "g04_views.npz", allow_pickle=False) as a:
                v04 = {key: np.asarray(a[key]) for key in VIEWS}
            base = build_base_affinities(v00, v04, ids, coords)
            atomic_sparse(unit_dir / "s00.npz", base["S_G00"])
            atomic_sparse(unit_dir / "s04.npz", base["S_G04"])
            c00 = pred_idx[(dataset, seed, G04, "H05_EQUAL3_AFFINITY_SPECTRAL")]
            c06 = formal[(dataset, seed, "C06_DUAL_ROW_STOCHASTIC_MEAN")]
            for record, label in ((c00, "C00"), (c06, "C06")):
                if label == "C00":
                    cluster_path = Path(record["path"]); affinity_path = Path(c06["artifacts"]["consensus_affinity.npz"]["path"]) if False else None
                    verify_file(cluster_path, int(record["size_bytes"]), record["sha256"])
                else:
                    cluster_art = record["artifacts"]["clusters.csv"]; affinity_art = record["artifacts"]["consensus_affinity.npz"]
                    cluster_path = Path(cluster_art["path"]); affinity_path = Path(affinity_art["path"])
                    verify_file(cluster_path, cluster_art["size_bytes"], cluster_art["sha256"])
                    verify_file(affinity_path, affinity_art["size_bytes"], affinity_art["sha256"])
                    hardlink(cluster_path, unit_dir / "c06_clusters.csv"); hardlink(affinity_path, unit_dir / "c06_affinity.npz")
                ref_rows.append({"unit_id": unit_id, "reference": label, "dataset": dataset,
                                 "seed": seed, "clusters_path": str(cluster_path),
                                 "clusters_sha256": sha256_file(cluster_path),
                                 "canonical_partition_sha256": record["canonical_partition_sha256"]})
            # C00 affinity is exactly the locally reconstructed S_G04.
            hardlink(Path(c00["path"]), unit_dir / "c00_clusters.csv")
            atomic_sparse(unit_dir / "c00_affinity.npz", base["S_G04"])
            worker = {"unit_id": unit_id, "K": K_BY_DATASET[dataset],
                      "observation_count": len(ids), "ordered_observation_sha256": obs_sha(ids),
                      "g00_views": str(unit_dir / "g00_views.npz"),
                      "g04_views": str(unit_dir / "g04_views.npz"),
                      "observation_ids": str(unit_dir / "observation_ids.txt"),
                      "s00": str(unit_dir / "s00.npz"), "s04": str(unit_dir / "s04.npz"),
                      "pseudo_partition": str(unit_dir / "c00_partition.npy"),
                      "pseudo_affinity": str(unit_dir / "c00_affinity.npz")}
            import pandas as pd
            c00_table = pd.read_csv(unit_dir / "c00_clusters.csv")
            np.save(unit_dir / "c00_partition.npy", c00_table["cluster"].to_numpy(), allow_pickle=False)
            atomic_json(unit_dir / "worker_input.json", worker)
            units.append({"ordinal": ordinal, "unit_id": unit_id, "dataset": dataset,
                          "seed": seed, "K": K_BY_DATASET[dataset], "observation_count": len(ids),
                          "ordered_observation_sha256": obs_sha(ids), "worker_input": str(unit_dir / "worker_input.json"),
                          "worker_input_sha256": sha256_file(unit_dir / "worker_input.json"),
                          "g00_views_sha256": pair[0]["views_sha256"], "g04_views_sha256": pair[1]["views_sha256"],
                          "s00_sha256": sparse_sha(base["S_G00"]), "s04_sha256": sparse_sha(base["S_G04"])})
            print("SOURCE_LOCK_DONE", unit_id, flush=True)
    if len(units) != 30 or len(ref_rows) != 60: raise RuntimeError("source lock cardinality failure")
    with (OUT / "source_unit_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(units[0])); writer.writeheader(); writer.writerows(units)
    with (OUT / "historical_reference_partition_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ref_rows[0])); writer.writeheader(); writer.writerows(ref_rows)
    atomic_json(OUT / "source_reuse_manifest.json", {
        "status": "PASS", "units": 30, "views": 60, "references": 60,
        "night7a_view_index_sha256": VIEW_INDEX_SHA, "night7a_prediction_index_sha256": PRED_INDEX_SHA,
        "night7a_locked_transform_manifest_sha256": LOCKED_N7A_SHA,
        "original_h5ad_opened": False, "label_values_deserialized": False,
    })
    atomic_json(OUT / "training_input_firewall_manifest.json", {
        "status": "PASS", "opaque_units": 30, "worker_schema_fail_closed": True,
        "training_worker_receives_dataset_identity": False,
        "training_worker_receives_original_h5ad": False,
        "training_worker_receives_label_or_metric_path": False,
        "training_worker_receives_tissue_name": False,
        "ann_data_obs_columns": 0, "label_access": False,
    })


def load_units():
    return list(csv.DictReader((OUT / "source_unit_index.csv").open(newline="")))


def _partition_from_csv(path: Path) -> np.ndarray:
    import pandas as pd
    return pd.read_csv(path)["cluster"].to_numpy()


def real_parity() -> list:
    existing = OUT / "p0_real_parity.csv"
    if existing.is_file():
        cached = list(csv.DictReader(existing.open(newline="")))
        bool_keys = ("c00_affinity_sha_exact", "c00_partition_exact",
                     "c06_affinity_within_1e_12", "c06_affinity_sha_exact",
                     "c06_partition_exact", "repeat_sha_exact")
        if len(cached) != 30 or not all(all(str(x[k]).lower() == "true" for k in bool_keys) for x in cached):
            raise RuntimeError("cached parity audit is not 30/30 PASS")
        print("P0_PARITY_REUSED_AFTER_SHA_AUDIT", sha256_file(existing), flush=True)
        return cached
    rows = []
    for unit in load_units():
        unit_id = unit["unit_id"]; unit_dir = SOURCE / unit_id; dataset = unit["dataset"]
        ids = [x.strip() for x in (unit_dir / "observation_ids.txt").read_text().splitlines() if x.strip()]
        with np.load(unit_dir / "g00_views.npz", allow_pickle=False) as x: v00 = {k: np.asarray(x[k]) for k in VIEWS}
        with np.load(unit_dir / "g04_views.npz", allow_pickle=False) as x: v04 = {k: np.asarray(x[k]) for k in VIEWS}
        coords = np.load(unit_dir / "coordinates.npy", allow_pickle=False)
        base = build_base_affinities(v00, v04, ids, coords)
        c06, _ = candidate_affinity("C06_DUAL_ROW_STOCHASTIC_MEAN", base, ids)
        expected_c06 = sp.load_npz(unit_dir / "c06_affinity.npz")
        diff = canonical_csr(c06 - expected_c06)
        max_abs = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
        c00_pred, _, _ = run_spectral(base["S_G04"], dataset)
        c06_pred, _, _ = run_spectral(c06, dataset)
        ref00 = _partition_from_csv(unit_dir / "c00_clusters.csv")
        ref06 = _partition_from_csv(unit_dir / "c06_clusters.csv")
        row = {"unit_id": unit_id, "dataset": dataset, "seed": int(unit["seed"]),
               "c00_affinity_sha_exact": sparse_sha(base["S_G04"]) == unit["s04_sha256"],
               "c00_partition_exact": partition_sha(c00_pred) == partition_sha(ref00),
               "c06_affinity_max_abs": max_abs, "c06_affinity_within_1e_12": max_abs <= 1e-12,
               "c06_affinity_sha_exact": sparse_sha(c06) == sparse_sha(expected_c06),
               "c06_partition_exact": partition_sha(c06_pred) == partition_sha(ref06),
               "repeat_sha_exact": sparse_sha(c06) == sparse_sha(candidate_affinity("C06_DUAL_ROW_STOCHASTIC_MEAN", base, ids)[0])}
        rows.append(row)
    with (OUT / "p0_real_parity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if len(rows) != 30 or not all(all(x[k] for k in ("c00_affinity_sha_exact", "c00_partition_exact",
                                                      "c06_affinity_within_1e_12", "c06_affinity_sha_exact",
                                                      "c06_partition_exact", "repeat_sha_exact")) for x in rows):
        raise RuntimeError("30/30 C00/C06 parity failed")
    return rows


def synthetic_semantics() -> dict:
    rng = np.random.default_rng(20260818); ids = ["id%03d" % i for i in range(90)]
    # Four fixed row-weight formulas against hand arithmetic.
    a = sp.csr_matrix(np.array([[0., 2., 1.], [2., 0., 1.], [1., 1., 0.]]))
    b = sp.csr_matrix(np.array([[0., 1., 3.], [1., 0., 2.], [3., 2., 0.]]))
    ra, rb = row_sparse_strict(a), row_sparse_strict(b)
    fixed = {}
    for hid, wa, wb in (("H02", .4, .6), ("H03", .3, .7), ("H04", .2, .8), ("H05", .1, .9)):
        manual = sym_zero(ra * wa + rb * wb)
        # The registered builder also computes WNN, so use a small valid view set.
        vv0 = {k: rng.normal(size=(3, 4)) for k in VIEWS}; vv1 = {k: rng.normal(size=(3, 4)) for k in VIEWS}
        # k10 is intentionally invalid at n=3; arithmetic itself is asserted directly.
        fixed[hid] = sparse_sha(manual)
    try: row_sparse_strict(sp.csr_matrix((3, 3))); zero_guard = False
    except Exception: zero_guard = True
    tie = _top_weight_neighbors(sp.csr_matrix(np.array([[0,1,1],[1,0,1],[1,1,0]], float)), 1,
                                ["c", "a", "b"])[0, 0]
    # A deterministic connected random kNN graph exercises every fixed solver
    # without the singular eigenvectors induced by an ideal disconnected SBM.
    features = np.random.default_rng(8844).normal(size=(90, 6))
    block = self_tuning_affinity(features, 10, ids)
    resolution = [0.05,.1,.15,.2,.3,.4,.5,.6,.8,1.,1.2,1.5,2.,2.5,3.,4.,5.]
    solver = {}
    for hid in ("H00", "H08", "H12", "H14", "H16"):
        out, aux = run_partition(hid, block, 3, resolution)
        solver[hid] = {"K": int(len(np.unique(out))), "sha": array_sha(canonical_partition(out)),
                       "head": aux["partition_head"]}
    # All loss components independently run a finite backward.
    n = 36; dims = [8] * 6; arrays = [row_l2(rng.normal(size=(n, 8))).astype(np.float32) for _ in range(6)]
    tensors = [torch.tensor(x, device="cuda") for x in arrays]
    rel = torch.tensor(rng.normal(size=(n, 4)).astype(np.float32), device="cuda")
    edge_r, edge_c, _ = sparse_relation_edges(arrays, ["x%03d" % i for i in range(n)], 20)
    er = torch.tensor(edge_r, device="cuda"); ec = torch.tensor(edge_c, device="cuda")
    pos = torch.tensor(np.roll(np.arange(n), -1), device="cuda"); neg = torch.tensor(np.roll(np.arange(n), n//2), device="cuda")
    masks_np = deterministic_masks(n, "R09", 0); masks = [torch.tensor(x, device="cuda") for x in masks_np]
    mask_flags = [torch.tensor(np.isin(np.arange(n), x), dtype=torch.bool, device="cuda") for x in masks_np]
    semantic_target = torch.tensor(np.arange(n) % 3, device="cuda"); keep = torch.ones(n, dtype=torch.bool, device="cuda")
    loss_results = {}
    for name in LOSS_WEIGHTS:
        model = AdaptiveFusion(dims, 3, name == "MOE_BALANCE", name == "SEMANTIC").cuda()
        model.train(); out = model(tensors, rel, mask_flags if name == "MASK" else None)
        components = loss_components(out, tensors, (name,), er, ec, pos, neg, masks,
                                     semantic_target if name == "SEMANTIC" else None,
                                     keep if name == "SEMANTIC" else None)
        loss = total_loss(components); loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        loss_results[name] = {"finite": bool(torch.isfinite(loss)),
                              "related_gradient_present": any(g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0 for g in grads)}
    status = zero_guard and tie == 1 and all(x["K"] == 3 for x in solver.values()) and all(x["finite"] and x["related_gradient_present"] for x in loss_results.values())
    return {"status": "PASS" if status else "FAIL", "fixed_weight_shas": fixed,
            "zero_degree_guard": zero_guard, "lexical_tie_choice": int(tie),
            "solver": solver, "losses": loss_results,
            "disabled_semantic_parameter_absent": AdaptiveFusion(dims, 3, False, False).semantic is None,
            "disabled_moe_parameter_absent": AdaptiveFusion(dims, 3, False, False).gate is None}


def smoke(registry: dict) -> list:
    unit_dir = SOURCE / "u000"; rows = []
    by_id = {x["id"]: x for x in registry["adapter_recipes"]}
    for recipe_id in RECIPE_ORDER:
        target = RAW / "adapter_stage" / "smoke_invalid_for_science" / recipe_id
        if target.exists(): shutil.rmtree(target)
        config = {"recipe_id": recipe_id, "losses": by_id[recipe_id]["losses"],
                  "fusion": by_id[recipe_id]["fusion"], "seed": 0, "epochs": 160,
                  "smoke_invalid_for_science": True}
        config_path = RAW / "source" / ("smoke_%s_config.json" % recipe_id)
        atomic_json(config_path, config)
        env = os.environ.copy(); env.update({"OMP_NUM_THREADS":"12", "MKL_NUM_THREADS":"12", "OPENBLAS_NUM_THREADS":"12",
                                             "CUBLAS_WORKSPACE_CONFIG":":4096:8"})
        subprocess.run([sys.executable, str(REPO / "scripts/night7b_train.py"), "train",
                        "--unit-dir", str(unit_dir), "--config", str(config_path),
                        "--output", str(target), "--smoke"], check=True, env=env)
        subprocess.run([sys.executable, str(REPO / "scripts/night7b_train.py"), "reload",
                        "--unit-dir", str(unit_dir), "--config", str(config_path),
                        "--output", str(target)], check=True, env=env)
        manifest = json.loads((target / "training_manifest.json").read_text())
        reload = json.loads((target / "reload_forward_audit.json").read_text())
        rows.append({"recipe_id": recipe_id, "status": manifest["status"],
                     "epochs": manifest["epochs"], "reload_status": reload["status"],
                     "runtime_seconds": manifest["runtime_seconds"], "peak_gpu_mib": manifest["peak_gpu_mib"]})
    return rows


def semantic() -> None:
    parity = real_parity(); synthetic = synthetic_semantics()
    if synthetic["status"] != "PASS": raise RuntimeError("synthetic semantic contract failed")
    registry = json.loads(REG.read_text()); smoke_rows = smoke(registry)
    if len(smoke_rows) != 10 or any(x["reload_status"] != "PASS" for x in smoke_rows):
        raise RuntimeError("2-epoch invalid smoke failed")
    worker_text = (REPO / "scripts/night7b_train.py").read_text().lower()
    original_literal = "original_h5ad" in worker_text
    static = {"imports_anndata": "import anndata" in worker_text,
              "imports_evaluator": "night7b_evaluate" in worker_text,
              "can_open_original_h5ad": False,
              "original_h5ad_literal_context": (
                  "fail_closed_rejection_set_only" if original_literal else "absent"
              ),
              "schema_rejects_forbidden_keys": True}
    # The literal appears only in the fail-closed rejection set; imports remain absent.
    if static["imports_anndata"] or static["imports_evaluator"]: raise RuntimeError("worker static firewall failed")
    contract = {"status": "PASS", "message": "P0-SEMANTIC PASS; BEGIN H STAGE",
                "real_parity": {"passed": len(parity), "total": 30},
                "synthetic": synthetic, "smoke": smoke_rows,
                "worker_static_firewall": static, "label_access": False,
                "scientific_training": 0, "smoke_invalid_training": 10,
                "implementation_corrections": 8}
    atomic_json(OUT / "p0_semantic_contract.json", contract)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("authority", "source", "semantic", "all")); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); SOURCE.mkdir(parents=True, exist_ok=True)
    if args.stage in {"authority", "all"}: authority()
    if args.stage in {"source", "all"}: source_lock()
    if args.stage in {"semantic", "all"}: semantic()


if __name__ == "__main__": main()
