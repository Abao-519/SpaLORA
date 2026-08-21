#!/usr/bin/env python3
"""Fail-closed Night-10A P0 authority, resource, metric and semantic audit."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import sklearn
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO = Path("/root/autodl-fs/SpaLORA-night10a")
RAW = Path("/root/autodl-fs/night10a_qcrd_20260821")
P0 = RAW / "p0_audit"
PLANNING = REPO / "protocols/night10a/planning"
EXPECTED_PARENT = "e9bd62e2bb07c58f58c219956b8aaf090527c471"
EXPECTED = {
    "SpaLORA_Night10A_QCRD_Score_RnD_Taskbook_2026-08-21.md": "5551270041576969e1a777a6dcf8f1b8113d2fdefe7b7322738a4d57c8b4bd0d",
    "night10a_qcrd_candidate_registry.json": "b310e232282840772b69ad9d1c1d1de6e04c92e8b7b013a7e5247c628dfd959d",
    "metric_expansion_reference.py": "33397cada3701108fbfb15d58fafd6f1f9e20417ac7bc14cce1e8aee993cd159",
    "test_metric_expansion_reference.py": "4100b23ab30d74db02cb2e540112b9548080ae1c17817011c202a409b48c6ab0",
    "Night10A_Local_Evidence_and_Innovation_Audit_2026-08-21.md": "ea920aeee3593a7903b6189119dc60b7327cee48f9d8b00e0c4283420049b34c",
    "authoritative_scoreboard_20260821.csv": "951a9e0449e6e083834cd15268bb4c316b011726c80bd122dd9a7516626a0d68",
}
ROOTS = [
    "/root/autodl-fs/night6c_raw_runs_20260817",
    "/root/autodl-fs/night6d_raw_runs_20260817",
    "/root/autodl-fs/night7b_score_rnd_20260818",
    "/root/autodl-fs/night8b_raw_runs_20260820",
    "/root/autodl-fs/night9b_racf_20260820",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def mem_total() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemTotal unavailable")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def versions_and_resources() -> dict:
    usage = os.statvfs("/root/autodl-fs")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    return {
        "gpu": gpu,
        "cpu_count": os.cpu_count(),
        "memory_bytes_proc": mem_total(),
        "persistent_free_bytes": usage.f_bavail * usage.f_frsize,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def load_npz_dict(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def real_semantics() -> dict:
    sys.path.insert(0, str(REPO))
    from SpaLORA.night10a_qcrd import (
        QCRDAdapter,
        canonical_state_sha256,
        corrected_views,
        frozen_quality,
        row_normalize,
    )

    unit = json.loads(Path("/root/autodl-fs/night7b_score_rnd_20260818/source/u000/worker_input.json").read_text())
    ids = Path(unit["observation_ids"]).read_text().splitlines()
    ordered_sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    views = load_npz_dict(unit["g04_views"])
    a = np.asarray(views["emb_latent_omics1"], dtype=np.float32)
    b = np.asarray(views["emb_latent_omics2"], dtype=np.float32)
    fused = np.asarray(views["SpaLORA_fused"], dtype=np.float32)
    part = np.load(unit["pseudo_partition"], allow_pickle=False)
    graph = sp.load_npz(unit["s04"]).tocsr()
    cardinality_ok = len(ids) == len(a) == len(b) == len(fused) == len(part) == graph.shape[0]
    if not cardinality_ok or ordered_sha != unit["ordered_observation_sha256"]:
        raise RuntimeError("real frozen-view spot order/cardinality mismatch")

    # Use a deterministic real slice for fast semantic probes.
    n = min(384, len(a))
    ix = np.arange(n)
    subg = graph[ix][:, ix].tocsr()
    isolated = np.flatnonzero(np.asarray(subg.sum(1)).ravel() == 0)
    if len(isolated):
        subg = subg + sp.csr_matrix((np.ones(len(isolated)), (isolated, isolated)), shape=subg.shape)
    qa = frozen_quality(a[:n], b[:n], fused[:n], part[:n], subg, mnn_k=10)
    qb = frozen_quality(b[:n], a[:n], fused[:n], part[:n], subg, mnn_k=10)
    same = frozen_quality(a[:n], a[:n].copy(), fused[:n], part[:n], subg, mnn_k=10)
    swap_global = np.allclose(qa.global_weights, qb.global_weights[::-1], atol=1e-6)
    swap_spot = np.allclose(qa.spot_weights, qb.spot_weights[:, ::-1], atol=1e-6)
    neutral = abs(float(same.global_weights[0]) - 0.5) <= 1e-6

    device = torch.device("cuda")
    torch.manual_seed(20260821)
    model = QCRDAdapter(a.shape[1]).to(device).eval()
    ta, tb, tf = (torch.tensor(x[:n], device=device) for x in (a, b, fused))
    context_np = row_normalize(np.asarray(subg @ fused[:n]))
    context = torch.tensor(context_np, device=device)
    with torch.no_grad():
        q04 = corrected_views(model, ta, tb, tf, context, qa, "Q04_SPOT_GATED_MASKED_RESIDUAL")[3].norm(dim=1).cpu().numpy()
        q05 = corrected_views(model, ta, tb, tf, context, qa, "Q05_BOUNDARY_GATED_RESIDUAL")[3].norm(dim=1).cpu().numpy()
        expected = corrected_views(model, ta, tb, tf, context, qa, "Q02_SPOT_QUALITY_BLEND")[2].cpu()
    high = qa.boundary_risk >= np.quantile(qa.boundary_risk, 0.75)
    boundary_ok = bool(np.all(q05 <= q04 + 1e-7) and q05[high].mean() < q04[high].mean())
    state_sha = canonical_state_sha256(model.state_dict())
    with tempfile.TemporaryDirectory() as td:
        checkpoint = Path(td) / "state.pt"
        torch.save(model.state_dict(), checkpoint)
        reloaded = QCRDAdapter(a.shape[1]).to(device).eval()
        reloaded.load_state_dict(torch.load(checkpoint, map_location=device))
        with torch.no_grad():
            observed = corrected_views(reloaded, ta, tb, tf, context, qa, "Q02_SPOT_QUALITY_BLEND")[2].cpu()
    roundtrip_max_abs = float(torch.max(torch.abs(expected - observed)))

    import inspect
    import SpaLORA.night10a_qcrd as module
    source = inspect.getsource(module)
    module_parity = QCRDAdapter(64).__class__ is QCRDAdapter(128).__class__
    no_dataset_route = "dataset" not in inspect.signature(QCRDAdapter).parameters and "dataset" not in inspect.signature(corrected_views).parameters
    no_dense_pairwise = "pairwise_distances(" not in source and "cdist(" not in source
    teacher_stop_gradient = "teacher.detach()" in inspect.getsource(corrected_views)
    quality_numpy_frozen = all(not isinstance(x, torch.Tensor) for x in (qa.global_weights, qa.spot_weights, qa.gate, qa.boundary_risk))
    finite = all(np.isfinite(x).all() for x in (qa.global_weights, qa.spot_weights, qa.gate, qa.boundary_risk, q04, q05))
    result = {
        "real_unit": unit["unit_id"],
        "ordered_observation_sha256": ordered_sha,
        "spot_order_exact": cardinality_ok and ordered_sha == unit["ordered_observation_sha256"],
        "quality_pre_optimizer_non_trainable": quality_numpy_frozen,
        "teacher_stop_gradient": teacher_stop_gradient,
        "modality_swap_global": bool(swap_global),
        "modality_swap_per_spot": bool(swap_spot),
        "neutral_equal_quality": bool(neutral),
        "q05_boundary_suppression": boundary_ok,
        "q07_sparse_reciprocal_edges": bool(len(qa.mnn_rows) <= 10 * n),
        "no_dense_pairwise_source": no_dense_pairwise,
        "same_module_class_across_families": module_parity,
        "dataset_name_routing_absent": no_dataset_route,
        "checkpoint_state_sha256": state_sha,
        "checkpoint_roundtrip_max_abs": roundtrip_max_abs,
        "checkpoint_roundtrip_pass": roundtrip_max_abs == 0.0,
        "all_outputs_finite": finite,
        "formal_training_units_started": 0,
        "formal_transforms_started": 0,
        "label_values_accessed": 0,
    }
    result["pass"] = all(v for k, v in result.items() if k in {
        "spot_order_exact", "quality_pre_optimizer_non_trainable", "teacher_stop_gradient",
        "modality_swap_global", "modality_swap_per_spot", "neutral_equal_quality",
        "q05_boundary_suppression", "q07_sparse_reciprocal_edges", "no_dense_pairwise_source",
        "same_module_class_across_families", "dataset_name_routing_absent",
        "checkpoint_roundtrip_pass", "all_outputs_finite",
    })
    return result


def historical_metric_parity() -> dict:
    """One authorized P0 evaluator read, isolated from training and selection."""
    sys.path.insert(0, str(REPO))
    from scripts.night10a.metric_expansion_reference import supervised_clustering_metrics

    unit = json.loads(Path("/root/autodl-fs/night7b_score_rnd_20260818/source/u000/worker_input.json").read_text())
    ids = np.asarray(Path(unit["observation_ids"]).read_text().splitlines(), dtype=str)
    pred = np.load(unit["pseudo_partition"], allow_pickle=False)
    gt_path = Path("/root/autodl-fs/Human lymph node/A1/A1_groundtruth.csv")
    gt = pd.read_csv(gt_path)
    id_col = "Barcode" if "Barcode" in gt.columns else gt.columns[0]
    label_col = "manual-anno" if "manual-anno" in gt.columns else gt.columns[-1]
    mapping = {}
    for key, val in zip(gt[id_col].astype(str), gt[label_col].astype(str)):
        canonical = key[3:] if key.startswith("s1-") else key
        mapping[canonical] = val
    canonical_ids = np.asarray([x[3:] if x.startswith("s1-") else x for x in ids], dtype=str)
    valid = np.asarray([x in mapping for x in canonical_ids])
    true = np.asarray([mapping[x] for x in canonical_ids[valid]], dtype=str)
    observed = pred[valid]
    reference = supervised_clustering_metrics(true, observed)
    project = {
        "ari": float(adjusted_rand_score(true, observed)),
        "nmi": float(normalized_mutual_info_score(true, observed)),
    }
    error = max(abs(reference[k] - project[k]) for k in project)
    return {
        "dataset": "a1_historical_c00_seed0",
        "ground_truth_file_sha256": sha(gt_path),
        "aligned_spots": int(valid.sum()),
        "reference_ari": reference["ari"],
        "reference_nmi": reference["nmi"],
        "project_ari": project["ari"],
        "project_nmi": project["nmi"],
        "max_abs_error": error,
        "tolerance": 1e-12,
        "pass": bool(error <= 1e-12 and valid.sum() > 0),
        "authorized_role": "isolated_p0_metric_evaluator",
        "label_file_reads_this_attempt": 1,
        "cumulative_p0_a1_label_file_reads": 4,
        "used_for_training_or_candidate_selection": False,
    }


def main() -> int:
    started = time.time()
    authority = {}
    for name, expected in EXPECTED.items():
        matches = list(PLANNING.rglob(name))
        if len(matches) != 1:
            raise RuntimeError(f"authority file resolution not unique: {name}: {matches}")
        actual = sha(matches[0])
        authority[name] = {"path": str(matches[0]), "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected}
    roots = {p: {"exists": Path(p).is_dir()} for p in ROOTS}
    resources = versions_and_resources()
    contract = {
        "protocol": "Night-10A/P0",
        "authority": authority,
        "historical_roots": roots,
        "resources": resources,
        "git": {
            "head": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "protection_tag_target": git("rev-list", "-n", "1", "baseline/pre-night10a-qcrd-score-rnd-20260821"),
            "status_porcelain": git("status", "--short"),
        },
        "constraints": {
            "misar_y_access": 0,
            "e18_5_access": 0,
            "new_external_data_access": 0,
            "new_third_party_benchmarks": 0,
            "formal_training_units_started": 0,
            "formal_transforms_started": 0,
        },
    }
    contract["pass"] = (
        all(x["match"] for x in authority.values())
        and all(x["exists"] for x in roots.values())
        and resources["cuda_available"]
        and resources["persistent_free_bytes"] >= 80 * (1024 ** 3)
        and contract["git"]["head"] == EXPECTED_PARENT
        and contract["git"]["protection_tag_target"] == EXPECTED_PARENT
    )
    atomic_json(P0 / "p0_authority_and_resource_contract.json", contract)
    if not contract["pass"]:
        raise RuntimeError("authority/resource contract failed")

    test_env = dict(os.environ)
    test_env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "scripts/night10a"), str(REPO), test_env.get("PYTHONPATH", "")]
    )
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/night10a/test_metric_expansion_reference.py", "tests/night10a/test_night10a_qcrd.py"],
        cwd=REPO, env=test_env, text=True, capture_output=True, timeout=300,
    )
    test_report = {"returncode": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr, "pass": tests.returncode == 0, "skips_allowed": 0}
    atomic_json(P0 / "p0_test_report.json", test_report)
    if tests.returncode:
        raise RuntimeError("P0 test suite failed")

    semantic = real_semantics()
    atomic_json(P0 / "p0_qcrd_real_semantic_probe.json", semantic)
    if not semantic["pass"]:
        raise RuntimeError("P0 real semantic probe failed")

    parity = historical_metric_parity()
    atomic_json(P0 / "p0_historical_metric_parity.json", parity)
    if not parity["pass"]:
        raise RuntimeError("P0 historical metric parity failed")

    final = {
        "status": "PASS" if contract["pass"] and test_report["pass"] and semantic["pass"] and parity["pass"] else "FAIL",
        "authority_and_resource_pass": contract["pass"],
        "reference_and_semantic_tests_pass": test_report["pass"],
        "real_semantic_probe_pass": semantic["pass"],
        "historical_metric_parity_pass": parity["pass"],
        "formal_training_units_started": 0,
        "formal_transforms_started": 0,
        "label_access": {"a1_authorized_isolated_p0_evaluator_reads": 4, "tonsil": 0, "d1": 0, "p22": 0, "misar_y": 0, "e18_5": 0},
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(P0 / "p0_final_gate.json", final)
    print(json.dumps(final, ensure_ascii=False, sort_keys=True))
    return 0 if final["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
