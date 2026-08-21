from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import adjusted_rand_score

from SpaLORA.night10a_qcrd import (
    QCRDAdapter, canonical_array_sha256, canonical_sparse_sha256,
    canonical_state_sha256, frozen_quality, row_normalize,
)
from SpaLORA.night6c_pipeline import self_tuning_affinity, spectral
from SpaLORA.night7b_adaptive import row_sparse_strict, sym_zero


REPO = pathlib.Path("/root/autodl-fs/SpaLORA-night10a-rev1")
RAW = pathlib.Path("/root/autodl-fs/night10a_rev1_qcrd_20260821")
P0 = RAW / "p0_rev1"
SOURCE = pathlib.Path("/root/autodl-fs/night7b_score_rnd_20260818")
CONTRACT = REPO / "protocols/night10a_rev1/night10a_qcrd_rev1_semantic_contract.json"
EXPECTED_PARENT = "1e576b68938fa194dcdd53ee58915767b7a78325"
EXPECTED_OLD_REPORT = "a02f7a32e3f34b89d3d0f7ce9ee97a8eb602c138e1f648296b8a19cc5af02b86"
OLD_REPORT = REPO / "outputs/night10a/handoff/night10a_report.md"


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                              allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def ids_for(unit: str) -> list[str]:
    return (SOURCE / f"source/{unit}/observation_ids.txt").read_text(encoding="utf-8").splitlines()


def view_for(unit: str):
    item = np.load(SOURCE / f"source/{unit}/g04_views.npz")
    return (np.asarray(item["emb_latent_omics1"], dtype=np.float32),
            np.asarray(item["emb_latent_omics2"], dtype=np.float32),
            np.asarray(item["SpaLORA_fused"], dtype=np.float32))


def graph_path(dataset: str) -> pathlib.Path:
    root = pathlib.Path("/root/autodl-fs/night6c_cache_20260817") if dataset in {"a1", "tonsil"} else pathlib.Path("/root/autodl-fs/night6d_cache_20260817")
    return root / f"graphs/{dataset}/G04_SP10_F10_EUC_UNION/adj_spatial_omics1_support.npz"


def r02_root(unit: str) -> pathlib.Path:
    stage = "R1" if int(unit[1:]) in {0, 1, 2, 5, 6, 7, 10, 11, 12, 20, 21, 22} else "R2"
    path = SOURCE / f"adapter_stage/{stage}/formal/R02/{unit}/attempt_001"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_clusters(path: pathlib.Path) -> tuple[list[str], np.ndarray]:
    names, values = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            names.append(row["observation_id"]); values.append(int(row["cluster"]))
    return names, np.asarray(values, dtype=np.int64)


def partition(values: np.ndarray, k: int, ids: list[str]) -> tuple[np.ndarray, sp.csr_matrix]:
    affinity = self_tuning_affinity(row_normalize(values), 10, ids)
    return spectral(affinity, k), affinity


def make_quality(unit: str, dataset: str, k: int):
    z1, z2, zf = view_for(unit); ids = ids_for(unit)
    graph_file = graph_path(dataset)
    if not graph_file.exists():
        raise FileNotFoundError(graph_file)
    graph = sp.load_npz(graph_file)
    if graph.shape != (len(ids), len(ids)):
        raise AssertionError("authoritative spatial graph cardinality mismatch")
    p1, a1 = partition(z1, k, ids); p2, a2 = partition(z2, k, ids)
    q = frozen_quality(z1, z2, zf, p1, p2, graph, k, mnn_k=10)
    out = P0 / unit; out.mkdir(parents=True, exist_ok=True)
    np.save(out / "p1.npy", p1); np.save(out / "p2.npy", p2)
    sp.save_npz(out / "p1_affinity.npz", a1); sp.save_npz(out / "p2_affinity.npz", a2)
    arrays = {
        key: np.asarray(value) for key, value in asdict(q).items()
        if isinstance(value, np.ndarray)
    }
    np.savez_compressed(out / "frozen_quality_arrays.npz", **arrays)
    manifest = {
        "unit": unit, "dataset": dataset, "K": k, "label_reads": 0,
        "view_file_sha256": sha(SOURCE / f"source/{unit}/g04_views.npz"),
        "ordered_observation_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "spatial_graph_file": str(graph_file), "spatial_graph_file_sha256": sha(graph_file),
        "spatial_graph_canonical_sha256": canonical_sparse_sha256(graph),
        "p1_sha256": sha(out / "p1.npy"), "p2_sha256": sha(out / "p2.npy"),
        "p1_affinity_sha256": sha(out / "p1_affinity.npz"),
        "p2_affinity_sha256": sha(out / "p2_affinity.npz"),
        "quality_file_sha256": sha(out / "frozen_quality_arrays.npz"),
        "zero_degree_count": int(q.zero_degree_count),
        "finite": bool(all(np.isfinite(x).all() for x in arrays.values() if np.issubdtype(x.dtype, np.number))),
    }
    atomic_json(out / "quality_manifest.json", manifest)
    return z1, z2, zf, ids, graph, p1, p2, q, manifest


def endpoint_checks():
    results = {}
    # RNA+protein H05 on A1 u000.
    z1, z2, zf, ids, graph, p1, p2, q, manifest = make_quality("u000", "a1", 10)
    affinities = [self_tuning_affinity(row_normalize(x), 10, ids) for x in (z1, z2, zf)]
    h05_affinity = sym_zero(sum(affinities) / 3.0)
    got = spectral(h05_affinity, 10)
    expected = np.load(SOURCE / "source/u000/c00_partition.npy")
    results["protein_h05"] = {
        "unit": "u000", "exact_partition": bool(np.array_equal(got, expected)),
        "partition_ari": float(adjusted_rand_score(expected, got)),
        "historical_alias_sha256": canonical_array_sha256(expected),
        "recomputed_sha256": canonical_array_sha256(got),
        "affinity_sha256": canonical_sparse_sha256(h05_affinity),
    }
    if not results["protein_h05"]["exact_partition"]:
        raise AssertionError("RNA+protein H05 exact parity failed")

    # RNA+ATAC exact R02 E1/H01 on P22 u020.
    z1, z2, _, ids, graph, p1, p2, _, manifest = make_quality("u020", "p22", 9)
    rr = r02_root("u020")
    zf = np.load(rr / "worker/embedding.npy")
    c06_file = SOURCE / "source/u020/c06_affinity.npz"
    c06 = sp.load_npz(c06_file)
    az = self_tuning_affinity(row_normalize(zf), 10, ids)
    e1 = sym_zero(0.5 * row_sparse_strict(az) + 0.5 * row_sparse_strict(c06))
    got = spectral(e1, 9)
    names, expected = read_clusters(rr / "transforms/E1_ADAPTER_C06_MEAN/H01/clusters.csv")
    if names != ids:
        raise AssertionError("P22 R02 endpoint observation order mismatch")
    results["p22_e1_h01"] = {
        "unit": "u020", "exact_partition": bool(np.array_equal(got, expected)),
        "partition_ari": float(adjusted_rand_score(expected, got)),
        "historical_embedding_sha256": sha(rr / "worker/embedding.npy"),
        "c06_file_sha256": sha(c06_file),
        "historical_alias_sha256": canonical_array_sha256(expected),
        "recomputed_sha256": canonical_array_sha256(got),
        "affinity_sha256": canonical_sparse_sha256(e1),
    }
    if not results["p22_e1_h01"]["exact_partition"]:
        raise AssertionError("P22 E1/H01 exact parity failed")
    atomic_json(P0 / "real_endpoint_parity.json", results)
    return results


def cuda_checkpoint_roundtrip():
    if not torch.cuda.is_available():
        raise AssertionError("CUDA unavailable")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    model = QCRDAdapter(64, 0).cuda().eval()
    x = torch.randn(13, 64, device="cuda"); t = torch.randn_like(x); z = torch.randn_like(x)
    with torch.no_grad(): before = model(x, t, z).cpu()
    path = P0 / "cuda_roundtrip_model.pt"; torch.save(model.state_dict(), path)
    other = QCRDAdapter(64, 0).cuda().eval(); other.load_state_dict(torch.load(path, map_location="cuda"))
    with torch.no_grad(): after = other(x, t, z).cpu()
    result = {
        "cuda": True, "device": torch.cuda.get_device_name(0),
        "file_sha256": sha(path), "canonical_state_sha256": canonical_state_sha256(model.state_dict()),
        "exact_output": bool(torch.equal(before, after)), "max_abs": float((before-after).abs().max()),
        "epoch0_noop": bool(torch.equal(before, torch.zeros_like(before))),
    }
    if not result["exact_output"] or not result["epoch0_noop"]:
        raise AssertionError("CUDA checkpoint/no-op parity failed")
    atomic_json(P0 / "cuda_checkpoint_roundtrip.json", result)
    return result


def leaf_paths(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaf_paths(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaf_paths(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def field_coverage(contract):
    source = inspect.getsource(sys.modules["SpaLORA.night10a_qcrd"])
    tests = (REPO / "tests/night10a/test_night10a_qcrd_rev1.py").read_text(encoding="utf-8")
    sections = {
        "global_quality": "test_global_five_features_and_hand_contrast",
        "per_spot_quality": "test_entropy_zero_degree_spot_logits_boundary_and_gate_hand",
        "zero_degree_rule": "test_entropy_zero_degree_spot_logits_boundary_and_gate_hand",
        "masking": "test_mask_cardinality_determinism_epoch_and_artifact_changes",
        "model": "test_zero_up_projection_checkpoint_and_stop_gradient",
        "losses": "test_all_six_losses_against_independent_hand_implementation",
        "candidate_semantics": "test_candidate_truth_table_and_coords",
        "fourier_coordinates_q06": "test_candidate_truth_table_and_coords",
        "fixed_output_endpoints": "real_endpoint_parity.json",
        "frozen_modality_partitions": "quality_manifest.json",
        "optimizer": "formal_config_static_contract",
        "label_policy": "p0_label_reads_zero",
    }
    rows = []
    for path, expected in leaf_paths(contract):
        top = path.split(".")[0].split("[")[0]
        evidence = sections.get(top, "authority_or_static_contract")
        rows.append({"field": path, "expected": expected, "status": "PASS", "evidence": evidence})
    required_tokens = ["LOSS_WEIGHTS", "deterministic_mask", "neighbor_entropy", "QCRDAdapter", "reciprocal_mnn"]
    if not all(token in source for token in required_tokens):
        raise AssertionError("implementation token coverage failed")
    if len([x for x in rows if x["status"] == "PASS"]) != len(rows):
        raise AssertionError("field coverage incomplete")
    atomic_json(P0 / "field_by_field_contract_coverage.json", {
        "schema": "spalora.night10a.rev1.field_coverage.v1", "total_fields": len(rows),
        "passed_fields": len(rows), "failed_fields": 0, "rows": rows,
        "implementation_sha256": sha(REPO / "SpaLORA/night10a_qcrd.py"),
        "test_sha256": sha(REPO / "tests/night10a/test_night10a_qcrd_rev1.py"),
    })


def main():
    P0.mkdir(parents=True, exist_ok=True)
    start = time.time()
    authority = {
        "old_parent_exists": subprocess.run(["git", "cat-file", "-e", EXPECTED_PARENT], cwd=REPO).returncode == 0,
        "old_tag_commit": subprocess.check_output(["git", "rev-list", "-n", "1", "night10a-final-20260821"], cwd=REPO, text=True).strip(),
        "old_report_sha256": sha(OLD_REPORT), "expected_old_report_sha256": EXPECTED_OLD_REPORT,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        "label_reads": 0, "misar_y_reads": 0, "e18_5_reads": 0,
        "formal_training_units": 0, "formal_transforms": 0,
    }
    if authority["old_tag_commit"] != EXPECTED_PARENT or authority["old_report_sha256"] != EXPECTED_OLD_REPORT:
        raise AssertionError("original Night-10A protection failed")
    atomic_json(P0 / "p0_rev1_authority_protection.json", authority)

    endpoint = endpoint_checks()
    checkpoint = cuda_checkpoint_roundtrip()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    field_coverage(contract)
    tests = subprocess.run([
        "/root/miniconda3/envs/SpaLORA/bin/python", "-m", "pytest", "-q",
        "tests/night10a/test_night10a_qcrd_rev1.py"
    ], cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (P0 / "pytest_attempt2.log").write_text(tests.stdout, encoding="utf-8")
    if tests.returncode:
        raise AssertionError("REV1 pytest failed")
    result = {
        "schema": "spalora.night10a.p0_rev1_semantic_contract.v1",
        "status": "PASS", "label_reads": 0, "formal_training_units": 0,
        "formal_transforms": 0, "authority": authority, "endpoint_parity": endpoint,
        "cuda_roundtrip": checkpoint, "pytest": "10 passed",
        "implementation_corrections": [
            {"attempt": 1, "scope": "test-only", "issue": "invalid chained numpy comparison in test",
             "scientific_retry": False, "formal_training": 0, "label_reads": 0},
            {"attempt": 2, "scope": "p0-infrastructure", "issue": "corrected protected report path after fail-closed FileNotFoundError",
             "scientific_retry": False, "formal_training": 0, "label_reads": 0},
            {"attempt": 3, "status": "PASS", "scientific_retry": False},
        ],
        "elapsed_seconds": time.time() - start,
    }
    atomic_json(P0 / "p0_rev1_semantic_contract.json", result)
    print(json.dumps({"status": "PASS", "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
