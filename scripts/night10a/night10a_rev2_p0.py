from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import pathlib
import subprocess
import tempfile
import time

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night10a_qcrd import (
    MASKED_CANDIDATES, QCRDAdapter, canonical_array_sha256,
    canonical_state_sha256, corrected_views, deterministic_mask,
    fourier_coordinates, qcrd_forward, row_normalize,
)
from scripts.night10a.night10a_rev2_run import (
    CANDIDATES, CONTRACT, DATA, PREFLIGHT, PYTHON, RAW, REPO,
    atomic_json, canonical_json_sha256, git_head, load_quality,
    load_zf_aligned, prepare_one, quality_dir, reference, sha, torch_loss, unit, views,
)

P0 = RAW / "p0_rev2"
LIMIT_SECONDS = 90 * 60
EXPECTED_CONTRACT_SHA = "991e27573c31d0ce072ef6782055bbc0f43fbea008ae3e14a0ce4503fc969a9b"
EXPECTED_PARENT = "c7cb3ecd1631db7a21606e62c69eaec05fe31ac1"
EXPECTED_BRANCH = "revision/q2-night10a-rev2-real-schema-harmonization-20260821"


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    columns = sorted({key for row in rows for key in row})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def run_tests() -> dict:
    command = [
        PYTHON, "-m", "pytest", "-q",
        "tests/night10a/test_night10a_qcrd_rev1.py",
        "tests/night10a/test_night10a_rev1_runner.py",
        "tests/night10a/test_night10a_qcrd_rev2.py",
        "tests/night10a/test_night10a_rev2_runner.py",
    ]
    result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (P0 / "static_and_synthetic_tests.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode: raise AssertionError("REV1+REV2 static/synthetic tests failed")
    return {"command": command, "returncode": result.returncode,
            "log_sha256": sha(P0 / "static_and_synthetic_tests.log")}


def authority() -> dict:
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    head = git_head(); status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    parent_exists = subprocess.run(["git", "cat-file", "-e", EXPECTED_PARENT], cwd=REPO).returncode == 0
    final_local = subprocess.run(["git", "rev-parse", "-q", "--verify",
                                  "refs/tags/night10a-rev2-final-20260821"], cwd=REPO).returncode == 0
    source = inspect.getsource(__import__("SpaLORA.night10a_qcrd", fromlist=["*"]))
    result = {
        "schema": "spalora.night10a.rev2.p0_authority.v1", "branch": branch,
        "head": head, "expected_parent_exists": parent_exists, "worktree_clean": status == "",
        "contract_sha256": sha(CONTRACT), "expected_contract_sha256": EXPECTED_CONTRACT_SHA,
        "final_tag_absent": not final_local, "label_reads": 0, "misar_y_reads": 0,
        "e18_5_reads": 0, "formal_training_units": 0, "formal_transforms": 0,
        "implementation_sha256": sha(REPO / "SpaLORA/night10a_qcrd.py"),
        "runner_sha256": sha(REPO / "scripts/night10a/night10a_rev2_run.py"),
        "p0_sha256": sha(REPO / "scripts/night10a/night10a_rev2_p0.py"),
        "dataset_routing_signature_absent": "dataset" not in inspect.signature(
            __import__("SpaLORA.night10a_qcrd", fromlist=["*"]).frozen_reference_harmonizer).parameters,
        "label_loader_absent_from_training_source": "load_labels" not in source,
    }
    if not (branch == EXPECTED_BRANCH and parent_exists and result["worktree_clean"] and
            result["contract_sha256"] == EXPECTED_CONTRACT_SHA and result["final_tag_absent"] and
            result["dataset_routing_signature_absent"] and result["label_loader_absent_from_training_source"]):
        raise AssertionError("P0-REV2 authority gate failed")
    atomic_json(P0 / "p0_rev2_authority.json", result); return result


def matrix_row(manifest: dict) -> dict:
    return {key: manifest[key] for key in (
        "dataset", "seed", "unit_id", "N", "d_z1", "d_z2", "d_zf_raw",
        "d_zf_aligned", "d_coords", "K", "z1_dtype", "z2_dtype",
        "zf_raw_dtype", "zf_aligned_dtype", "all_finite",
        "ordered_observation_sha256", "views_file_sha256", "z1_canonical_sha256",
        "z2_canonical_sha256", "zf_raw_file_sha256", "zf_raw_canonical_sha256",
        "zf_aligned_canonical_sha256", "spatial_graph_file_sha256",
        "spatial_graph_canonical_sha256", "coordinates_file_sha256",
        "quality_sha256", "p1_sha256", "p2_sha256", "pf_sha256",
        "harmonizer_mode", "harmonizer_sha256", "registered_input_sha256",
        "contract_sha256", "implementation_commit")}


def prepare_matrix(started: float) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    rows = []; manifests = {}; timings = []
    for dataset, spec in DATA.items():
        for seed in spec["r2"]:
            before = time.time(); manifest = prepare_one(dataset, int(seed)); elapsed = time.time() - before
            timings.append({"dataset": dataset, "seed": int(seed), "seconds": elapsed})
            rows.append(matrix_row(manifest)); manifests[(dataset, int(seed))] = manifest
            if time.time() - started > LIMIT_SECONDS:
                performance_stop(started, "SCHEMA_MATRIX_TIMEOUT", timings, [])
    if len(rows) != 30 or len({(row["dataset"], row["seed"]) for row in rows}) != 30:
        raise AssertionError("real runtime schema matrix is not 30 unique units")
    atomic_json(P0 / "real_runtime_schema_matrix.json", {
        "schema": "spalora.night10a.rev2.real_schema_matrix.v1", "count": len(rows),
        "label_reads": 0, "rows": rows})
    write_csv(P0 / "real_runtime_schema_matrix.csv", rows)
    atomic_json(P0 / "schema_preparation_timings.json", {"rows": timings})
    return rows, manifests


def performance_stop(started: float, reason: str, unit_timings: list[dict],
                     row_timings: list[dict]) -> None:
    value = {
        "schema": "spalora.night10a.rev2.p0_performance_stop.v1",
        "status": "P0_REV2_PERFORMANCE_TIMEOUT", "reason": reason,
        "limit_seconds": LIMIT_SECONDS, "elapsed_seconds": time.time() - started,
        "completed_runtime_rows": len(row_timings), "unit_timings": unit_timings,
        "row_timings": row_timings, "formal_training_started": False,
        "label_reads": 0,
    }
    atomic_json(P0 / "p0_rev2_performance_diagnostic.json", value)
    atomic_json(PREFLIGHT, {"status": "P0_REV2_PERFORMANCE_TIMEOUT", "rows": [],
                            "label_reads": 0, "formal_training_started": False})
    raise TimeoutError("P0-REV2 exceeded the registered 90-minute wall clock")


def checkpoint_load(path: pathlib.Path, device: torch.device):
    try: return torch.load(path, map_location=device, weights_only=False)
    except TypeError: return torch.load(path, map_location=device)


def one_runtime_row(dataset: str, seed: int, candidate: str, manifest: dict,
                    device: torch.device, code_commit: str) -> dict:
    started = time.time(); unit_id = unit(dataset, seed)
    z1, z2, _ = views(unit_id); zf_raw, _, _ = reference(unit_id, DATA[dataset]["family"])
    zf_aligned = load_zf_aligned(manifest, z1, z2, zf_raw)
    quality = load_quality(pathlib.Path(manifest["quality_file"]))
    partition = np.load(manifest["pf_file"]); graph = sp.load_npz(manifest["spatial_graph_file"])
    support = graph.maximum(graph.T); upper = sp.triu(support, k=1).tocoo()
    keep = partition[upper.row] != partition[upper.col]
    br = torch.as_tensor(upper.row[keep], dtype=torch.long, device=device)
    bc = torch.as_tensor(upper.col[keep], dtype=torch.long, device=device)
    first = torch.as_tensor(row_normalize(z1), dtype=torch.float32, device=device)
    second = torch.as_tensor(row_normalize(z2), dtype=torch.float32, device=device)
    aligned = torch.as_tensor(zf_aligned, dtype=torch.float32, device=device)
    coord = None
    if candidate == "Q06_COORDINATE_PRIOR_RESIDUAL":
        coordinate_array = fourier_coordinates(np.load(manifest["coordinates_file"]))
        if coordinate_array.shape != (manifest["N"], 16):
            raise AssertionError("real Q06 coordinate schema mismatch")
        coord = torch.as_tensor(coordinate_array, dtype=torch.float32, device=device)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    model = QCRDAdapter(first.shape[1], 0 if coord is None else coord.shape[1], 64, 16, .1).to(device)
    expected_width = 3 * manifest["d_zf_aligned"] + (16 if coord is not None else 0)
    if model.input.in_features != expected_width:
        raise AssertionError("real adapter input width mismatch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    input_manifest_sha = sha(quality_dir(unit_id) / "input_manifest.json")
    mask = None
    if candidate in MASKED_CANDIDATES:
        mask = torch.as_tensor(deterministic_mask(
            candidate, input_manifest_sha, seed, 0,
            manifest["N"], manifest["d_z1"]), device=device)
    model.train(); optimizer.zero_grad(set_to_none=True)
    forward = qcrd_forward(model, first, second, aligned, quality, candidate, coord, mask)
    losses = torch_loss(forward, first, second, aligned, quality, candidate, mask, br, bc)
    if set(losses) != {"align", "mask", "anchor", "correction", "boundary", "mnn", "total"}:
        raise AssertionError("registered loss component coverage mismatch")
    if not all(torch.isfinite(value) for value in losses.values()):
        raise FloatingPointError("non-finite real preflight loss")
    losses["total"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if not gradients or any(value is None or not torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError("missing or non-finite real preflight gradient")
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).detach().cpu())
    optimizer.step()  # The one and only temporary optimizer step for this row.
    state_sha = canonical_state_sha256(model.state_dict())
    temp_root = P0 / "temporary_checkpoints"; temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="row_", dir=temp_root) as directory:
        checkpoint = pathlib.Path(directory) / "checkpoint.pt"
        torch.save({"state_dict": model.state_dict(), "candidate": candidate,
                    "unit_id": unit_id, "optimizer_steps": 1}, checkpoint)
        checkpoint_sha = sha(checkpoint); loaded = checkpoint_load(checkpoint, device)
        replay = QCRDAdapter(first.shape[1], 0 if coord is None else coord.shape[1], 64, 16, .1).to(device)
        replay.load_state_dict(loaded["state_dict"])
        if loaded["optimizer_steps"] != 1 or canonical_state_sha256(replay.state_dict()) != state_sha:
            raise AssertionError("real preflight checkpoint state round-trip mismatch")
        replay.eval()
        with torch.no_grad(): outputs = corrected_views(replay, first, second, aligned, quality, candidate, coord)
    if any(value.shape != first.shape or not torch.isfinite(value).all() for value in outputs):
        raise FloatingPointError("real preflight eval output shape/finite mismatch")
    torch.cuda.synchronize()
    stable = {
        "dataset": dataset, "seed": seed, "unit_id": unit_id, "candidate": candidate,
        "N": manifest["N"], "d_z1": manifest["d_z1"], "d_z2": manifest["d_z2"],
        "d_zf_raw": manifest["d_zf_raw"], "d_zf_aligned": manifest["d_zf_aligned"],
        "d_coords": 16 if coord is not None else 0, "K": manifest["K"],
        "input_manifest_sha256": input_manifest_sha,
        "registered_input_sha256": manifest["registered_input_sha256"],
        "ordered_observation_sha256": manifest["ordered_observation_sha256"],
        "harmonizer_mode": manifest["harmonizer_mode"],
        "harmonizer_sha256": manifest["harmonizer_sha256"],
        "contract_sha256": manifest["contract_sha256"], "code_commit": code_commit,
        "implementation_sha256": sha(REPO / "SpaLORA/night10a_qcrd.py"),
        "runner_sha256": sha(REPO / "scripts/night10a/night10a_rev2_run.py"),
        "forward_complete": True, "loss_components": sorted(losses),
        "backward_complete": True, "gradient_finite": True,
        "optimizer_steps": 1, "checkpoint_roundtrip": True,
        "checkpoint_file_sha256": checkpoint_sha, "checkpoint_state_sha256": state_sha,
        "eval_shapes_finite": True, "label_reads": 0,
    }
    row = {**stable, "status": "PASS", "gradient_norm_before_clip": gradient_norm,
           "runtime_seconds": time.time() - started}
    row["preflight_row_sha256"] = canonical_json_sha256(stable)
    return row


def main() -> None:
    P0.mkdir(parents=True, exist_ok=True); started = time.time()
    auth = authority(); tests = run_tests(); matrix, manifests = prepare_matrix(started)
    code_commit = auth["head"]; rows = []; row_timings = []
    device = torch.device("cuda")
    if not torch.cuda.is_available(): raise AssertionError("P0 real runtime requires CUDA")
    for dataset, spec in DATA.items():
        for seed in spec["r2"]:
            for candidate in CANDIDATES:
                row = one_runtime_row(dataset, int(seed), candidate,
                                      manifests[(dataset, int(seed))], device, code_commit)
                rows.append(row); row_timings.append({
                    "dataset": dataset, "seed": int(seed), "candidate": candidate,
                    "seconds": row["runtime_seconds"]})
                atomic_json(PREFLIGHT.with_name("real_runtime_preflight_partial.json"), {
                    "status": "IN_PROGRESS", "expected": 210, "completed": len(rows),
                    "label_reads": 0, "rows": rows})
                if time.time() - started > LIMIT_SECONDS:
                    unit_timings = json.loads((P0 / "schema_preparation_timings.json").read_text())["rows"]
                    performance_stop(started, "RUNTIME_MATRIX_TIMEOUT", unit_timings, row_timings)
    keys = {(row["dataset"], row["seed"], row["candidate"]) for row in rows}
    if len(rows) != 210 or len(keys) != 210 or not all(row["status"] == "PASS" for row in rows):
        raise AssertionError("P0-REV2 real runtime gate is not 210/210")
    result = {
        "schema": "spalora.night10a.rev2.real_runtime_preflight.v1", "status": "PASS",
        "expected_rows": 210, "passed_rows": 210, "failed_rows": 0,
        "unique_row_sha256": len({row["preflight_row_sha256"] for row in rows}),
        "real_inputs_only": True, "synthetic_substitution_rows": 0,
        "p22_mixed_dimension_rows": sum(row["dataset"] == "p22" for row in rows),
        "p22_q06_rows": sum(row["dataset"] == "p22" and row["candidate"] ==
                             "Q06_COORDINATE_PRIOR_RESIDUAL" for row in rows),
        "device": torch.cuda.get_device_name(0), "code_commit": code_commit,
        "contract_sha256": sha(CONTRACT), "elapsed_seconds": time.time() - started,
        "label_reads": 0, "formal_training_started": False,
        "static_tests": tests, "schema_matrix_count": len(matrix), "rows": rows,
    }
    if result["unique_row_sha256"] != 210 or result["p22_mixed_dimension_rows"] != 70 or result["p22_q06_rows"] != 10:
        raise AssertionError("P0-REV2 coverage summary mismatch")
    atomic_json(PREFLIGHT, result); write_csv(P0 / "real_runtime_preflight.csv", rows)
    atomic_json(P0 / "p0_rev2_summary.json", {key: value for key, value in result.items() if key != "rows"})
    print(json.dumps({"status": "PASS", "rows": 210,
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
