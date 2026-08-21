from __future__ import annotations

import json
import os
import pathlib
import time

import numpy as np
import torch

from SpaLORA.night10a_qcrd import (
    QCRDAdapter, canonical_array_sha256, canonical_state_sha256,
    corrected_views, fourier_coordinates, frozen_reference_harmonizer,
    row_normalize,
)
from scripts.night10a.night10a_rev2_run import (
    CANDIDATES, DATA, PREFLIGHT, RAW, REPO, REV1_RAW, atomic_json,
    canonical_json_sha256, cell_dir, config_path, load_quality,
    prepare_one, reference, rev1_artifact_index, sha, unit,
    verify_rev1_artifact, views,
)

OUT = RAW / "reuse_audit"
EXPECTED_REV1_SOURCE_SHA = "69e420790003ec62c49a8807a67fa7c412e10b7975096fffb1c4a7777a4caad9"


def checkpoint_load(path: pathlib.Path, device: torch.device):
    try: return torch.load(path, map_location=device, weights_only=False)
    except TypeError: return torch.load(path, map_location=device)


def safe_symlink(source: pathlib.Path, target: pathlib.Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve(): return
        raise FileExistsError(f"refusing to replace existing REV2 artifact: {target}")
    os.symlink(source, target)


def old_cell(dataset: str, seed: int, candidate: str) -> pathlib.Path:
    return REV1_RAW / f"r1/{dataset}/seed_{seed}/{candidate}"


def replay_one(dataset: str, seed: int, candidate: str,
               artifact_index: dict[str, dict], device: torch.device) -> dict:
    started = time.time(); source = old_cell(dataset, seed, candidate)
    required = ["training_manifest.json", "reload_audit.json", "model_final.pt",
                "corrected_views.npz", "loss_curve.csv"]
    evidence = [verify_rev1_artifact(source / name, artifact_index) for name in required]
    old_manifest = json.loads((source / "training_manifest.json").read_text(encoding="utf-8"))
    old_reload = json.loads((source / "reload_audit.json").read_text(encoding="utf-8"))
    if old_manifest.get("status") != "CHECKPOINT_ROUNDTRIP_PASS" or old_reload.get("status") != "PASS":
        raise AssertionError("REV1 checkpoint was not round-trip locked")
    cfg = old_manifest["config"]
    if cfg["dataset"] != dataset or int(cfg["seed"]) != seed or cfg["candidate"] != candidate:
        raise AssertionError("REV1 checkpoint config identity mismatch")
    manifest = prepare_one(dataset, seed)
    z1, z2, _ = views(unit(dataset, seed)); zf_raw, _, _ = reference(unit(dataset, seed), DATA[dataset]["family"])
    ordered = manifest["ordered_observation_sha256"]
    harmonizer = frozen_reference_harmonizer(z1, z2, zf_raw, ordered, ordered, ordered)
    if harmonizer.mode != "identity" or harmonizer.zf_aligned is not zf_raw:
        raise AssertionError("reuse-eligible dataset did not take byte-exact identity path")
    if canonical_array_sha256(harmonizer.zf_aligned) != canonical_array_sha256(zf_raw):
        raise AssertionError("identity harmonizer canonical SHA mismatch")
    quality = load_quality(pathlib.Path(manifest["quality_file"]))
    first = torch.as_tensor(row_normalize(z1), dtype=torch.float32, device=device)
    second = torch.as_tensor(row_normalize(z2), dtype=torch.float32, device=device)
    aligned = torch.as_tensor(harmonizer.zf_aligned, dtype=torch.float32, device=device)
    coord = None
    if candidate == "Q06_COORDINATE_PRIOR_RESIDUAL":
        coord = torch.as_tensor(fourier_coordinates(np.load(manifest["coordinates_file"])),
                                dtype=torch.float32, device=device)
    checkpoint = checkpoint_load(source / "model_final.pt", device)
    model = QCRDAdapter(128, 0 if coord is None else 16, 64, 16, .1).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    state_sha = canonical_state_sha256(model.state_dict())
    if state_sha != old_manifest["canonical_state_sha256"]:
        raise AssertionError("REV1 checkpoint canonical state SHA mismatch")
    with torch.no_grad(): replay = corrected_views(model, first, second, aligned, quality, candidate, coord)
    expected = np.load(source / "corrected_views.npz")
    keys = ("z1c", "z2c", "zc", "correction"); checks = {}
    for key, observed in zip(keys, replay):
        got = observed.cpu().numpy(); wanted = expected[key]
        maximum = float(np.max(np.abs(got - wanted)))
        checks[key] = {"shape": list(got.shape), "finite": bool(np.isfinite(got).all()),
                       "max_abs": maximum, "within_1e_7": maximum <= 1e-7}
    if not all(value["finite"] and value["within_1e_7"] for value in checks.values()):
        raise AssertionError("REV1 corrected-view replay mismatch")
    stored_pf = np.load(manifest["pf_file"])
    if canonical_array_sha256(stored_pf) != manifest["reference_partition_canonical_sha256"]:
        raise AssertionError("REV1 reference partition mismatch")
    transform_evidence = []
    if dataset in {"a1", "tonsil"}:
        for name in ("transform_manifest.json", "clusters.csv", "affinity.npz"):
            transform_evidence.append(verify_rev1_artifact(source / name, artifact_index))
        transform = json.loads((source / "transform_manifest.json").read_text(encoding="utf-8"))
        if transform.get("status") != "PASS": raise AssertionError("REV1 transform is not locked PASS")
    stable = {
        "dataset": dataset, "seed": seed, "candidate": candidate,
        "checkpoint_state_sha256": state_sha,
        "identity_reference_sha256": canonical_array_sha256(harmonizer.zf_aligned),
        "input_manifest_sha256": sha(RAW / f"raw/inputs/{unit(dataset, seed)}/input_manifest.json"),
        "reference_partition_sha256": canonical_array_sha256(stored_pf),
        "corrected_view_checks": checks, "artifact_evidence": evidence,
        "transform_evidence": transform_evidence, "label_reads": 0,
    }
    return {**stable, "status": "PASS", "row_sha256": canonical_json_sha256(stable),
            "runtime_seconds": time.time() - started}


def certify_dataset(dataset: str, artifact_index: dict[str, dict],
                    device: torch.device) -> dict:
    rows = []; failures = []
    for seed in DATA[dataset]["r1"]:
        for candidate in CANDIDATES:
            try: rows.append(replay_one(dataset, int(seed), candidate, artifact_index, device))
            except Exception as exc:
                failures.append({"dataset": dataset, "seed": int(seed), "candidate": candidate,
                                 "error": repr(exc)})
    decision = "REUSE_ALL_21" if len(rows) == 21 and not failures else "RETRAIN_ALL_21"
    return {"dataset": dataset, "decision": decision, "expected_cells": 21,
            "passed_cells": len(rows), "failed_cells": len(failures),
            "rows": rows, "failures": failures, "label_reads": 0}


def materialize_reuse(group: dict) -> list[dict]:
    if group["decision"] != "REUSE_ALL_21": return []
    dataset = group["dataset"]; materialized = []
    by_key = {(row["seed"], row["candidate"]): row for row in group["rows"]}
    for seed in DATA[dataset]["r1"]:
        for candidate in CANDIDATES:
            cfg_file = config_path("r1", dataset, int(seed), candidate)
            if not cfg_file.is_file(): raise FileNotFoundError("REV2 formal config must exist before reuse materialization")
            cfg = json.loads(cfg_file.read_text(encoding="utf-8")); source = old_cell(dataset, int(seed), candidate)
            target = cell_dir(cfg); target.mkdir(parents=True, exist_ok=True)
            for name in ("model_final.pt", "corrected_views.npz", "loss_curve.csv", "reload_audit.json"):
                safe_symlink(source / name, target / name)
            old_manifest = json.loads((source / "training_manifest.json").read_text(encoding="utf-8"))
            row = by_key[(int(seed), candidate)]
            wrapper = {
                "schema": "spalora.night10a.rev2.reused_training.v1",
                "status": "REUSED_CHECKPOINT_ROUNDTRIP_PASS", "config": cfg,
                "config_file_sha256": sha(cfg_file), "reuse_row_sha256": row["row_sha256"],
                "source_rev1_cell": str(source), "source_training_manifest_sha256": sha(source / "training_manifest.json"),
                "model_file_sha256": sha(source / "model_final.pt"),
                "canonical_state_sha256": row["checkpoint_state_sha256"],
                "runtime_seconds": old_manifest["runtime_seconds"],
                "peak_gpu_bytes": old_manifest["peak_gpu_bytes"],
                "device": old_manifest["device"], "label_reads": 0,
                "scientific_retry": 0, "fallback": 0,
            }
            atomic_json(target / "training_manifest.json", wrapper)
            if dataset in {"a1", "tonsil"}:
                safe_symlink(source / "clusters.csv", target / "clusters.csv")
                safe_symlink(source / "affinity.npz", target / "affinity.npz")
                old_transform = json.loads((source / "transform_manifest.json").read_text(encoding="utf-8"))
                atomic_json(target / "transform_manifest.json", {
                    "schema": "spalora.night10a.rev2.reused_transform.v1",
                    "status": "REUSED_PASS", "endpoint": old_transform["endpoint"],
                    "source_rev1_cell": str(source),
                    "source_transform_manifest_sha256": sha(source / "transform_manifest.json"),
                    "clusters_sha256": sha(source / "clusters.csv"),
                    "affinity_sha256": sha(source / "affinity.npz"),
                    "partition_sha256": old_transform["partition_sha256"],
                    "runtime_seconds": old_transform["runtime_seconds"], "label_reads": 0,
                })
            materialized.append({"dataset": dataset, "seed": int(seed), "candidate": candidate,
                                 "target": str(target), "training_wrapper_sha256": sha(target / "training_manifest.json")})
    return materialized


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("passed_rows") != 210:
        raise AssertionError("reuse audit requires passing 210-row P0")
    compact_source = REV1_RAW / "official_compact/source/night10a_qcrd.py"
    if sha(compact_source) != EXPECTED_REV1_SOURCE_SHA:
        raise AssertionError("preserved REV1 source SHA drift")
    artifact_index = rev1_artifact_index(); device = torch.device("cuda")
    if not torch.cuda.is_available(): raise AssertionError("checkpoint replay audit requires CUDA")
    groups = [certify_dataset(dataset, artifact_index, device)
              for dataset in ("a1", "tonsil", "d1")]
    decision = {
        "schema": "spalora.night10a.rev2.outcome_blind_group_reuse.v1",
        "status": "PASS", "preflight_sha256": sha(PREFLIGHT),
        "rev1_source_sha256": sha(compact_source), "label_reads_before_decision": 0,
        "metric_tables_read": 0, "misar_y_reads": 0, "e18_5_reads": 0,
        "grouping_rule": "all 21 cells reuse or all 21 cells retrain",
        "groups": groups, "p22": {"decision": "RETRAIN_ALL_21_REV2_REQUIRED",
                                    "reason": "REV1 failed before first optimizer step"},
    }
    atomic_json(OUT / "reuse_certification.json", decision)
    materialized = []
    for group in groups: materialized.extend(materialize_reuse(group))
    atomic_json(OUT / "reuse_materialization.json", {
        "status": "PASS", "count": len(materialized), "rows": materialized,
        "label_reads": 0, "source_raw_modified": False})
    print(json.dumps({"status": "PASS", "decisions": {
        group["dataset"]: group["decision"] for group in groups},
        "materialized": len(materialized), "label_reads": 0}, indent=2))


if __name__ == "__main__":
    main()
