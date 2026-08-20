#!/usr/bin/env python3
"""Night-8A evaluation-only recovery pre-label authority and view lock.

This program is deliberately label-blind.  It hashes existing evidence and
constructs a manifest of read-only evaluation references; it never imports or
opens a label snapshot.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get("NIGHT8A_RECOVERY_REPO", "/root/autodl-fs/SpaLORA-night8a-eval-recovery"))
ORIGINAL_REPO = Path("/root/autodl-fs/SpaLORA-night8a")
RAW = Path("/root/autodl-fs/night8a_raw_runs_20260820")
RECOVERY = Path("/root/autodl-fs/night8a_eval_recovery_20260820")
OUT = REPO / "outputs/night8a_eval_recovery"
ORIGINAL_OUT = ORIGINAL_REPO / "outputs/night8a_handoff"
LABEL_ROOT = Path("/root/autodl-fs/night7a_consensus_20260818/evaluation_label_snapshots")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class Evidence:
    def __init__(self) -> None:
        self.expected: dict[str, dict[str, Any]] = {}

    def add(self, path: str | Path, expected: str | None, category: str, source: str) -> None:
        p = str(Path(path))
        if not expected:
            expected = sha(Path(p)) if Path(p).is_file() else None
        row = self.expected.setdefault(p, {"path": p, "expected_sha256": expected, "categories": [], "sources": []})
        if row["expected_sha256"] and expected and row["expected_sha256"] != expected:
            raise RuntimeError(f"conflicting expected SHA for {p}")
        row["expected_sha256"] = row["expected_sha256"] or expected
        if category not in row["categories"]:
            row["categories"].append(category)
        if source not in row["sources"]:
            row["sources"].append(source)

    def hash_all(self) -> dict[str, Any]:
        rows = []
        for p in sorted(self.expected):
            item = dict(self.expected[p])
            path = Path(p)
            item["exists"] = path.is_file()
            item["size_bytes"] = path.stat().st_size if path.is_file() else None
            item["actual_sha256"] = sha(path) if path.is_file() else None
            item["match"] = bool(item["exists"] and item["expected_sha256"] == item["actual_sha256"])
            rows.append(item)
        failures = [x for x in rows if not x["match"]]
        return {
            "schema_version": "night8a-eval-recovery-original-artifacts-v1",
            "status": "PASS" if not failures else "RECOVERY_BLOCKED_ARTIFACT_MISMATCH",
            "file_count": len(rows),
            "verified_count": len(rows) - len(failures),
            "failure_count": len(failures),
            "failures": failures,
            "files": rows,
        }


def add_embedded(evidence: Evidence, value: Any, category: str, source: str) -> None:
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            evidence.add(value["path"], value["sha256"], category, source)
        for child in value.values():
            add_embedded(evidence, child, category, source)
    elif isinstance(value, list):
        for child in value:
            add_embedded(evidence, child, category, source)


def source_index() -> dict[str, dict[str, str]]:
    path = REPO / "outputs/night7b_handoff/source_unit_index.csv"
    return {x["unit_id"]: x for x in csv.DictReader(path.open(newline="", encoding="utf-8"))}


def night7a_c00() -> dict[tuple[str, int], dict[str, Any]]:
    source = read_json(REPO / "outputs/night7a_handoff/locked_consensus_transform_manifest.json")
    result = {}
    for row in source["transforms"]:
        if row.get("candidate_id") == "C00_G04_H05_CONFIRMED" and row["dataset"] in {"a1", "tonsil", "d1"}:
            result[(row["dataset"], int(row["seed"]))] = row
    return result


def night7a_c06() -> dict[tuple[str, int], dict[str, Any]]:
    source = read_json(REPO / "outputs/night7a_handoff/locked_consensus_transform_manifest.json")
    return {(x["dataset"], int(x["seed"])): x for x in source["transforms"]
            if x.get("candidate_id") == "C06_DUAL_ROW_STOCHASTIC_MEAN"}


def night7b_r02() -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for source_stage in ("R1", "R2"):
        locked = read_json(REPO / f"outputs/night7b_handoff/locked_{source_stage}_manifest.json")
        for row in locked["transforms"]:
            if (row.get("recipe_id") == "R02" and row.get("endpoint") == "E1_ADAPTER_C06_MEAN"
                    and row.get("head_id") == "H01" and row.get("dataset") == "p22"):
                row = dict(row); row["night7b_source_stage"] = source_stage
                result[("p22", int(row["seed"]))] = row
    return result


def build() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    r1 = read_json(ORIGINAL_OUT / "r1_stage_manifest.json")
    r2 = read_json(ORIGINAL_OUT / "r2_stage_manifest.json")
    global_manifest = read_json(ORIGINAL_OUT / "training_manifest.json")
    checkpoint = read_json(ORIGINAL_OUT / "checkpoint_and_roundtrip_audit.json")
    invalidation = read_json(ORIGINAL_OUT / "dev_window1_semantic_invalidation.json")
    prelabel = read_json(ORIGINAL_OUT / "label_firewall_prelabel_audit.json")
    if r1["status"] != "LOCKED_PRE_LABEL" or r2["status"] != "LOCKED_PRE_LABEL":
        raise RuntimeError("original stages were not locked pre-label")
    if global_manifest["actual_cuda_training_attempts"] != 116 or checkpoint["audited_checkpoints"] != 116:
        raise RuntimeError("116-training authority mismatch")

    gm = {(x["stage"], x["config_id"], x["unit_id"]): x for x in global_manifest["cells"]}
    ck = {(x["stage"], x["config_id"], x["unit_id"]): x for x in checkpoint["rows"]}
    evidence = Evidence()
    evidence.add(ORIGINAL_OUT / "r1_stage_manifest.json", "15000596a8db3d20d057fd4d3d5d081e275ce7d81d2300edd6683ee89ad17911", "authority", "taskbook")
    evidence.add(ORIGINAL_OUT / "r2_stage_manifest.json", "c7be512d335d6e6d45f1d4e566987a23df7ad6d96fc6ff9b696e3b75388537a2", "authority", "taskbook")
    evidence.add(ORIGINAL_OUT / "dev_window1_semantic_invalidation.json", "47bfafbefc0d37e36ad63fa264976fac724fe82a447cd568e03fde215f0aee8c", "authority", "taskbook")
    evidence.add(ORIGINAL_OUT / "failure_and_retry_audit.json", "f5f6547653db417cd627f2383a3cc94545b0c55579ce8ddfc02852ead3df84ac", "authority", "taskbook")
    evidence.add(ORIGINAL_OUT / "external_dataset_lock.json", "706395103ded06d0c4c4d0d79010b5685adab34e77a1fcc692a72cb6a658e386", "authority", "taskbook")

    dependency_rows = []
    candidate_checked = 0
    for stage_manifest in (r1, r2):
        for cell in stage_manifest["cells"]:
            key = (cell["stage"], cell["config_id"], cell["unit_id"])
            src = f"{cell['stage']}|{cell['config_id']}|{cell['unit_id']}"
            training = cell.get("training_manifest")
            if training:
                add_embedded(evidence, training, "training_declared", src)
                g = gm[key]
                evidence.add(Path(cell["root"]) / "training/training_manifest.json", g.get("training_manifest_sha256"), "training_manifest", src)
                c = ck[key]
                evidence.add(c["checkpoint_path"], c["checkpoint_file_sha256"], "checkpoint", src)
                evidence.add(Path(c["checkpoint_path"]).parent / "reload_audit.json", c["reload_audit_sha256"], "reload_audit", src)
            if cell["status"] == "CONTENT_ADDRESSED_NOOP_ALIAS_PRE_LABEL":
                evidence.add(cell["alias_target_transform_manifest"], cell["alias_target_transform_manifest_sha256"], "alias_target_manifest", src)
                evidence.add(cell["alias_target_training_manifest"], cell["alias_target_training_manifest_sha256"], "alias_target_manifest", src)
            elif cell.get("transform_status") == "SUCCESS_PRE_LABEL":
                tm = cell["transform_manifest"]
                tdir = Path(cell["transform_dir"])
                evidence.add(tdir / "affinity.npz", tm["affinity_file_sha256"], "transform", src)
                evidence.add(tdir / "clusters.csv", tm["cluster_file_sha256"], "transform", src)
                evidence.add(tdir / "metrics_placeholder.json", tm["metrics_placeholder_sha256"], "transform", src)
                evidence.add(tdir / "transform_manifest.json", gm[key]["transform_manifest_sha256"], "transform_manifest", src)
            elif cell.get("transform_status") == "SCIENTIFIC_NUMERICAL_FAILURE_NO_RETRY":
                evidence.add(Path(cell["root"]) / "transform.log", cell["transform_log_sha256"], "fixed_failure_log", src)

            is_baseline = cell["config_id"] in {"A00_FAMILY_REFERENCE", "B00_FAMILY_REFERENCE"}
            is_a05_protein_noop = (cell["config_id"] == "A05_RNA_ANCHOR" and cell["family"] == "RNA_PROTEIN")
            if not is_baseline and not is_a05_protein_noop:
                candidate_checked += 1
                refs = json.dumps(cell, sort_keys=True)
                b00_dependency = "/A00_FAMILY_REFERENCE/" in refs or "/B00_FAMILY_REFERENCE/" in refs
                valid_b05 = True
                if cell["config_id"] == "B05_SP_RR10_PROTO_RNAANCHOR" and cell["family"] == "RNA_PROTEIN":
                    valid_b05 = "/B04_SP_RR10_PROTO/" in refs and "/B00_FAMILY_REFERENCE/" not in refs
                uses_own = (cell["status"] == "CONTENT_ADDRESSED_NOOP_ALIAS_PRE_LABEL" or
                            f"/{cell['config_id']}/{cell['unit_id']}/" in str(cell["root"]))
                passed = not b00_dependency and valid_b05 and uses_own
                dependency_rows.append({"stage": cell["stage"], "config_id": cell["config_id"],
                                        "dataset": cell["dataset"], "seed": int(cell["seed"]),
                                        "unit_id": cell["unit_id"], "passed": passed,
                                        "b00_dependency": b00_dependency, "valid_b05_alias": valid_b05,
                                        "own_or_registered_alias_root": uses_own})

    # Locked label files are byte-hashed without opening their contents.
    for name, spec in prelabel["label_file_byte_hash_only"].items():
        evidence.add(LABEL_ROOT / name, spec["sha256"], "sealed_label_snapshot_byte_hash_only", "prelabel_audit")

    source = source_index()
    c00 = night7a_c00()
    c06 = night7a_c06()
    r02 = night7b_r02()
    invalid = {x["unit_id"]: x for x in invalidation["rows"] if x["family"] == "RNA_PROTEIN"}
    comparator = {}
    for dataset, units in {"a1": ["u000", "u001", "u002"], "tonsil": ["u005", "u006", "u007"],
                           "d1": ["u010", "u011", "u012"]}.items():
        for unit in units:
            s = source[unit]; seed = int(s["seed"]); old = invalid[unit]; locked = c00[(dataset, seed)]
            evidence.add(s["worker_input"], s["worker_input_sha256"], "night7b_locked_worker_input", unit)
            evidence.add(Path(s["worker_input"]).parent / "g04_views.npz", s["g04_views_sha256"], "night7b_locked_source_view", unit)
            evidence.add(old["locked_affinity_path"], old["locked_affinity_file_sha256"], "locked_C00_affinity", unit)
            evidence.add(old["locked_partition_path"], old["locked_partition_file_sha256"], "locked_C00_partition", unit)
            cluster_path = Path(s["worker_input"]).parent / "c00_clusters.csv"
            evidence.add(cluster_path, locked["artifacts"]["clusters.csv"]["sha256"], "locked_C00_cluster_table", unit)
            comparator[(dataset, seed)] = {
                "reference_id": "C00_G04_H05_CONFIRMED", "cluster_path": str(cluster_path),
                "cluster_sha256": locked["artifacts"]["clusters.csv"]["sha256"],
                "partition_path": old["locked_partition_path"], "partition_sha256": old["locked_partition_file_sha256"],
                "affinity_path": old["locked_affinity_path"], "affinity_sha256": old["locked_affinity_file_sha256"],
                "embedding_path": str(Path(s["worker_input"]).parent / "g04_views.npz"),
                "embedding_sha256": s["g04_views_sha256"], "source_unit": unit,
            }

    for unit in ("u020", "u021", "u022"):
        s = source[unit]; seed = int(s["seed"]); locked = r02[("p22", seed)]
        source_stage = locked["night7b_source_stage"]
        base = Path(f"/root/autodl-fs/night7b_score_rnd_20260818/adapter_stage/{source_stage}/formal/R02/{unit}/attempt_001")
        tdir = base / "transforms/E1_ADAPTER_C06_MEAN/H01"
        evidence.add(s["worker_input"], s["worker_input_sha256"], "night7b_locked_worker_input", unit)
        # ``embedding_sha256`` in the adapter transform manifest is the
        # canonical array hash, not the .npy byte hash.  The byte hash is
        # already declared by Night-8A's family_reference entry; preserve it.
        evidence.add(base / "worker/embedding.npy", None, "locked_R02_embedding", unit)
        evidence.add(tdir / "affinity.npz", locked["affinity_file_sha256"], "locked_R02_affinity", unit)
        evidence.add(tdir / "clusters.csv", locked["clusters_file_sha256"], "locked_R02_partition", unit)
        evidence.add(tdir / "fresh_transform_reload_audit.json", locked["fresh_transform_reload_audit_sha256"], "locked_R02_reload", unit)
        evidence.add(tdir / "transform_manifest.json", None, "locked_R02_manifest", unit)
        c06row = c06[("p22", seed)]["artifacts"]["consensus_affinity.npz"]
        c06path = Path(s["worker_input"]).parent / "c06_affinity.npz"
        evidence.add(c06path, c06row["sha256"], "locked_C06_affinity", unit)
        comparator[("p22", seed)] = {
            "reference_id": "R02_P22_FRONTIER_DEVELOPMENT_REFERENCE",
            "cluster_path": str(tdir / "clusters.csv"), "cluster_sha256": locked["clusters_file_sha256"],
            "partition_path": str(tdir / "clusters.csv"), "partition_sha256": locked["clusters_file_sha256"],
            "affinity_path": str(tdir / "affinity.npz"), "affinity_sha256": locked["affinity_file_sha256"],
            "embedding_path": str(base / "worker/embedding.npy"),
            "embedding_sha256": sha(base / "worker/embedding.npy"),
            "source_unit": unit,
        }

    if candidate_checked != 109 or any(not x["passed"] for x in dependency_rows):
        dependency_status = "RECOVERY_BLOCKED_CANDIDATE_CONTAMINATION"
    else:
        dependency_status = "PASS"
    dependency = {
        "schema_version": "night8a-eval-recovery-dependency-v1", "status": dependency_status,
        "candidate_cells_expected": 109, "candidate_cells_checked": candidate_checked,
        "passed": sum(x["passed"] for x in dependency_rows), "failed": sum(not x["passed"] for x in dependency_rows),
        "excluded_from_109": ["A00_FAMILY_REFERENCE", "B00_FAMILY_REFERENCE", "A05 RNA_PROTEIN designed no-op"],
        "rows": dependency_rows,
    }

    view_rows = []
    for stage_manifest in (r1, r2):
        for cell in stage_manifest["cells"]:
            key = (cell["dataset"], int(cell["seed"]))
            row = {"stage": cell["stage"], "config_id": cell["config_id"], "dataset": cell["dataset"],
                   "family": cell["family"], "seed": int(cell["seed"]), "unit_id": cell["unit_id"]}
            is_comparator = cell["config_id"] in {"A00_FAMILY_REFERENCE", "B00_FAMILY_REFERENCE"}
            is_a05_protein = cell["config_id"] == "A05_RNA_ANCHOR" and cell["family"] == "RNA_PROTEIN"
            if is_comparator or is_a05_protein:
                row.update(comparator[key]); row["status"] = "EVALUABLE_LOCKED_REFERENCE"
                row["role"] = "COMPARATOR" if is_comparator else "DESIGNED_NOOP_CANDIDATE"
                row["invalid_night8a_transform_excluded"] = True
            elif cell.get("transform_status") == "SCIENTIFIC_NUMERICAL_FAILURE_NO_RETRY":
                row.update({"status": "INELIGIBLE_UPSTREAM_FAILURE", "role": "NEW_CANDIDATE",
                            "failure_reason": "B03/P22/seed1 fixed cluster-K numerical failure"})
            else:
                tdir = Path(cell["transform_dir"])
                if cell["status"] == "CONTENT_ADDRESSED_NOOP_ALIAS_PRE_LABEL":
                    tm = read_json(Path(cell["alias_target_transform_manifest"]))
                    train_dir = Path(cell["alias_target_training_manifest"]).parent
                else:
                    tm = cell["transform_manifest"]
                    train_dir = Path(cell["root"]) / "training"
                row.update({"status": "EVALUABLE_LOCKED_CANDIDATE", "role": "NEW_CANDIDATE",
                            "cluster_path": str(tdir / "clusters.csv"), "cluster_sha256": tm["cluster_file_sha256"],
                            "partition_path": str(tdir / "clusters.csv"), "partition_sha256": tm["cluster_file_sha256"],
                            "affinity_path": str(tdir / "affinity.npz"), "affinity_sha256": tm["affinity_file_sha256"],
                            "embedding_path": str(train_dir / "embeddings.npz"),
                            "embedding_sha256": sha(train_dir / "embeddings.npz"),
                            "source_unit": cell["unit_id"]})
            view_rows.append(row)

    counts = {
        "r1_total": sum(x["stage"] == "R1" for x in view_rows),
        "r1_evaluable": sum(x["stage"] == "R1" and x["status"].startswith("EVALUABLE") for x in view_rows),
        "r2_total": sum(x["stage"] == "R2" for x in view_rows),
        "r2_evaluable": sum(x["stage"] == "R2" and x["status"].startswith("EVALUABLE") for x in view_rows),
        "r2_failures": sum(x["stage"] == "R2" and x["status"] == "INELIGIBLE_UPSTREAM_FAILURE" for x in view_rows),
        "r2_new_candidate_evaluable": sum(x["stage"] == "R2" and x["role"] == "NEW_CANDIDATE" and x["status"].startswith("EVALUABLE") for x in view_rows),
    }
    expected_counts = {"r1_total": 32, "r1_evaluable": 32, "r2_total": 96, "r2_evaluable": 95,
                       "r2_failures": 1, "r2_new_candidate_evaluable": 83}
    view = {"schema_version": "night8a-eval-recovery-view-v1",
            "status": "LOCKED_PRE_LABEL" if counts == expected_counts else "RECOVERY_BLOCKED_VIEW_COVERAGE",
            "training": 0, "transform": 0, "external_benchmark": 0, "label_access": False,
            "counts": counts, "expected_counts": expected_counts, "rows": view_rows}
    return evidence.hash_all(), dependency, view


def main() -> None:
    RECOVERY.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    before, dependency, view = build()
    atomic_json(RECOVERY / "original_artifact_manifest_before.json", before)
    atomic_json(OUT / "original_artifact_manifest_before.json", before)
    atomic_json(OUT / "candidate_dependency_audit.json", dependency)
    atomic_json(OUT / "evaluation_view_manifest.json", view)
    p0_status = "PASS"
    if before["status"] != "PASS": p0_status = before["status"]
    elif dependency["status"] != "PASS": p0_status = dependency["status"]
    elif view["status"] != "LOCKED_PRE_LABEL": p0_status = view["status"]
    p0 = {"schema_version": "night8a-eval-recovery-p0-v1", "status": p0_status,
          "base_commit": "d09aa00b5e25269e66712dd47d01064b7c9422cf",
          "base_tag": "night8a-final-20260820", "original_night8a_status": "IMPLEMENTATION_SEMANTICS_INVALID",
          "training": 0, "transform": 0, "external_benchmark": 0, "gpu_used": False,
          "artifact_files_verified": before["verified_count"], "artifact_failures": before["failure_count"],
          "candidate_dependency_passed": dependency["passed"], "candidate_dependency_expected": 109,
          "evaluation_view_counts": view["counts"], "label_values_deserialized": False}
    atomic_json(OUT / "p0_recovery_authority.json", p0)
    rule = {
        "schema_version": "night8a-eval-recovery-rule-lock-v1", "status": "LOCKED_PRE_LABEL",
        "source_registry_sha256": "cfe056489f0d42692cc78049cbc987ef47fe814d8d1594a6d958413ca8ccf9c1",
        "rules": {
            "metrics_spatial_weights_seeds_tiebreak": "unchanged from Night-8A registry",
            "B00_FAMILY_REFERENCE": "comparator only; excluded from shortlist, new-candidate Pareto and R3 finalist",
            "B03_RR10_PROTO": "incomplete 11/12; descriptive only; cannot advance",
            "threshold_or_configuration_changes": 0,
            "maximum_distinct_finalist_slots": 3,
            "slots": ["unified_balanced", "human_lymph_frontier", "P22_frontier"],
            "r3_training_in_this_task": 0,
        },
        "label_values_deserialized_before_lock": False,
    }
    atomic_json(OUT / "recovery_rule_lock.json", rule)
    print(json.dumps({"status": p0_status, "artifacts": before["verified_count"],
                      "candidate_dependencies": dependency["passed"], "view": view["counts"]}, sort_keys=True))
    if p0_status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
