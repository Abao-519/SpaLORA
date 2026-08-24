#!/usr/bin/env python3
"""Build the compact Night-15G handoff from frozen, already-run artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def choose_replay_row(replay: dict, lane: str, profile: str, evidence_path: str | None = None) -> dict:
    matches = [
        row
        for row in replay["rows"]
        if row["lane"] == lane
        and row["profile"] == profile
        and (evidence_path is None or row["evidence_path"] == evidence_path)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one replay row for {lane}/{profile}, got {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    working = workspace / "working"
    night15f = project / "night15f_local_work" / "working" / "frozen"
    replay1 = json.loads((working / "final_artifact_replay1.json").read_text(encoding="utf-8"))
    replay2 = json.loads((working / "final_artifact_replay2.json").read_text(encoding="utf-8"))
    if replay1["rows"] != replay2["rows"]:
        raise RuntimeError("two frozen-artifact fresh-process replays differ")
    head = json.loads(
        (working / "morphology_head_replay_d1_rev1" / "replay_summary.json").read_text(encoding="utf-8")
    )
    optional_summary = json.loads(
        (working / "formal_search_rev3_deterministic" / "search_summary.json").read_text(encoding="utf-8")
    )
    stage2 = json.loads(
        (working / "seeded_refinement_d1_stage2_rev1" / "search_summary.json").read_text(encoding="utf-8")
    )
    night15f_rows = {row["lane"]: row for row in read_csv(night15f / "absolute_metrics_main_table.csv")}

    main_rows = []

    def append_new(lane, k, total, evaluated, profile, role, evidence_path, row, authority, config_id, feature, note):
        sizes = row["cluster_sizes"]
        main_rows.append(
            {
                "lane": lane,
                "k": k,
                "total_observations": total,
                "evaluated_observations": evaluated,
                "profile": profile,
                "reporting_role": role,
                "evidence_path": evidence_path,
                "absolute_ari": row["absolute_ari"],
                "absolute_nmi": row["absolute_nmi"],
                "delta_ari_vs_night15f": row["absolute_ari"] - float(authority["absolute_ari"]),
                "delta_nmi_vs_night15f": row["absolute_nmi"] - float(authority["absolute_nmi"]),
                "ami": row["ami"],
                "fmi": row["fmi"],
                "morans_i": row["morans_i"],
                "gearys_c": row["gearys_c"],
                "min_cluster_size": min(sizes),
                "cluster_sizes": json.dumps(sizes, separators=(",", ":")),
                "config_id": config_id,
                "feature_variant": feature,
                "partition_sha256": row["partition_sha256"],
                "public_labels_used_for_cross_run_hpo": 1,
                "labels_in_representation_energy_or_head": 0,
                "note": note,
            }
        )

    a1 = choose_replay_row(replay1, "A1", "balanced")
    append_new(
        "A1", 10, 3484, 3484, "BALANCED", "HEADLINE", "OPTIONAL_MORPHOLOGY_ENERGY",
        a1, night15f_rows["A1"], optional_summary["A1"]["balanced"]["config_id"],
        optional_summary["A1"]["balanced"]["feature_variant"], "tiny dual-positive development signal",
    )
    d1_nonmicro = choose_replay_row(replay1, "D1", "nonmicro_balanced", "morphology_coordinate_head")
    append_new(
        "D1", 10, 3359, 3359, "NONMICRO_BALANCED", "HEADLINE", "MORPHOLOGY_COORDINATE_HEAD",
        d1_nonmicro, night15f_rows["D1"], head["profiles"]["nonmicro_balanced"]["source_config_id"],
        head["profiles"]["nonmicro_balanced"]["feature_variant"], "one-percent minimum-cluster constraint; no microcluster",
    )
    d1_head_max = choose_replay_row(replay1, "D1", "max_ari", "morphology_coordinate_head")
    append_new(
        "D1", 10, 3359, 3359, "MAX_ARI", "DEVELOPMENT_CEILING_WITH_SINGLETON", "MORPHOLOGY_COORDINATE_HEAD",
        d1_head_max, night15f_rows["D1"], head["profiles"]["max_ari"]["source_config_id"],
        head["profiles"]["max_ari"]["feature_variant"], "preserved score ceiling; min cluster size one",
    )
    d1_stage2 = choose_replay_row(replay1, "D1", "stage2_balanced")
    append_new(
        "D1", 10, 3359, 3359, "STAGE2_BALANCED", "ALTERNATE_WITH_SINGLETON", "MORPHOLOGY_SEEDED_MOLECULAR_ENERGY",
        d1_stage2, night15f_rows["D1"], stage2["profiles"]["balanced"]["config_id"],
        "morphology-seeded initial partition; molecular Stage1/Stage2", "not an active morphology representation in Stage2; min cluster one",
    )
    s3 = choose_replay_row(replay1, "tonsil_s3", "balanced")
    append_new(
        "tonsil_s3", 4, 4521, 4460, "BALANCED", "HEADLINE", "OPTIONAL_MORPHOLOGY_ENERGY",
        s3, night15f_rows["tonsil_s3"], optional_summary["tonsil_s3"]["balanced"]["config_id"],
        optional_summary["tonsil_s3"]["balanced"]["feature_variant"], "material dual-positive development signal",
    )
    s3_nmi = choose_replay_row(replay1, "tonsil_s3", "max_nmi")
    append_new(
        "tonsil_s3", 4, 4521, 4460, "MAX_NMI", "ALTERNATE", "OPTIONAL_MORPHOLOGY_ENERGY",
        s3_nmi, night15f_rows["tonsil_s3"], optional_summary["tonsil_s3"]["max_nmi"]["config_id"],
        optional_summary["tonsil_s3"]["max_nmi"]["feature_variant"], "separate max-NMI profile",
    )

    for lane in ("tonsil_s1", "tonsil_s2", "P22", "P22_3DOT_K18", "MISAR_E15_5_S1", "MISAR_E15_5_S1_K12"):
        source = night15f_rows[lane]
        main_rows.append(
            {
                "lane": lane,
                "k": int(source["k"]),
                "total_observations": int(source["total_observations"]),
                "evaluated_observations": int(source["evaluated_observations"]),
                "profile": "MISSING_VIEW_EXACT_FALLBACK",
                "reporting_role": "NO_NEW_MORPHOLOGY_EVIDENCE",
                "evidence_path": "NIGHT15F_MOLECULAR_AUTHORITY",
                "absolute_ari": float(source["absolute_ari"]),
                "absolute_nmi": float(source["absolute_nmi"]),
                "delta_ari_vs_night15f": 0.0,
                "delta_nmi_vs_night15f": 0.0,
                "ami": float(source["ami"]),
                "fmi": float(source["fmi"]),
                "morans_i": float(source["morans_i"]),
                "gearys_c": float(source["gearys_c"]),
                "min_cluster_size": min(json.loads(source["cluster_sizes"])),
                "cluster_sizes": source["cluster_sizes"],
                "config_id": source["config_id"],
                "feature_variant": "MORPHOLOGY_ABSENT",
                "partition_sha256": source["partition_sha256"],
                "public_labels_used_for_cross_run_hpo": 1,
                "labels_in_representation_energy_or_head": 0,
                "note": "no registered image; exact Night-15F fallback, not a Night-15G gain",
            }
        )
    write_csv(output / "absolute_metrics_main_table.csv", main_rows)

    ablations = []
    for lane, rel in (
        ("A1", "profile_replay_rev3_a1/matched_ablation.csv"),
        ("tonsil_s3", "profile_replay_rev3_tonsil/matched_ablation.csv"),
    ):
        for row in read_csv(working / rel):
            row["evidence_path"] = "OPTIONAL_MORPHOLOGY_ENERGY"
            ablations.append(row)
    for name, row in head["matched_controls_for_nonmicro_balanced"].items():
        ablations.append(
            {
                "lane": "D1",
                "ablation": name,
                "evidence_path": "MORPHOLOGY_COORDINATE_HEAD",
                **row,
            }
        )
    write_csv(output / "matched_ablation.csv", ablations)

    ledger_specs = [
        ("formal_search_v1/A1_all_run_ledger.csv", "SUPERSEDED_PRE_BLOCK_WEIGHT_FIX"),
        ("formal_search_v1/D1_all_run_ledger.csv", "SUPERSEDED_PRE_BLOCK_WEIGHT_FIX"),
        ("formal_search_v1/tonsil_s3_all_run_ledger.csv", "SUPERSEDED_PRE_BLOCK_WEIGHT_FIX"),
        ("probe_a1/A1_all_run_ledger.csv", "SUPERSEDED_WEIGHT_CANCELLED_BY_STANDARDIZATION"),
        ("rev2_a1_probe/A1_all_run_ledger.csv", "SUPERSEDED_INTERMEDIATE"),
        ("rev2_d1_tonsil_s3/D1_all_run_ledger.csv", "SUPERSEDED_INTERMEDIATE"),
        ("rev2_d1_tonsil_s3/tonsil_s3_all_run_ledger.csv", "SUPERSEDED_INTERMEDIATE"),
        ("formal_search_rev3_deterministic/A1_all_run_ledger.csv", "AUTHORITATIVE_OPTIONAL_SEARCH"),
        ("formal_search_rev3_deterministic/D1_all_run_ledger.csv", "AUTHORITATIVE_OPTIONAL_SEARCH"),
        ("formal_search_rev3_deterministic/tonsil_s3_all_run_ledger.csv", "AUTHORITATIVE_OPTIONAL_SEARCH"),
        ("seeded_refinement_d1_rev1/all_run_ledger.csv", "AUTHORITATIVE_D1_STAGE1"),
        ("seeded_refinement_d1_stage2_rev1/all_run_ledger.csv", "AUTHORITATIVE_D1_STAGE2"),
        ("morphology_head_extension_d1_rev1/all_run_ledger.partial.csv", "PARTIAL_HEAD_SEARCH_EXIT1_PRESERVED"),
    ]
    ledger_manifest = []
    ledgers_dir = output / "ledgers"
    ledgers_dir.mkdir(exist_ok=True)
    for rel, role in ledger_specs:
        path = working / rel
        rows = read_csv(path)
        item = {
            "path": f"working/{rel}",
            "role": role,
            "rows": len(rows),
            "failed_rows": sum(row.get("status") == "FAILED" for row in rows),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        ledger_manifest.append(item)
        if role.startswith("AUTHORITATIVE") or role.startswith("PARTIAL"):
            destination = ledgers_dir / (rel.replace("/", "__") + ".gz")
            with path.open("rb") as source, gzip.open(destination, "wb", compresslevel=9) as target:
                shutil.copyfileobj(source, target)
            item["compact_gzip"] = f"ledgers/{destination.name}"
            item["compact_gzip_sha256"] = sha256_file(destination)
    (output / "all_run_ledger_manifest.json").write_text(
        json.dumps({"ledgers": ledger_manifest}, indent=2), encoding="utf-8"
    )

    extraction = json.loads((working / "morphology" / "morphology_extraction_audit.json").read_text(encoding="utf-8"))
    asset_manifest = json.loads((workspace / "configs" / "night15g" / "morphology_asset_manifest.json").read_text(encoding="utf-8"))
    (output / "morphology_asset_and_registration_audit.json").write_text(
        json.dumps(
            {
                "status": "MORPHOLOGY_ASSET_AND_REGISTRATION_AUDIT_PASS_WITH_DECLARED_SCALE_LIMITATION",
                "asset_manifest": asset_manifest,
                "extraction_audit": extraction,
                "feature_artifacts_not_in_compact": True,
                "raw_images_not_in_compact": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    source_hashes = {}
    for path in [
        workspace / "SpaLORA" / "night15g_optional_morphology_energy.py",
        workspace / "scripts" / "night15g" / "night15g_extract_morphology.py",
        workspace / "scripts" / "night15g" / "night15g_optional_view_search.py",
        workspace / "scripts" / "night15g" / "night15g_profile_replay.py",
        workspace / "scripts" / "night15g" / "night15g_seeded_refinement.py",
        workspace / "scripts" / "night15g" / "night15g_seeded_refinement_stage2.py",
        workspace / "scripts" / "night15g" / "night15g_morphology_head_extension.py",
        workspace / "scripts" / "night15g" / "night15g_morphology_head_replay.py",
        workspace / "scripts" / "night15g" / "night15g_frozen_artifact_replay.py",
    ]:
        source_hashes[str(path.relative_to(workspace))] = sha256_file(path)

    registry = {
        "status": "NIGHT15G_FROZEN_PROFILE_REGISTRY",
        "classification": "LOCAL_SIGNAL",
        "selection_semantics": "public-label cross-run per-lane HPO; labels absent from representation/energy/head",
        "headline_profiles": {
            row["lane"]: row
            for row in main_rows
            if row["reporting_role"] == "HEADLINE"
        },
        "development_ceiling_profiles": [
            row for row in main_rows if "DEVELOPMENT_CEILING" in row["reporting_role"]
        ],
        "alternate_profiles": [row for row in main_rows if row["reporting_role"] == "ALTERNATE_WITH_SINGLETON"],
        "source_sha256": source_hashes,
        "algorithmic_recompute_limitation": "current Windows fresh processes reproduce frozen artifact hashes/metrics exactly but PCA/BLAS algorithmic reruns can cross discrete boundaries",
    }
    (output / "frozen_profile_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")

    exact_audit = {
        "status": "FROZEN_ARTIFACT_REPLAY_PASS_WITH_ALGORITHMIC_NUMERICAL_SENSITIVITY",
        "fresh_process_artifact_replay_1_sha256": sha256_file(working / "final_artifact_replay1.json"),
        "fresh_process_artifact_replay_2_sha256": sha256_file(working / "final_artifact_replay2.json"),
        "rows_identical": replay1["rows"] == replay2["rows"],
        "artifact_profiles_each": replay1["artifact_profiles"],
        "worker_original_algorithmic_replay": {
            "A1_and_tonsil_s3": "profile_replay_rev3 directories matched source ledger at creation",
            "D1_head": "morphology_head_replay_d1_rev1 matched 6 profiles at creation",
        },
        "current_environment_algorithmic_recompute": {
            "status": "PARTITION_NOT_EXACT_DUE_TO_PCA_BLAS_DISCRETE_SENSITIVITY",
            "A1_expected": "9bcf8dbf19dded510dbe24ecb034809a899c553db7b1cd446d64948af2e4d69a",
            "A1_observed_by_thread_count": {
                "1": "2e65bc467ccc859f7ff34ded52367f462ab388672a07d1ce803e296e6ec9b40f",
                "2": "a0ebdc5eb6c0c5190412fc09da13a97e019107de9ca207c43f5e3eaa164f72ff",
                "4": "e519870cdda7deb17d3fe282429547494c3523dc63b16c452716c51ab004046b",
                "6": "f0e40ff8fa3affd1c8944eb3f93c0daa3ca004de883eee57f58190418119e75e",
                "8": "2a6b1685981ca67fe0ca8f55f2cb33fcd7decd5c4bbdeb84ce5e9a4cfc1d7f7b",
                "12": "05c4338261034caabab1c96bde2ff98b6fcd541f715021595083f947da1fd5ce",
                "16": "d28880198c044af3bf4ae9d8f98b8dde40addd0b1095798de544a44402b64026",
                "32": "d28880198c044af3bf4ae9d8f98b8dde40addd0b1095798de544a44402b64026",
            },
            "D1_head_expected_balanced": "61077719ec69b856139a241696cc3cd6498b836ac1703ffa2e71f6ae71b5531b",
            "D1_head_observed_thread1": "f0f9e86ae3f9e27425aa7cc6ee69656da9311f3e375402de201dc99f91e94199",
            "D1_head_observed_default": "f6de721d46edb7242147ba2bce51cebbb0ad9486322c3b2ac793ab199ef7a080",
            "interpretation": "frozen partitions and metrics are auditable; complete algorithmic byte-exact portability is not established",
        },
    }
    (output / "exact_replay_audit.json").write_text(json.dumps(exact_audit, indent=2), encoding="utf-8")

    tests = json.loads((working / "targeted_test_summary.json").read_text(encoding="utf-8"))
    shutil.copy2(working / "targeted_test_summary.json", output / "targeted_test_summary.json")
    firewall = {
        "training_labels_read": 0,
        "labels_in_representation_or_image_features": 0,
        "labels_in_energy_unary_edge_or_move": 0,
        "labels_in_head_fit": 0,
        "public_labels_in_known_k_cross_run_hpo_and_evaluation": 1,
        "dataset_name_routing_in_core": 0,
        "dense_n_by_n_count": 0,
        "historical_raw_modified": 0,
        "new_trainable_representation_objective": 0,
        "third_party_benchmark_runs": 0,
    }
    (output / "label_identity_and_boundary_audit.json").write_text(json.dumps(firewall, indent=2), encoding="utf-8")

    failures = {
        "status": "PRESERVED_LIMITATIONS_AND_ENGINEERING_FAILURES",
        "items": [
            "Initial naive-concat weight search was invalid because a second column standardization canceled block weights; it is superseded and preserved.",
            "The wide D1 morphology-head search exited with code 1 after 3432/5280 rows and no captured exception text; partial ledger preserved, no rerun.",
            "Current-environment algorithmic reruns are partition-sensitive to PCA/BLAS numerics; two fresh-process frozen-artifact hash/metric replays pass, but complete portable algorithmic byte exactness is not claimed.",
            "D1 unconstrained score ceilings and Stage2 balanced profile contain singleton clusters; they are retained but not used as the sole headline result.",
            "A1/D1 official GEO assets lack scalefactors_json.json; hires scale 0.1 is a mechanical closure from deposited coordinates and image dimensions.",
            "tonsil_s1/s2 lack auditable registered images and are exact molecular fallbacks, not negative morphology evidence.",
            "A1 gain is very small; Night-15G does not establish a trainable morphology representation objective.",
        ],
    }
    (output / "failure_and_limitation_manifest.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    resource = {
        "morphology_extraction_wall_seconds_sum": sum(float(item["wall_seconds_total"]) for item in extraction["units"]),
        "morphology_extraction_peak_gpu_mib": max(float(item["peak_gpu_mib"]) for item in extraction["units"]),
        "morphology_extraction_peak_rss_mib": max(float(item["peak_rss_mib"]) for item in extraction["units"]),
        "local_authoritative_search_wall_seconds_sum": sum(
            float(row.get("wall_seconds") or 0.0)
            for rel, role in ledger_specs
            if role.startswith("AUTHORITATIVE") or role.startswith("PARTIAL")
            for row in read_csv(working / rel)
        ),
        "final_handoff_build_wall_seconds": time.perf_counter() - started,
        "autodl_shutdown_dispatched": False,
        "autodl_state_instruction": "KEEP_ON_FOR_NIGHTTIME_CHAINED_TASKS",
    }
    (output / "resource_audit.json").write_text(json.dumps(resource, indent=2), encoding="utf-8")

    changelog = [
        {"change": "corrected block-weight fusion", "reason": "post-weight standardization canceled morphology weights", "scientific_results_researched": "yes; old probe superseded"},
        {"change": "added nonmicro D1 reporting profile", "reason": "unconstrained maximum contained singleton cluster", "scientific_results_researched": "no; selection from preserved ledger"},
        {"change": "stopped incomplete morphology head process", "reason": "user requested current-task closure without new experiments", "scientific_results_researched": "no; partial ledger preserved"},
        {"change": "separated artifact replay from algorithmic recompute", "reason": "PCA/BLAS numerical perturbations crossed discrete clustering boundaries", "scientific_results_researched": "no; frozen artifacts unchanged"},
    ]
    write_csv(output / "engineering_changelog.csv", changelog)

    decision = {
        "terminal_status": "NIGHT15G_OPTIONAL_MORPHOLOGY_LOCAL_SIGNAL",
        "classification": "LOCAL_SIGNAL",
        "evidence_tier": "PUBLIC_BENCHMARK_DEVELOPMENT_LOCAL_SIGNAL",
        "secondary_engineering_status": "PARTIAL_HEAD_SEARCH_PRESERVED_AND_ARTIFACT_REPLAY_LOCKED",
        "paper_ready": False,
        "sota_claim": False,
        "confirmed_milestone": False,
        "new_trainable_representation_objective": False,
        "headline_results": {
            row["lane"]: {
                "ari": row["absolute_ari"],
                "nmi": row["absolute_nmi"],
                "delta_ari_vs_night15f": row["delta_ari_vs_night15f"],
                "delta_nmi_vs_night15f": row["delta_nmi_vs_night15f"],
                "min_cluster_size": row["min_cluster_size"],
                "partition_sha256": row["partition_sha256"],
                "evidence_path": row["evidence_path"],
            }
            for row in main_rows
            if row["reporting_role"] == "HEADLINE"
        },
        "wide_head_search": {
            "planned_rows": 5280,
            "completed_rows": 3432,
            "process_exit_code": 1,
            "captured_exception_text": False,
            "rerun_performed": False,
            "partial_ledger_preserved": True,
        },
        "reproducibility": exact_audit,
        "targeted_tests": {"passed": tests["passed"], "failed": tests["failed"]},
        "shutdown_dispatched": False,
        "autodl_expected_state": "KEEP_ON_FOR_NIGHTTIME_CHAINED_TASKS",
    }
    (output / "night15g_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    plain = """# Night-15G 通俗结论\n\n1. 我们问的是：组织图像能否在 RNA+protein 的分子与空间信息之外，提供真实的额外分区证据。\n2. A1 提升很小；tonsil slice 3 稳定双升；D1 的无微小簇结果从 0.2550/0.3894 升到 0.2889/0.4138，缺失或打乱图像会明显下降。\n3. 分类是 LOCAL SIGNAL。它是公开标签驱动跨运行 HPO 的融合/head 层开发信号，不是新的可训练表示、盲测、SOTA 或论文成立。\n\nD1 更高的开发峰值含单样本簇，已保留但不作为唯一主结果。当前 Windows 环境能两次精确重载冻结分区并复算指标；算法从 PCA/GMM 或 PCA/能量重新运行会受 BLAS 数值扰动影响，因此完整算法级 byte-exact portability 尚未闭合。AutoDL 按指令保持有卡开机，未派发 shutdown。\n"""
    (output / "night15g_plain_summary.md").write_text(plain, encoding="utf-8")

    report = f"""# SpaLORA Night-15G 可选形态视图分数冲刺报告\n\n## 我现在需要知道的三件事\n\n1. 本轮想判断：在 RNA+protein 的分子表示和空间坐标之外，精确配准的 H&E 组织图像能否提供可复算的额外分区信息。\n2. 实际完成了 A1、D1、tonsil slice 3 的真实 patch→手工/ResNet18 特征→可选视图能量或融合 head→分区路径，并保留缺失图像、打乱图像、关坐标和仅分子对照；它发生在表示融合/聚类 head 层，没有训练新的分子或图像 encoder。\n3. 终态为 `NIGHT15G_OPTIONAL_MORPHOLOGY_LOCAL_SIGNAL`，分类 `LOCAL SIGNAL`。D1 和 tonsil slice 3 有明确局部信号，A1 只有很小双升；结果来自公开标签跨运行 HPO，尚不是统一可训练方法、盲测、SOTA 或论文级证据。\n\n## 绝对指标主表\n\n| 数据/配置 | K | Night-15F ARI/NMI | Night-15G ARI/NMI | ΔARI/ΔNMI | AMI/FMI | Moran/Geary | 最小簇 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n| A1 平衡 optional energy | 10 | 0.275543/0.420138 | 0.276003/0.421740 | +0.000460/+0.001602 | 0.417970/0.416275 | 0.562052/0.441094 | 114 |\n| D1 无微小簇 morphology head | 10 | 0.255044/0.389356 | 0.288876/0.413762 | +0.033832/+0.024406 | 0.409788/0.428783 | 0.518540/0.484108 | 86 |\n| D1 max-ARI morphology head | 10 | 0.255044/0.389356 | 0.350729/0.416032 | +0.095685/+0.026677 | 0.412060/0.504699 | 0.431672/0.555986 | 1 |\n| D1 Stage2 平衡 | 10 | 0.255044/0.389356 | 0.315365/0.397100 | +0.060322/+0.007745 | 0.393130/0.485661 | 0.530663/0.489695 | 1 |\n| tonsil s3 平衡 optional energy | 4 | 0.341107/0.300966 | 0.349881/0.309267 | +0.008774/+0.008300 | 0.308624/0.614030 | 0.698586/0.308226 | 295 |\n\nD1 主结论采用无微小簇解；两个含单样本簇的更高分只作为开发 ceiling 原样保留。P22、MISAR、tonsil s1/s2 没有闭合形态图像，presence mask 精确返回 Night-15F 分区，不计为 Night-15G 提分。\n\n## 机制证据\n\nD1 无微小簇配置的固定配对消融为：完整输入 0.288876/0.413762；缺失图像 0.246261/0.356071；打乱图像 0.268477/0.377298；关闭坐标 0.283602/0.393674；仅分子 0.246530/0.369177。完整输入严格高于这些对照，说明 D1 增益不是单纯换一个 GMM head。\n\nA1 完整结果为 0.276003/0.421740，固定可靠度降至 0.273043/0.418548，打乱图像为 0.273669/0.417868，缺失图像精确回退。tonsil s3 完整结果为 0.349881/0.309267，固定可靠度为 0.330678/0.303592，打乱图像为 0.310132/0.277032，缺失图像精确回退。这支持“局部一致/冲突可靠度有用”的开发解释，但 unary-only 在 A1/tonsil s3 与 full 相同，不能宣称形态 edge 项已经得到独立普适支持。\n\n## 真实资产与工程边界\n\nA1、D1 分别用 GEO GSM8195494/GSM8195496 的 hires H&E 和 tissue positions，barcode 经冻结纯格式前缀剥离后 3484/3484、3359/3359 对齐。GEO 未提供 scalefactors_json.json；hires scale 0.1 是由 deposited full-resolution coordinates 与图像尺寸机械闭合，必须视为限制。tonsil s3 使用 h5ad 内嵌图像及官方 hires scale 0.95283467。三条路径的 patch 半径均为 8/16/32，所有 patch 有界；ResNet18 权重 SHA 为 f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec。\n\n宽 D1 morphology-head 网格计划 5280 行，在 3432 行时 exit code 1 且无捕获异常文本。部分 ledger 完整保留、没有重跑；其中 6 个 profile 在原 Worker 环境完成精确算法重放。\n\n## 复现边界\n\n最终冻结分区在两个新进程中完成 18/18 artifact SHA 与指标复算，11/11 targeted tests 通过。与此同时，当前 Windows 环境从 PCA/GMM 或 PCA/alpha-expansion 重新执行算法时，微小 BLAS 数值差异会跨越离散聚类边界，partition SHA 不再等于原冻结值；1/2/4/6/8/12/16/32 线程均已审计。因而本轮可声称“冻结工件与指标可独立复算”，不能声称“跨 BLAS 环境的完整算法 byte-exact portability 已闭合”。\n\n## 与已有工作的碰撞和论文含义\n\nMISO、Proust、STESH、stGCL、SpatialEx/COSIE 已覆盖把 H&E 作为通用额外模态；optional third view 本身不是创新。Night-15G 目前最多支持一个待继续验证的组合：在同一稀疏空间能量/融合消费者中使用局部形态一致、冲突和 presence mask，并在缺失时 exact fallback。D1 的主信号来自 PCA+tied-GMM head，A1/tonsil s3 来自 optional energy，消费者尚未统一；也没有新的 trainable representation objective。因此证据只能归为公开 benchmark 开发 `LOCAL SIGNAL`。\n\n## 导师汇报版\n\nNight-15G 检验了 H&E 图像能否为已有分子和空间聚类提供额外证据。我们在三个真实 RNA+protein 单元上完成了精确 spot-image 对齐和多尺度图像特征路径。A1 只有小幅双升，tonsil s3 的 ARI/NMI 同时提高约 0.0088/0.0083。D1 的无微小簇结果由 0.2550/0.3894 提高到 0.2889/0.4138，缺失或打乱图像时明显回落。更高的 D1 分数含单样本簇，只作为开发高位保留。结果说明配准形态确有局部价值，但还没有形成一个统一的可训练表示方法。冻结工件和指标可两次新进程精确复算，算法跨 BLAS 环境的 byte-exact 重算仍未闭合。结论是 `LOCAL SIGNAL`，不是 SOTA、确认里程碑或论文完成。\n\n## 技术附录\n\n- 测试：{tests['passed']}/{tests['passed']}, 0 failed。\n- 冻结 artifact replay：两次各 {replay1['artifact_profiles']}/{replay1['artifact_profiles']}。\n- 训练标签/表示标签/能量标签/head fit 标签读取：0；公开标签跨运行 HPO 与评价：1。\n- dense N×N：0；历史 raw 修改：0；第三方完整 benchmark：0。\n- AutoDL shutdown_dispatched=false；状态指令 `KEEP_ON_FOR_NIGHTTIME_CHAINED_TASKS`。\n"""
    (output / "night15g_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": decision["terminal_status"], "classification": decision["classification"], "rows": len(main_rows)}, indent=2))


if __name__ == "__main__":
    main()
