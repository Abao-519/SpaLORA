#!/usr/bin/env python3
"""Build the compact Night-19C scientific handoff after the frozen seed0 gate."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/root/SpaLORA-night16h")
WORK = Path("/root/night19c_working")
OUT = REPO / "outputs/night19c_handoff"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def main() -> None:
    if OUT.exists():
        assert OUT.resolve() == Path("/root/SpaLORA-night16h/outputs/night19c_handoff")
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    gate = json.loads((WORK / "summary/placenta_seed0_gate.json").read_text())
    if gate["seed0_transfer_gate_passed"] or gate["seeds_1_2_authorized"] or gate["stage_c_authorized"]:
        raise AssertionError("handoff builder cannot seal a failed gate as authorized")
    metrics = {row["run_id"]: row for row in read_csv(WORK / "placenta_seed0/metrics.csv")}
    producer = json.loads((WORK / "placenta_seed0/producer.producer.json").read_text())
    replay = json.loads((WORK / "placenta_seed0/fresh_replay.json").read_text())
    evaluator = json.loads((WORK / "placenta_seed0/metrics.evaluator.json").read_text())
    bank = json.loads((WORK / "bank/PLACENTA_UNBIASED_BANK_V1.manifest.json").read_text())
    if not replay["all_representation_exact"] or not replay["all_partition_exact"]:
        raise AssertionError("fresh replay is not exact")
    if producer["producer_label_reads"] != 0 or evaluator["evaluator_label_reads"] != 1:
        raise AssertionError("label-flow audit failed")

    copies = {
        "formula_freeze.json": REPO / "configs/night19c/formula_freeze.json",
        "placenta_relation_bank_authority.json": WORK / "bank/PLACENTA_UNBIASED_BANK_V1.manifest.json",
        "placenta_seed0_producer.json": WORK / "placenta_seed0/producer.producer.json",
        "placenta_seed0_fresh_replay.json": WORK / "placenta_seed0/fresh_replay.json",
        "placenta_seed0_metrics.csv": WORK / "placenta_seed0/metrics.csv",
        "placenta_seed0_evaluator.json": WORK / "placenta_seed0/metrics.evaluator.json",
        "placenta_seed0_gate.json": WORK / "summary/placenta_seed0_gate.json",
        "night17c_three_lane_authority_recovery.csv": WORK / "summary/night17c_three_lane_authority_recovery.csv",
        "night17c_authority_recovery_audit.json": WORK / "summary/night17c_authority_recovery_audit.json",
        "night17c_original_head_integration_matched_controls.csv": Path("/root/night17c_p0_working/official_compact/outputs/night17c_handoff/head_integration_matched_controls.csv"),
        "targeted_tests.log": WORK / "targeted_tests.log",
    }
    for name, source in copies.items():
        shutil.copy2(source, OUT / name)

    label_flow = {
        "schema": "night19c-label-flow-audit-v1",
        "producer_label_reads": 0,
        "bank_builder_label_reads": 0,
        "candidate_partitions_locked_and_hashed_before_evaluator": True,
        "producer_carrier_accessed_keys": producer["carrier_accessed_keys"],
        "producer_annotation_arrays_accessed": producer["carrier_annotation_arrays_accessed"],
        "evaluator_label_reads": 1,
        "reference_path": evaluator["reference_path"],
        "reference_sha256": evaluator["reference_sha256"],
        "n_evaluated": evaluator["n_evaluated"],
        "reference_k": evaluator["reference_k"],
        "public_labels_used_for": "POST_LOCK_EVALUATION_ONLY",
        "public_labels_used_for_training_or_checkpoint_selection": False,
    }
    (OUT / "label_flow_audit.json").write_text(json.dumps(label_flow, indent=2, sort_keys=True) + "\n")

    correction = [{
        "id": "E00",
        "stage": "PRE_PRODUCER_PREFLIGHT",
        "status": "CORRECTED_BEFORE_BANK_OR_TRAINING_OR_LABEL_ACCESS",
        "issue": "The first launcher omitted repository PYTHONPATH and failed at module import.",
        "evidence": "ModuleNotFoundError: No module named SpaLORA.night17b_sfrd",
        "scientific_artifacts_created": 0,
        "labels_read": 0,
        "formula_changed": False,
        "correction": "Set PYTHONPATH=/root/SpaLORA-night16h and reran the frozen path from bank construction.",
    }]
    (OUT / "implementation_and_correction_ledger.json").write_text(json.dumps(correction, indent=2) + "\n")

    source_paths = [
        "SpaLORA/night17c_zero_start.py", "SpaLORA/night19c_zero_start_transfer.py",
        "scripts/night19c/build_placenta_bank.py", "scripts/night19c/night19c_producer.py",
        "scripts/night19c/night19c_replay.py", "scripts/night19c/night19c_evaluator.py",
        "scripts/night19c/build_scientific_summary.py", "scripts/night19c/build_handoff.py",
        "tests/test_night17c_zero_start.py", "tests/test_night19c_zero_start_transfer.py",
    ]
    source_audit = {
        "schema": "night19c-source-authority-audit-v1",
        "source_sha256": {path: sha256(REPO / path) for path in source_paths},
        "night17c_core_matches_final_compact": sha256(REPO / "SpaLORA/night17c_zero_start.py") == "635eac0e264165fedc617640e67ae1b2f14b5def8b8fed61c8d2175b688a367c",
        "night17c_compact_25_of_25_reverified": True,
        "night17c_recovered_lane_seed_units": 9,
        "placenta_bank_candidate_count": bank["candidate_count"],
        "placenta_bank_authority_sha256": bank["bank_authority_sha256"],
        "z01_formula_modified": False,
    }
    (OUT / "source_and_authority_audit.json").write_text(json.dumps(source_audit, indent=2, sort_keys=True) + "\n")

    stat = os.statvfs("/")
    data_stat = os.statvfs("/autodl-fs/data")
    resource_audit = {
        "schema": "night19c-resource-audit-v1",
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "root_available_bytes": stat.f_bavail * stat.f_frsize,
        "root_available_inodes": stat.f_favail,
        "persistent_available_bytes": data_stat.f_bavail * data_stat.f_frsize,
        "persistent_available_inodes": data_stat.f_favail,
        "night19c_working_bytes": sum(path.stat().st_size for path in WORK.rglob("*") if path.is_file()),
        "new_downloads": 0,
        "new_environment": False,
        "dense_n_by_n": 0,
        "training_wall_seconds": producer["wall_seconds"],
        "peak_gpu_mib": producer["peak_gpu_mib"],
        "peak_rss_mib": producer["peak_rss_mib"],
    }
    (OUT / "resource_and_disk_audit.json").write_text(json.dumps(resource_audit, indent=2, sort_keys=True) + "\n")

    test_log = (WORK / "targeted_tests.log").read_text()
    if "12 passed" not in test_log:
        raise AssertionError("targeted test log does not report 12 passed")
    (OUT / "targeted_test_summary.json").write_text(json.dumps({
        "schema": "night19c-targeted-tests-v1", "status": "PASS", "passed": 12, "failed": 0,
        "command": "pytest -q tests/test_night17c_zero_start.py tests/test_night19c_zero_start_transfer.py",
        "log_sha256": sha256(WORK / "targeted_tests.log"),
    }, indent=2, sort_keys=True) + "\n")

    report = f"""# Night-19C 报告

## 我现在需要知道的三件事

1. **问题**：Night-17C 的零起步可训练表示核心能否从 P22/MISAR/人海马转移到独立 human placenta，而不是把后续 selector/direct-cut 的失败误算到它头上？
2. **实际动作所在层**：我们保持 `Z01_CONSERVATIVE` 网络、损失、40 steps 与 KMeans endpoint 完全不变；只把 Night-19B 在标签前锁定的 16 个非消融/非置换 placenta 分区按原 ID、等权构成 relation bank。
3. **论文含义**：旧三 lane 的 9 个 lane×seed artifact/checkpoint/replay 重新核验通过，Night-17C 旧的 `LOCAL_SIGNAL` 仍成立；但 placenta 的可训练 full 明显低于 deterministic relation smooth，因此本轮分类为 **SCIENTIFIC_NEGATIVE**，不授权多 seed 或结构 head 集成。

## Placenta seed0 绝对指标（同一 KMeans endpoint）

| arm | ARI | NMI | AMI | FMI | min cluster | changed vs zero |
|---|---:|---:|---:|---:|---:|---:|
| Frozen retained | {float(metrics['FROZEN_RETAINED']['ari']):.6f} | {float(metrics['FROZEN_RETAINED']['nmi']):.6f} | {float(metrics['FROZEN_RETAINED']['ami']):.6f} | {float(metrics['FROZEN_RETAINED']['fmi']):.6f} | {metrics['FROZEN_RETAINED']['min_cluster_size']} | {metrics['FROZEN_RETAINED']['changed_spots_vs_zero']} |
| Relation smooth | {float(metrics['RELATION_SMOOTH']['ari']):.6f} | {float(metrics['RELATION_SMOOTH']['nmi']):.6f} | {float(metrics['RELATION_SMOOTH']['ami']):.6f} | {float(metrics['RELATION_SMOOTH']['fmi']):.6f} | {metrics['RELATION_SMOOTH']['min_cluster_size']} | 0 |
| Zero residual | {float(metrics['ZERO_RESIDUAL']['ari']):.6f} | {float(metrics['ZERO_RESIDUAL']['nmi']):.6f} | {float(metrics['ZERO_RESIDUAL']['ami']):.6f} | {float(metrics['ZERO_RESIDUAL']['fmi']):.6f} | {metrics['ZERO_RESIDUAL']['min_cluster_size']} | 0 |
| Permuted relation | {float(metrics['PERMUTED_RELATION']['ari']):.6f} | {float(metrics['PERMUTED_RELATION']['nmi']):.6f} | {float(metrics['PERMUTED_RELATION']['ami']):.6f} | {float(metrics['PERMUTED_RELATION']['fmi']):.6f} | {metrics['PERMUTED_RELATION']['min_cluster_size']} | {metrics['PERMUTED_RELATION']['changed_spots_vs_zero']} |
| **Z01 full** | **{gate['full_ari']:.6f}** | **{gate['full_nmi']:.6f}** | {float(metrics['Z01_FULL']['ami']):.6f} | {float(metrics['Z01_FULL']['fmi']):.6f} | {gate['full_min_cluster_size']} | {metrics['Z01_FULL']['changed_spots_vs_zero']} |

Full 相对 coordinate-wise strongest matched control 的差值为 **{gate['delta_ari']:+.6f} ARI / {gate['delta_nmi']:+.6f} NMI**，远低于预注册的 `+0.005/+0.005` 门。没有 singleton/empty cluster，但结构健康不能补偿方法分数失败。

## 旧证据恢复与贡献边界

- Night-17C final compact 25/25 size+SHA 重新通过；当前 frozen core SHA 与 final compact 一致。
- P22、MISAR、人海马的 seeds 0–2 共 9 个 producer artifact、checkpoint、fresh replay 和 metrics authority 均验证通过，未重跑旧科学结果。
- 旧 Z01 common-KMeans mean：P22 `0.480888/0.609897`，MISAR `0.363704/0.541180`，human `0.199789/0.270691`。这是旧的局部表示信号，不是 Night-19C 新确认。
- Placenta relation smooth 相对 frozen retained 提高 **{float(metrics['RELATION_SMOOTH']['ari'])-float(metrics['FROZEN_RETAINED']['ari']):+.6f}/{float(metrics['RELATION_SMOOTH']['nmi'])-float(metrics['FROZEN_RETAINED']['nmi']):+.6f}**，但它是确定性 control/head signal，不是可训练核心贡献；且低于 Night-19B concatenated-feature control 的 `0.499999/0.631180`。
- Stage B seed0 失败后，严格没有运行 seeds 1–2，也没有进入 Stage C 或新增参数补救。

## 真实 P0 与技术边界

- Shape：N=1662，RNA=30，ATAC=30，retained=30，pair bank=11650，K=10。
- 训练确有非零 gradient/parameter change；zero-start 在 step0 对 smooth byte-exact，zero-gate self-return 由原 frozen core 保证。
- checkpoint strict load、fresh-process learned/permuted representation 与 partition 2/2 exact。
- Producer 只读取 numeric allow-list；标签在所有 partition 写出并哈希之后由独立 evaluator 读取。
- 首次启动因缺少 `PYTHONPATH` 在 import 前失败；未建 bank、未训练、未读标签，修正 launcher 后从头执行冻结路径。

## 导师汇报版

1. 我们先把 Night-17C 的旧证据重新核清，确认它的零起步训练信号没有被后来失败的 selector 或 direct-cut 自动推翻。
2. 新实验完全复用原 Z01，不改网络、损失、步数和阈值。
3. Placenta 的教师来自 Night-19B 标签前锁定的 16 个分区，保留原 ID 并严格等权，没有伪装成旧候选前缀。
4. 工程路径真实完成了训练、参数更新、checkpoint reload 和 fresh-process exact replay。
5. 但 Z01 full 只有 `{gate['full_ari']:.6f}/{gate['full_nmi']:.6f}`，明显低于 relation smooth 的 `{gate['coordinate_wise_strongest_control_ari']:.6f}/{gate['coordinate_wise_strongest_control_nmi']:.6f}`。
6. 因此 Night-17C 的局部信号没有迁移到 placenta，训练 residual 反而破坏了更好的 deterministic smooth carrier。
7. Relation smooth 本身相对 retained 有明显 control signal，但低于 Night-19B concat frontier，不能当作可训练方法成功。
8. 本轮按预注册门停止，不补 seed、不接结构 head，结论为 `SCIENTIFIC_NEGATIVE`。
"""
    (OUT / "night19c_report.md").write_text(report, encoding="utf-8")
    mentor = "\n".join(report.split("## 导师汇报版", 1)[1].strip().splitlines()) + "\n"
    (OUT / "mentor_oral_report.md").write_text("# Night-19C 导师口述稿\n\n" + mentor, encoding="utf-8")
    decision = {
        "schema": "night19c-decision-v1",
        "classification": "SCIENTIFIC_NEGATIVE",
        "status": "NIGHT19C_Z01_PLACENTA_TRANSFER_NO_INDEPENDENT_SIGNAL",
        "night17c_original_local_signal_preserved": True,
        "placenta_seed0_transfer_gate_passed": False,
        "seeds_1_2_authorized": False,
        "stage_c_authorized": False,
        "trainable_core_and_score_milestone": False,
        "deterministic_relation_smooth_control_signal": True,
        "shutdown_dispatched": False,
    }
    if decision["placenta_seed0_transfer_gate_passed"] != gate["seed0_transfer_gate_passed"]:
        raise AssertionError("decision differs from mechanical gate")
    (OUT / "night19c_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    (OUT / "reviewer_risk_register.md").write_text(
        "# Reviewer risk register\n\n"
        "- The placenta relation bank is transductive and derived from 16 locked Night-19B partitions; it is label-isolated but not an external pretrained teacher.\n"
        "- Equal-weight relation smoothing is a deterministic control, not a trainable-core contribution.\n"
        "- Common KMeans and Night-16H structural head are different endpoints; Stage C was not authorized.\n"
        "- The new physical study failed the preregistered transfer gate; old three-lane local evidence is not a confirmation on placenta.\n",
        encoding="utf-8",
    )

    files = []
    for path in sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "handoff_file_index.json"):
        files.append({"path": path.name, "size": path.stat().st_size, "sha256": sha256(path)})
    (OUT / "handoff_file_index.json").write_text(json.dumps({
        "schema": "night19c-handoff-index-v1", "file_count": len(files), "files": files,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"classification": decision["classification"], "handoff_files": len(files) + 1,
                      "stage_c_authorized": False}, indent=2))


if __name__ == "__main__":
    main()
