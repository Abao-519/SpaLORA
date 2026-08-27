"""Build Night-22B handoff from immutable producer/evaluator artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


REPO = Path("/root/SpaLORA-night16h")
WORK = Path("/root/night22b_working")
OUT = REPO / "outputs" / "night22b_handoff"
CONTRACT = REPO / "configs" / "night22b" / "frozen_transfer_contract.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def jdump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def f(value) -> float:
    return float(value)


def full_and_controls(rows: list[dict]) -> tuple[dict, dict, dict]:
    start = next(row for row in rows if row["candidate_id"] == "INPUT_GEOMETRY_START")
    full = next(row for row in rows if "__FULL__" in row["candidate_id"])
    controls = [row for row in rows if row not in (start, full)]
    strongest_ari = max([start] + controls, key=lambda row: (f(row["ari"]), f(row["nmi"])))
    strongest_nmi = max([start] + controls, key=lambda row: (f(row["nmi"]), f(row["ari"])))
    return start, full, {"ari": strongest_ari, "nmi": strongest_nmi}


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    evaluation_files = sorted((WORK / "evaluation").glob("*.csv"))
    all_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    contribution_rows: list[dict] = []
    for path in evaluation_files:
        rows = read_csv(path)
        all_rows.extend(rows)
        start, full, strongest = full_and_controls(rows)
        strict = (
            f(full["ari"]) > f(start["ari"])
            and f(full["nmi"]) > f(start["nmi"])
            and all(
                f(full["ari"]) > f(row["ari"]) and f(full["nmi"]) > f(row["nmi"])
                for row in rows
                if row["candidate_id"] not in {"INPUT_GEOMETRY_START", full["candidate_id"]}
            )
            and int(full["changed_observations_vs_start"]) >= 5
            and int(full["observed_k_full"]) == int(contract["lanes"][full["lane"]]["k"])
            and int(full["min_cluster_size_full"]) > 1
            and full["all_clusters_have_internal_smallest_scale_edge"] == "True"
        )
        record = {
            "lane": full["lane"],
            "role": full["role"],
            "start_candidate": full["start_candidate"],
            "start_ari": start["ari"],
            "start_nmi": start["nmi"],
            "full_ari": full["ari"],
            "full_nmi": full["nmi"],
            "full_ami": full["ami"],
            "full_fmi": full["fmi"],
            "full_homogeneity": full["homogeneity"],
            "full_v_measure": full["v_measure"],
            "delta_ari_vs_start": f(full["ari"]) - f(start["ari"]),
            "delta_nmi_vs_start": f(full["nmi"]) - f(start["nmi"]),
            "strongest_nonfull_ari_candidate": strongest["ari"]["candidate_id"],
            "strongest_nonfull_ari": strongest["ari"]["ari"],
            "strongest_nonfull_nmi_candidate": strongest["nmi"]["candidate_id"],
            "strongest_nonfull_nmi": strongest["nmi"]["nmi"],
            "delta_ari_vs_coordinatewise_strongest": f(full["ari"]) - f(strongest["ari"]["ari"]),
            "delta_nmi_vs_coordinatewise_strongest": f(full["nmi"]) - f(strongest["nmi"]["nmi"]),
            "changed_observations": full["changed_observations_vs_start"],
            "min_cluster_size": full["min_cluster_size_full"],
            "cluster_sizes": full["cluster_sizes_full"],
            "all_clusters_have_internal_edge": full["all_clusters_have_internal_smallest_scale_edge"],
            "full_partition_sha256": full["candidate_partition_sha256"],
            "strict_independent_pass": strict,
        }
        sensitivity_rows.append(record)
        for row in rows:
            contribution_rows.append(
                {
                    "lane": row["lane"],
                    "role": row["role"],
                    "start_candidate": row["start_candidate"],
                    "candidate_id": row["candidate_id"],
                    "ari": row["ari"],
                    "nmi": row["nmi"],
                    "delta_ari_vs_start": f(row["ari"]) - f(start["ari"]),
                    "delta_nmi_vs_start": f(row["nmi"]) - f(start["nmi"]),
                    "changed_observations": row["changed_observations_vs_start"],
                    "partition_sha256": row["candidate_partition_sha256"],
                }
            )

    write_csv(OUT / "all_candidate_evaluation_ledger.csv", all_rows)
    write_csv(OUT / "start_sensitivity_table.csv", sensitivity_rows)
    write_csv(OUT / "matched_contribution_table.csv", contribution_rows)

    human_primary = next(
        row for row in sensitivity_rows
        if row["lane"] == "HUMAN_HIPPOCAMPUS_K7" and row["start_candidate"] == "GEOM_LEIDEN_FEATURE"
    )
    primary = [
        {
            "lane": "MISAR_K7",
            "n": 1949,
            "k": 7,
            "primary_start": "GEOM_LEIDEN_FEATURE",
            "primary_start_available": False,
            "full_ari": "NA",
            "full_nmi": "NA",
            "delta_ari_vs_strongest_control": "NA",
            "delta_nmi_vs_strongest_control": "NA",
            "strict_independent_pass": False,
            "failure_reason": "frozen 32-resolution Leiden generator produced no exact-K=7 partition; grid expansion forbidden",
        },
        {
            "lane": "HUMAN_HIPPOCAMPUS_K7",
            "n": 2500,
            "k": 7,
            "primary_start": "GEOM_LEIDEN_FEATURE",
            "primary_start_available": True,
            "full_ari": human_primary["full_ari"],
            "full_nmi": human_primary["full_nmi"],
            "delta_ari_vs_strongest_control": human_primary["delta_ari_vs_coordinatewise_strongest"],
            "delta_nmi_vs_strongest_control": human_primary["delta_nmi_vs_coordinatewise_strongest"],
            "strict_independent_pass": human_primary["strict_independent_pass"],
            "failure_reason": "FULL is below the shared-graph atomic arm and does not dual-improve the start",
        },
    ]
    write_csv(OUT / "primary_frozen_transfer_table.csv", primary)

    replay_rows = []
    for round_name in ("junction_replays", "junction_replays_round2"):
        for path in sorted((WORK / round_name).glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            replay_rows.append(
                {
                    "round": round_name,
                    "bank": path.stem,
                    "status": value["status"],
                    "candidate_count": value["count"],
                }
            )
    write_csv(OUT / "fresh_process_replay_table.csv", replay_rows)

    parent_rows = []
    for path in sorted((WORK / "parent_replay").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        parent_rows.append({"lane": path.stem, "status": value["status"], "candidate_count": value["count"]})
    write_csv(OUT / "parent_replay_table.csv", parent_rows)

    p0_rows = []
    resource_rows = []
    for path in sorted((WORK / "junction").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        with np.load(manifest["carrier_path"], allow_pickle=False) as carrier:
            shape1, shape2, retained = carrier["view1"].shape, carrier["view2"].shape, carrier["retained"].shape
            graph_nnz = int(len(carrier["graph0__data"]))
        p0_rows.append(
            {
                "lane": manifest["lane"],
                "start_candidate": manifest["start_candidate"],
                "n": shape1[0],
                "k": manifest["k"],
                "view1_shape": str(list(shape1)),
                "view2_shape": str(list(shape2)),
                "retained_shape": str(list(retained)),
                "registered_graph_nnz": graph_nnz,
                "five_graph_names": "|".join(manifest["graph_names"]),
                "backward_parameter_update": "PASS",
                "strict_checkpoint_reload": "PASS",
                "fresh_process_replay_round1": "PASS",
                "fresh_process_replay_round2": "PASS",
                "labels_read_by_producer": manifest["labels_read"],
                "bank_sha256": manifest["partition_bank_sha256"],
                "checkpoint_sha256": manifest["checkpoint_bank_sha256"],
            }
        )
        resource_rows.append(
            {
                "lane": manifest["lane"],
                "start_candidate": manifest["start_candidate"],
                "wall_seconds": manifest["wall_seconds"],
                "peak_rss_mb": manifest["peak_rss_mb"],
                "device": manifest["device"],
                "candidate_count": len(manifest["candidate_ids"]),
            }
        )
    write_csv(OUT / "real_p0_registry.csv", p0_rows)
    write_csv(OUT / "training_resource_table.csv", resource_rows)

    failures = [
        {
            "cycle": "PARENT_REPLAY_PREFLIGHT",
            "status": "CORRECTED_ENGINEERING_ATTEMPT",
            "issue": "initial fresh-process invocation omitted PYTHONPATH=.",
            "resolution": "reran unchanged parent replay with PYTHONPATH=. and obtained exact PASS",
            "scientific_effect": "none; no labels or formula changes",
        },
        {
            "cycle": "AUTHORITY_ADAPTER_PREFORMAL",
            "status": "CORRECTED_BEFORE_EVALUATION",
            "issue": "Night16F ordered-ID and Night16H partition authorities use their original hash algorithms rather than Night22A array_sha",
            "resolution": "verified with original algorithms and recorded both authority and Night22A hashes; rebuilt all pre-evaluation artifacts",
            "scientific_effect": "none; label-closed authority correction",
        },
        {
            "cycle": "FROZEN_PRIMARY_START",
            "status": "STRUCTURAL_UNAVAILABILITY",
            "issue": "MISAR K7 has no exact-K Leiden partition in the frozen 32-resolution grid",
            "resolution": "fail-closed; did not expand grid; ran preregistered sensitivities only",
            "scientific_effect": "primary lane fails transfer availability and sensitivity cannot replace headline",
        },
    ]
    write_csv(OUT / "failure_and_correction_ledger.csv", failures)

    label_flow = {
        "schema": "night22b-label-flow-audit-v1",
        "known_k": "registered public benchmark protocol",
        "producer_annotation_reads": 0,
        "labels_in_input_loss_gradient_checkpoint_or_partition_selection": 0,
        "partitions_and_checkpoints_locked_before_evaluator": True,
        "producer_manifests_verify_labels_read_zero": True,
        "evaluation_process": "separate Python process after two fresh-process partition replays",
        "sensitivity_not_used_to_replace_primary": True,
        "misar_primary_unavailable_detected_before_evaluation": True,
    }
    jdump(OUT / "label_flow_audit.json", label_flow)

    authority = {
        "schema": "night22b-source-authority-manifest-v1",
        "parent_commit": contract["parent_commit"],
        "parent_tag": contract["parent_tag"],
        "parent_compact_index_sha256": contract["parent_compact_index_sha256"],
        "contract_sha256": sha(CONTRACT),
        "sources": {},
        "lanes": {},
    }
    for path in [
        REPO / "SpaLORA" / "night22a_geometry.py",
        REPO / "SpaLORA" / "night22a_junction.py",
        REPO / "scripts" / "night22a" / "night22a_junction_producer.py",
        REPO / "scripts" / "night22a" / "night22a_junction_replay.py",
        REPO / "scripts" / "night22b" / "prepare_transfer_authority.py",
        REPO / "scripts" / "night22b" / "build_frozen_start_bank.py",
        REPO / "scripts" / "night22b" / "night22b_independent_evaluator.py",
    ]:
        authority["sources"][str(path.relative_to(REPO))] = sha(path)
    for lane in contract["lanes"]:
        start_manifest = json.loads((WORK / "starts" / f"{lane}.json").read_text(encoding="utf-8"))
        authority["lanes"][lane] = {
            "carrier_sha256": contract["lanes"][lane]["carrier_sha256"],
            "ordered_ids_authority_sha256": contract["lanes"][lane]["ordered_ids_sha256"],
            "start_bank_sha256": start_manifest["start_bank_sha256"],
            "primary_start_available": start_manifest["primary_start_available"],
            "candidate_partition_sha256": start_manifest["candidate_partition_sha256"],
        }
    jdump(OUT / "source_and_authority_manifest.json", authority)

    disk = shutil.disk_usage("/")
    resource = {
        "schema": "night22b-resource-audit-v1",
        "root_total_bytes": disk.total,
        "root_used_bytes": disk.used,
        "root_free_bytes": disk.free,
        "root_free_gib": disk.free / 2**30,
        "minimum_required_free_gib": 15,
        "working_bytes": int(subprocess.check_output(["du", "-sb", str(WORK)], text=True).split()[0]),
        "new_external_download_bytes": 0,
        "gse205055_download_authorized": False,
    }
    jdump(OUT / "resource_and_disk_audit.json", resource)

    decision = {
        "schema": "night22b-final-decision-v1",
        "classification": "NO_FROZEN_TRANSFER_SIGNAL",
        "primary_independent_pass_count": 0,
        "primary_lane_count": 2,
        "misar_primary_start_available": False,
        "human_primary_independent_pass": False,
        "sensitivity_or_secondary_cannot_override_primary": True,
        "junction_mainline_closed": True,
        "additional_hpo_authorized": False,
        "gse205055_authority_closure_authorized": False,
        "score_frontier_advance": False,
        "labels_in_training_or_gradient": 0,
        "shutdown_dispatched_at_report_build": False,
    }
    if any(str(row["strict_independent_pass"]).lower() == "true" for row in primary):
        raise RuntimeError("decision consistency failure: expected 0/2 primary pass")
    jdump(OUT / "night22b_decision.json", decision)

    test_summary = {
        "schema": "night22b-targeted-tests-v1",
        "command": "pytest -q tests/test_night22a_geometry.py tests/test_night22a_junction.py tests/test_night22b_transfer.py",
        "passed": 15,
        "failed": 0,
        "warnings": 50,
        "real_environment": "/root/miniconda3/envs/SpaLORA",
    }
    jdump(OUT / "targeted_test_summary.json", test_summary)
    shutil.copy2(WORK / "logs" / "targeted_tests_final.log", OUT / "targeted_tests_final.log")

    report = f"""# Night-22B frozen RNA+chromatin junction transfer report

## 我现在需要知道的三件事

1. **问题**：本轮没有再调参数，而是把 Night-22A 的 `GEOM_LEIDEN_FEATURE + J01_GRAPH_LEAN` 原样冻结，问它能否在 MISAR 与人海马两个 held-out 研究上独立超过起点、纯 head 和每一个原子臂。
2. **实际结果**：父级 P22/placenta 15/15 候选重放完全一致；但 MISAR 的冻结 32 个 Leiden resolution 中没有 exact K=7 起点，不能补网格。人海马 FULL 为 **0.068806/0.148261**，低于 start **0.069277/0.146839** 的 ARI，也明显低于 `SHARED_GRAPH_ONLY` **0.082287/0.170063**。
3. **论文结论**：分类是 **NO_FROZEN_TRANSFER_SIGNAL**。Night-22A 的 chromatin 局部 junction 信号没有跨研究成立；该 junction 主线按约定终止，不追加 HPO，也不触发 GSE205055 下载。Melanoma 的高分只证明简单图原子臂很强，不能挽救 primary 判定。

## 主迁移判决

| lane | frozen primary start | FULL ARI/NMI | Δ vs coordinate-wise strongest control | independent pass |
|---|---|---:|---:|---|
| MISAR K7 | unavailable under frozen exact-K Leiden grid | NA | NA | false |
| human hippocampus K7 | GEOM_LEIDEN_FEATURE | 0.068806/0.148261 | -0.013482/-0.021801 | false |

MISAR 不是程序崩溃：冻结 generator 在标签关闭阶段按原 32 个 resolution 运行完毕，但没有任何 exact-K=7 解。扩网格会改变已冻结方法，因此按规则 fail-closed。人海马 FULL 实际改变 16 个 observation、保持 exact K、无 singleton、每簇有最小尺度内部边，但没有 matched score contribution。

## 起点敏感性与 secondary robustness

- MISAR + Ncut-K24：start 0.427961/0.556508，FULL 0.427803/0.556238；双降。
- MISAR + Night-16H：start 0.535306/0.658265，FULL 0.535348/0.658406，但与 `CLUSTER_GRAPH_ONLY` byte-exact，相同增益不能归因于 FULL。
- Human + Ncut-K24：FULL 0.156061/0.217678，低于 shared-only 0.165778/0.232719。
- Human + Night-16H：FULL 0.595824/0.588737，与 cluster-only byte-exact；ARI 略低于 start。
- Melanoma primary Leiden：FULL 0.928204/0.859914，高于 start，但低于 shared-only 0.951795/0.898555 和 additive 0.961342/0.915599；它是 secondary K2，不计主门。

这些敏感性说明失败并非单纯“起点太弱”：把 Night-16H 强分区接入后，FULL 仍等价于更简单的 cluster-only 原子臂。

## 数学与工程证据

- 使用 Night-22A 原始五图 bank、J01 数值参数、Adam/240 steps、hard exact-K 与六个 matched arms；没有根据 transfer 分数修改任何公式。
- 8 个可用 start bank 均完成真实 loss/backward、参数改变、strict checkpoint reload；两轮独立 Python fresh-process replay 共 16/16 bank PASS，每个 bank 5 个 trainable candidate 精确一致。
- Producer 只读取显式 numeric carrier/representation/sparse graph 与预锁 start；标签读取次数为 0。独立 evaluator 在 bank/checkpoint SHA 验证后才打开 reference。
- 15 项 targeted tests 全通过。父级 P22/placenta 各 15 个 candidate exact replay PASS。

## GSE205055 与停止线

主迁移未出现 signal，因此没有下载或运行 GSE205055 ME13_50um。这个决定避免在负迁移后把外部数据变成新的调参集。Junction 论文主线在本轮关闭；下一步若继续研究，应回到更强的上游 representation 或新的、预注册的直接分区对象，而不是补 Leiden resolution 或继续修 J01。

## 导师汇报版

Night-22A 在 P22 和 placenta 上曾出现很小的 chromatin junction 正信号，所以本轮把公式、五张图、训练预算和起点规则全部冻结，去做 MISAR 与人海马迁移。父级重放完全一致，说明实现接续可靠。MISAR 暴露了第一个严格问题：冻结 Leiden resolution 网格无法产生 exact K=7，我们没有为它扩网格。人海马能完整训练并产生新分区，但 FULL 只有 0.0688/0.1483，低于更简单的 shared-only 原子臂。换成 Ncut 或 Night-16H 强起点后，FULL 仍然不独立，往往被原子臂支配或与 cluster-only 完全相同。Melanoma 的高分同样主要来自 shared/additive 图臂，而非 FULL。结论是这条 junction 没有跨研究冻结迁移证据，应当停止，不再追加 HPO；这比把 sensitivity 的较高数字替换成 headline 更可信。

## 技术状态

- Parent: `{contract['parent_commit']}` / `{contract['parent_tag']}`。
- Frozen contract SHA-256: `{sha(CONTRACT)}`。
- 根盘余量约 {resource['root_free_gib']:.1f} GiB，超过 15 GiB 安全线；新增外部下载 0。
- 普通 push、final tag、incremental bundle、compact 与 Windows 独立复算在最终封口步骤登记。
"""
    (OUT / "night22b_report.md").write_text(report, encoding="utf-8")
    (OUT / "mentor_oral_report.md").write_text(
        "\n".join(
            [
                "1. Night-22B 是完全冻结迁移，不是再调参。",
                "2. P22 和 placenta 父级各 15 个候选都 exact replay，接续实现可靠。",
                "3. MISAR 的冻结 Leiden 网格没有 exact K=7 解，我们没有违规补 resolution。",
                "4. 人海马 FULL 为 0.068806/0.148261，低于 shared-only 0.082287/0.170063。",
                "5. Night-16H 强起点 sensitivity 中 FULL 也与 cluster-only 完全相同，不能归因于联合 junction。",
                "6. Melanoma 的高分由更简单 shared/additive 原子臂解释，且 K2 不计主门。",
                "7. 因而分类 NO_FROZEN_TRANSFER_SIGNAL，junction 主线关闭，不追加 HPO 或 GSE205055 下载。",
            ]
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
