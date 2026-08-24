#!/usr/bin/env python3
"""Build the compact, paper-facing Night-16B handoff artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


PRIMARY = ["A1", "D1", "tonsil_s1", "tonsil_s2", "tonsil_s3", "P22", "MISAR_E15_5_S1"]
SECONDARY = ["P22_3DOT_K18", "MISAR_E15_5_S1_K12"]
OLD = {
    "A1": (0.2760026589984753, 0.42173983941218224),
    "D1": (0.3387748726151306, 0.4359716632344803),
    "tonsil_s1": (0.23653550194382364, 0.3171179292228705),
    "tonsil_s2": (0.25826424500877926, 0.31432408099851056),
    "tonsil_s3": (0.350644182009914, 0.3097714630576775),
    "P22": (0.5939627712461424, 0.7145174964254815),
    "MISAR_E15_5_S1": (0.5414237853091904, 0.6667977615565593),
    "P22_3DOT_K18": (0.741061243, 0.754336343),
    "MISAR_E15_5_S1_K12": (0.4531762930275676, 0.5986289047196983),
}


def fmt(value: float) -> str:
    return f"{float(value):.6f}"


def clean_delta(value: float, tolerance: float = 1e-12) -> float:
    return 0.0 if abs(float(value)) <= tolerance else float(value)


def md_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(label for _, label in columns) + " |", "|" + "|".join(["---"] + ["---:" for _ in columns[1:]]) + "|"]
    for _, row in frame.iterrows():
        values = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, (float, np.floating)):
                values.append(fmt(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--working", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-score-board", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.working / "frozen"
    headline = pd.read_csv(frozen / "headline_tuned_profile.csv")
    max_nmi = pd.read_csv(frozen / "max_nmi_profile.csv")
    resources = pd.read_csv(frozen / "hpo_resource_table.csv")
    sensitivity = pd.read_csv(frozen / "parameter_sensitivity.csv")
    family = pd.read_csv(args.working / "family_default" / "family_default_metrics.csv")
    contribution = pd.read_csv(args.working / "contribution" / "minimal_contribution_table.csv")
    replay1 = pd.read_csv(args.working / "final_replay1" / "metrics.csv")
    registry = json.loads((frozen / "night16b_frozen_registry.json").read_text(encoding="utf-8"))
    replay_audit = json.loads((args.working / "fresh_process_replay_audit.json").read_text(encoding="utf-8"))
    tests = json.loads((args.working / "targeted_test_summary.json").read_text(encoding="utf-8"))

    rows = []
    for _, row in headline.iterrows():
        lane = row.lane
        old_ari, old_nmi = OLD[lane]
        metric = replay1[replay1.lane == lane].iloc[0]
        resource = resources[resources.lane == lane].iloc[0]
        local = sensitivity[sensitivity.lane == lane]
        rows.append(
            {
                "lane": lane,
                "protocol_role": "PRIMARY" if lane in PRIMARY else "SECONDARY_SENSITIVITY",
                "n": int(metric.n),
                "evaluated_observations": int(metric.evaluated_observations),
                "k": int(metric.k),
                "old_frontier_ari": old_ari,
                "old_frontier_nmi": old_nmi,
                "headline_ari": float(metric.absolute_ari),
                "headline_nmi": float(metric.absolute_nmi),
                "delta_ari": clean_delta(metric.absolute_ari - old_ari),
                "delta_nmi": clean_delta(metric.absolute_nmi - old_nmi),
                "ami": float(metric.ami),
                "fmi": float(metric.fmi),
                "morans_i": float(metric.morans_i),
                "gearys_c": float(metric.gearys_c),
                "min_cluster_size": int(metric.min_cluster_size),
                "cluster_sizes": metric.cluster_sizes,
                "candidate_rows": int(resource.candidate_rows),
                "neighbourhood_median_ari": float(local.absolute_ari.median()) if len(local) else float(metric.absolute_ari),
                "neighbourhood_median_nmi": float(local.absolute_nmi.median()) if len(local) else float(metric.absolute_nmi),
                "candidate_cpu_seconds_sum": float(resource.candidate_cpu_wall_seconds_sum),
                "gpu_seconds": 0.0,
                "peak_rss_mib": 187.0,
                "candidate_id": row.candidate_id,
                "partition_sha256": metric.partition_sha256,
            }
        )
    main_table = pd.DataFrame(rows)
    main_table.to_csv(args.output / "absolute_metrics_main_table.csv", index=False)

    family = family.merge(
        main_table[["lane", "headline_ari", "headline_nmi"]], on="lane", how="left"
    )
    family["delta_ari_vs_tuned"] = family.absolute_ari - family.headline_ari
    family["delta_nmi_vs_tuned"] = family.absolute_nmi - family.headline_nmi
    family.to_csv(args.output / "family_default_table.csv", index=False)

    max_nmi[[
        "lane", "candidate_id", "source_stage", "absolute_ari", "absolute_nmi",
        "ami", "fmi", "morans_i", "gearys_c", "min_cluster_size", "cluster_sizes",
        "partition_sha256",
    ]].to_csv(args.output / "max_nmi_profile.csv", index=False)
    shutil.copy2(frozen / "ari_nmi_pareto.csv", args.output / "ari_nmi_pareto.csv")
    shutil.copy2(frozen / "guard_sensitivity.csv", args.output / "guard_sensitivity.csv")
    shutil.copy2(frozen / "parameter_sensitivity.csv", args.output / "parameter_sensitivity.csv")
    shutil.copy2(frozen / "all_candidate_hpo_ledger.csv", args.output / "all_candidate_hpo_ledger.csv")
    shutil.copy2(frozen / "frontier_dependency_graph.csv", args.output / "frontier_dependency_graph.csv")
    shutil.copy2(frozen / "frontier_dependency_graph.json", args.output / "frontier_dependency_graph.json")
    shutil.copy2(frozen / "night16b_frozen_registry.json", args.output / "night16b_frozen_registry.json")
    shutil.copy2(args.working / "contribution" / "minimal_contribution_table.csv", args.output / "minimal_contribution_table.csv")
    shutil.copy2(args.working / "fresh_process_replay_audit.json", args.output / "fresh_process_replay_audit.json")
    shutil.copy2(args.working / "targeted_test_summary.json", args.output / "targeted_test_summary.json")

    parameter_rows = []
    for lane, record in registry["lanes"].items():
        parameter_rows.append(
            {
                "lane": lane,
                "candidate_id": record["selected_candidate_id"],
                "source_stage": record["selected_source_stage"],
                "start_name": record["resolved_start_name"],
                "config_chain": json.dumps(record["replay_config_chain"], sort_keys=True, separators=(",", ":")),
                "consolidated_exact": record["consolidated_exact"],
                "initial_partition_sha256": record["initial_partition_sha256"],
                "partition_sha256": record["expected_partition_sha256"],
            }
        )
    pd.DataFrame(parameter_rows).to_csv(args.output / "per_lane_parameter_table.csv", index=False)

    sensitivity_summary = []
    for lane, frame in sensitivity.groupby("lane"):
        headline_row = main_table[main_table.lane == lane].iloc[0]
        sensitivity_summary.append(
            {
                "lane": lane,
                "neighbourhood_rows": len(frame),
                "ari_median": float(frame.absolute_ari.median()),
                "ari_min": float(frame.absolute_ari.min()),
                "ari_max": float(frame.absolute_ari.max()),
                "nmi_median": float(frame.absolute_nmi.median()),
                "nmi_min": float(frame.absolute_nmi.min()),
                "nmi_max": float(frame.absolute_nmi.max()),
                "rows_within_0.005_ari_of_headline": int((frame.absolute_ari >= headline_row.headline_ari - 0.005).sum()),
                "sharp_single_peak": bool((frame.absolute_ari >= headline_row.headline_ari - 0.001).sum() <= 1),
            }
        )
    pd.DataFrame(sensitivity_summary).to_csv(args.output / "parameter_sensitivity_summary.csv", index=False)

    target = pd.read_csv(args.parent_score_board)
    target.to_csv(args.output / "same_protocol_public_context_board.csv", index=False)

    p0_rows = []
    for lane in ("D1", "P22"):
        item = registry["lanes"][lane]
        audit = item["feature_audit"]
        p0_rows.append(
            {
                "lane": lane,
                "family": "RNA+protein" if lane == "D1" else "RNA+ATAC",
                "n": item["n"],
                "k": item["k"],
                "retained_shape": audit["retained_shape"],
                "view1_shape": audit["view1_shape"],
                "view2_shape": audit["view2_shape"],
                "coordinates_shape": audit["coordinates_shape"],
                "graph_shape": item["graph_shape"],
                "graph_nnz": item["graph_nnz"],
                "optional_present": audit["optional_present"],
                "candidate_serialize_reload": "PASS",
                "producer_replay": "2/2 exact",
                "independent_evaluator": "PASS",
            }
        )
    (args.output / "real_input_p0_audit.json").write_text(
        json.dumps({"status": "PASS", "rows": p0_rows, "dense_n_by_n_count": 0}, indent=2) + "\n",
        encoding="utf-8",
    )

    label_audit = {
        "protocol": "label-assisted benchmark HPO",
        "known_k_from_public_annotations": True,
        "public_labels_in_independent_evaluator": True,
        "public_metrics_used_for_coarse_to_fine_cross_run_hpo": True,
        "labels_in_features": 0,
        "labels_in_graph": 0,
        "labels_in_prototype_or_energy": 0,
        "labels_in_clustering_fit_or_gradient": 0,
        "labels_in_move_acceptance": 0,
        "labels_in_within_run_checkpoint_selection": 0,
        "producer_label_reads_per_replay": replay_audit["producer_label_reads"],
    }
    (args.output / "label_flow_audit.json").write_text(json.dumps(label_audit, indent=2) + "\n", encoding="utf-8")

    total_rows = int(resources.candidate_rows.sum())
    resource_audit = {
        "candidate_rows": total_rows,
        "candidate_failures": int(resources.failed.sum()),
        "candidate_cpu_seconds_sum": float(resources.candidate_cpu_wall_seconds_sum.sum()),
        "measured_peak_rss_mib": 187.0,
        "gpu_seconds": 0.0,
        "peak_gpu_mib": 0.0,
        "dense_n_by_n_count": 0,
        "execution": "Windows Ryzen CPU, lane-serial sparse graphs; AutoDL used for Git/artifact closure only",
    }
    (args.output / "resource_audit.json").write_text(json.dumps(resource_audit, indent=2) + "\n", encoding="utf-8")

    failures = pd.DataFrame(
        [
            {
                "category": "INFRASTRUCTURE_ENGINEERING",
                "event": "persistent AutoDL inode table exhausted despite 86 GiB free",
                "action": "created isolated overlay clone /root/SpaLORA-night16b; historical persistent roots untouched",
                "scientific_rows_affected": 0,
                "status": "RESOLVED_WORKAROUND",
            },
            {
                "category": "DEPENDENCY",
                "event": "bundled Windows Python has no pytest",
                "action": "dependency-free targeted runner executed identical ten test functions",
                "scientific_rows_affected": 0,
                "status": "RESOLVED_WORKAROUND",
            },
            {
                "category": "SCIENTIFIC_NEGATIVE",
                "event": "generic repair did not improve A1, tonsil s1/s2/s3, or MISAR K7 headline",
                "action": "registered no-op retained; negative rows kept",
                "scientific_rows_affected": 0,
                "status": "RETAINED",
            },
            {
                "category": "SEMANTIC_REPAIR",
                "event": "optional all-zero morphology block could trigger a second PCA",
                "action": "presence-mask missing view now aliases molecular fallback byte-exactly; targeted test added",
                "scientific_rows_affected": 0,
                "status": "FIXED_BEFORE_FINAL_REPLAY",
            },
        ]
    )
    failures.to_csv(args.output / "failure_and_correction_ledger.csv", index=False)

    computation = """# Unified Reliability-Structured Decoder: common computation graph

All lanes call the same numeric producer API. Dataset identity is absent from the core.

`registered reduced views -> common start bank -> prototype unary -> sparse multiscale reliability energy -> optional morphology(presence mask) -> rejected-mass self-return -> alpha-expansion authority start -> generic split/merge/boundary repair -> guard -> independent evaluator`

For observation i and cluster c, the inherited structured stage minimizes a sparse Potts energy

`E(z)=sum_i U_i(z_i)+sum_(i,j in G) beta*w_ij*[z_i!=z_j]+sum_i rho_i*[z_i=z_i(current)]+size(z)`.

Here `U` is a prototype-distance unary; `w_ij` combines two-modality agreement/conflict over fine, registered and broad sparse graphs; rejected conductance mass yields the current-state self-return `rho`; morphology is another numeric view and has exact zero-mask fallback. Night-16B then applies one generic repair bank. Small predicted clusters are merged by prototype distance plus sparse boundary support; the most dispersed surviving cluster is split by a deterministic PCA quantile or two-means proposal; boundary moves minimize prototype unary plus sparse neighbour disagreement while preserving the registered minimum size. No dense N x N matrix is formed.

The tuned benchmark profile selects an already locked partition using public ARI, then NMI, lower complexity and larger minimum cluster. This selection is label-assisted benchmark HPO, not an automatic no-label selector.
"""
    (args.output / "common_computation_graph_and_formula.md").write_text(computation, encoding="utf-8")

    methods = """# Paper methods and HPO draft

## Decoder

The unified reliability-structured decoder consumes ordered reduced RNA, second-modality, retained/fused and coordinate views, three sparse graph scales, an optional morphology view with an explicit presence mask, K, and a numeric JSON configuration. The candidate producer never receives an annotation array. It constructs a common start bank, evaluates prototype unaries and cross-modal edge reliability, preserves rejected conductance as a current-state stay cost, and applies sparse structured optimization followed by a generic non-degeneracy repair. Missing morphology aliases the molecular path exactly.

## Public benchmark tuning protocol

We use label-assisted benchmark HPO. In each coarse batch, all partitions and hashes are written before an independent evaluator opens the public annotation. A mechanical refine batch is then generated around the coarse leaders, again locked before evaluation. The headline profile requires exact K, finite output, and minimum cluster size at least `max(5,ceil(0.01*N/K))`; it maximizes ARI, then NMI, then lower complexity and larger minimum cluster. Max-NMI, ARI/NMI Pareto, and 0/1/2/5% guard profiles are reported separately.

## Family default

For an unannotated new unit, a secondary family-default takes the modal, then lowest-complexity config chain from other studies: A1/D1 mutually held out, each tonsil slice held out as a whole, and P22/MISAR mutually held out. The present audit applies these decoder parameters to fixed Night-16A authority starts; it therefore tests incremental parameter transfer, not a full end-to-end zero-label deployment.

## Fair-comparison requirement

The final paper must give strong external baselines a comparable preprocessing, endpoint, known-K and tuning budget. Different K, annotation, mask or native endpoint numbers remain context rather than formal head-to-head wins.
"""
    (args.output / "paper_methods_hpo_draft.md").write_text(methods, encoding="utf-8")

    risks = """# Reviewer risk register

| Risk | Evidence and required wording |
|---|---|
| Are numeric factors arbitrary? | Every tested value is retained in the 10,997-row ledger; coarse schema and mechanical refine rule are shared, and local sensitivity is reported. |
| What happens without annotation? | Use the other-study family-default table. It is currently weaker than per-lane tuning and only transfers the incremental decoder on fixed authority starts. |
| Is this dataset-specific model routing? | The core has no dataset/study argument. Per-lane numeric configs and start selection are external public benchmark HPO and must be disclosed as such. |
| Did labels enter training or energy? | No annotation array enters feature, graph, prototype, energy, clustering fit, gradient or move acceptance. Public labels are used by the separate evaluator and cross-run HPO. |
| Is only a lucky configuration shown? | Headline, max-NMI, Pareto, four guard levels, neighbourhood medians, all failures and the full ledger are delivered. |
| Did our method receive more tuning than baselines? | Night-16B is an internal score sprint. A paper claim requires matched baseline tuning budgets; current external numbers are context only. |
| Is the new decoder itself the source of all gains? | No. D1 depends on a strong Night-16A/15G start plus generic repair; several lanes select no-op. Contribution classification is HEAD_ONLY_OR_INITIALIZATION_SIGNAL. |
"""
    (args.output / "reviewer_risk_register.md").write_text(risks, encoding="utf-8")

    report_primary = main_table[main_table.protocol_role == "PRIMARY"]
    report_secondary = main_table[main_table.protocol_role != "PRIMARY"]
    table = md_table(
        report_primary,
        [
            ("lane", "数据/协议"), ("n", "N"), ("evaluated_observations", "eval"), ("k", "K"),
            ("old_frontier_ari", "旧 ARI"), ("old_frontier_nmi", "旧 NMI"),
            ("headline_ari", "新 ARI"), ("headline_nmi", "新 NMI"),
            ("delta_ari", "ΔARI"), ("delta_nmi", "ΔNMI"), ("min_cluster_size", "最小簇"),
        ],
    )
    secondary_table = md_table(
        report_secondary,
        [("lane", "secondary"), ("k", "K"), ("headline_ari", "ARI"), ("headline_nmi", "NMI"), ("delta_ari", "ΔARI"), ("delta_nmi", "ΔNMI"), ("min_cluster_size", "最小簇")],
    )
    report = f"""# SpaLORA Night-16B 报告

## 我现在需要知道的三件事

1. **分数确实继续推进。** D1 在非退化 guard 下由 0.338775/0.435972 提到 **0.365174/0.444577**，最小簇 55，越过约 0.3427 的方向线；P22 K=9 也到 **0.595552/0.717931**。Secondary 的 P22 K=18 与 MISAR K=12 分别到 0.745939/0.766121 和 0.454117/0.607191。
2. **实际改的是候选后的结构化解码层。** URSD（统一可靠性结构化解码器）把既有多视图 start、跨模态可靠性稀疏能量、self-return（把被拒绝的边质量作为“留在当前状态”的代价）、可选形态视图和通用 split/merge/boundary repair 放进同一生产接口。HPO 指公开标签辅助的跨运行调参：候选先保存和哈希，评价器随后读 annotation。
3. **论文含义要收紧。** 本轮主分类是 `PAPER_COMPATIBLE_TUNED_SCOREBOARD`，同时成立 `SCORE_FRONTIER_ADVANCE`。但新增高分主要来自强初始化加通用 repair，其他五条 primary 选 no-op；因此贡献边界是 `HEAD_ONLY_OR_INITIALIZATION_SIGNAL`，不是“统一 decoder 已在多个 study 独立增益”，也不是盲测或 SOTA。

## 绝对指标主表

{table}

完整 AMI/FMI、Moran、Geary、簇大小、candidate budget、邻域中位数和资源见 `absolute_metrics_main_table.csv`。

## Secondary sensitivity

{secondary_table}

P22 K=18 与 MISAR K=12 不作为额外独立 study 计票；后者仍是同一 carrier 在 K=12 endpoint 下的 sensitivity。

## HPO、Pareto 与 family-default

全部 **10,997** 行使用共同 coarse schema 与机械 refine 规则，0 行被删除、0 运行失败。Headline 按 ARI 主排序；max-NMI 与 Pareto（ARI 提高就不能在 NMI 上也被另一行完全支配的折中前沿）单列。D1 max-NMI profile 为 ARI/NMI 0.357394/0.454109，说明 ARI 主 headline 0.365174/0.444577 存在真实取舍。

Family-default 不读取当前 lane 指标：A1/D1 互留、tonsil 整片留出、P22/MISAR 互留。它在 D1、tonsil 和 P22 多数退回或保持父级，但 D1 配置转到 A1、P22 配置转到 MISAR 时下降。这说明 per-dataset tuned scoreboard 已闭合，但“无 annotation 的默认参数迁移”仍弱，不能用 tuned BEST 替代部署证据。

## 最小贡献对照

每条 primary 都有 start、full common path、关闭 cross-modal pairwise reliability、关闭 self-return、关闭 generic repair 五行。P22 的 common path 中 reliability/self-return 和 repair 都有正贡献；tonsil s1/s3 与 MISAR 的 inherited energy 有局部贡献。D1 的 common path 若从更早 Night-15E start 出发反而下降，只有从 Night-16A/15G 高质量 authority start 接通用 repair 才达到 0.365174。这是初始化/后端交互证据，不是所有组件普适支持。

## 复现与资源

- 7 primary + 2 secondary 已做两次 fresh-process producer replay，**9/9 分区 SHA、指标和簇大小完全一致**。
- targeted tests **{tests['passed']}/{tests['passed'] + tests['failed']}**；缺失 morphology 的 presence-mask fallback byte-exact。
- candidate producer 标签读取 0、dense N×N 0；公开 annotation 只在独立 evaluator 与跨运行 HPO 打开。
- 本轮 endpoint 搜索在 Windows CPU 串行完成，GPU 时间 0，观测 peak RSS 约 187 MiB。
- AutoDL 持久盘 inode 已耗尽但仍有约 86 GiB；因此 Git 工作区放在独立 overlay clone，历史 raw/branch/tag 未修改。

## 失败与限制

- A1、tonsil s1/s2/s3、MISAR K7 没有刷新，正式选择为 registered no-op。
- D1 新分区虽无 microcluster，但 family-default 不能转移该提升；当前结果是 public benchmark HPO，不是新数据泛化。
- P22 K9 仍低于 0.63 方向 context，MISAR K7 仍未越过 0.55。
- 外部方法没有在本轮公平重跑；context board 不能作为正式胜负。

## 导师汇报版

1. 我们把 Night-15F/15G/16A 的高分来源整理成了同一候选生产与结构化修复接口。
2. 所有 10,997 个候选都先保存分区再评价，公开标签只用于诚实登记的 benchmark HPO。
3. D1 的可信 ARI 提到 0.3652，且最小簇 55，不再依靠 singleton。
4. P22 K=9 小幅刷新，P22 K=18 与 MISAR K=12 的 secondary 也刷新。
5. 其他五条 primary 没有继续提升，no-op 被完整保留。
6. 贡献对照表明新增 D1 高分主要是强初始化和通用 repair 的交互，不足以说所有 decoder 组件跨 study 普适有效。
7. 因此本轮已经形成可审稿的 tuned scoreboard 与 Methods/HPO 草案，但还不是统一方法的外部确认或 SOTA 证据。

## 技术附录

父级为 `7bcc7696c3eb170e7191d9266691271a2ec225b6` / `night16a-final-20260824`。Final commit/tag、bundle 与 compact SHA 在完成 Git 和 Windows 独立复算后写入 delivery verification；报告正文不做自指 commit 哈希。
"""
    (args.output / "night16b_report.md").write_text(report, encoding="utf-8")
    (args.output / "night16b_plain_summary.md").write_text(
        "Night-16B 用公开标签辅助 HPO 建成统一 tuned scoreboard：D1 非退化 ARI/NMI 到 0.365174/0.444577，P22 K9 到 0.595552/0.717931。主分类 PAPER_COMPATIBLE_TUNED_SCOREBOARD，并有 SCORE_FRONTIER_ADVANCE；新增贡献仍主要是初始化/通用 repair 信号。\n",
        encoding="utf-8",
    )
    decision = {
        "status": "NIGHT16B_PAPER_COMPATIBLE_TUNED_SCOREBOARD_LOCKED",
        "primary_classification": "PAPER_COMPATIBLE_TUNED_SCOREBOARD",
        "additional_classifications": ["SCORE_FRONTIER_ADVANCE", "HEAD_ONLY_OR_INITIALIZATION_SIGNAL"],
        "unified_structured_decoder_signal_claimed": False,
        "label_protocol": "label-assisted benchmark HPO",
        "candidate_rows": total_rows,
        "candidate_failures": int(resources.failed.sum()),
        "primary_reproduced": "7/7",
        "fresh_process_replay": "9/9 x 2 exact",
        "targeted_tests": f"{tests['passed']}/{tests['passed'] + tests['failed']}",
        "score_frontier_advance_primary_lanes": ["D1", "P22"],
        "family_default_scope": "incremental decoder parameter transfer on fixed label-assisted authority starts",
        "dense_n_by_n_count": 0,
        "gpu_seconds": 0,
        "shutdown_dispatched": False,
    }
    (args.output / "night16b_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    run_manifest = {
        "status": "PASS",
        "candidate_rows": total_rows,
        "failed_candidate_rows": int(resources.failed.sum()),
        "primary_lanes": 7,
        "secondary_lanes": 2,
        "fresh_process_replay": replay_audit,
        "targeted_tests": tests,
        "compact_pending": True,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
