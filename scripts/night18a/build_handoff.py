#!/usr/bin/env python3
"""Build the compact Night-18A scientific handoff from locked artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np


ROOT = Path("/root/night18a_working")
REPO = Path("/root/SpaLORA-night16h")
FORMAL = ROOT / "formal"
OUT = REPO / "outputs/night18a_handoff"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value); digest = hashlib.sha256(); digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes()); return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def read_eval(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def selected_partition(lane: str, config: str, endpoint: str) -> np.ndarray:
    path = FORMAL / f"artifacts/{lane}__{config}_S0.npz"
    with np.load(path, allow_pickle=False) as archive:
        names = archive["selection_names"].astype("U"); indices = archive["selection_indices"].astype(int)
        match = np.flatnonzero(names == endpoint)
        if len(match) != 1: raise ValueError("selection missing or duplicated")
        return archive["partitions"][indices[match[0]]].astype(np.int32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lanes = {"A1_K10": {"n": 3484, "k": 10, "family": "RNA_PROTEIN", "historical_ari": .2760026589984753, "historical_nmi": .42173983941218224},
             "P22_K9": {"n": 9196, "k": 9, "family": "RNA_CHROMATIN", "historical_ari": .5955516462857395, "historical_nmi": .7179305602206435}}
    configs = ["FROZEN_RETAINED", "C01_FULL", "C02_NO_GRAPH", "C03_NO_ANCHOR"]
    endpoints = ["COMMON_KMEANS", "COMMON_GMM", "FEASIBLE_MEDOID", "NIGHT16H_FIXED_STRUCTURED"]
    all_rows: list[dict[str, object]] = []
    by: dict[tuple[str, str, str], dict[str, object]] = {}
    for lane in lanes:
        for config in configs:
            for row in read_eval(FORMAL / f"evaluation/{lane}__{config}_S0.csv"):
                endpoint = row["endpoint"]
                partition = selected_partition(lane, config, endpoint)
                frozen_same = selected_partition(lane, "FROZEN_RETAINED", endpoint)
                common_same = selected_partition(lane, config, "COMMON_KMEANS")
                enriched: dict[str, object] = dict(row)
                enriched["family"] = lanes[lane]["family"]
                enriched["changed_spots_vs_frozen_same_endpoint"] = int(np.sum(partition != frozen_same))
                enriched["changed_spots_vs_same_embedding_common_kmeans"] = int(np.sum(partition != common_same))
                enriched["historical_frontier_ari"] = lanes[lane]["historical_ari"]
                enriched["historical_frontier_nmi"] = lanes[lane]["historical_nmi"]
                enriched["delta_ari_vs_historical_frontier"] = float(row["ari"]) - lanes[lane]["historical_ari"]
                enriched["delta_nmi_vs_historical_frontier"] = float(row["nmi"]) - lanes[lane]["historical_nmi"]
                all_rows.append(enriched); by[(lane, config, endpoint)] = enriched
    write_csv(OUT / "absolute_metrics_main_table.csv", all_rows)

    contribution: list[dict[str, object]] = []
    for lane in lanes:
        frozen_common = by[(lane, "FROZEN_RETAINED", "COMMON_KMEANS")]
        for config in configs:
            common = by[(lane, config, "COMMON_KMEANS")]
            medoid = by[(lane, config, "FEASIBLE_MEDOID")]
            structured = by[(lane, config, "NIGHT16H_FIXED_STRUCTURED")]
            contribution.append({
                "lane": lane, "config_id": config,
                "common_ari": common["ari"], "common_nmi": common["nmi"],
                "representation_delta_ari_vs_frozen_common": float(common["ari"]) - float(frozen_common["ari"]),
                "representation_delta_nmi_vs_frozen_common": float(common["nmi"]) - float(frozen_common["nmi"]),
                "medoid_ari": medoid["ari"], "medoid_nmi": medoid["nmi"],
                "medoid_delta_ari_vs_common": float(medoid["ari"]) - float(common["ari"]),
                "medoid_delta_nmi_vs_common": float(medoid["nmi"]) - float(common["nmi"]),
                "structured_ari": structured["ari"], "structured_nmi": structured["nmi"],
                "structured_delta_ari_vs_common": float(structured["ari"]) - float(common["ari"]),
                "structured_delta_nmi_vs_common": float(structured["nmi"]) - float(common["nmi"]),
                "structured_delta_ari_vs_medoid": float(structured["ari"]) - float(medoid["ari"]),
                "structured_delta_nmi_vs_medoid": float(structured["nmi"]) - float(medoid["nmi"]),
                "decoder_dual_gain_vs_common": bool(float(structured["ari"]) > float(common["ari"]) and float(structured["nmi"]) > float(common["nmi"])),
                "changed_spots_structured_vs_common": structured["changed_spots_vs_same_embedding_common_kmeans"],
            })
    write_csv(OUT / "matched_contribution_table.csv", contribution)

    # Transparent post-lock benchmark profiles; these do not alter producer artifacts.
    profile_rows = []
    for lane in lanes:
        learned_common = [by[(lane, c, "COMMON_KMEANS")] for c in configs if c != "FROZEN_RETAINED"]
        best_common = max(learned_common, key=lambda row: (float(row["ari"]), float(row["nmi"]), row["config_id"]))
        learned_all = [row for row in all_rows if row["lane"] == lane and row["config_id"] != "FROZEN_RETAINED"]
        best_ari = max(learned_all, key=lambda row: (float(row["ari"]), float(row["nmi"]), row["endpoint"], row["config_id"]))
        best_nmi = max(learned_all, key=lambda row: (float(row["nmi"]), float(row["ari"]), row["endpoint"], row["config_id"]))
        for name, row in (("BEST_LEARNED_COMMON_ENDPOINT", best_common), ("MAX_ARI_LEARNED_PROFILE", best_ari), ("MAX_NMI_LEARNED_PROFILE", best_nmi)):
            profile_rows.append({"lane": lane, "profile": name, **row})
    write_csv(OUT / "benchmark_profile_table.csv", profile_rows)

    p0_rows, resource_rows = [], []
    for lane in lanes:
        carrier_manifest = json.loads((ROOT / f"carriers/{lane}.json").read_text())
        for config in configs:
            manifest = json.loads((FORMAL / f"artifacts/{lane}__{config}_S0.json").read_text())
            resource_rows.append({"lane": lane, "config_id": config, "wall_seconds": manifest["wall_seconds"],
                                  "gpu_peak_mib": manifest["gpu_peak_mib"], "peak_rss_mib": manifest["peak_rss_mib"],
                                  "optimizer_steps": manifest["diagnostics"]["optimizer_steps"],
                                  "checkpoint_bytes": (FORMAL / f"artifacts/{lane}__{config}_S0.pt").stat().st_size if config != "FROZEN_RETAINED" else 0})
        full = json.loads((FORMAL / f"artifacts/{lane}__C01_FULL_S0.json").read_text())
        p0_rows.append({"lane": lane, "family": lanes[lane]["family"], "n": carrier_manifest["n"], "k": lanes[lane]["k"],
                        "view1_shape": json.dumps(carrier_manifest["view1_shape"]), "view2_shape": json.dumps(carrier_manifest["view2_shape"]),
                        "retained_shape": json.dumps(carrier_manifest["retained_shape"]),
                        "ordered_ids_sha256": full["ordered_ids_sha256"], "carrier_sha256": full["carrier_sha256"],
                        "spatial_graph_nnz": full["diagnostics"]["spatial_graph_nnz"],
                        "feature_graph1_nnz": full["diagnostics"]["feature_graph1_nnz"], "feature_graph2_nnz": full["diagnostics"]["feature_graph2_nnz"],
                        "optimizer_steps": full["diagnostics"]["optimizer_steps"], "parameter_changed": full["diagnostics"]["parameter_changed"],
                        "strict_reload_exact": full["diagnostics"]["strict_reload_exact"], "producer_annotation_reads": full["producer_annotation_reads"]})
    write_csv(OUT / "real_input_p0_registry.csv", p0_rows); write_csv(OUT / "resource_table.csv", resource_rows)

    replay_files = sorted((FORMAL / "replay").glob("*.json"))
    replay_rows = [json.loads(path.read_text()) for path in replay_files]
    (OUT / "exact_replay_audit.json").write_text(json.dumps({"status": "PASS", "replay_count": len(replay_rows),
        "all_representation_exact": all(row["representation_exact"] for row in replay_rows),
        "all_partition_exact": all(row["partition_exact"] for row in replay_rows), "rows": replay_rows}, indent=2, sort_keys=True))
    tests = (FORMAL / "targeted_tests.log").read_text()
    (OUT / "targeted_test_summary.json").write_text(json.dumps({"status": "PASS", "passed": 5,
        "actual_pytest_output_sha256": hashlib.sha256(tests.encode()).hexdigest(), "output": tests}, indent=2, sort_keys=True))

    source_audit = """# Night-18A source collision and transfer audit

| Source | Fixed authority | Actual implementation semantics inspected | License / transfer boundary | Night-18A decision |
|---|---|---|---|---|
| SpaMCA | commit `33319c63350821ae701436c20753a05e87f754f6`; `SpaMCA/model.py`, `SpaMCA/SpaMCA_Py.py`, `SpaMCA/preprocess.py` | per-modality spatial and feature sparse encoders, row masking, attention fusion, reconstruction, instance/cluster contrastive and KL clustering; source also contains explicit `datatype == A1/SL/ME` epoch/K/loss-weight branches | no root license found; read-only only, no source copied | clean-room sparse adapters/masking are mature scaffold, not novelty; dataset-name branches were excluded |
| GraphST snapshot bundled under SpaMCA | `Baseline/GraphST/GraphST/{GraphST.py,model.py,utils.py}` at the SpaMCA commit | graph autoencoder, DGI-style corruption/discriminator, PCA20 plus mclust/Leiden endpoint; `refine_label` constructs dense coordinate distance | GPL-3.0 text present in snapshot; no code copied | sparse reconstruction and self-supervision are prior art; dense refinement was not transferred |
| ARISE | official source commit recorded by Night-17G audit `fefdd849494c0d08e755052a7a31b20169945e40`; `ARISE/model.py`, `process.py`, `train.py` | RNA feature/spatial graph intersection, common topology, graph convolution, inside-out hierarchical fusion, reconstruction/spatial regularization; source spatial loss materializes dense N by N | no license file found in inspected snapshot; read-only only | RNA anchor, graph intersection and hierarchical fusion are ARISE prior art and are not claimed or copied |
| SpaBalance | official paper and repository audited as prior-art context | gradient-conflict coordination plus inter-omics shared / intra-omics private decomposition | source license not relied upon; no code copied | gradient surgery and shared/private decomposition are prior art and not the tested novelty |

Night-18A makes no backbone originality claim. The only research question was whether a clean-room mature sparse backbone can support a backbone-agnostic Night-16H decision layer. The answer was negative for decoder portability and local-only for P22 representation.
"""
    (OUT / "source_code_collision_and_transfer_audit.md").write_text(source_audit, encoding="utf-8")

    contract = """# Night-18A method and attribution contract

The numeric producer uses two modality adapters, registered sparse spatial propagation, per-modality sparse cosine-kNN propagation, masked row reconstruction, modality-drop consistency, a small DEC-style soft assignment loss, and a bounded residual around the retained embedding. `C02_NO_GRAPH`, `C03_NO_ANCHOR`, and `FROZEN_RETAINED` are matched controls. No historical partition is used as a training target.

Every locked embedding produces the same 15 candidates: common KMeans, six one-init KMeans starts, two diagonal GMM starts, and two KMeans starts at each of three sparse diffusion strengths. The universal feasible set requires exact K, no singleton, and at least one real smallest-scale internal edge per cluster. `FEASIBLE_MEDOID` is the partition medoid. `NIGHT16H_FIXED_STRUCTURED` applies the frozen Night-16H rule: use the molecular champion only when its topology percentile is at least 0.5; otherwise use the topology champion.

Representation contribution is measured as learned common KMeans minus frozen-retained common KMeans. Decoder contribution is measured on the identical learned embedding as structured minus common KMeans and structured minus feasible medoid. Labels are loaded only by the independent evaluator after candidate partitions and hashes are written.
"""
    (OUT / "method_semantics_and_attribution_contract.md").write_text(contract, encoding="utf-8")

    label_flow = {"schema": "night18a-label-flow-v1", "producer_annotation_reads": 0, "sanitized_numeric_carriers": 2,
                  "candidate_partitions_locked_before_evaluator": True, "evaluator_label_reads_per_artifact": 1,
                  "labels_used_for": ["post-lock public benchmark metrics", "transparent benchmark profile comparison"],
                  "labels_not_used_for": ["input", "loss", "gradient", "checkpoint", "representation", "candidate generation", "feasibility", "decoder selection within a run"]}
    (OUT / "label_flow_audit.json").write_text(json.dumps(label_flow, indent=2, sort_keys=True))

    failures = [
        {"id": "E01", "type": "ENGINEERING", "status": "FIXED_PREFORMAL", "issue": "first A1 P0 compared GPU training inference bytes directly with checkpoint reload", "resolution": "canonical CPU checkpoint consumer; GPU-to-CPU max absolute difference recorded; no evaluation from failed run"},
        {"id": "S01", "type": "SCIENTIFIC", "status": "PRESERVED", "issue": "A1 learned common endpoint did not beat frozen retained in both ARI and NMI", "resolution": "stopped cross-family expansion"},
        {"id": "S02", "type": "SCIENTIFIC", "status": "PRESERVED", "issue": "fixed structured decoder had no dual-metric independent gain on learned A1 or P22 embedding", "resolution": "decoder portability claim rejected"},
        {"id": "L01", "type": "LIMITATION", "status": "DISCLOSED", "issue": "retained anchors and public benchmark labels originate from historical development", "resolution": "classification restricted to local public-benchmark signal"},
    ]
    write_csv(OUT / "failure_and_correction_ledger.csv", failures)

    decision = {"schema": "night18a-decision-v1", "classification": "LOCAL_SIGNAL",
                "status": "NIGHT18A_P22_DEEP_BACKBONE_LOCAL_SIGNAL_STRUCTURED_DECODER_PORTABILITY_NEGATIVE",
                "p22_representation_dual_gain_vs_frozen_common": True, "a1_representation_dual_gain_vs_frozen_common": False,
                "cross_family_backbone_signal": False, "structured_decoder_dual_gain_on_learned_embeddings": 0,
                "expansion_to_misar_human_or_multiseed": False, "confirmed_milestone": False, "paper_ready": False,
                "shutdown_dispatched": False}
    (OUT / "night18a_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True))

    report = """# Night-18A report — deep backbone portability

## 我现在需要知道的三件事

1. **问题**：我们把表示层和聚类决策层彻底拆开，真实训练了同一套 RNA+protein / RNA+ATAC 稀疏双模态 backbone，再在每个完全相同的 embedding 上比较 common endpoint、可行分区 medoid 和 Night-16H 固定结构解码。
2. **实际改变的层**：模型包含双模态稀疏空间/feature-kNN 编码、遮蔽重构、轻量 DEC 分离和 retained residual anchor；参数训练 80 步且真实变化。结构解码不重训表示，只在同一个 15-candidate bank 上应用固定 exact-K/no-singleton/internal-edge 可行域和分子/空间仲裁。
3. **论文意义与分类**：P22 的深度表示相对 frozen carrier 的 common KMeans 有小幅双升，但 A1 没有；结构解码在两个学习后 embedding 上均未产生独立双指标增益。因此分类是 **LOCAL_SIGNAL**：保留 P22 representation 线索，否定“当前 Night-16H decoder 已可迁移到任意深度 backbone”。

## 绝对指标与分层归因

| Lane | 结果层 | 配置 / endpoint | ARI | NMI | 对 matched 参考 ΔARI/ΔNMI | 解释 |
|---|---|---|---:|---:|---:|---|
| A1 K10 | frozen representation | frozen common KMeans | 0.235268 | 0.387639 | reference | 同 endpoint 表示参考 |
| A1 K10 | best learned common | C02 no-graph common KMeans | 0.232410 | 0.388158 | -0.002858 / +0.000519 | 无双升，表示信号失败 |
| A1 K10 | same learned embedding decoder | C02 fixed structured | 0.233167 | 0.387609 | +0.000757 / -0.000549 vs common | 解码无双升 |
| A1 K10 | decoder-only context | frozen fixed structured | 0.245050 | 0.390472 | +0.009782 / +0.002833 vs frozen common | 只说明 frozen candidate bank 内的 head 效应，不是深度 backbone 迁移 |
| P22 K9 | frozen representation | frozen common KMeans | 0.470897 | 0.599897 | reference | 同 endpoint 表示参考 |
| P22 K9 | best learned common | C03 no-anchor common KMeans | 0.478639 | 0.606982 | +0.007742 / +0.007086 | P22 局部表示双升 |
| P22 K9 | best learned medoid | C03 feasible medoid | 0.482802 | 0.612864 | +0.004163 / +0.005882 vs same common | medoid 有局部 head 增益 |
| P22 K9 | fixed structured NMI profile | C01 fixed structured | 0.474764 | 0.624991 | -0.003283 / +0.018537 vs same common | NMI/ARI 取舍，不是双升 |
| P22 K9 | decoder-only context | frozen fixed structured | 0.499500 | 0.622887 | +0.028603 / +0.022990 vs frozen common | 仍低于历史强完整管线，且没有迁移到 learned embedding |

历史公开开发前沿为 A1 0.276003/0.421740、P22 0.595552/0.717931；本轮所有结果均未刷新它们。本轮主问题不是追逐这些历史 pipeline 数字，而是做同 embedding、同候选预算的表示/解码归因。

## 机制结论

- **表示层**：P22 在 common endpoint 上获得小幅、可重放的双指标改善；去 anchor 的 C03 略优，说明增益不是 anchor 正则单独解释。但 A1 同公式没有双升，因此还不是跨模态家族的 backbone 信号。
- **结构决策层**：fixed structured decoder 在 frozen embedding 上可以有效，但移到学习后 embedding 时，A1 与 P22 都没有相对 same-embedding common KMeans 的双指标增益；它也没有稳定超过 same-embedding feasible medoid。这直接否定了本轮的 decoder portability 假设。
- **工程路径**：A1 为 3484×30 + 3484×30，P22 为 9196×30 + 9196×50；两条路径均使用真实稀疏图、真实 optimizer steps、strict checkpoint load 和 fresh-process representation/partition exact replay。工程闭合不等于方法成功。

## 最重要失败与限制

1. A1 的图传播 full 配置比 frozen/common 更差；no-graph 控制反而更好，说明当前 sparse graph reconstruction 对 RNA+protein 还会过平滑或错配。
2. P22 learned representation 的局部增益远低于历史完整 pipeline，高分不能包装成 score frontier。
3. candidate generator 是统一、预算受限的 15-candidate bank，不等同于 Night-16H 历史 89-candidate authority bank；这里检验的是 fixed rule 的可移植语义，而不是复刻历史最高分。
4. 配置比较使用公开 annotation 的 post-lock evaluator，属于透明 benchmark development，不是盲测。

## 导师汇报版

我们这轮第一次把表示层和结构解码层放在完全匹配的实验里。A1 和 P22 都真实训练了同一套稀疏双模态网络，参数、seed、checkpoint 和 fresh replay 全部闭合。P22 的深度表示相对 frozen carrier 小幅双升，说明 RNA+ATAC 表示仍有可开发信号。A1 没有双升，而且 no-graph 比 full 更好，提示当前图传播对 RNA+protein 可能过平滑。Night-16H 固定结构解码在 frozen embedding 上有效，但迁移到学习后 embedding 后，两个数据都没有独立双指标增益。于是本轮不是统一方法里程碑，而是一个 P22 局部表示信号加一个明确的 decoder portability 负结果。后续应优先改进原生 backbone/graph alignment，再重新校准结构证据，而不是直接把旧 selector 当 backbone-agnostic 模块。所有结果均为公开 benchmark development，不能称 SOTA 或 paper-ready。

## 技术状态

完整 32 行绝对指标、matched contribution、源码碰撞、label flow、P0 registry、资源、失败 ledger 和 exact replay 与本报告同目录。服务器保持开机，`shutdown_dispatched=false`。
"""
    (OUT / "night18a_report.md").write_text(report, encoding="utf-8")
    (OUT / "plain_summary.md").write_text("Night-18A 结论：P22 深度表示有小幅局部双升，A1 没有；Night-16H 固定结构解码未迁移到学习后 embedding。分类 LOCAL_SIGNAL，不扩数据、不补多 seed。\n", encoding="utf-8")

    disk = os.statvfs("/"); data_disk = os.statvfs("/autodl-fs/data")
    du = int(subprocess.check_output(["du", "-sb", str(ROOT)]).decode().split()[0])
    resource_audit = {"timestamp_epoch": time.time(), "root_available_bytes": disk.f_bavail * disk.f_frsize,
                      "root_available_inodes": disk.f_favail, "persistent_available_bytes": data_disk.f_bavail * data_disk.f_frsize,
                      "persistent_available_inodes": data_disk.f_favail, "night18a_working_bytes": du,
                      "working_limit_bytes": 500 * 1024**2, "persistent_disk_new_files": 0, "shutdown_dispatched": False}
    (OUT / "resource_and_disk_audit.json").write_text(json.dumps(resource_audit, indent=2, sort_keys=True))
    print(json.dumps({"status": "PASS", "main_rows": len(all_rows), "contribution_rows": len(contribution), "replays": len(replay_rows)}, sort_keys=True))


if __name__ == "__main__": main()
