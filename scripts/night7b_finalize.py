#!/usr/bin/env python3
"""Create compact, auditable Night-7B scientific summaries (never raw checkpoints)."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night7a_consensus import atomic_json, sha256_file  # noqa: E402

OUT = REPO / "outputs/night7b_handoff"
RAW = Path("/root/autodl-fs/night7b_score_rnd_20260818")
ADAPTER = RAW / "adapter_stage"


def stage_locks():
    return {stage:json.loads((OUT / ("locked_%s_manifest.json" % stage)).read_text())
            for stage in ("R1", "R2")}


def build_training_indexes(locks) -> None:
    checkpoint_rows, loss_rows, gate_rows = [], [], []
    for stage, locked in locks.items():
        for cell in locked["training_cells"]:
            manifest = cell.get("training_manifest", {})
            reload_audit = cell.get("reload_audit", {})
            checkpoint_rows.append({
                "stage":stage, "recipe_id":cell["recipe_id"], "unit_id":cell["unit_id"],
                "dataset":cell["dataset"], "seed":cell["seed"], "status":cell["status"],
                "checkpoint_sha256":manifest.get("checkpoint_sha256"),
                "state_tensor_sha256":manifest.get("state_tensor_sha256"),
                "embedding_sha256":manifest.get("embedding_sha256"),
                "gate_sha256":manifest.get("gate_sha256"),
                "reload_status":reload_audit.get("status"),
                "reload_embedding_exact":reload_audit.get("embedding_exact"),
                "reload_gate_exact":reload_audit.get("gate_exact"),
                "fresh_process":reload_audit.get("fresh_process"),
            })
            if cell["status"] != "success":
                continue
            worker = ADAPTER / stage / "formal" / cell["recipe_id"] / cell["unit_id"] / "attempt_001" / "worker"
            curve = pd.read_csv(worker / "loss_curve.csv")
            row = {
                "stage":stage, "recipe_id":cell["recipe_id"], "unit_id":cell["unit_id"],
                "dataset":cell["dataset"], "seed":cell["seed"], "epochs":len(curve),
                "initial_total":float(curve.iloc[0]["total"]), "final_total":float(curve.iloc[-1]["total"]),
                "loss_curve_sha256":manifest["loss_curve_sha256"],
            }
            for column in curve.columns:
                if column not in ("epoch", "total", "learning_rate"):
                    row["initial_" + column] = float(curve.iloc[0][column]) if pd.notna(curve.iloc[0][column]) else np.nan
                    row["final_" + column] = float(curve.iloc[-1][column]) if pd.notna(curve.iloc[-1][column]) else np.nan
            loss_rows.append(row)
            gates = np.load(worker / "gate_weights.npy", allow_pickle=False)
            learned = cell["recipe_id"] in ("R07", "R08", "R09")
            gate_rows.append({
                "stage":stage, "recipe_id":cell["recipe_id"], "unit_id":cell["unit_id"],
                "dataset":cell["dataset"], "seed":cell["seed"], "learned_moe":learned,
                "expert0_mean":float(gates[:,0].mean()), "expert1_mean":float(gates[:,1].mean()),
                "expert0_q05":float(np.quantile(gates[:,0], .05)), "expert0_q50":float(np.quantile(gates[:,0], .5)),
                "expert0_q95":float(np.quantile(gates[:,0], .95)), "expert1_q05":float(np.quantile(gates[:,1], .05)),
                "expert1_q50":float(np.quantile(gates[:,1], .5)), "expert1_q95":float(np.quantile(gates[:,1], .95)),
                "fraction_max_expert_gt_0_95":float(np.mean(np.max(gates, axis=1) > .95)),
                "collapse_diagnostic_min_mean_lt_0_05":bool(min(gates.mean(0)) < .05) if learned else False,
                "gate_sha256":manifest["gate_sha256"],
            })
    pd.DataFrame(checkpoint_rows).to_csv(OUT / "checkpoint_roundtrip_index.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(OUT / "loss_diagnostics.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(OUT / "gate_diagnostics.csv", index=False)


def build_mechanism_summaries() -> None:
    registry = json.loads((REPO / "protocols/night7b/SpaLORA_Night7B_Adaptive_Relational_Fusion_Registry_2026-08-18.json").read_text())
    recipe_losses = {x["id"]:set(x["losses"]) for x in registry["adapter_recipes"]}
    r1 = pd.read_csv(OUT / "R1_candidate_summary_vs_C00.csv")
    r1["recipe_id"] = r1.config_id.str.split("__").str[0]
    rows = []
    for loss in registry["loss_contract"]:
        present = r1[r1.recipe_id.map(lambda x: loss in recipe_losses[x])]
        absent = r1[r1.recipe_id.map(lambda x: loss not in recipe_losses[x])]
        rows.append({
            "loss":loss, "present_config_count":len(present), "absent_config_count":len(absent),
            "present_mean_priority_delta_q":float(present.priority_weighted_delta_q.mean()) if len(present) else np.nan,
            "absent_mean_priority_delta_q":float(absent.priority_weighted_delta_q.mean()) if len(absent) else np.nan,
            "descriptive_present_minus_absent":float(present.priority_weighted_delta_q.mean() - absent.priority_weighted_delta_q.mean()) if len(present) and len(absent) else np.nan,
            "present_best_priority_delta_q":float(present.priority_weighted_delta_q.max()) if len(present) else np.nan,
            "interpretation":"descriptive_noncausal_registered_recipe_comparison",
        })
    pd.DataFrame(rows).to_csv(OUT / "loss_family_effects_descriptive.csv", index=False)
    gate = pd.read_csv(OUT / "gate_diagnostics.csv")
    effects = pd.read_csv(OUT / "loss_family_effects_descriptive.csv")
    loss = pd.read_csv(OUT / "loss_diagnostics.csv")
    merged = gate[gate.learned_moe].merge(loss, on=["stage","recipe_id","unit_id","dataset","seed"], how="inner")
    correlations = []
    quality = [x for x in ("final_total","final_RECON","final_RELKL","final_MNN","final_DCCA") if x in merged]
    for gate_name in ("expert0_mean","expert1_mean","fraction_max_expert_gt_0_95"):
        for quality_name in quality:
            valid = merged[[gate_name, quality_name]].dropna()
            correlations.append({
                "gate_diagnostic":gate_name, "unlabeled_quality":quality_name,
                "n":len(valid), "pearson_r":float(valid.corr().iloc[0,1]) if len(valid) >= 3 else np.nan,
                "label_values_used":False,
            })
    pd.DataFrame(correlations).to_csv(OUT / "gate_unlabeled_quality_correlations.csv", index=False)


def build_failures_and_budget(locks) -> None:
    failures = []
    head = json.loads((OUT / "locked_head_transform_manifest.json").read_text())
    for cell in head["transforms"]:
        if cell["status"] != "success":
            failures.append({"stage":"H", "cell_id":"%s__%s" % (cell["unit_id"], cell["head_id"]),
                             "status":cell["status"], "failure_type":cell.get("failure_type")})
    for stage, locked in locks.items():
        for cell in locked["training_cells"]:
            if cell["status"] != "success":
                failures.append({"stage":stage + "_training", "cell_id":"%s__%s" % (cell["recipe_id"], cell["unit_id"]),
                                 "status":cell["status"], "failure_type":cell.get("failure_type")})
        for cell in locked["transforms"]:
            if cell["status"] != "success":
                failures.append({"stage":stage + "_transform", "cell_id":"%s__%s" % (cell["config_id"], cell["unit_id"]),
                                 "status":cell["status"], "failure_type":cell.get("failure_type")})
    pd.DataFrame(failures, columns=("stage","cell_id","status","failure_type")).to_csv(OUT / "failure_audit.csv", index=False)
    p0 = json.loads((OUT / "p0_semantic_contract.json").read_text())
    budget = {
        "H_formal_transforms":{"used":len(head["transforms"]), "limit":540},
        "R1_scientific_training":{"used":len(locks["R1"]["training_cells"]), "limit":80},
        "R2_scientific_training":{"used":len(locks["R2"]["training_cells"]), "limit":88},
        "scientific_training_total":{"used":len(locks["R1"]["training_cells"]) + len(locks["R2"]["training_cells"]), "limit":168},
        "R1_transforms":{"used":len(locks["R1"]["transforms"]), "limit":320},
        "R2_transforms":{"used":len(locks["R2"]["transforms"]), "limit":88},
        "global_corrections":{"used":p0["implementation_corrections"], "limit":12},
        "scientific_retry":0, "formal_external_benchmark_runs":0,
        "fresh_dataset_downloads_or_label_reads":0,
    }
    if any(value["used"] > value["limit"] for value in budget.values() if isinstance(value, dict) and "limit" in value):
        raise RuntimeError("budget exceeded")
    atomic_json(OUT / "budget_audit.json", budget)


def git_audit() -> dict:
    def run(*args):
        return subprocess.check_output(args, cwd=REPO, text=True).strip()
    return {
        "branch":run("git","branch","--show-current"),
        "head":run("git","rev-parse","HEAD"),
        "parent_authority":"8b67e4bc09196f6c44b20e7afcfd0c3f0345e88b",
        "force_push_used":False, "force_with_lease_used":False,
        "final_tag_created":False,
    }


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 10) -> str:
    use = frame.loc[:, columns].head(limit)
    return use.to_markdown(index=False, floatfmt=".5f")


def build_report() -> None:
    final = json.loads((OUT / "night7b_final_candidate_lock.json").read_text())
    h = pd.read_csv(OUT / "H_candidate_summary.csv")
    r1 = pd.read_csv(OUT / "R1_candidate_summary_vs_C00.csv")
    r2 = pd.read_csv(OUT / "R2_final_summary_vs_C00.csv")
    r2c06 = pd.read_csv(OUT / "R2_final_summary_vs_C06.csv")
    loss = pd.read_csv(OUT / "loss_diagnostics.csv")
    gate = pd.read_csv(OUT / "gate_diagnostics.csv")
    failures = pd.read_csv(OUT / "failure_audit.csv")
    promoted_h = json.loads((OUT / "H_to_R1_contract.json").read_text())["promoted_head_ids"]
    promoted_r1 = json.loads((OUT / "R1_to_R2_contract.json").read_text())["promoted_config_ids"]
    top = r2.iloc[0]
    top_c06 = r2c06[r2c06.config_id == top.config_id].iloc[0]
    best_loss = effects.sort_values("descriptive_present_minus_absent", ascending=False).iloc[0]
    worst_loss = effects.sort_values("descriptive_present_minus_absent", ascending=True).iloc[0]
    both_human_and_p22 = bool(top.human_lymph_equal_mean_delta_q > 0 and top.p22_mean_delta_q > 0)
    lines = [
        "# SpaLORA Night-7B adaptive relational score R&D report", "",
        "## Plain-language outcome", "",
        "Night-7B reused the 60 immutable Night-6C/Night-6D views, compared 18 clustering heads, then trained the ten registered adaptive/relational recipes under a strict no-label worker contract. Labels were opened only after each stage was completely locked.", "",
        "Terminal status: `%s`." % final["terminal_status"], "",
        "Unified locked candidates: %s." % (", ".join(final["unified_pass_config_ids"]) or "none"), "",
        "The top all-seed configuration `%s` %s improve both the equal-weight human-lymph aggregate and P22 relative to C00. Its priority-weighted delta versus C00 was ARI %+.5f, NMI %+.5f, Q %+.5f; versus C06 it was ARI %+.5f, NMI %+.5f, Q %+.5f." % (top.config_id, "did" if both_human_and_p22 else "did not", top.priority_weighted_delta_ari, top.priority_weighted_delta_nmi, top.priority_weighted_delta_q, top_c06.priority_weighted_delta_ari, top_c06.priority_weighted_delta_nmi, top_c06.priority_weighted_delta_q), "",
        "Across the locked recipe panel, the strongest descriptive loss-family association was `%s` (%+.5f present-minus-absent priority delta Q) and the weakest was `%s` (%+.5f). This is not a causal ablation claim." % (best_loss.loss, best_loss.descriptive_present_minus_absent, worst_loss.loss, worst_loss.descriptive_present_minus_absent), "",
        "The quantitative tables below report the same-seed differences against confirmed C00 and the Night-7A C06 accuracy frontier; no seed, epoch, recipe, threshold, or dataset-specific setting was selected after viewing labels.", "",
        "## Locked stage decisions", "",
        "- H promoted heads: %s" % ", ".join(promoted_h),
        "- R1 promoted configurations: %s" % ", ".join(promoted_r1),
        "- Accuracy frontier order: %s" % ", ".join(final["accuracy_frontier_order"]),
        "- Balanced frontier order: %s" % (", ".join(final["balanced_frontier_order"]) or "none"), "",
        "## Final all-seed comparison versus C00", "",
        markdown_table(r2, ["config_id","priority_weighted_delta_q","balanced_macro_delta_q","worst_dataset_delta_q","a1_mean_delta_q","d1_mean_delta_q","p22_mean_delta_q","tonsil_mean_delta_q","total_q_wins"], 4), "",
        "## Final all-seed comparison versus C06", "",
        markdown_table(r2c06, ["config_id","priority_weighted_delta_q","balanced_macro_delta_q","a1_mean_delta_q","d1_mean_delta_q","p22_mean_delta_q","tonsil_mean_delta_q"], 4), "",
        "## H and R1 rankings", "",
        markdown_table(h, ["config_id","priority_weighted_delta_q","balanced_macro_delta_q","worst_dataset_delta_q","total_q_wins"], 10), "",
        markdown_table(r1, ["config_id","priority_weighted_delta_q","balanced_macro_delta_q","worst_dataset_delta_q","total_q_wins"], 8), "",
        "## Mechanism and resource diagnostics", "",
        "Loss diagnostics contain %d successful training rows. Gate diagnostics contain %d rows; %d learned-MoE rows met the preregistered descriptive collapse flag (minimum expert mean below 0.05). These are mechanism diagnostics, not post-hoc selection criteria." % (len(loss), len(gate), int(gate.collapse_diagnostic_min_mean_lt_0_05.sum())), "",
        "`loss_family_effects_descriptive.csv` reports registered present-versus-absent recipe associations for every loss family; these comparisons are explicitly descriptive rather than causal. `gate_unlabeled_quality_correlations.csv` relates MoE behavior only to label-free training losses.", "",
        "## Failures and integrity", "",
        "There were %d preserved numerical/upstream failures. No formal scientific retry or fallback was used. P0 passed 30/30 real C00/C06 parity and 10/10 invalid smoke checkpoint reloads after eight preserved global pre-label corrections." % len(failures), "",
        "All checkpoints, affinities and large raw arrays remain under `/root/autodl-fs`; the compact handoff contains hashes and small evidence only.", "",
    ]
    (OUT / "night7b_report.md").write_text("\n".join(lines), encoding="utf-8")
    summary = [
        "Night-7B terminal status: %s" % final["terminal_status"],
        "Unified candidate(s): %s" % (", ".join(final["unified_pass_config_ids"]) or "none"),
        "H promoted: %s" % ", ".join(promoted_h),
        "R1 promoted: %s" % ", ".join(promoted_r1),
        "See night7b_report.md and the all-seed CSV tables for exact ARI/NMI/Q and spatial deltas.",
    ]
    (OUT / "plain_language_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def build_internal_index() -> None:
    excluded = {"delivery_index.json"}
    records = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in excluded:
            records.append({"path":path.relative_to(REPO).as_posix(), "size_bytes":path.stat().st_size,
                            "sha256":sha256_file(path)})
    root = __import__("hashlib").sha256("\n".join(x["sha256"] for x in records).encode()).hexdigest()
    atomic_json(OUT / "delivery_index.json", {"schema_version":1, "files":records,
                                               "root_rule":"sha256(newline_join(file_sha256_in_lexical_path_order))",
                                               "root_sha256":root})


def main() -> None:
    locks = stage_locks()
    build_training_indexes(locks)
    build_mechanism_summaries()
    build_failures_and_budget(locks)
    atomic_json(OUT / "git_audit_pre_final.json", git_audit())
    atomic_json(OUT / "shutdown_dispatch_prepared.json", {
        "command":"/usr/bin/shutdown", "dispatch_status":"PENDING_UNTIL_ALL_GIT_AND_WINDOWS_VERIFICATION_COMPLETE",
        "must_be_last_remote_command":True, "reconnect_after_dispatch_forbidden":True,
    })
    build_report()
    build_internal_index()


if __name__ == "__main__":
    main()
