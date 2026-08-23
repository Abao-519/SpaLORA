#!/usr/bin/env python3
"""Independent study-balanced ranking for completed Night-13C runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED = ["A1", "tonsil_s1", "P22", "MISAR_E15_5_S1"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    files = []
    for root in map(Path, args.roots):
        files.extend(sorted(root.glob("*_summary.csv")))
    if not files:
        raise RuntimeError("no candidate summaries")
    rows = pd.concat([pd.read_csv(path).assign(source_summary=str(path)) for path in files],
                     ignore_index=True)
    rows = rows.sort_values("source_summary").drop_duplicates(
        ["candidate_id", "dataset", "training_seed"], keep="last")
    discovery = rows[(rows.training_seed == 0) & rows.dataset.isin(REQUIRED)].copy()
    counts = discovery.groupby("candidate_id").dataset.nunique()
    complete = counts[counts == len(REQUIRED)].index
    discovery = discovery[discovery.candidate_id.isin(complete)]
    records = []
    for candidate, table in discovery.groupby("candidate_id"):
        protein = table[table.dataset.isin(["A1", "tonsil_s1"])]
        atac = table[table.dataset.isin(["P22", "MISAR_E15_5_S1"])]
        protein_ari, protein_nmi = protein.delta_ari_mean.mean(), protein.delta_nmi_mean.mean()
        atac_ari, atac_nmi = atac.delta_ari_mean.mean(), atac.delta_nmi_mean.mean()
        protein_score = (protein_ari + protein_nmi) / 2.0
        atac_score = (atac_ari + atac_nmi) / 2.0
        records.append({
            "candidate_id": candidate, "mechanism": table.mechanism.iloc[0],
            "protein_delta_ari": protein_ari, "protein_delta_nmi": protein_nmi,
            "atac_delta_ari": atac_ari, "atac_delta_nmi": atac_nmi,
            "weak_family_score": min(protein_score, atac_score),
            "macro_score": (protein_score + atac_score) / 2.0,
            "worst_dataset_score": ((table.delta_ari_mean + table.delta_nmi_mean) / 2.0).min(),
            "mean_consensus_delta_ari": table.consensus_delta_ari.mean(),
            "mean_consensus_delta_nmi": table.consensus_delta_nmi.mean(),
            "all_four_complete": True,
        })
    ranking = pd.DataFrame(records).sort_values(
        ["weak_family_score", "macro_score", "worst_dataset_score"], ascending=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output, index=False)
    recommendation = {
        "ranking_source_files": [str(x) for x in files],
        "complete_candidate_count": len(ranking),
        "recommended_candidate": None if ranking.empty else ranking.iloc[0].candidate_id,
        "selection_rule": "weak family score, then macro, then worst dataset; public labels used only across completed runs",
        "labels_in_loss_gradient_or_within_run_checkpoint_selection": False,
    }
    output.with_suffix(".json").write_text(
        json.dumps(recommendation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(recommendation, sort_keys=True))


if __name__ == "__main__":
    main()
