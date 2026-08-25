#!/usr/bin/env python3
"""Build the frozen Night-17E scoreboard without modifying scientific rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LANES = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7")
CONTROL_ARMS = (
    "RELATION_DISABLED",
    "ZERO_RELATION",
    "UNIFORM_MASS_MATCHED",
    "PERMUTED_RELATION",
)
TOL = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def metric(row: dict[str, str], key: str) -> float:
    return float(row[key])


def baseline(rows: list[dict[str, str]]) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["start_id"] == "METHOD_NIGHT16H_START" and row["arm"] == "INPUT_START"
    ]
    if len(matches) != 1:
        raise ValueError("ambiguous Night-16H input authority")
    return matches[0]


def learned_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["arm"] == "LEARNED_RELATION" and row["status"] == "PASS"]


def choose_balanced(rows: list[dict[str, str]], reference: dict[str, str]) -> dict[str, str]:
    def key(row: dict[str, str]):
        d_ari = metric(row, "absolute_ari") - metric(reference, "absolute_ari")
        d_nmi = metric(row, "absolute_nmi") - metric(reference, "absolute_nmi")
        return (
            d_ari > TOL and d_nmi > TOL,
            min(d_ari, d_nmi),
            metric(row, "absolute_ari"),
            metric(row, "absolute_nmi"),
            row["candidate_id"],
        )
    return max(rows, key=key)


def profile_row(lane: str, profile: str, row: dict[str, str], reference: dict[str, str]) -> dict:
    output = dict(row)
    output.update(
        profile=profile,
        delta_ari_vs_night16h=metric(row, "absolute_ari") - metric(reference, "absolute_ari"),
        delta_nmi_vs_night16h=metric(row, "absolute_nmi") - metric(reference, "absolute_nmi"),
        night16h_ari=metric(reference, "absolute_ari"),
        night16h_nmi=metric(reference, "absolute_nmi"),
        lane=lane,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--loso-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    formal_root = Path(args.formal_root)
    loso_root = Path(args.loso_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    main_rows: list[dict] = []
    controls: list[dict] = []
    strict_rows: list[dict] = []
    gate_rows = []

    for lane in LANES:
        rows = read_csv(formal_root / lane / "evaluation.csv")
        if len(rows) != 126 or any(row["status"] != "PASS" for row in rows):
            raise ValueError(f"formal row contract failed for {lane}")
        all_rows.extend(rows)
        reference = baseline(rows)
        learned = learned_rows(rows)
        fixed = [
            row
            for row in learned
            if row["start_id"] == "METHOD_NIGHT16H_START" and row["config_id"] == "L02_MIX050"
        ]
        if len(fixed) != 1:
            raise ValueError("fixed global row missing")
        main_rows.append(profile_row(lane, "NIGHT16H_INPUT_AUTHORITY", reference, reference))
        main_rows.append(profile_row(lane, "FIXED_GLOBAL_L02", fixed[0], reference))
        balanced = choose_balanced(learned, reference)
        main_rows.append(profile_row(lane, "DIRECT_HPO_BALANCED", balanced, reference))
        max_ari = max(learned, key=lambda row: (metric(row, "absolute_ari"), metric(row, "absolute_nmi")))
        max_nmi = max(learned, key=lambda row: (metric(row, "absolute_nmi"), metric(row, "absolute_ari")))
        main_rows.append(profile_row(lane, "DIRECT_HPO_MAX_ARI", max_ari, reference))
        main_rows.append(profile_row(lane, "DIRECT_HPO_MAX_NMI", max_nmi, reference))

        selection = json.loads((loso_root / f"{lane}.selection.json").read_text(encoding="utf-8"))
        strict = [row for row in rows if row["candidate_id"] == selection["candidate_id"]]
        if len(strict) != 1 or strict[0]["partition_sha256"] != selection["partition_sha256"]:
            raise ValueError("strict LOSO selection/evaluation mismatch")
        strict_profile = profile_row(lane, "STRICT_LOSO", strict[0], reference)
        strict_rows.append(strict_profile)
        main_rows.append(strict_profile)

        matched = [
            row
            for row in rows
            if row["start_id"] == strict[0]["start_id"]
            and row["config_id"] == strict[0]["config_id"]
            and row["arm"] in CONTROL_ARMS + ("LEARNED_RELATION",)
        ]
        if len(matched) != 5:
            raise ValueError("matched-control set incomplete")
        learned_strict = [row for row in matched if row["arm"] == "LEARNED_RELATION"][0]
        learned_signature = learned_strict["partition_sha256"]
        equivalent_controls = [row["arm"] for row in matched if row["arm"] != "LEARNED_RELATION" and row["partition_sha256"] == learned_signature]
        dominating_controls = [
            row["arm"]
            for row in matched
            if row["arm"] != "LEARNED_RELATION"
            and metric(row, "absolute_ari") >= metric(learned_strict, "absolute_ari") - TOL
            and metric(row, "absolute_nmi") >= metric(learned_strict, "absolute_nmi") - TOL
            and (
                metric(row, "absolute_ari") > metric(learned_strict, "absolute_ari") + TOL
                or metric(row, "absolute_nmi") > metric(learned_strict, "absolute_nmi") + TOL
            )
        ]
        for row in matched:
            value = profile_row(lane, "STRICT_LOSO_MATCHED_CONTROL", row, reference)
            value["delta_ari_vs_learned"] = metric(row, "absolute_ari") - metric(learned_strict, "absolute_ari")
            value["delta_nmi_vs_learned"] = metric(row, "absolute_nmi") - metric(learned_strict, "absolute_nmi")
            controls.append(value)
        d_ari = strict_profile["delta_ari_vs_night16h"]
        d_nmi = strict_profile["delta_nmi_vs_night16h"]
        gate_rows.append(
            {
                "lane": lane,
                "dual_positive_vs_night16h": d_ari > TOL and d_nmi > TOL,
                "third_lane_within_drawdown_bound": not (d_ari < -0.01 and d_nmi < -0.01),
                "learned_partition_changed_vs_disabled": learned_signature
                != [row for row in matched if row["arm"] == "RELATION_DISABLED"][0]["partition_sha256"],
                "equivalent_controls": equivalent_controls,
                "dominating_controls": dominating_controls,
                "control_independent": not equivalent_controls and not dominating_controls,
                "min_cluster_size_full": int(float(strict_profile["min_cluster_size_full"])),
            }
        )

    write_csv(output_root / "all_candidate_hpo_ledger.csv", all_rows)
    write_csv(output_root / "absolute_metrics_main_table.csv", main_rows)
    write_csv(output_root / "strict_loso_transfer_table.csv", strict_rows)
    write_csv(output_root / "matched_control_table.csv", controls)

    gate = {
        "schema": "night17e-mechanism-gate-v1",
        "strict_loso_dual_positive_lanes": sum(row["dual_positive_vs_night16h"] for row in gate_rows),
        "strict_loso_within_drawdown_bound_lanes": sum(row["third_lane_within_drawdown_bound"] for row in gate_rows),
        "strict_loso_control_independent_lanes": sum(row["control_independent"] for row in gate_rows),
        "required_dual_positive_lanes": 2,
        "required_control_independent_lanes": 2,
        "gate_passed": (
            sum(row["dual_positive_vs_night16h"] for row in gate_rows) >= 2
            and all(row["third_lane_within_drawdown_bound"] for row in gate_rows)
            and sum(row["control_independent"] for row in gate_rows) >= 2
        ),
        "lane_details": gate_rows,
        "classification_if_failed": "SCIENTIFIC_NEGATIVE",
        "expansion_allowed": False,
    }
    (output_root / "mechanism_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
