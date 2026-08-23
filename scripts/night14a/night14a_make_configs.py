#!/usr/bin/env python3
"""Generate the frozen Night-14A development filter registry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def record(identifier, variant, max_low, max_high=0.0, global_center=0.4,
           conflict_scale=8.0):
    return {
        "conflict_scale": float(conflict_scale),
        "filter_id": identifier,
        "global_center": float(global_center),
        "global_scale": 10.0,
        "high_center": 0.30,
        "max_high": float(max_high),
        "max_low": float(max_low),
        "node_center": 0.40,
        "node_scale": 8.0,
        "roughness_center": 0.23,
        "roughness_scale": 0.0,
        "support_center": 0.90,
        "support_scale": 20.0,
        "variant": variant,
    }


def registry():
    rows = [record("D00_IDENTITY", "IDENTITY", 0.0, conflict_scale=0.0)]
    for index, maximum in enumerate((0.10, 0.20, 0.30, 0.40, 0.50, 0.60), 1):
        rows.append(record("D%02d_FIXED_L%02d" % (index, round(100 * maximum)),
                           "FIXED_LOW", maximum, conflict_scale=0.0))
    for offset, maximum in enumerate((0.20, 0.30, 0.40, 0.50, 0.60), 7):
        rows.append(record("D%02d_SUPPORT_L%02d" % (offset, round(100 * maximum)),
                           "SUPPORT_LOW", maximum, conflict_scale=0.0))
    index = 12
    for center in (0.30, 0.35, 0.40, 0.45):
        for maximum in (0.30, 0.40, 0.50, 0.60):
            rows.append(record(
                "D%02d_TCF_C%02d_L%02d" %
                (index, round(100 * center), round(100 * maximum)),
                "TCF_LOW_HIGH", maximum, 0.0, center, 8.0,
            ))
            index += 1
    for center in (0.35, 0.40):
        for maximum in (0.40, 0.50):
            for high in (0.05, 0.10):
                rows.append(record(
                    "D%02d_TCF_C%02d_L%02d_H%02d" %
                    (index, round(100 * center), round(100 * maximum), round(100 * high)),
                    "TCF_LOW_HIGH", maximum, high, center, 8.0,
                ))
                index += 1
    if len(rows) != 36 or len({x["filter_id"] for x in rows}) != 36:
        raise RuntimeError("development registry cardinality mismatch")
    return rows


def targeted_registry():
    """Adaptive second screen after the first grid exposed scale compression."""
    rows = [record("R00_IDENTITY", "IDENTITY", 0.0, conflict_scale=0.0),
            record("R01_FIXED_L60", "FIXED_LOW", 0.60, conflict_scale=0.0),
            record("R02_SUPPORT_L60", "SUPPORT_LOW", 0.60, conflict_scale=0.0)]
    index = 3
    for scale in (20.0, 30.0, 40.0):
        for center in (0.40, 0.45, 0.50):
            for maximum in (0.60, 0.80, 1.00):
                row = record(
                    "R%02d_TCF_G%02d_C%02d_L%03d" %
                    (index, round(scale), round(100 * center), round(100 * maximum)),
                    "TCF_LOW_HIGH", maximum, 0.0, center, 8.0,
                )
                row["global_scale"] = scale
                row["roughness_scale"] = 30.0
                row["support_center"] = 0.75
                row["support_scale"] = 10.0
                rows.append(row)
                index += 1
    if len(rows) != 30 or len({x["filter_id"] for x in rows}) != 30:
        raise RuntimeError("targeted registry cardinality mismatch")
    return rows


def final_registry():
    """Focused roughness/conflict interaction grid before the formal freeze."""
    rows = [record("Q00_IDENTITY", "IDENTITY", 0.0, conflict_scale=0.0),
            record("Q01_FIXED_L60", "FIXED_LOW", 0.60, conflict_scale=0.0)]
    index = 2
    for global_scale in (20.0, 30.0, 40.0):
        for roughness_scale in (40.0, 60.0, 80.0):
            for roughness_center in (0.22, 0.23, 0.24):
                row = record(
                    "Q%02d_TCF_G%02d_R%02d_RC%02d" %
                    (index, round(global_scale), round(roughness_scale),
                     round(100 * roughness_center)),
                    "TCF_LOW_HIGH", 1.0, 0.0, 0.40, 8.0,
                )
                row["global_scale"] = global_scale
                row["roughness_scale"] = roughness_scale
                row["roughness_center"] = roughness_center
                row["support_center"] = 0.75
                row["support_scale"] = 10.0
                rows.append(row)
                index += 1
    if len(rows) != 29 or len({x["filter_id"] for x in rows}) != 29:
        raise RuntimeError("final filter registry cardinality mismatch")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=("broad", "targeted", "final"),
                        default="broad")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "broad":
        rows = registry()
    elif args.kind == "targeted":
        rows = targeted_registry()
    else:
        rows = final_registry()
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
