#!/usr/bin/env python3
"""Build deterministic bounded Night-16E family screen registries."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


POSITIVE_BLEND = {
    "beta",
    "edge_floor",
    "conflict_temperature",
    "conflict_union_weight",
    "conflict_penalty",
    "mass_temperature",
    "neighbor_capacity",
    "low_weight",
    "twohop_weight",
    "high_weight",
    "unary_temperature",
    "trust_scale",
    "trust_temperature",
    "move_threshold",
    "move_fraction",
    "sweeps",
    "pairwise_beta",
    "self_return_strength",
    "size_prior",
    "scale_fine",
    "scale_registered",
    "scale_broad",
    "expansion_cycles",
}


def blend_number(first: float, second: float, key: str) -> float:
    if key in POSITIVE_BLEND and first > 0 and second > 0:
        return float(np.sqrt(float(first) * float(second)))
    return 0.5 * (float(first) + float(second))


def blend_base(first: dict[str, object], second: dict[str, object]) -> dict[str, object]:
    output = copy.deepcopy(first)
    for key in output:
        if key == "local":
            for local_key in output[key]:
                output[key][local_key] = blend_number(
                    first[key][local_key], second[key][local_key], local_key
                )
        elif key in second and isinstance(output[key], (int, float)):
            output[key] = blend_number(first[key], second[key], key)
    output["expansion_cycles"] = max(1, int(round(float(output["expansion_cycles"]))))
    output["local"]["sweeps"] = max(1, int(round(float(output["local"]["sweeps"]))))
    return output


def conservative(base: dict[str, object]) -> dict[str, object]:
    value = copy.deepcopy(base)
    value["expansion_cycles"] = 1
    value["pairwise_beta"] = min(float(value["pairwise_beta"]), 12.0)
    value["self_return_strength"] = max(float(value["self_return_strength"]), 4.0)
    value["size_prior"] = min(float(value["size_prior"]), 0.20)
    value["local"]["trust_scale"] = max(float(value["local"]["trust_scale"]), 0.50)
    value["local"]["retained_bias"] = max(float(value["local"]["retained_bias"]), 1.0)
    return value


def make_candidate(base_id: str, variant: str, config: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256((base_id + variant + canonical).encode()).hexdigest()[:14]
    return {
        "candidate_id": f"{variant}_{digest}",
        "profile_id": f"PROFILE_{digest}",
        "base_id": base_id,
        "variant": variant,
        "config": config,
    }


def tsre_config(
    base: dict[str, object],
    support_mix: float,
    relation_temperature: float,
    boundary_strength: float,
    private_strength: float,
    relation_stay_strength: float,
) -> dict[str, object]:
    return {
        "base": copy.deepcopy(base),
        "support_mix": float(support_mix),
        "relation_temperature": float(relation_temperature),
        "boundary_strength": float(boundary_strength),
        "private_strength": float(private_strength),
        "relation_stay_strength": float(relation_stay_strength),
    }


def build(args: argparse.Namespace) -> None:
    frozen = json.loads(Path(args.night15f_registry).read_text())["lanes"]
    if args.family == "RNA_PROTEIN":
        first = frozen["A1"]["config"]
        second = frozen["tonsil_s1"]["config"]
        midpoint = blend_base(first, second)
        bases = {
            "P_A1": first,
            "P_S1": second,
            "P_GEO": midpoint,
            "P_CONSERVATIVE": conservative(midpoint),
        }
        seed = 160501
    else:
        first = frozen["P22"]["config"]
        second = frozen["MISAR_E15_5_S1"]["config"]
        midpoint = blend_base(first, second)
        bases = {
            "C_P22": first,
            "C_GEO": midpoint,
            "C_CONSERVATIVE": conservative(first),
        }
        seed = 160502

    candidates = [
        {
            "candidate_id": "INPUT_STRONG_START",
            "profile_id": "INPUT_STRONG_START",
            "base_id": "NONE",
            "variant": "INPUT_STRONG_START",
            "config": None,
        }
    ]
    for base_id, base in bases.items():
        neutral = tsre_config(base, 0.0, 1.0, 0.0, 0.0, 0.0)
        candidates.append(make_candidate(base_id, "NIGHT15F_DIRECT", neutral))
        for support_mix in (0.25, 0.60, 1.0):
            support = tsre_config(base, support_mix, 1.0, 0.0, 0.0, 0.0)
            candidates.append(make_candidate(base_id, "TSRE_SUPPORT_ONLY", support))
        for strength in (0.05, 0.20, 0.80):
            full = tsre_config(base, 0.60, 1.0, strength, 0.5 * strength, 2.0 * strength)
            candidates.append(make_candidate(base_id, "TSRE_FULL", full))

    rng = np.random.default_rng(seed)
    target = int(args.candidate_count)
    base_items = list(bases.items())
    while len(candidates) < target:
        base_id, base = base_items[int(rng.integers(0, len(base_items)))]
        support_mix = float(rng.uniform(0.10, 1.0))
        relation_temperature = float(np.exp(rng.uniform(np.log(0.45), np.log(2.2))))
        boundary_strength = float(np.exp(rng.uniform(np.log(0.015), np.log(5.0))))
        private_strength = float(np.exp(rng.uniform(np.log(0.01), np.log(2.5))))
        relation_stay_strength = float(np.exp(rng.uniform(np.log(0.05), np.log(12.0))))
        config = tsre_config(
            base,
            support_mix,
            relation_temperature,
            boundary_strength,
            private_strength,
            relation_stay_strength,
        )
        candidates.append(make_candidate(base_id, "TSRE_FULL", config))

    output = {
        "schema": "night16e-bounded-family-screen-v1",
        "family": args.family,
        "seed": seed,
        "candidate_count": len(candidates),
        "labels_used_to_generate_configs": 0,
        "candidates": candidates,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--night15f-registry", required=True)
    parser.add_argument("--family", choices=("RNA_PROTEIN", "RNA_CHROMATIN"), required=True)
    parser.add_argument("--candidate-count", type=int, default=80)
    parser.add_argument("--output", required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()

