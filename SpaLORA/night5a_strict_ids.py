"""Generic strict observation-ID contracts; contains no real-label loader."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def strict_id_positions(model_ids: Sequence[str], evaluation_ids: Sequence[str],
                        expected_count: Optional[int] = None) -> np.ndarray:
    model = pd.Index(map(str, model_ids))
    evaluation = pd.Index(map(str, evaluation_ids))
    if model.has_duplicates or evaluation.has_duplicates:
        raise ValueError("Observation IDs must be unique")
    if expected_count is not None and (len(model) != int(expected_count) or len(evaluation) != int(expected_count)):
        raise ValueError("Observation count does not match strict interface")
    missing = model.difference(evaluation)
    extra = evaluation.difference(model)
    if len(missing) or len(extra):
        raise ValueError("Observation IDs are missing or extra")
    positions = evaluation.get_indexer(model)
    if np.any(positions < 0):
        raise ValueError("Observation ID alignment failed")
    return positions.astype(np.int64)


def d1_strict_positions(model_ids: Sequence[str], evaluation_ids: Sequence[str]) -> np.ndarray:
    return strict_id_positions(model_ids, evaluation_ids, expected_count=3359)


__all__ = ["strict_id_positions", "d1_strict_positions"]
