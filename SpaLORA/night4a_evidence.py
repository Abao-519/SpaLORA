"""Night-4A evidence-repair primitives.

These helpers operate only on locked Night-3B tabular outputs.  They do not
train a model and intentionally keep metric direction separate from the
numeric delta definition (FULL_IGE - ablation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MetricDirection:
    metric: str
    higher_is_better: bool

    @property
    def full_win_rule(self) -> str:
        return "delta > 0" if self.higher_is_better else "delta < 0"

    def full_wins(self, values: Iterable[float]) -> int:
        array = np.asarray(list(values), dtype=float)
        array = array[np.isfinite(array)]
        return int(np.sum(array > 0)) if self.higher_is_better else int(np.sum(array < 0))


METRIC_DIRECTIONS = {
    "ari": MetricDirection("ari", True),
    "nmi": MetricDirection("nmi", True),
    "spatial_neighbor_agreement": MetricDirection("spatial_neighbor_agreement", True),
    "spatial_cluster_moran_mean": MetricDirection("spatial_cluster_moran_mean", True),
    "spatial_cluster_geary_mean": MetricDirection("spatial_cluster_geary_mean", False),
    "boundary_disagreement": MetricDirection("boundary_disagreement", False),
}


def corrected_count(values: Iterable[float], metric: str) -> dict:
    """Return unambiguous counts for a FULL-minus-ablation delta vector."""
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    direction = METRIC_DIRECTIONS[metric]
    return {
        "n_finite": int(len(finite)),
        "n_missing": int(len(array) - len(finite)),
        "positive_delta_count": int(np.sum(finite > 0)),
        "negative_delta_count": int(np.sum(finite < 0)),
        "tie_count": int(np.sum(finite == 0)),
        "full_win_count_corrected": direction.full_wins(finite),
        "direction": "higher_is_better" if direction.higher_is_better else "lower_is_better",
        "full_win_rule": direction.full_win_rule,
    }

