"""Loss-decomposition helpers for the gated Night-2 causal audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch


LOSS_LOG_FIELDS = (
    "epoch",
    "raw_rna_reconstruction",
    "weighted_rna_before_global_scale",
    "final_rna_contribution",
    "raw_modality2_reconstruction",
    "final_modality2_contribution",
    "raw_corr1",
    "final_corr1_contribution",
    "raw_corr2",
    "final_corr2_contribution",
    "total_loss",
    "global_scale_multiplier",
    "m_bad",
    "gene_weight_mean",
    "gene_weight_min",
    "gene_weight_max",
)


def legacy_bug_weight_vector(features_omics1: torch.Tensor) -> torch.Tensor:
    """Reproduce the public argsort bug exactly; this is not an abundance score."""
    if features_omics1.ndim != 2:
        raise ValueError("features_omics1 must be a 2D locations-by-genes tensor")
    d = features_omics1.shape[1]
    average = features_omics1.mean(dim=0)
    bad_expr_percentile = torch.argsort(average, descending=False).to(features_omics1.dtype) / d
    return 1.0 + 5.0 * torch.sigmoid(-10.0 * (bad_expr_percentile - 0.25))


def per_gene_mse(diff: torch.Tensor) -> torch.Tensor:
    return torch.mean(diff ** 2, dim=0)


def loss_decomposition(per_gene: torch.Tensor, legacy_bug_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
    m_bad = torch.mean(legacy_bug_weights)
    unweighted = torch.mean(per_gene)
    uniform_legacy_scale = m_bad * unweighted
    legacy_shape_normalized = torch.sum(legacy_bug_weights * per_gene) / torch.sum(legacy_bug_weights)
    legacy_loss_replay = torch.mean(legacy_bug_weights * per_gene)
    return {
        "m_bad": m_bad,
        "corrected_unweighted": unweighted,
        "uniform_legacy_scale": uniform_legacy_scale,
        "legacy_shape_normalized": legacy_shape_normalized,
        "legacy_loss_replay": legacy_loss_replay,
    }


def required_checkpoint_epochs(epochs: int) -> List[int]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    final = epochs - 1
    return sorted({0, int(round(0.1 * final)), int(round(0.5 * final)), final})


@dataclass
class LossDynamicsRecorder:
    epochs: int
    records: List[dict] = field(default_factory=list)

    def should_record(self, epoch: int) -> bool:
        return epoch in required_checkpoint_epochs(self.epochs)

    def add(self, record: dict) -> None:
        missing = set(LOSS_LOG_FIELDS) - set(record)
        if missing:
            raise AssertionError("Incomplete Night-2 loss log: %s" % sorted(missing))
        self.records.append({name: record[name] for name in LOSS_LOG_FIELDS})

    def assert_complete(self) -> None:
        observed = sorted(int(item["epoch"]) for item in self.records)
        expected = required_checkpoint_epochs(self.epochs)
        if observed != expected:
            raise AssertionError("Loss log epochs %r != required %r" % (observed, expected))
