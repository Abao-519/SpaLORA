"""Pure fail-closed resolver for the frozen Night-10B family policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .family_recipes import FROZEN_RECIPES, FrozenRecipe, canonical_json_sha256


class UnsupportedAssayPair(ValueError):
    """The explicit assay pair is absent, ambiguous, reversed, or unsupported."""


@dataclass(frozen=True)
class ResolvedPolicy:
    recipe: FrozenRecipe

    def as_dict(self) -> dict[str, Any]:
        value = self.recipe.as_dict()
        value["config_sha256"] = self.recipe.config_sha256
        value["policy_contract"] = "spalora.night10b.resolved_family_policy.v1"
        value["resolved_policy_sha256"] = canonical_json_sha256(value)
        return value


def _canonical_assay(value: str) -> str:
    if not isinstance(value, str):
        raise UnsupportedAssayPair("FAIL_CLOSED_UNSUPPORTED_ASSAY_PAIR")
    token = value.strip().upper()
    if not token or any(mark in token for mark in (",", "+", "/", "|", ";")):
        raise UnsupportedAssayPair("FAIL_CLOSED_UNSUPPORTED_ASSAY_PAIR")
    return token


def resolve_family_policy(
    primary_assay: str, auxiliary_assay: str,
) -> ResolvedPolicy:
    """Resolve only from an explicitly supplied ordered assay pair."""
    key = (_canonical_assay(primary_assay), _canonical_assay(auxiliary_assay))
    recipe = FROZEN_RECIPES.get(key)
    if recipe is None:
        raise UnsupportedAssayPair("FAIL_CLOSED_UNSUPPORTED_ASSAY_PAIR")
    return ResolvedPolicy(recipe)


__all__ = ["ResolvedPolicy", "UnsupportedAssayPair", "resolve_family_policy"]
