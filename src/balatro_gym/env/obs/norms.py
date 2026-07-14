"""All observation normalization in one place."""

from __future__ import annotations

import math

LOG_SCORE_CAP = math.log1p(100_000.0)  # White-stake ante-8 requirement ~50k -> ~0.98
LOG_MONEY_CAP = math.log1p(100.0)
LOG_COUNTER_CAP = math.log1p(50.0)


def norm_score(x: float) -> float:
    return math.log1p(max(0.0, x)) / LOG_SCORE_CAP


def norm_money(x: float) -> float:
    return math.log1p(max(0.0, float(x))) / LOG_MONEY_CAP


def norm_money_linear(x: float) -> float:
    """Linear channel: interest breakpoints ($5 steps up to $25) stay resolvable."""
    return min(max(float(x), 0.0), 25.0) / 25.0


def norm_counter(x: float) -> float:
    return math.log1p(max(0.0, float(x))) / LOG_COUNTER_CAP


def norm_progress(scored: float, needed: float) -> float:
    if needed <= 0:
        return 1.0
    return min(scored / needed, 2.0) / 2.0
