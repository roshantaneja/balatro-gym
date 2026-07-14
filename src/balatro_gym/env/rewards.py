"""Pluggable reward computation — driven by the engine event log, not state diffs."""

from __future__ import annotations

from dataclasses import dataclass

from balatro_gym.engine import BlindCleared, Event, HandScored, RunWon


@dataclass(frozen=True)
class RewardConfig:
    mode: str = "blind"  # "blind" | "shaped"
    w_blind: float = 1.0
    w_win: float = 5.0
    w_progress: float = 0.5   # shaped mode only; ante-weighted, capped potential
    w_step: float = 0.0       # optional per-step cost (shop dithering)

    def compute(self, events: list[Event], ante: int) -> float:
        r = self.w_step * -1.0 if self.w_step else 0.0
        for e in events:
            if isinstance(e, BlindCleared):
                r += self.w_blind
            elif isinstance(e, RunWon):
                r += self.w_win
            elif isinstance(e, HandScored) and self.mode == "shaped":
                after = min(e.chips_scored / e.chips_needed, 1.0)
                before = min(
                    (e.chips_scored - e.result.score) / e.chips_needed, 1.0
                )
                r += self.w_progress * (ante / 8.0) * max(after - before, 0.0)
        return r
