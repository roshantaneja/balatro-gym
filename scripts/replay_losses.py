"""Diagnose the play policy: replay every blind the planner lost.

    python scripts/replay_losses.py 0 1 2 3        # seeds

For each seed the planner plays a full run; the engine state at the first
hand of the blind it lost is snapshotted and restored into the env, and the
blind is replayed (a) by the same agent with a different rollout RNG and
(b) by a perfect-information greedy oracle that sees the deck order. A
blind the oracle clears was winnable with better in-round play; one the
agent clears on replay is a rollout-variance loss.
"""

from __future__ import annotations

import sys

from balatro_gym.engine import HandType, Phase, RunLost
from balatro_gym.engine import round as round_mod
from balatro_gym.env.envs import BalatroEnv
from balatro_gym.env.planner import (
    PlannerAgent,
    _best_plays,
    _draw_discards,
    _simple_discard,
    _with_fillers,
)


def restore(env: BalatroEnv, clone) -> tuple:
    env.game.run = clone.clone()
    env._mask_cache = None
    env._pending = None
    env._refresh_hand_order()
    return env._obs(), env._info()


def play_round(env: BalatroEnv, agent) -> bool:
    obs, info = env._obs(), env._info()
    while env.run.phase == Phase.PLAYING:
        a = agent.act(env, obs, info["action_mask"])
        obs, _r, term, _trunc, info = env.step(a)
        if term:
            break
    return env.run.phase != Phase.GAME_OVER


def _greedy_finish(trial) -> float:
    guard = 0
    while trial.phase == Phase.PLAYING and trial.round is not None and guard < 12:
        guard += 1
        plays = _best_plays(trial, 6)
        if not plays:
            break
        sc, pl = plays[0]
        rnd = trial.round
        need = rnd.chips_needed - rnd.chips_scored
        if sc < need and sc < need / max(rnd.hands_left, 1) and rnd.discards_left > 0:
            d = _simple_discard(trial)
            if d:
                round_mod.discard(trial, d)
                continue
        round_mod.play_hand(trial, _with_fillers(trial, tuple(sorted(pl))))
    if trial.phase != Phase.GAME_OVER and (
        trial.round is None or trial.round.chips_scored >= trial.round.chips_needed
    ):
        return 1.0
    return trial.round.chips_scored / trial.round.chips_needed if trial.round else 0.0


def oracle(run) -> bool:
    """Perfect information: every option is scored by a greedy continuation
    on the TRUE deck order; the best option is taken."""
    sim = run.clone()
    guard = 0
    while sim.phase == Phase.PLAYING and sim.round is not None and guard < 12:
        guard += 1
        rnd = sim.round
        options = [("play", idxs) for _, idxs in _best_plays(sim, 8)[:4]]
        if rnd.discards_left > 0:
            d = _simple_discard(sim)
            if d:
                options.append(("discard", d))
            options.extend(("discard", d) for d in _draw_discards(sim))
        best = None
        for kind, idxs in options:
            trial = sim.clone()
            if kind == "play":
                round_mod.play_hand(trial, _with_fillers(trial, tuple(sorted(idxs))))
            else:
                round_mod.discard(trial, tuple(sorted(idxs)))
            score = _greedy_finish(trial)
            if best is None or score > best[0]:
                best = (score, kind, idxs)
        if best is None:
            break
        if best[1] == "play":
            round_mod.play_hand(sim, _with_fillers(sim, tuple(sorted(best[2]))))
        else:
            round_mod.discard(sim, tuple(sorted(best[2])))
    return sim.phase != Phase.GAME_OVER and (
        sim.round is None or sim.round.chips_scored >= sim.round.chips_needed
    )


def main() -> None:
    seeds = [int(x) for x in sys.argv[1:]] or list(range(10))
    for seed in seeds:
        agent = PlannerAgent(commit=HandType.PAIR)
        env = BalatroEnv()
        obs, info = env.reset(seed=seed)
        done, steps, snap, lost = False, 0, None, False
        while not done and steps < 4000:
            run = env.run
            rnd = run.round
            if (
                run.phase == Phase.PLAYING
                and rnd is not None
                and rnd.hands_played == 0
                and rnd.discards_used == 0
                and (snap is None or snap[0] != (run.ante, run.blind_index))
            ):
                snap = ((run.ante, run.blind_index), run.clone())
            a = agent.act(env, obs, info["action_mask"])
            obs, _r, term, _trunc, info = env.step(a)
            done, steps = term, steps + 1
            lost = lost or any(isinstance(e, RunLost) for e in info.get("events", []))
        if not lost or snap is None:
            print(f"seed {seed}: {'WON' if env.run.phase == Phase.WON else 'no snapshot'}", flush=True)
            continue
        key, clone = snap
        replays = []
        for k in range(3):
            again = PlannerAgent(commit=HandType.PAIR, seed=100 + k)
            restore(env, clone)
            replays.append(play_round(env, again))
        print(
            f"seed {seed}: lost at {key} boss={clone.round.boss_id} "
            f"need={clone.round.chips_needed:.0f} | replay clears {sum(replays)}/3 | "
            f"oracle {oracle(clone)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
