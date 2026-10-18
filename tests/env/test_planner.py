"""PlannerAgent: simulation-based valuation and rollout arbitration."""

from __future__ import annotations

import random

from balatro_gym.engine import HandType, cardops, new_blind_scenario
from balatro_gym.env.envs import BalatroEnv
from balatro_gym.env.planner import PlannerAgent, _best_play, rollout_clear
from tests.helpers import joker


def test_power_increases_with_a_scoring_joker():
    env = BalatroEnv()
    env.reset(seed=1)
    agent = PlannerAgent(samples=4)
    run = env.run
    base = agent.power(run)
    with_joker = agent.value_of_joker(run, "j_joker")
    assert with_joker > base
    assert agent.value_of_planet(run, "c_mercury") >= base


def test_scaler_projection_values_green_joker():
    env = BalatroEnv()
    env.reset(seed=2)
    agent = PlannerAgent(samples=4)
    run = env.run
    projected = agent.projected(run)
    assert projected is not run
    run.jokers = [joker("j_green_joker")]
    p = agent.projected(run)
    assert p.jokers[0].state["mult"] > 0


def test_rollout_and_best_play_run_on_scenarios():
    g = new_blind_scenario("roll", hand_levels={HandType.PAIR: 3})
    score, play = _best_play(g.run)
    assert play is not None and score > 0
    outcomes = [rollout_clear(g.run, ("play", play), random.Random(i)) for i in range(3)]
    assert all(0.0 <= o <= 1.0 for o in outcomes)
    assert g.run.round.hands_left == 4  # rollouts never touch the real state


def test_planner_plays_legal_full_run_prefix():
    env = BalatroEnv()
    agent = PlannerAgent(samples=3, top_k=6)
    agent.rollouts = 2
    obs, info = env.reset(seed=5)
    done = False
    steps = 0
    while not done and steps < 80:
        a = agent.act(env, obs, info["action_mask"])
        assert info["action_mask"][a]
        obs, r, term, trunc, info = env.step(a)
        assert "illegal_action" not in info
        done = term
        steps += 1
    assert steps > 10
    assert cardops.effective_hands(env.run) >= 1
