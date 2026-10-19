"""PlannerAgent: simulation-based valuation and rollout arbitration."""

from __future__ import annotations

import random

from balatro_gym.engine import Enhancement, HandType, cardops, new_blind_scenario, score_play
from balatro_gym.env.envs import BalatroEnv
from balatro_gym.env.planner import PlannerAgent, _best_play, _with_fillers, rollout_clear
from tests.helpers import cards, joker


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
        obs, _r, term, _trunc, info = env.step(a)
        assert "illegal_action" not in info
        done = term
        steps += 1
    assert steps > 10
    assert cardops.effective_hands(env.run) >= 1


def _hand_scenario(specs, joker_ids=None):
    g = new_blind_scenario("pad", joker_ids=joker_ids)
    g.run.round.hand = cards(*specs)
    return g.run


def test_fillers_pad_a_pair_with_junk_and_keep_assets():
    run = _hand_scenario(["9S", "9H", "KD", "3C", "4D(steel)", "2H", "7C", "7D"])
    padded = _with_fillers(run, (0, 1))
    # the pair plus the three cheapest junk cards; the steel card, the
    # paired sevens and the pair itself stay in hand
    assert padded == (0, 1, 2, 3, 5)
    hand = run.round.hand
    assert all(hand[i].enhancement != Enhancement.STEEL for i in padded)
    assert all(int(hand[i].rank) != 7 for i in padded)


def test_fillers_respect_half_and_square_jokers():
    specs = ["9S", "9H", "KD", "3C", "4D", "2H", "7C", "5D"]
    assert len(_with_fillers(_hand_scenario(specs, ["j_half"]), (0, 1))) == 3
    assert len(_with_fillers(_hand_scenario(specs, ["j_square"]), (0, 1))) == 4
    assert len(_with_fillers(_hand_scenario(specs), (0, 1))) == 5
    # five-card plays and flushes are left alone
    run = _hand_scenario(["2S", "4S", "6S", "8S", "TS", "3D", "5C", "9H"])
    assert _with_fillers(run, (0, 1, 2, 3, 4)) == (0, 1, 2, 3, 4)


def test_fillers_never_lower_the_score():
    run = _hand_scenario(["9S", "9H", "KD", "3C", "4D", "2H", "7C", "5D"])
    padded = _with_fillers(run, (0, 1))
    base = score_play(run.clone(), [run.round.hand[i] for i in (0, 1)], held=[]).score
    trial = run.clone()
    played = [trial.round.hand[i] for i in padded]
    held = [c for i, c in enumerate(trial.round.hand) if i not in padded]
    assert score_play(trial, played, held=held).score >= base
    assert _best_play(run)[1] is not None


def test_lookahead_rollout_scores_and_leaves_state_untouched():
    from balatro_gym.engine import Phase
    from balatro_gym.env import action_codec as codec
    from balatro_gym.env.lookahead import LookaheadPlannerAgent

    env = BalatroEnv()
    agent = LookaheadPlannerAgent(samples=2, top_k=4, samples_ahead=1)
    obs, info = env.reset(seed=53)
    steps = 0
    while env.run.phase != Phase.SHOP and steps < 200:
        a = agent.act(env, obs, info["action_mask"])
        obs, _r, _term, _trunc, info = env.step(a)
        steps += 1
    assert env.run.phase == Phase.SHOP
    before = env.run.to_dict()
    score = agent._rollout(env, codec.LEAVE_SHOP_IDX, 0)
    assert -1.0 <= score <= 3.0
    assert env.run.to_dict() == before  # rollouts never touch the real run
    a = agent.act(env, obs, info["action_mask"])
    assert info["action_mask"][a]


def test_joker_ordering_settles_with_brainstorm():
    """Brainstorm copies the leftmost joker: the strategist moves its best
    joker there and must then stop swapping (it used to flip slots 0 and 1
    forever in the shop)."""
    from balatro_gym.env import action_codec as codec
    from balatro_gym.env.baselines import StrategistAgent

    env = BalatroEnv(deck="d_black")
    env.reset(seed=0)
    run = env.run
    run.jokers.clear()
    for jid in ("j_even_steven", "j_hologram", "j_sly", "j_mime", "j_ramen", "j_brainstorm"):
        cardops.add_joker(run, cardops.new_joker(run, jid))
    env.step(codec.SELECT_BLIND)
    agent = StrategistAgent(seed=0)
    swaps = 0
    while True:
        mask = env.action_masks()
        a = agent._swap_action(env, mask)
        if a is None:
            break
        env.step(a)
        swaps += 1
        assert swaps < 20, "joker ordering never settles"
    assert run.jokers[0].id == "j_hologram"
