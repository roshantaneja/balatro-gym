"""Round loop and Game facade: phases, legality, full-run fuzz, determinism."""

import random

import tests.helpers  # noqa: F401  (loads content)
from balatro_gym.engine import (
    BlindCleared,
    BlindKind,
    Discard,
    Game,
    HandScored,
    Phase,
    PlayHand,
    RoundPayout,
    RunLost,
    SelectBlind,
    SkipBlind,
    new_blind_scenario,
)


def test_game_starts_at_blind_select():
    g = Game(seed="s1")
    assert g.run.phase == Phase.BLIND_SELECT
    acts = g.legal_actions()
    assert any(isinstance(a, SelectBlind) for a in acts)
    assert any(isinstance(a, SkipBlind) for a in acts)  # small blind is skippable


def test_red_deck_extra_discard():
    g = Game(seed="s1", deck_id="d_red")
    g.apply(SelectBlind())
    assert g.run.round is not None
    assert g.run.round.discards_left == 4  # 3 + red deck bonus


def test_select_blind_deals_hand():
    g = Game(seed="s1")
    g.apply(SelectBlind())
    assert g.run.phase == Phase.PLAYING
    assert len(g.run.round.hand) == 8
    assert len(g.run.round.deck_remaining) == 44


def test_play_hand_refills():
    g = Game(seed="s1")
    g.apply(SelectBlind())
    events = g.apply(PlayHand((0, 1)))
    assert any(isinstance(e, HandScored) for e in events)
    rnd = g.run.round
    if rnd is not None:  # blind not yet cleared
        assert len(rnd.hand) == 8
        assert rnd.hands_left == 3


def test_discard_consumes_and_refills():
    g = Game(seed="s1")
    g.apply(SelectBlind())
    before = g.run.round.discards_left
    g.apply(Discard((0, 1, 2)))
    assert g.run.round.discards_left == before - 1
    assert len(g.run.round.hand) == 8


def test_small_blind_requirement():
    g = Game(seed="s1")
    g.apply(SelectBlind())
    assert g.run.round.chips_needed == 300  # ante 1 small = 1.0 x 300


def test_skip_blind_advances():
    g = Game(seed="s1")
    g.apply(SkipBlind())
    assert g.run.blind_index == 1
    assert g.run.phase == Phase.BLIND_SELECT
    g.apply(SkipBlind())
    assert g.run.blind_index == 2
    # Boss can't be skipped.
    assert not any(isinstance(a, SkipBlind) for a in g.legal_actions())


def test_clearing_blind_pays_out():
    from balatro_gym.engine import HandType

    # Leveled hands guarantee any play clears the ante-1 small blind (300).
    g = new_blind_scenario("s1", hand_levels={HandType.PAIR: 20, HandType.HIGH_CARD: 20})
    start_money = g.run.money
    # Play best pair or any hand; leveled hands guarantee > 300.
    events = g.apply(PlayHand((0, 1)))
    assert any(isinstance(e, BlindCleared) for e in events)
    payout = next(e for e in events if isinstance(e, RoundPayout))
    assert payout.blind_reward == 3          # small blind
    assert payout.unused_hands == 3          # 4 hands - 1 played
    assert payout.interest == 0              # $4 held -> $0 interest
    assert g.run.money == start_money + payout.total
    assert g.run.phase == Phase.SHOP


def test_interest_calculation():
    from balatro_gym.engine import HandType

    g = new_blind_scenario(
        "s1", money=23, hand_levels={HandType.PAIR: 20, HandType.HIGH_CARD: 20}
    )
    events = g.apply(PlayHand((0, 1)))
    payout = next(e for e in events if isinstance(e, RoundPayout))
    assert payout.interest == 4  # 23 // 5 = 4


def test_interest_cap():
    from balatro_gym.engine import HandType

    g = new_blind_scenario(
        "s1", money=100, hand_levels={HandType.PAIR: 20, HandType.HIGH_CARD: 20}
    )
    events = g.apply(PlayHand((0, 1)))
    payout = next(e for e in events if isinstance(e, RoundPayout))
    assert payout.interest == 5  # capped


def test_running_out_of_hands_loses():
    g = new_blind_scenario("s1", ante=8, blind_kind=BlindKind.SMALL)  # 50k requirement
    events = []
    for _ in range(4):
        if g.run.phase != Phase.PLAYING:
            break
        events = g.apply(PlayHand((0,)))
    assert g.run.phase == Phase.GAME_OVER
    assert any(isinstance(e, RunLost) for e in events)


def _play_random_run(seed: str, max_steps: int = 5000) -> tuple[Phase, int, list]:
    """Random legal-action playthrough; returns (final phase, steps, events)."""
    g = Game(seed=seed)
    py_rng = random.Random(seed)
    all_events = []
    steps = 0
    while g.run.phase not in (Phase.WON, Phase.GAME_OVER) and steps < max_steps:
        acts = g.legal_actions()
        assert acts, f"no legal actions in phase {g.run.phase}"
        all_events.extend(g.apply(py_rng.choice(acts)))
        steps += 1
    return g.run.phase, steps, all_events


def test_full_run_random_agent_terminates():
    for seed in ["a", "b", "c", "d", "e"]:
        phase, steps, _ = _play_random_run(seed)
        assert phase in (Phase.WON, Phase.GAME_OVER)
        assert steps < 5000


def test_full_run_money_never_negative():
    for seed in ["m1", "m2", "m3"]:
        g = Game(seed=seed)
        py_rng = random.Random(seed)
        steps = 0
        while g.run.phase not in (Phase.WON, Phase.GAME_OVER) and steps < 3000:
            g.apply(py_rng.choice(g.legal_actions()))
            assert g.run.money >= 0, f"negative money at step {steps}"
            steps += 1


def test_run_determinism():
    """Same seed + same action choices => identical final state."""

    def run_once():
        g = Game(seed="det")
        py_rng = random.Random(42)
        steps = 0
        while g.run.phase not in (Phase.WON, Phase.GAME_OVER) and steps < 2000:
            acts = g.legal_actions()
            g.apply(acts[py_rng.randrange(len(acts))])
            steps += 1
        return g.run.to_dict()

    assert run_once() == run_once()


def test_serialization_roundtrip_mid_round():
    from balatro_gym.engine import RunState

    g = Game(seed="ser")
    g.apply(SelectBlind())
    g.apply(Discard((0, 1)))
    d = g.run.to_dict()
    restored = RunState.from_dict(d)
    assert restored.to_dict() == d
    # Round cards must alias full_deck objects after deserialization.
    full_by_uid = {c.uid: c for c in restored.full_deck}
    for c in restored.round.hand:
        assert full_by_uid[c.uid] is c


def test_clone_preserves_card_identity():
    g = Game(seed="cl")
    g.apply(SelectBlind())
    cloned = g.run.clone()
    full_by_uid = {c.uid: c for c in cloned.full_deck}
    for c in cloned.round.hand + cloned.round.deck_remaining:
        assert full_by_uid[c.uid] is c
