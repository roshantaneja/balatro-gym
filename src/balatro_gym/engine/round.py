"""Blind lifecycle: start, draw/refill, play, discard, payout.

Round card containers (hand, deck_remaining, discarded) reference the SAME
Card objects as run.full_deck — mutations (tarot enhancements) persist for
the run without any syncing.
"""

from __future__ import annotations

from .effects import BossEffect, boss_def, joker_def
from .events import (
    BlindCleared,
    Discarded,
    Event,
    HandScored,
    RoundPayout,
    RunLost,
    RunWon,
)
from .levels import (
    ANTE_BASE_CHIPS,
    BLIND_REWARD,
    BLIND_SCORE_MULT,
    INTEREST_CAP,
    INTEREST_PER,
    UNUSED_HAND_PAYOUT,
    WIN_ANTE,
)
from .scoring import get_boss, score_play
from .state import RoundState, RunState
from .types import BlindKind, Enhancement, HandType, Phase


def blind_requirement(run: RunState, kind: BlindKind, boss: BossEffect | None) -> float:
    base = ANTE_BASE_CHIPS[run.ante]
    if kind == BlindKind.BOSS and boss is not None:
        return base * boss.score_multiplier()
    return base * BLIND_SCORE_MULT[kind]


def pick_boss(run: RunState) -> str | None:
    """Choose a boss for the current ante from the registry (no repeats until
    the eligible pool is exhausted). Returns None when no bosses registered."""
    from .effects import BOSSES

    eligible = [
        bid
        for bid, d in sorted(BOSSES.items())
        if d.effect.min_ante <= run.ante and d.effect.is_finisher == (run.ante % 8 == 0)
    ]
    if not eligible:
        return None
    fresh = [b for b in eligible if b not in run.bosses_seen]
    pool = fresh if fresh else eligible
    choice = run.rng.stream("boss").choice(pool)
    run.bosses_seen.append(choice)
    return choice


def start_blind(run: RunState, kind: BlindKind) -> None:
    boss_id = pick_boss(run) if kind == BlindKind.BOSS else None
    boss = boss_def(boss_id).effect if boss_id else None

    deck = list(run.full_deck)
    run.rng.stream("shuffle").shuffle(deck)

    run.round = RoundState(
        blind_kind=kind,
        boss_id=boss_id,
        chips_needed=blind_requirement(run, kind, boss),
        hands_left=run.hands_per_round,
        discards_left=run.discards_per_round,
        deck_remaining=deck,
    )
    run.phase = Phase.PLAYING
    if boss is not None:
        boss.on_blind_start(run)
    # Tag round-start hooks (Juggle) — may self-consume from run.tags.
    from .effects import TAGS

    for tag_id in list(run.tags):
        tag = TAGS.get(tag_id)
        if tag and tag.on_round_start:
            tag.on_round_start(run)  # type: ignore[operator]
    for j in run.jokers:
        joker_def(j.id).effect.on_blind_selected(run, j)
    refill_hand(run)


def refill_hand(run: RunState) -> None:
    rnd = run.round
    assert rnd is not None
    boss = boss_def(rnd.boss_id).effect if rnd.boss_id else None
    target = run.hand_size + rnd.boss_state.get("hand_size_bonus", 0)
    while len(rnd.hand) < target and rnd.deck_remaining:
        card = rnd.deck_remaining.pop()
        if boss is not None and boss.draw_face_down(run, card):
            rnd.face_down_uids.append(card.uid)
        rnd.hand.append(card)


def play_hand(run: RunState, indices: tuple[int, ...]) -> list[Event]:
    """Play 1-5 cards from hand positions. Assumes legality was checked."""
    rnd = run.round
    assert rnd is not None and run.phase == Phase.PLAYING
    events: list[Event] = []

    played = [rnd.hand[i] for i in indices]
    rnd.hand = [c for i, c in enumerate(rnd.hand) if i not in set(indices)]
    if rnd.face_down_uids:  # played cards flip face-up
        played_uids = {c.uid for c in played}
        rnd.face_down_uids = [u for u in rnd.face_down_uids if u not in played_uids]

    result = score_play(run, played, held=rnd.hand)
    rnd.chips_scored += result.score
    rnd.hands_left -= 1
    rnd.hands_played += 1
    rnd.played_uids.extend(c.uid for c in played)
    rnd.hand_types_played.append(int(result.hand_result.hand_type))
    events.append(HandScored(result, rnd.chips_scored, rnd.chips_needed))

    boss = get_boss(run)
    if boss is not None:
        boss.after_hand(run)

    if rnd.chips_scored >= rnd.chips_needed:
        events.extend(_win_blind(run))
    elif rnd.hands_left <= 0:
        run.phase = Phase.GAME_OVER
        events.append(RunLost(run.ante, rnd.blind_kind, rnd.chips_scored, rnd.chips_needed))
    else:
        refill_hand(run)
    return events


def discard(run: RunState, indices: tuple[int, ...]) -> list[Event]:
    rnd = run.round
    assert rnd is not None and run.phase == Phase.PLAYING
    cards = [rnd.hand[i] for i in indices]

    # Joker discard hooks: scoring hooks for resolved slots, updates for originals.
    from .effects import is_copy_slot, resolve_joker

    last = rnd.discards_left == 1
    for i in range(len(run.jokers)):
        r = resolve_joker(run.jokers, i)
        if r is not None:
            r[0].on_discard(run, r[1], cards, last)
        if not is_copy_slot(run.jokers, i):
            j = run.jokers[i]
            joker_def(j.id).effect.update_on_discard(run, j, cards)

    # Purple seal: creates a Tarot when discarded (consumable creation lands in
    # M5; count it so behavior is observable/testable now).
    purple = sum(1 for c in cards if c.seal.value == "purple")
    if purple:
        run.stats["purple_seals_discarded"] = run.stats.get("purple_seals_discarded", 0) + purple

    rnd.hand = [c for i, c in enumerate(rnd.hand) if i not in set(indices)]
    rnd.discarded.extend(cards)
    if rnd.face_down_uids:
        gone = {c.uid for c in cards}
        rnd.face_down_uids = [u for u in rnd.face_down_uids if u not in gone]
    rnd.discards_left -= 1
    refill_hand(run)
    return [Discarded(len(cards))]


def _win_blind(run: RunState) -> list[Event]:
    rnd = run.round
    assert rnd is not None
    events: list[Event] = [BlindCleared(run.ante, rnd.blind_kind)]

    boss = get_boss(run)
    if boss is not None:
        boss.on_defeat(run)

    # --- cash out -----------------------------------------------------------
    reward = boss.reward() if (rnd.blind_kind == BlindKind.BOSS and boss) else BLIND_REWARD[rnd.blind_kind]
    unused = rnd.hands_left * UNUSED_HAND_PAYOUT
    interest = min(run.money // INTEREST_PER, INTEREST_CAP)

    other = 0
    # Gold cards held in hand at end of round: $3 each.
    other += 3 * sum(1 for c in rnd.hand if c.enhancement == Enhancement.GOLD)
    # Blue seals held at end of round: create the Planet of the final played
    # hand (content registers PLANET_FOR_HAND; no-op until planets exist).
    from .effects import CONSUMABLES, PLANET_FOR_HAND
    from .state import ConsumableInst
    from .types import Seal

    if rnd.hand_types_played and PLANET_FOR_HAND:
        final_hand = HandType(rnd.hand_types_played[-1])
        planet_id = PLANET_FOR_HAND.get(final_hand)
        if planet_id is not None:
            for c in rnd.hand:
                if c.seal == Seal.BLUE and len(run.consumables) < run.consumable_slots:
                    run.consumables.append(
                        ConsumableInst(
                            id=planet_id,
                            kind=CONSUMABLES[planet_id].kind,
                            uid=run.new_uid(),
                        )
                    )
    # Joker round-end money (Golden Joker) + round-end state updates (decay,
    # Ancient Joker re-roll, Egg) — originals only.
    for j in run.jokers:
        eff = joker_def(j.id).effect
        other += eff.round_end_money(run, j)
        eff.update_on_round_end(run, j)
    # Track unused discards across the run (Garbage tag reads this).
    run.stats["discards_unused_run"] = run.stats.get("discards_unused_run", 0) + rnd.discards_left

    # Tag money hooks (Investment) — hooks may self-consume from run.tags.
    from .effects import TAGS

    for tag_id in list(run.tags):
        tag = TAGS.get(tag_id)
        if tag and tag.round_end_money:
            other += tag.round_end_money(run)  # type: ignore[operator]

    total = reward + unused + interest + other
    run.money += total
    events.append(RoundPayout(reward, unused, interest, other, total))

    # --- advance ------------------------------------------------------------
    if rnd.blind_kind == BlindKind.BOSS and run.ante >= WIN_ANTE:
        run.phase = Phase.WON
        run.round = None
        events.append(RunWon(run.ante))
        return events

    if rnd.blind_kind == BlindKind.BOSS:
        run.ante += 1
        run.blind_index = 0
    else:
        run.blind_index += 1
    run.round = None
    run.phase = Phase.SHOP
    from .shop import open_shop

    open_shop(run)
    return events


def skip_blind(run: RunState) -> list[Event]:
    """Skip Small/Big blind for a tag (tag granting arrives with M3 content)."""
    from .events import BlindSkipped

    kind = BlindKind(run.blind_index)
    assert kind != BlindKind.BOSS
    run.skips_used.append(int(kind))

    tag_id = _grant_skip_tag(run)
    run.blind_index += 1
    run.phase = Phase.BLIND_SELECT
    return [BlindSkipped(run.ante, kind, tag_id)]


def _grant_skip_tag(run: RunState) -> str | None:
    from .effects import TAGS

    if not TAGS:
        return None
    tag_id = run.rng.stream("misc").choice(sorted(TAGS))
    run.tags.append(tag_id)
    tag = TAGS[tag_id]
    if tag.on_acquire:
        tag.on_acquire(run)  # type: ignore[operator]
    return tag_id
