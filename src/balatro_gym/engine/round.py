"""Blind lifecycle: ante setup, start, draw/refill, play, discard, payout.

Round card containers (hand, deck_remaining, discarded) reference the SAME
Card objects as run.full_deck — mutations (tarot enhancements) persist for
the run without any syncing.

Ante setup (`begin_ante`) pre-rolls the boss, the ante voucher and the two
skip tags so the agent can see them before choosing to play or skip — exactly
what the in-game blind-select screen shows.
"""

from __future__ import annotations

from .cardops import (
    create_consumable,
    destroy_joker,
    effective_discards,
    effective_hand_size,
    effective_hands,
    flush_pending_destroys,
    has_consumable_room,
)
from .effects import (
    BOSSES,
    PLANET_FOR_HAND,
    TAGS,
    BossEffect,
    active_jokers,
    boss_def,
    joker_def,
    resolve_joker,
)
from .events import (
    BlindCleared,
    BlindSkipped,
    Discarded,
    Event,
    HandScored,
    RoundPayout,
    RunLost,
    RunWon,
)
from .levels import (
    BLIND_REWARD,
    BLIND_SCORE_MULT,
    INTEREST_PER,
    UNUSED_HAND_PAYOUT,
    WIN_ANTE,
    ante_base_chips,
)
from .scoring import get_boss, score_play
from .state import RoundState, RunState, TagInst
from .types import (
    RENTAL_FEE,
    STICKER_RENTAL,
    BlindKind,
    Enhancement,
    HandType,
    Phase,
    Seal,
)

# ---------------------------------------------------------------------------
# Ante setup
# ---------------------------------------------------------------------------

def blind_requirement(run: RunState, kind: BlindKind, boss: BossEffect | None) -> float:
    base = ante_base_chips(run.ante, run.stake) * run.mods.get("ante_scaling", 1)
    if kind == BlindKind.BOSS and boss is not None:
        return base * boss.score_multiplier()
    return base * BLIND_SCORE_MULT[kind]


def pick_boss(run: RunState) -> str | None:
    """Choose a boss for the current ante from the registry (no repeats until
    the eligible pool is exhausted). Returns None when no bosses registered."""
    finisher = run.ante > 0 and run.ante % WIN_ANTE == 0
    eligible = [
        bid
        for bid, d in sorted(BOSSES.items())
        if d.effect.min_ante <= max(run.ante, 1) and d.effect.is_finisher == finisher
    ]
    if not eligible:
        return None
    fresh = [b for b in eligible if b not in run.bosses_seen]
    pool = fresh if fresh else eligible
    choice = run.rng.stream("boss").choice(pool)
    return choice


def roll_tag(run: RunState, exclude: set[str] | None = None) -> TagInst | None:
    pool = sorted(
        tid for tid, d in TAGS.items()
        if d.min_ante <= run.ante and (not exclude or tid not in exclude)
    )
    if not pool:
        return None
    tid = run.rng.stream("misc").choice(pool)
    tag = TagInst(id=tid, state=dict(TAGS[tid].init_state))
    # Per-instance randomisation seams (Orbital's hand) via "orbital_hand".
    if "hand" in tag.state and tag.state["hand"] is None:
        tag.state["hand"] = int(run.rng.stream("misc").choice(sorted(run.hands_discovered)))
    return tag


def roll_voucher(run: RunState) -> str | None:
    from .effects import VOUCHERS

    pool = sorted(
        vid
        for vid, d in VOUCHERS.items()
        if vid not in run.vouchers and (d.requires is None or d.requires in run.vouchers)
    )
    if not pool:
        return None
    return run.rng.stream("shop").choice(pool)


def begin_ante(run: RunState) -> None:
    """Pre-roll everything the blind-select screen shows for this ante."""
    run.blind_index = 0
    run.ante_boss = pick_boss(run)
    if run.ante_boss is not None:
        run.bosses_seen.append(run.ante_boss)
    run.ante_voucher = roll_voucher(run)
    run.blind_tags = [roll_tag(run), roll_tag(run)]
    run.boss_rerolls_used = 0
    run.ante_played_uids = []


def reroll_boss(run: RunState) -> None:
    """Director's Cut / Retcon / Boss Tag: pick a different boss for this ante."""
    old = run.ante_boss
    if old is not None and old in run.bosses_seen:
        run.bosses_seen.remove(old)
    new = pick_boss(run)
    if new == old:
        # pick_boss prefers unseen bosses; with everything seen it may repeat —
        # try once more among the rest of the pool.
        finisher = run.ante > 0 and run.ante % WIN_ANTE == 0
        others = [
            bid for bid, d in sorted(BOSSES.items())
            if d.effect.min_ante <= max(run.ante, 1) and d.effect.is_finisher == finisher and bid != old
        ]
        if others:
            new = run.rng.stream("boss").choice(others)
    run.ante_boss = new
    if new is not None:
        run.bosses_seen.append(new)


# ---------------------------------------------------------------------------
# Blind start
# ---------------------------------------------------------------------------

def start_blind(run: RunState, kind: BlindKind) -> None:
    if kind == BlindKind.BOSS and run.ante_boss is None:
        run.ante_boss = pick_boss(run)
        if run.ante_boss is not None:
            run.bosses_seen.append(run.ante_boss)
    boss_id = run.ante_boss if kind == BlindKind.BOSS else None
    boss = boss_def(boss_id).effect if boss_id else None

    deck = list(run.full_deck)
    run.rng.stream("shuffle").shuffle(deck)

    run.round = RoundState(
        blind_kind=kind,
        boss_id=boss_id,
        chips_needed=blind_requirement(run, kind, boss),
        hands_left=effective_hands(run),
        discards_left=effective_discards(run),
        deck_remaining=deck,
    )
    run.phase = Phase.PLAYING
    # Jokers first (Chicot may disable the boss; Burglar sets hands/discards),
    # then the boss, then round-start tags.
    for j in active_jokers(run):
        joker_def(j.id).effect.on_blind_selected(run, j)
    if boss is not None and not run.round.boss_state.get("disabled"):
        boss.on_blind_start(run)
    for tag in list(run.tags):
        d = TAGS.get(tag.id)
        if d and d.effect.on_round_start(run, tag):
            _consume_tag(run, tag)
    refill_hand(run)


def _consume_tag(run: RunState, tag: TagInst) -> None:
    run.tags = [t for t in run.tags if t is not tag]


def refill_hand(run: RunState) -> None:
    rnd = run.round
    assert rnd is not None
    boss = get_boss(run)
    target = effective_hand_size(run)
    fixed = boss.draw_count(run) if boss is not None else None
    if fixed is not None and (rnd.hands_played > 0 or rnd.discards_used > 0):
        # The Serpent: always draw exactly N after a play or discard.
        n = fixed
        while n > 0 and rnd.deck_remaining:
            _draw_one(run, boss)
            n -= 1
    else:
        while len(rnd.hand) < target and rnd.deck_remaining:
            _draw_one(run, boss)
    if boss is not None:
        boss.after_refill(run)


def _draw_one(run: RunState, boss: BossEffect | None) -> None:
    rnd = run.round
    assert rnd is not None
    card = rnd.deck_remaining.pop()
    if boss is not None and boss.draw_face_down(run, card):
        rnd.face_down_uids.append(card.uid)
    rnd.hand.append(card)


# ---------------------------------------------------------------------------
# Play / discard
# ---------------------------------------------------------------------------

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

    rnd.boss_state.pop("triggered", None)
    result = score_play(run, played, held=rnd.hand)
    rnd.chips_scored += result.score
    rnd.hands_left -= 1
    rnd.hands_played += 1
    rnd.played_uids.extend(c.uid for c in played)
    run.ante_played_uids.extend(c.uid for c in played)
    rnd.hand_types_played.append(int(result.hand_result.hand_type))
    events.append(HandScored(result, rnd.chips_scored, rnd.chips_needed))

    boss = get_boss(run)
    if boss is not None:
        boss.after_hand(run)

    if rnd.chips_scored >= rnd.chips_needed:
        events.extend(_win_blind(run))
    elif rnd.hands_left <= 0:
        events.extend(_lose_blind(run))
    else:
        refill_hand(run)
        if is_stuck(run):
            events.extend(_lose_blind(run))
    return events


def is_stuck(run: RunState) -> bool:
    """True when no play or discard is possible (empty hand, or a boss
    restriction no selection can satisfy, e.g. The Psychic with 4 cards).
    The game cannot progress, which is a lost blind."""
    from itertools import combinations

    rnd = run.round
    if rnd is None or run.phase != Phase.PLAYING:
        return False
    if not rnd.hand:
        return True
    boss = get_boss(run)
    if boss is None:
        return False
    n = len(rnd.hand)
    for k in range(1, min(5, n) + 1):
        for idxs in combinations(range(n), k):
            cards = [rnd.hand[i] for i in idxs]
            if rnd.hands_left > 0 and boss.validate_play(run, cards) is None:
                return False
            if rnd.discards_left > 0 and boss.validate_discard(run, cards) is None:
                return False
    return True


def _lose_blind(run: RunState) -> list[Event]:
    rnd = run.round
    assert rnd is not None
    # Mr. Bones: survive if at least 25% of the requirement was scored.
    for j in active_jokers(run):
        if joker_def(j.id).effect.prevent_loss(run, j):
            destroy_joker(run, j)
            return _win_blind(run, saved=True)
    run.phase = Phase.GAME_OVER
    return [RunLost(run.ante, rnd.blind_kind, rnd.chips_scored, rnd.chips_needed)]


def discard(run: RunState, indices: tuple[int, ...]) -> list[Event]:
    rnd = run.round
    assert rnd is not None and run.phase == Phase.PLAYING
    cards = [rnd.hand[i] for i in indices]

    # Joker discard hooks: scoring hooks for resolved slots, updates for originals.
    from .effects import is_copy_slot

    last = rnd.discards_left == 1
    for i in range(len(run.jokers)):
        r = resolve_joker(run.jokers, i, run)
        if r is not None:
            r[0].on_discard(run, r[1], cards, last)
    for j in active_jokers(run):
        if not is_copy_slot(run.jokers, run.jokers.index(j)):
            joker_def(j.id).effect.update_on_discard(run, j, cards)

    # Purple seal: creates a Tarot card when discarded (if there is room).
    purple = [c for c in cards if c.seal == Seal.PURPLE]
    if purple:
        from .cardops import create_random_consumable
        from .types import ConsumableKind

        run.stats["purple_seals_discarded"] = run.stats.get("purple_seals_discarded", 0) + len(purple)
        for _ in purple:
            create_random_consumable(run, ConsumableKind.TAROT)

    # Cards may have been destroyed by discard hooks (Trading Card): keep only
    # the ones still in hand.
    still = {c.uid for c in rnd.hand}
    cards = [c for c in cards if c.uid in still]
    gone = {c.uid for c in cards}
    rnd.hand = [c for c in rnd.hand if c.uid not in gone]
    rnd.discarded.extend(cards)
    if rnd.face_down_uids:
        rnd.face_down_uids = [u for u in rnd.face_down_uids if u not in gone]
    rnd.discards_left -= 1
    rnd.discards_used += 1
    run.stats["cards_discarded_run"] = run.stats.get("cards_discarded_run", 0) + len(cards)
    boss = get_boss(run)
    if boss is not None:
        boss.after_discard(run)
    flush_pending_destroys(run)
    refill_hand(run)
    events: list[Event] = [Discarded(len(cards))]
    if is_stuck(run):
        events.extend(_lose_blind(run))
    return events


# ---------------------------------------------------------------------------
# Round end
# ---------------------------------------------------------------------------

def _win_blind(run: RunState, saved: bool = False) -> list[Event]:
    rnd = run.round
    assert rnd is not None
    events: list[Event] = [BlindCleared(run.ante, rnd.blind_kind)]

    boss = get_boss(run)
    if boss is not None:
        boss.on_defeat(run)
    if rnd.blind_kind == BlindKind.BOSS:
        run.stats["bosses_defeated"] = run.stats.get("bosses_defeated", 0) + 1
        for j in active_jokers(run):
            joker_def(j.id).effect.update_on_boss_defeated(run, j)

    # --- cash out -----------------------------------------------------------
    if rnd.blind_kind == BlindKind.BOSS and boss is not None:
        reward = boss.reward()
    elif rnd.blind_kind == BlindKind.BOSS and rnd.boss_id is not None:
        reward = boss_def(rnd.boss_id).effect.reward()
    else:
        reward = BLIND_REWARD[rnd.blind_kind]
    if rnd.blind_kind == BlindKind.SMALL and run.mods.get("no_small_reward"):
        reward = 0
    unused = rnd.hands_left * UNUSED_HAND_PAYOUT
    if run.mods.get("no_interest"):
        interest = 0
    else:
        amount = run.interest_amount + sum(
            joker_def(j.id).effect.interest_bonus(run, j) for j in active_jokers(run)
        )
        interest = amount * min(max(run.money, 0) // INTEREST_PER, run.interest_cap // INTEREST_PER)

    other = 0
    # Green Deck: $2 per remaining hand, $1 per remaining discard (no interest).
    other += run.mods.get("extra_hand_bonus", 0) * rnd.hands_left
    other += run.mods.get("extra_discard_bonus", 0) * rnd.discards_left
    # Gold cards held in hand at end of round: $3 each.
    other += 3 * sum(1 for c in rnd.hand if c.enhancement == Enhancement.GOLD)
    # Blue seals held at end of round: create the Planet of the final played hand.
    if rnd.hand_types_played and PLANET_FOR_HAND:
        final_hand = HandType(rnd.hand_types_played[-1])
        planet_id = PLANET_FOR_HAND.get(final_hand)
        if planet_id is not None:
            for c in rnd.hand:
                if c.seal == Seal.BLUE and has_consumable_room(run):
                    create_consumable(run, planet_id)
    # Joker round-end money (Golden Joker) + round-end state updates (decay,
    # Ancient Joker re-roll, Egg) — originals only, debuffed jokers skipped.
    for j in active_jokers(run):
        eff = joker_def(j.id).effect
        other += eff.round_end_money(run, j)
        eff.update_on_round_end(run, j)
    # Stickers: rental fee, perishable countdown.
    for j in list(run.jokers):
        if STICKER_RENTAL in j.stickers:
            other -= RENTAL_FEE
        if j.perish_left > 0:
            j.perish_left -= 1
            if j.perish_left == 0:
                j.debuffed = True
    # Track unused discards across the run (Garbage tag reads this).
    run.stats["discards_unused_run"] = run.stats.get("discards_unused_run", 0) + rnd.discards_left
    run.stats["rounds_played"] = run.stats.get("rounds_played", 0) + 1

    # Tag money hooks (Investment) — hooks may self-consume.
    for tag in list(run.tags):
        d = TAGS.get(tag.id)
        if d:
            dollars, consumed = d.effect.round_end_money(run, tag)
            other += dollars
            if consumed:
                _consume_tag(run, tag)

    total = reward + unused + interest + other
    run.money += total
    events.append(RoundPayout(reward, unused, interest, other, total))
    flush_pending_destroys(run)

    # --- advance ------------------------------------------------------------
    if rnd.blind_kind == BlindKind.BOSS and run.ante >= WIN_ANTE:
        run.phase = Phase.WON
        run.round = None
        events.append(RunWon(run.ante))
        return events

    if rnd.blind_kind == BlindKind.BOSS:
        run.ante += 1
        begin_ante(run)
    else:
        run.blind_index += 1
    run.round = None
    run.phase = Phase.SHOP
    from .shop import open_shop

    open_shop(run)
    return events


# ---------------------------------------------------------------------------
# Skipping
# ---------------------------------------------------------------------------

def skip_blind(run: RunState) -> list[Event]:
    """Skip the Small/Big blind for the pre-rolled tag."""
    kind = BlindKind(run.blind_index)
    assert kind != BlindKind.BOSS
    run.skips_used.append(int(kind))
    run.stats["blinds_skipped"] = run.stats.get("blinds_skipped", 0) + 1

    tag = run.blind_tags[int(kind)] if int(kind) < len(run.blind_tags) else None
    tag_id = tag.id if tag else None
    run.blind_index += 1
    run.phase = Phase.BLIND_SELECT
    if tag is not None:
        acquire_tag(run, tag)
    for j in active_jokers(run):
        joker_def(j.id).effect.update_on_blind_skipped(run, j)
    return [BlindSkipped(run.ante, kind, tag_id)]


def acquire_tag(run: RunState, tag: TagInst) -> None:
    """Add a tag: Double Tags watching for it fire first, then immediate
    effects (which may consume the tag straight away)."""
    for other in list(run.tags):
        d = TAGS.get(other.id)
        if d and d.effect.on_tag_acquired(run, other, tag):
            _consume_tag(run, other)
    run.tags.append(tag)
    d = TAGS.get(tag.id)
    if d and d.effect.on_acquire(run, tag):
        _consume_tag(run, tag)
