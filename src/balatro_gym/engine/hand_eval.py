"""Poker-hand evaluation, parameterized by EvalRules.

Rule-modifying jokers (Four Fingers, Shortcut, Smeared, Splash, Pareidolia)
only ever flip EvalRules fields — the evaluator itself is pure and content-free.

Semantics from Poker Hands.wiki:
- Stone cards are excluded from detection but always score.
- Debuffed cards still participate in detection (they just contribute nothing
  at scoring time — that is scoring.py's concern, not ours).
- Two Pair / Full House require distinct ranks (Four of a Kind contains no
  Two Pair; Five of a Kind contains no Full House).
- Secret hands are conjunctions judged independently: Flush House = Full House
  AND Flush; Flush Five = Five of a Kind AND Flush; Straight Flush likewise.
- Ace is high or low in straights (no wraparound) but always scores 11 chips.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .cards import Card
from .types import HandType, Rank, Suit

_BLACK = frozenset({Suit.SPADES, Suit.CLUBS})
_RED = frozenset({Suit.HEARTS, Suit.DIAMONDS})


@dataclass(frozen=True, slots=True)
class EvalRules:
    flush_size: int = 5        # Four Fingers -> 4
    straight_size: int = 5     # Four Fingers -> 4
    straight_max_gap: int = 1  # Shortcut -> 2
    smeared_suits: bool = False
    splash: bool = False
    all_faces: bool = False    # Pareidolia (consumed via is_face, not detection)


DEFAULT_RULES = EvalRules()


@dataclass(slots=True)
class HandResult:
    hand_type: HandType
    contained: frozenset[HandType]
    scoring_cards: list[Card]  # play order; includes Stone cards always
    is_royal: bool = False


def suit_matches(card: Card, suit: Suit, rules: EvalRules) -> bool:
    """The one suit-comparison used everywhere (jokers included via ctx)."""
    if card.is_stone:
        return False
    from .types import Enhancement

    if card.enhancement == Enhancement.WILD:
        return True
    if rules.smeared_suits:
        cls = _BLACK if card.suit in _BLACK else _RED
        return suit in cls
    return card.suit == suit


def is_face(card: Card, rules: EvalRules) -> bool:
    if card.is_stone:
        return False
    return rules.all_faces or card.rank.is_face


def _flush_cards(cards: list[Card], rules: EvalRules) -> list[Card]:
    """Largest suit-matching group if it reaches flush_size, else []."""
    best: list[Card] = []
    for suit in Suit:
        group = [c for c in cards if suit_matches(c, suit, rules)]
        if len(group) > len(best):
            best = group
    return best if len(best) >= rules.flush_size else []


def _straight_ranks(cards: list[Card], rules: EvalRules) -> set[int] | None:
    """Rank *values* (ace = 14 and 1) forming a straight, or None.

    A straight is `straight_size` distinct values in ascending order where each
    consecutive step is <= straight_max_gap (Shortcut allows gaps of 2).
    """
    values = set()
    for c in cards:
        values.add(int(c.rank))
        if c.rank == Rank.ACE:
            values.add(1)
    ordered = sorted(values)
    n = rules.straight_size
    for i in range(len(ordered)):
        run = [ordered[i]]
        for j in range(i + 1, len(ordered)):
            if ordered[j] - run[-1] <= rules.straight_max_gap:
                run.append(ordered[j])
                if len(run) == n:
                    # Ace-high straights must not wrap: exclude runs using both 1 and 14
                    if not (1 in run and 14 in run):
                        return set(run)
            else:
                break
    return None


def evaluate_hand(played: list[Card], rules: EvalRules = DEFAULT_RULES) -> HandResult:
    """Classify a played selection of 1-5 cards."""
    detect = [c for c in played if not c.is_stone]
    [c for c in played if c.is_stone]

    contained: set[HandType] = set()
    if detect:
        contained.add(HandType.HIGH_CARD)

    rank_counts = Counter(c.rank for c in detect)
    counts = sorted(rank_counts.values(), reverse=True)

    if counts and counts[0] >= 2:
        contained.add(HandType.PAIR)
    if counts and counts[0] >= 3:
        contained.add(HandType.THREE_KIND)
    if counts and counts[0] >= 4:
        contained.add(HandType.FOUR_KIND)
    if counts and counts[0] >= 5:
        contained.add(HandType.FIVE_KIND)
    if len(counts) >= 2 and counts[0] >= 2 and counts[1] >= 2:
        contained.add(HandType.TWO_PAIR)
    if len(counts) >= 2 and counts[0] >= 3 and counts[1] >= 2:
        contained.add(HandType.FULL_HOUSE)

    flush_group = _flush_cards(detect, rules)
    if flush_group:
        contained.add(HandType.FLUSH)
    straight_vals = _straight_ranks(detect, rules) if len(detect) >= rules.straight_size else None
    if straight_vals:
        contained.add(HandType.STRAIGHT)

    if HandType.STRAIGHT in contained and HandType.FLUSH in contained:
        contained.add(HandType.STRAIGHT_FLUSH)
    if HandType.FULL_HOUSE in contained and HandType.FLUSH in contained:
        contained.add(HandType.FLUSH_HOUSE)
    if HandType.FIVE_KIND in contained and HandType.FLUSH in contained:
        contained.add(HandType.FLUSH_FIVE)

    if not contained:
        # Only stone cards were played: scores as High Card with only stones scoring.
        hand_type = HandType.HIGH_CARD
    else:
        hand_type = max(contained)

    scoring = _scoring_cards(hand_type, played, detect, rank_counts, flush_group, straight_vals, rules)

    is_royal = (
        hand_type == HandType.STRAIGHT_FLUSH
        and straight_vals is not None
        and straight_vals == {10, 11, 12, 13, 14}
    )
    return HandResult(
        hand_type=hand_type,
        contained=frozenset(contained),
        scoring_cards=scoring,
        is_royal=is_royal,
    )


def _scoring_cards(
    hand_type: HandType,
    played: list[Card],
    detect: list[Card],
    rank_counts: Counter,
    flush_group: list[Card],
    straight_vals: set[int] | None,
    rules: EvalRules,
) -> list[Card]:
    if rules.splash:
        return list(played)

    in_pattern: set[int] = set()  # card uids

    if hand_type == HandType.HIGH_CARD:
        if detect:
            best = max(detect, key=lambda c: int(c.rank))
            in_pattern.add(best.uid)
    elif hand_type in (HandType.PAIR, HandType.THREE_KIND, HandType.FOUR_KIND, HandType.FIVE_KIND):
        need = {
            HandType.PAIR: 2,
            HandType.THREE_KIND: 3,
            HandType.FOUR_KIND: 4,
            HandType.FIVE_KIND: 5,
        }[hand_type]
        best_rank = max((r for r, n in rank_counts.items() if n >= need), key=int)
        in_pattern.update(c.uid for c in detect if c.rank == best_rank)
    elif hand_type == HandType.TWO_PAIR:
        pair_ranks = sorted((r for r, n in rank_counts.items() if n >= 2), key=int, reverse=True)[:2]
        in_pattern.update(c.uid for c in detect if c.rank in pair_ranks)
    elif hand_type in (HandType.FULL_HOUSE, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE):
        in_pattern.update(c.uid for c in detect)  # always exactly 5 cards
    elif hand_type == HandType.FLUSH:
        in_pattern.update(c.uid for c in flush_group)
    elif hand_type in (HandType.STRAIGHT, HandType.STRAIGHT_FLUSH):
        assert straight_vals is not None
        for c in detect:
            vals = {int(c.rank)} | ({1} if c.rank == Rank.ACE else set())
            if vals & straight_vals:
                in_pattern.add(c.uid)

    # Preserve play order; Stone cards always score.
    return [c for c in played if c.uid in in_pattern or c.is_stone]


def best_hand_from(cards: list[Card], rules: EvalRules = DEFAULT_RULES) -> HandResult:
    """Brute-force best play over all 1-5 card subsets (tests, obs features, Mouth/Eye masks)."""
    from itertools import combinations

    best: HandResult | None = None
    for k in range(1, min(5, len(cards)) + 1):
        for combo in combinations(cards, k):
            hr = evaluate_hand(list(combo), rules)
            if best is None or hr.hand_type > best.hand_type:
                best = hr
    assert best is not None
    return best
