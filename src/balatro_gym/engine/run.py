"""Game facade: phase machine, legal-action enumeration, action application.

This is the boundary the env layer consumes. legal_actions() is exact — the
env's action mask is a straight mapping of its output.
"""

from __future__ import annotations

from itertools import combinations

from . import round as round_mod
from . import shop as shop_mod
from .actions import (
    Action,
    BuyPack,
    BuyShopCard,
    BuyVoucher,
    Discard,
    LeaveShop,
    PickPackItem,
    PlayHand,
    Reroll,
    RerollBoss,
    SelectBlind,
    SellConsumable,
    SellJoker,
    SkipBlind,
    SkipPack,
    SwapJokers,
    UseConsumable,
)
from .cardops import can_afford, current_hand, has_consumable_room, has_joker_room
from .effects import DECKS, STAKES, VOUCHERS, consumable_def
from .events import ConsumableUsed, Event
from .levels import STAKE_ORDER
from .scoring import get_boss
from .state import RunState, new_run_state
from .types import STICKER_ETERNAL, BlindKind, HandType, Phase, Stake

MAX_PLAY = 5
BOSS_REROLL_COST = 10


class IllegalActionError(Exception):
    pass


class Game:
    def __init__(
        self,
        seed: str,
        deck_id: str = "d_red",
        stake: Stake | str = Stake.WHITE,
        enable_skip_blind: bool = True,
    ):
        self.enable_skip_blind = enable_skip_blind
        stake = Stake(stake)
        self.run: RunState = new_run_state(seed=seed, deck_id=deck_id, stake=stake)
        # Stakes are cumulative: apply every stake up to and including this one.
        for s in STAKE_ORDER[: STAKE_ORDER.index(stake) + 1]:
            sd = STAKES.get(s)
            if sd is not None and sd.apply:
                sd.apply(self.run)  # type: ignore[operator]
        deck = DECKS.get(deck_id)
        if deck is not None and deck.apply:
            deck.apply(self.run)  # type: ignore[operator]
        self.run.phase = Phase.BLIND_SELECT
        round_mod.begin_ante(self.run)

    # ------------------------------------------------------------------ helpers
    @property
    def current_blind_kind(self) -> BlindKind:
        return BlindKind(min(self.run.blind_index, 2))

    def _boss(self):
        return get_boss(self.run)

    def _can_reroll_boss(self) -> bool:
        run = self.run
        if run.phase != Phase.BLIND_SELECT or run.ante_boss is None:
            return False
        allowed = run.mods.get("boss_rerolls", 0)  # 1 Director's Cut, -1 Retcon (unlimited)
        if allowed == 0:
            return False
        if allowed > 0 and run.boss_rerolls_used >= allowed:
            return False
        return can_afford(run, BOSS_REROLL_COST)

    # ------------------------------------------------------------- legal actions
    def legal_actions(self) -> list[Action]:
        run = self.run
        phase = run.phase
        acts: list[Action] = []

        if phase == Phase.BLIND_SELECT:
            acts.append(SelectBlind())
            if self.enable_skip_blind and self.current_blind_kind != BlindKind.BOSS:
                acts.append(SkipBlind())
            if self._can_reroll_boss():
                acts.append(RerollBoss())
            acts.extend(self._consumable_actions())
            acts.extend(self._board_actions())

        elif phase == Phase.PLAYING:
            rnd = run.round
            assert rnd is not None
            n = len(rnd.hand)
            boss = self._boss()
            can_play = rnd.hands_left > 0
            can_discard = rnd.discards_left > 0
            for k in range(1, min(MAX_PLAY, n) + 1):
                for idxs in combinations(range(n), k):
                    if boss is not None:
                        cards = [rnd.hand[i] for i in idxs]
                        if can_play and boss.validate_play(run, cards) is None:
                            acts.append(PlayHand(idxs))
                        if can_discard and boss.validate_discard(run, cards) is None:
                            acts.append(Discard(idxs))
                    else:
                        if can_play:
                            acts.append(PlayHand(idxs))
                        if can_discard:
                            acts.append(Discard(idxs))
            acts.extend(self._consumable_actions())
            acts.extend(self._board_actions())

        elif phase == Phase.SHOP:
            shop = run.shop
            assert shop is not None
            for i, item in enumerate(shop.card_slots):
                if not can_afford(run, item.cost):
                    continue
                if item.kind == "joker" and not has_joker_room(run, item.edition):
                    continue
                if item.kind in ("tarot", "planet", "spectral") and not has_consumable_room(run):
                    continue
                acts.append(BuyShopCard(i))
            for i, offer in enumerate(shop.pack_slots):
                if can_afford(run, offer.cost):
                    acts.append(BuyPack(i))
            for i, vid in enumerate(shop.vouchers):
                if can_afford(run, VOUCHERS[vid].cost):
                    acts.append(BuyVoucher(i))
            if can_afford(run, shop_mod.reroll_cost_now(run)):
                acts.append(Reroll())
            acts.append(LeaveShop())
            acts.extend(self._consumable_actions())
            acts.extend(self._board_actions())

        elif phase == Phase.PACK_OPEN:
            pack = run.open_pack
            assert pack is not None
            hand_n = len(pack.hand)
            for i, kind in enumerate(pack.option_kinds):
                if not shop_mod.pack_pick_legal(run, i):
                    continue
                if kind in ("tarot", "planet", "spectral"):
                    lo, hi = consumable_def(pack.option_ids[i]).effect.target_arity
                    if lo > 0:
                        for k in range(lo, min(hi, hand_n) + 1):
                            for idxs in combinations(range(hand_n), k):
                                acts.append(PickPackItem(i, idxs))
                        continue
                acts.append(PickPackItem(i))
            acts.append(SkipPack())
            acts.extend(self._consumable_actions())
            acts.extend(self._board_actions())

        return acts

    def _board_actions(self) -> list[Action]:
        run = self.run
        acts: list[Action] = [
            SellJoker(i)
            for i, j in enumerate(run.jokers)
            if STICKER_ETERNAL not in j.stickers
        ]
        acts.extend(SellConsumable(i) for i in range(len(run.consumables)))
        if not (run.round is not None and run.round.boss_state.get("jokers_hidden")):
            acts.extend(SwapJokers(i) for i in range(len(run.jokers) - 1))
        return acts

    def _consumable_actions(self) -> list[Action]:
        run = self.run
        acts: list[Action] = []
        hand = current_hand(run)
        hand_n = len(hand)
        for slot, inst in enumerate(run.consumables):
            eff = consumable_def(inst.id).effect
            if not eff.can_use(run):
                continue
            lo, hi = eff.target_arity
            if lo == 0:
                acts.append(UseConsumable(slot))
            elif hand_n >= lo:
                # Targeted consumables need visible cards (round hand or pack hand).
                for k in range(lo, min(hi, hand_n) + 1):
                    for idxs in combinations(range(hand_n), k):
                        acts.append(UseConsumable(slot, idxs))
        return acts

    # ---------------------------------------------------------------- application
    def apply(self, action: Action) -> list[Event]:
        run = self.run
        phase = run.phase

        if isinstance(action, SelectBlind):
            self._check(phase == Phase.BLIND_SELECT, action)
            round_mod.start_blind(run, self.current_blind_kind)
            return []

        if isinstance(action, SkipBlind):
            self._check(
                phase == Phase.BLIND_SELECT
                and self.enable_skip_blind
                and self.current_blind_kind != BlindKind.BOSS,
                action,
            )
            return round_mod.skip_blind(run)

        if isinstance(action, RerollBoss):
            self._check(self._can_reroll_boss(), action)
            run.money -= BOSS_REROLL_COST
            run.boss_rerolls_used += 1
            round_mod.reroll_boss(run)
            return []

        if isinstance(action, PlayHand):
            rnd = run.round
            self._check(
                phase == Phase.PLAYING
                and rnd is not None
                and rnd.hands_left > 0
                and 1 <= len(action.indices) <= MAX_PLAY
                and all(0 <= i < len(rnd.hand) for i in action.indices)
                and len(set(action.indices)) == len(action.indices),
                action,
            )
            boss = self._boss()
            if boss is not None:
                assert rnd is not None
                reason = boss.validate_play(run, [rnd.hand[i] for i in action.indices])
                self._check(reason is None, action, reason)
            return round_mod.play_hand(run, tuple(sorted(action.indices)))

        if isinstance(action, Discard):
            rnd = run.round
            self._check(
                phase == Phase.PLAYING
                and rnd is not None
                and rnd.discards_left > 0
                and 1 <= len(action.indices) <= MAX_PLAY
                and all(0 <= i < len(rnd.hand) for i in action.indices)
                and len(set(action.indices)) == len(action.indices),
                action,
            )
            boss = self._boss()
            if boss is not None:
                assert rnd is not None
                reason = boss.validate_discard(run, [rnd.hand[i] for i in action.indices])
                self._check(reason is None, action, reason)
            return round_mod.discard(run, tuple(sorted(action.indices)))

        if isinstance(action, SellJoker):
            self._check(
                phase != Phase.WON and phase != Phase.GAME_OVER
                and 0 <= action.slot < len(run.jokers)
                and STICKER_ETERNAL not in run.jokers[action.slot].stickers,
                action,
            )
            return shop_mod.sell_joker(run, action.slot)

        if isinstance(action, SellConsumable):
            self._check(
                phase not in (Phase.WON, Phase.GAME_OVER)
                and 0 <= action.slot < len(run.consumables),
                action,
            )
            return shop_mod.sell_consumable(run, action.slot)

        if isinstance(action, SwapJokers):
            self._check(
                phase not in (Phase.WON, Phase.GAME_OVER)
                and 0 <= action.slot < len(run.jokers) - 1
                and not (run.round is not None and run.round.boss_state.get("jokers_hidden")),
                action,
            )
            return shop_mod.swap_jokers(run, action.slot)

        if isinstance(action, UseConsumable):
            self._check(0 <= action.slot < len(run.consumables), action)
            inst = run.consumables[action.slot]
            eff = consumable_def(inst.id).effect
            lo, hi = eff.target_arity
            targets = list(action.targets)
            self._check(eff.can_use(run), action, "can_use failed")
            hand = current_hand(run)
            if lo > 0:
                self._check(
                    lo <= len(targets) <= hi
                    and all(0 <= i < len(hand) for i in targets)
                    and len(set(targets)) == len(targets),
                    action,
                )
                cards = [hand[i] for i in targets]
            else:
                self._check(len(targets) == 0, action)
                cards = []
            run.consumables.pop(action.slot)
            eff.use(run, cards)
            shop_mod._note_consumable_used(run, inst.id)
            return [ConsumableUsed(inst.id)]

        if isinstance(action, BuyShopCard):
            shop = run.shop
            self._check(
                phase == Phase.SHOP
                and shop is not None
                and 0 <= action.slot < len(shop.card_slots)
                and can_afford(run, shop.card_slots[action.slot].cost),
                action,
            )
            assert shop is not None
            item = shop.card_slots[action.slot]
            if item.kind == "joker":
                self._check(has_joker_room(run, item.edition), action, "joker slots full")
            elif item.kind != "card":
                self._check(has_consumable_room(run), action, "consumable slots full")
            return shop_mod.buy_shop_card(run, action.slot)

        if isinstance(action, BuyPack):
            shop = run.shop
            self._check(
                phase == Phase.SHOP
                and shop is not None
                and 0 <= action.slot < len(shop.pack_slots)
                and can_afford(run, shop.pack_slots[action.slot].cost),
                action,
            )
            return shop_mod.buy_pack(run, action.slot)

        if isinstance(action, BuyVoucher):
            shop = run.shop
            self._check(
                phase == Phase.SHOP
                and shop is not None
                and 0 <= action.slot < len(shop.vouchers)
                and can_afford(run, VOUCHERS[shop.vouchers[action.slot]].cost),
                action,
            )
            return shop_mod.buy_voucher(run, action.slot)

        if isinstance(action, Reroll):
            shop = run.shop
            self._check(
                phase == Phase.SHOP and shop is not None
                and can_afford(run, shop_mod.reroll_cost_now(run)),
                action,
            )
            return shop_mod.reroll(run)

        if isinstance(action, LeaveShop):
            self._check(phase == Phase.SHOP, action)
            return shop_mod.leave_shop(run)

        if isinstance(action, PickPackItem):
            pack = run.open_pack
            self._check(
                phase == Phase.PACK_OPEN
                and pack is not None
                and 0 <= action.index < len(pack.option_kinds)
                and shop_mod.pack_pick_legal(run, action.index),
                action,
            )
            assert pack is not None
            kind = pack.option_kinds[action.index]
            targets = tuple(action.targets)
            if kind in ("tarot", "planet", "spectral"):
                lo, hi = consumable_def(pack.option_ids[action.index]).effect.target_arity
                self._check(
                    lo <= len(targets) <= hi
                    and all(0 <= i < len(pack.hand) for i in targets)
                    and len(set(targets)) == len(targets),
                    action, "bad pack targets",
                )
            else:
                self._check(not targets, action)
            return shop_mod.pick_pack_item(run, action.index, targets)

        if isinstance(action, SkipPack):
            self._check(phase == Phase.PACK_OPEN and run.open_pack is not None, action)
            return shop_mod.skip_pack(run)

        raise IllegalActionError(f"unknown action {action!r}")

    @staticmethod
    def _check(cond: bool, action: Action, reason: str | None = None) -> None:
        if not cond:
            raise IllegalActionError(f"illegal action {action!r}" + (f": {reason}" if reason else ""))


# ---------------------------------------------------------------------------
# Curriculum scenarios
# ---------------------------------------------------------------------------

def new_blind_scenario(
    seed: str,
    *,
    ante: int = 1,
    blind_kind: BlindKind = BlindKind.SMALL,
    deck_id: str = "d_red",
    stake: Stake | str = Stake.WHITE,
    joker_ids: list[str] | None = None,
    hand_levels: dict[HandType, int] | None = None,
    hands: int | None = None,
    discards: int | None = None,
    money: int = 4,
    boss_id: str | None = None,
) -> Game:
    """A Game dropped directly into PLAYING on a single blind (BalatroBlind-v0)."""
    game = Game(seed=seed, deck_id=deck_id, stake=stake, enable_skip_blind=False)
    run = game.run
    run.ante = ante
    run.money = money
    if hands is not None:
        run.hands_per_round = hands
    if discards is not None:
        run.discards_per_round = discards
    if hand_levels:
        for ht, lvl in hand_levels.items():
            run.hand_levels[ht] = lvl
    if joker_ids:
        from .cardops import add_joker, new_joker

        for jid in joker_ids:
            add_joker(run, new_joker(run, jid))
    round_mod.begin_ante(run)
    run.blind_index = int(blind_kind)
    if boss_id is not None:
        run.ante_boss = boss_id
    round_mod.start_blind(run, blind_kind)
    return game
