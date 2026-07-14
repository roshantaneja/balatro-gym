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
    SelectBlind,
    SellJoker,
    SkipBlind,
    SkipPack,
    UseConsumable,
)
from .effects import CONSUMABLES, DECKS, boss_def, consumable_def
from .events import ConsumableUsed, Event
from .state import RunState, new_run_state
from .types import BlindKind, HandType, Phase

MAX_PLAY = 5


class IllegalActionError(Exception):
    pass


class Game:
    def __init__(
        self,
        seed: str,
        deck_id: str = "d_red",
        enable_skip_blind: bool = True,
    ):
        self.enable_skip_blind = enable_skip_blind
        self.run: RunState = new_run_state(seed=seed, deck_id=deck_id)
        deck = DECKS.get(deck_id)
        if deck is not None and deck.apply:
            deck.apply(self.run)  # type: ignore[operator]
        self.run.phase = Phase.BLIND_SELECT

    # ------------------------------------------------------------------ helpers
    @property
    def current_blind_kind(self) -> BlindKind:
        return BlindKind(min(self.run.blind_index, 2))

    def _boss(self):
        if self.run.round is not None and self.run.round.boss_id:
            return boss_def(self.run.round.boss_id).effect
        return None

    # ------------------------------------------------------------- legal actions
    def legal_actions(self) -> list[Action]:
        run = self.run
        phase = run.phase
        acts: list[Action] = []

        if phase == Phase.BLIND_SELECT:
            acts.append(SelectBlind())
            if self.enable_skip_blind and self.current_blind_kind != BlindKind.BOSS:
                acts.append(SkipBlind())
            acts.extend(self._consumable_actions())
            acts.extend(self._sell_actions())

        elif phase == Phase.PLAYING:
            rnd = run.round
            assert rnd is not None
            n = len(rnd.hand)
            boss = self._boss()
            can_play = rnd.hands_left > 0
            can_discard = rnd.discards_left > 0
            for k in range(1, min(MAX_PLAY, n) + 1):
                for idxs in combinations(range(n), k):
                    if can_play:
                        if boss is not None:
                            cards = [rnd.hand[i] for i in idxs]
                            if boss.validate_play(run, cards) is not None:
                                pass  # illegal under this boss
                            else:
                                acts.append(PlayHand(idxs))
                        else:
                            acts.append(PlayHand(idxs))
                    if can_discard:
                        acts.append(Discard(idxs))
            acts.extend(self._consumable_actions())
            acts.extend(self._sell_actions())

        elif phase == Phase.SHOP:
            shop = run.shop
            assert shop is not None
            for i, item in enumerate(shop.card_slots):
                if run.money >= item.cost:
                    if item.kind == "joker" and len(run.jokers) >= run.joker_slots:
                        continue
                    if item.kind != "joker" and len(run.consumables) >= run.consumable_slots:
                        continue
                    acts.append(BuyShopCard(i))
            for i, offer in enumerate(shop.pack_slots):
                if run.money >= offer.cost:
                    acts.append(BuyPack(i))
            if shop.voucher is not None:
                from .effects import VOUCHERS

                if run.money >= VOUCHERS[shop.voucher].cost:
                    acts.append(BuyVoucher())
            if run.money >= shop.reroll_cost:
                acts.append(Reroll())
            acts.append(LeaveShop())
            acts.extend(self._consumable_actions())
            acts.extend(self._sell_actions())

        elif phase == Phase.PACK_OPEN:
            shop = run.shop
            assert shop is not None and shop.open_pack is not None
            pack = shop.open_pack
            for i, kind in enumerate(pack.option_kinds):
                if kind == "joker" and len(run.jokers) >= run.joker_slots:
                    continue
                if kind in ("tarot", "planet", "spectral") and len(run.consumables) >= run.consumable_slots:
                    continue
                acts.append(PickPackItem(i))
            acts.append(SkipPack())

        return acts

    def _sell_actions(self) -> list[Action]:
        return [
            SellJoker(i)
            for i, j in enumerate(self.run.jokers)
            if "eternal" not in j.stickers
        ]

    def _consumable_actions(self) -> list[Action]:
        run = self.run
        acts: list[Action] = []
        hand_n = len(run.round.hand) if run.round else 0
        for slot, inst in enumerate(run.consumables):
            eff = consumable_def(inst.id).effect
            if not eff.can_use(run):
                continue
            lo, hi = eff.target_arity
            if lo == 0:
                acts.append(UseConsumable(slot))
            elif run.phase == Phase.PLAYING and hand_n > 0:
                # Targeted consumables only usable with a hand on the table.
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
            return round_mod.discard(run, tuple(sorted(action.indices)))

        if isinstance(action, SellJoker):
            self._check(
                phase in (Phase.BLIND_SELECT, Phase.PLAYING, Phase.SHOP)
                and 0 <= action.slot < len(run.jokers)
                and "eternal" not in run.jokers[action.slot].stickers,
                action,
            )
            return shop_mod.sell_joker(run, action.slot)

        if isinstance(action, UseConsumable):
            self._check(0 <= action.slot < len(run.consumables), action)
            inst = run.consumables[action.slot]
            eff = consumable_def(inst.id).effect
            lo, hi = eff.target_arity
            targets = list(action.targets)
            self._check(eff.can_use(run), action, "can_use failed")
            if lo > 0:
                rnd = run.round
                self._check(
                    phase == Phase.PLAYING
                    and rnd is not None
                    and lo <= len(targets) <= hi
                    and all(0 <= i < len(rnd.hand) for i in targets),
                    action,
                )
                cards = [rnd.hand[i] for i in targets]
            else:
                self._check(len(targets) == 0, action)
                cards = []
            run.consumables.pop(action.slot)
            eff.use(run, cards)
            run.stats["consumables_used"] = run.stats.get("consumables_used", 0) + 1
            key = f"{inst.kind.value}s_used"  # tarots_used / planets_used / spectrals_used
            run.stats[key] = run.stats.get(key, 0) + 1
            from .effects import joker_def as _jd

            for j in run.jokers:
                _jd(j.id).effect.on_consumable_used(run, j, inst.kind)
            return [ConsumableUsed(inst.id)]

        if isinstance(action, BuyShopCard):
            shop = run.shop
            self._check(
                phase == Phase.SHOP
                and shop is not None
                and 0 <= action.slot < len(shop.card_slots)
                and run.money >= shop.card_slots[action.slot].cost,
                action,
            )
            assert shop is not None
            item = shop.card_slots[action.slot]
            if item.kind == "joker":
                self._check(len(run.jokers) < run.joker_slots, action, "joker slots full")
            else:
                self._check(
                    len(run.consumables) < run.consumable_slots, action, "consumable slots full"
                )
            return shop_mod.buy_shop_card(run, action.slot)

        if isinstance(action, BuyPack):
            shop = run.shop
            self._check(
                phase == Phase.SHOP
                and shop is not None
                and 0 <= action.slot < len(shop.pack_slots)
                and run.money >= shop.pack_slots[action.slot].cost,
                action,
            )
            return shop_mod.buy_pack(run, action.slot)

        if isinstance(action, BuyVoucher):
            shop = run.shop
            from .effects import VOUCHERS

            self._check(
                phase == Phase.SHOP
                and shop is not None
                and shop.voucher is not None
                and run.money >= VOUCHERS[shop.voucher].cost,
                action,
            )
            return shop_mod.buy_voucher(run)

        if isinstance(action, Reroll):
            shop = run.shop
            self._check(
                phase == Phase.SHOP and shop is not None and run.money >= shop.reroll_cost,
                action,
            )
            return shop_mod.reroll(run)

        if isinstance(action, LeaveShop):
            self._check(phase == Phase.SHOP, action)
            return shop_mod.leave_shop(run)

        if isinstance(action, PickPackItem):
            shop = run.shop
            self._check(
                phase == Phase.PACK_OPEN
                and shop is not None
                and shop.open_pack is not None
                and 0 <= action.index < len(shop.open_pack.option_kinds),
                action,
            )
            assert shop is not None and shop.open_pack is not None
            kind = shop.open_pack.option_kinds[action.index]
            if kind == "joker":
                self._check(len(run.jokers) < run.joker_slots, action, "joker slots full")
            elif kind in ("tarot", "planet", "spectral"):
                self._check(
                    len(run.consumables) < run.consumable_slots, action, "consumable slots full"
                )
            return shop_mod.pick_pack_item(run, action.index)

        if isinstance(action, SkipPack):
            self._check(
                phase == Phase.PACK_OPEN and run.shop is not None and run.shop.open_pack is not None,
                action,
            )
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
    joker_ids: list[str] | None = None,
    hand_levels: dict[HandType, int] | None = None,
    hands: int | None = None,
    discards: int | None = None,
    money: int = 4,
) -> Game:
    """A Game dropped directly into PLAYING on a single blind (BalatroBlind-v0)."""
    game = Game(seed=seed, deck_id=deck_id, enable_skip_blind=False)
    run = game.run
    run.ante = ante
    run.blind_index = int(blind_kind)
    run.money = money
    if hands is not None:
        run.hands_per_round = hands
    if discards is not None:
        run.discards_per_round = discards
    if hand_levels:
        for ht, lvl in hand_levels.items():
            run.hand_levels[ht] = lvl
    if joker_ids:
        from .effects import JOKERS, new_joker_state
        from .state import JokerInst

        for jid in joker_ids:
            run.jokers.append(
                JokerInst(
                    id=jid,
                    uid=run.new_uid(),
                    sell_value=max(1, JOKERS[jid].base_cost // 2),
                    state=new_joker_state(jid),
                )
            )
    round_mod.start_blind(run, blind_kind)
    return game
