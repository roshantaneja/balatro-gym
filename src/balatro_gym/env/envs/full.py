"""BalatroEnv: the full-run Gymnasium environment with legal-action masking."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from balatro_gym.content import load_all
from balatro_gym.engine import (
    Discard,
    Game,
    IllegalActionError,
    Phase,
    PickPackItem,
    PlayHand,
    RunLost,
    RunWon,
    Stake,
    UseConsumable,
    cardops,
    consumable_def,
)

from .. import action_codec as codec
from ..obs.encoder import ObsEncoder, canonical_order
from ..rewards import RewardConfig

# (kind, index, lo, hi): kind is "consumable" (slot) or "pack" (option index)
Pending = tuple[str, int, int, int]


class BalatroEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        deck: str = "d_red",
        stake: str = "white",
        reward_config: RewardConfig | None = None,
        reward_mode: str | None = None,
        enable_skip_blind: bool = True,
        strict_masking: bool = False,
        render_mode: str | None = None,
    ):
        load_all()
        self.deck_id = deck
        self.stake = Stake(stake)
        self.reward_config = reward_config or RewardConfig(mode=reward_mode or "blind")
        self.enable_skip_blind = enable_skip_blind
        self.strict_masking = strict_masking  # raise on illegal action (debug builds)
        self.render_mode = render_mode

        self.encoder = ObsEncoder()
        self.observation_space = self.encoder.space()
        self.action_space = gym.spaces.Discrete(codec.N_ACTIONS)

        self.game: Game | None = None
        self._hand_order: list[int] = []
        self._pending: Pending | None = None
        self._episode_steps = 0
        self._mask_cache: np.ndarray | None = None  # invalidated on every state change

    # -------------------------------------------------------------- helpers
    @property
    def run(self):
        assert self.game is not None
        return self.game.run

    def _env_phase(self) -> str:
        if self._pending is not None:
            return "target"
        phase = self.run.phase
        if phase in (Phase.WON, Phase.GAME_OVER):
            return "terminal"
        return str(phase.value)

    def _visible_hand(self) -> list:
        return cardops.current_hand(self.run)

    def _refresh_hand_order(self) -> None:
        hand = self._visible_hand()
        self._hand_order = canonical_order(hand) if hand else []

    def _make_game(self, seed_str: str) -> Game:
        return Game(
            seed=seed_str, deck_id=self.deck_id, stake=self.stake,
            enable_skip_blind=self.enable_skip_blind,
        )

    # -------------------------------------------------------------- masking
    def action_masks(self) -> np.ndarray:
        """MaskablePPO looks for exactly this method name."""
        if self._mask_cache is not None:
            return self._mask_cache
        self._mask_cache = self._compute_mask()
        return self._mask_cache

    def _compute_mask(self) -> np.ndarray:
        if self.game is None or self.run.phase in (Phase.WON, Phase.GAME_OVER):
            return np.zeros(codec.N_ACTIONS, dtype=bool)
        hand_n = min(len(self._hand_order), codec.HAND_SLOTS)
        if self._pending is not None:
            _, _, lo, hi = self._pending
            return codec.build_mask([], pending_targets=(lo, hi), hand_n=hand_n)

        engine_actions = self.game.legal_actions()
        # Remap play/discard subsets from engine-hand indices to canonical slots.
        remap = {eng_i: slot for slot, eng_i in enumerate(self._hand_order)}
        remapped = []
        for a in engine_actions:
            if isinstance(a, (PlayHand, Discard)):
                slots = tuple(sorted(remap[i] for i in a.indices))
                remapped.append(type(a)(slots))
            else:
                remapped.append(a)
        return codec.build_mask(remapped, hand_n=hand_n)

    # ------------------------------------------------------------------ core
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        seed_str = f"run-{self.np_random.integers(0, 2**63)}"
        self.game = self._make_game(seed_str)
        self._pending = None
        self._episode_steps = 0
        self._mask_cache = None
        self._refresh_hand_order()
        obs = self._obs()
        return obs, self._info()

    def step(self, action: int):
        assert self.game is not None
        action = int(action)
        events = []
        reward = 0.0
        info_extra: dict[str, Any] = {}
        self._episode_steps += 1

        kind, payload = codec.decode_kind(action)
        self._mask_cache = None
        try:
            events = self._dispatch(kind, payload)
        except IllegalActionError as exc:
            if self.strict_masking:
                raise
            info_extra["illegal_action"] = str(exc)

        reward = self.reward_config.compute(events, self.run.ante)
        self._refresh_hand_order()

        terminated = self.run.phase in (Phase.WON, Phase.GAME_OVER)
        obs = self._obs()
        info = self._info()
        info.update(info_extra)
        info["events"] = events
        for e in events:
            if isinstance(e, (RunWon, RunLost)):
                info["episode_stats"] = self._episode_stats(won=isinstance(e, RunWon))
        return obs, reward, terminated, False, info

    def _engine_targets(self, payload) -> tuple[int, ...]:
        if any(p >= len(self._hand_order) for p in payload):
            raise IllegalActionError("target position beyond hand")
        return tuple(self._hand_order[p] for p in payload)

    def _dispatch(self, kind: str, payload):
        from balatro_gym.engine import (
            BuyPack,
            BuyShopCard,
            BuyVoucher,
            LeaveShop,
            Reroll,
            RerollBoss,
            SelectBlind,
            SellConsumable,
            SellJoker,
            SkipBlind,
            SkipPack,
            SwapJokers,
        )

        game = self.game
        assert game is not None

        if kind == "select_targets":
            if self._pending is None:
                raise IllegalActionError("no pending consumable")
            pkind, index, lo, hi = self._pending
            if not (lo <= len(payload) <= hi):
                raise IllegalActionError("bad target arity")
            targets = self._engine_targets(payload)
            self._pending = None
            if pkind == "pack":
                return game.apply(PickPackItem(index, targets))
            return game.apply(UseConsumable(index, targets))

        if self._pending is not None:
            raise IllegalActionError("must select targets first")

        if kind == "use_consumable":
            slot = payload
            if not (0 <= slot < len(self.run.consumables)):
                raise IllegalActionError("empty consumable slot")
            inst = self.run.consumables[slot]
            eff = consumable_def(inst.id).effect
            lo, hi = eff.target_arity
            if lo > 0:
                hand = self._visible_hand()
                if not hand or len(hand) < lo:
                    raise IllegalActionError("targeted consumable needs a hand")
                if not eff.can_use(self.run):
                    raise IllegalActionError("consumable cannot be used now")
                self._pending = ("consumable", slot, lo, min(hi, len(hand)))
                return []  # env-level micro-phase; no engine step
            return game.apply(UseConsumable(slot))

        if kind == "pick_pack":
            pack = self.run.open_pack
            if pack is None or not (0 <= payload < len(pack.option_kinds)):
                raise IllegalActionError("no such pack option")
            okind = pack.option_kinds[payload]
            if okind in ("tarot", "planet", "spectral"):
                lo, hi = consumable_def(pack.option_ids[payload]).effect.target_arity
                if lo > 0:
                    from balatro_gym.engine.shop import pack_pick_legal

                    if not pack_pick_legal(self.run, payload):
                        raise IllegalActionError("pack option cannot be used")
                    self._pending = ("pack", payload, lo, min(hi, len(pack.hand)))
                    return []
            return game.apply(PickPackItem(payload))

        if kind in ("play", "discard"):
            engine_idxs = tuple(sorted(self._engine_targets(payload)))
            act = PlayHand(engine_idxs) if kind == "play" else Discard(engine_idxs)
            return game.apply(act)

        simple = {
            "select_blind": SelectBlind,
            "skip_blind": SkipBlind,
            "reroll_boss": RerollBoss,
            "reroll": Reroll,
            "leave_shop": LeaveShop,
            "skip_pack": SkipPack,
        }
        if kind in simple:
            return game.apply(simple[kind]())
        if kind == "sell_joker":
            return game.apply(SellJoker(payload))
        if kind == "sell_consumable":
            return game.apply(SellConsumable(payload))
        if kind == "swap_jokers":
            return game.apply(SwapJokers(payload))
        if kind == "buy_card":
            return game.apply(BuyShopCard(payload))
        if kind == "buy_pack":
            return game.apply(BuyPack(payload))
        if kind == "buy_voucher":
            return game.apply(BuyVoucher(payload))
        raise IllegalActionError(f"reserved/unknown action kind {kind!r}")

    # ------------------------------------------------------------------ obs/info
    def _obs(self) -> dict:
        return self.encoder.encode(
            self.run,
            env_phase=self._env_phase(),
            hand_order=self._hand_order,
            pending=self._pending,
            enable_skip=self.enable_skip_blind,
        )

    def _info(self) -> dict:
        return {
            "action_mask": self.action_masks(),
            "state": self.run,
            "phase": self._env_phase(),
        }

    def _episode_stats(self, won: bool) -> dict:
        run = self.run
        return {
            "win": won,
            "ante_reached": run.ante,
            "money_final": run.money,
            "hands_played": run.stats.get("hands_played_run", 0),
            "jokers_final": [j.id for j in run.jokers],
            "steps": self._episode_steps,
            "deck": run.deck_id,
            "stake": str(run.stake),
        }

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode != "ansi":
            return None
        run = self.run
        lines = [
            f"ante {run.ante} | blind {run.blind_index} | phase {self._env_phase()}"
            f" | ${run.money} | jokers: {[j.id for j in run.jokers]}"
            f" | boss: {run.ante_boss} | tags: {[t.id for t in run.tags]}"
        ]
        rnd = run.round
        if rnd is not None:
            lines.append(
                f"score {rnd.chips_scored:.0f}/{rnd.chips_needed:.0f}"
                f" | hands {rnd.hands_left} | discards {rnd.discards_left}"
            )
        hand = self._visible_hand()
        if hand:
            lines.append("hand: " + " ".join(f"[{slot}]{hand[i]!r}" for slot, i in enumerate(self._hand_order)))
        shop = run.shop
        if shop is not None:
            items = ", ".join(f"{s.item_id or s.card!r}(${s.cost})" for s in shop.card_slots)
            packs = ", ".join(f"{p.pack_id}(${p.cost})" for p in shop.pack_slots)
            lines.append(f"shop: {items} | packs: {packs} | vouchers: {shop.vouchers}"
                         f" | reroll ${shop.reroll_cost}")
        pack = run.open_pack
        if pack is not None:
            opts = ", ".join(
                f"[{i}]{pack.option_ids[i] or pack.option_cards[i]!r}"
                for i in range(len(pack.option_kinds))
            )
            lines.append(f"pack ({pack.picks_left} picks): {opts}")
        return "\n".join(lines)
