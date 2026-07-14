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
    PlayHand,
    RunLost,
    RunWon,
    UseConsumable,
    consumable_def,
)

from .. import action_codec as codec
from ..obs.encoder import ObsEncoder, canonical_order
from ..rewards import RewardConfig


class BalatroEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        deck: str = "d_red",
        reward_config: RewardConfig | None = None,
        reward_mode: str | None = None,
        enable_skip_blind: bool = True,
        strict_masking: bool = False,
        render_mode: str | None = None,
    ):
        load_all()
        self.deck_id = deck
        self.reward_config = reward_config or RewardConfig(mode=reward_mode or "blind")
        self.enable_skip_blind = enable_skip_blind
        self.strict_masking = strict_masking  # raise on illegal action (debug builds)
        self.render_mode = render_mode

        self.encoder = ObsEncoder()
        self.observation_space = self.encoder.space()
        self.action_space = gym.spaces.Discrete(codec.N_ACTIONS)

        self.game: Game | None = None
        self._hand_order: list[int] = []
        self._pending: tuple[int, int, int] | None = None  # (slot, lo, hi) TARGET phase
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

    def _refresh_hand_order(self) -> None:
        rnd = self.run.round
        self._hand_order = canonical_order(rnd.hand) if rnd is not None else []

    def _make_game(self, seed_str: str) -> Game:
        return Game(seed=seed_str, deck_id=self.deck_id, enable_skip_blind=self.enable_skip_blind)

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
        rnd = self.run.round
        hand_n = len(rnd.hand) if rnd is not None else 0
        if self._pending is not None:
            return codec.build_mask([], pending_targets=self._pending, hand_n=hand_n)

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

    def _dispatch(self, kind: str, payload):
        from balatro_gym.engine import (
            BuyPack,
            BuyShopCard,
            BuyVoucher,
            LeaveShop,
            PickPackItem,
            Reroll,
            SelectBlind,
            SellJoker,
            SkipBlind,
            SkipPack,
        )

        game = self.game
        assert game is not None

        if kind == "select_targets":
            if self._pending is None:
                raise IllegalActionError("no pending consumable")
            slot, lo, hi = self._pending
            if not (lo <= len(payload) <= hi):
                raise IllegalActionError("bad target arity")
            if any(p >= len(self._hand_order) for p in payload):
                raise IllegalActionError("target position beyond hand")
            engine_targets = tuple(self._hand_order[p] for p in payload)
            self._pending = None
            return game.apply(UseConsumable(slot, engine_targets))

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
                rnd = self.run.round
                if self.run.phase != Phase.PLAYING or rnd is None or not rnd.hand:
                    raise IllegalActionError("targeted consumable needs a hand")
                self._pending = (slot, lo, min(hi, len(rnd.hand)))
                return []  # env-level micro-phase; no engine step
            return game.apply(UseConsumable(slot))

        if kind in ("play", "discard"):
            if any(p >= len(self._hand_order) for p in payload):
                raise IllegalActionError("card position beyond hand")
            engine_idxs = tuple(sorted(self._hand_order[p] for p in payload))
            act = PlayHand(engine_idxs) if kind == "play" else Discard(engine_idxs)
            return game.apply(act)

        simple = {
            "select_blind": SelectBlind,
            "skip_blind": SkipBlind,
            "buy_voucher": BuyVoucher,
            "reroll": Reroll,
            "leave_shop": LeaveShop,
            "skip_pack": SkipPack,
        }
        if kind in simple:
            return game.apply(simple[kind]())
        if kind == "sell_joker":
            return game.apply(SellJoker(payload))
        if kind == "buy_card":
            return game.apply(BuyShopCard(payload))
        if kind == "buy_pack":
            return game.apply(BuyPack(payload))
        if kind == "pick_pack":
            return game.apply(PickPackItem(payload))
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
        }

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode != "ansi":
            return None
        run = self.run
        lines = [
            f"ante {run.ante} | blind {run.blind_index} | phase {self._env_phase()}"
            f" | ${run.money} | jokers: {[j.id for j in run.jokers]}"
        ]
        rnd = run.round
        if rnd is not None:
            lines.append(
                f"score {rnd.chips_scored:.0f}/{rnd.chips_needed:.0f}"
                f" | hands {rnd.hands_left} | discards {rnd.discards_left}"
            )
            hand = " ".join(
                f"[{slot}]{rnd.hand[i]!r}" for slot, i in enumerate(self._hand_order)
            )
            lines.append(f"hand: {hand}")
        shop = run.shop
        if shop is not None:
            items = ", ".join(f"{s.item_id}(${s.cost})" for s in shop.card_slots)
            packs = ", ".join(f"{p.pack_id}(${p.cost})" for p in shop.pack_slots)
            lines.append(f"shop: {items} | packs: {packs} | voucher: {shop.voucher}"
                         f" | reroll ${shop.reroll_cost}")
            if shop.open_pack:
                opts = ", ".join(
                    f"[{i}]{shop.open_pack.option_ids[i] or shop.open_pack.option_cards[i]!r}"
                    for i in range(len(shop.open_pack.option_kinds))
                )
                lines.append(f"pack ({shop.open_pack.picks_left} picks): {opts}")
        return "\n".join(lines)
