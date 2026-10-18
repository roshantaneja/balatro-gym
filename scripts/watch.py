"""Live game viewer: watch an agent play full Balatro runs in the browser.

Usage:
  python scripts/watch.py                               # heuristic agent
  python scripts/watch.py --agent random
  python scripts/watch.py --agent model --model ppo_balatro.zip
  python scripts/watch.py --port 8712

Then open http://localhost:8712 — play/pause, speed, step, new run.
Stdlib-only server; the trained-model agent needs the ".[train]" extra.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from balatro_gym.engine import (
    BlindCleared,
    BlindSkipped,
    Discarded,
    Edition,
    HandScored,
    ItemBought,
    JokerSold,
    RoundPayout,
    RunLost,
    RunWon,
    boss_def,
)
from balatro_gym.env import action_codec as codec
from balatro_gym.env.baselines import HeuristicAgent, RandomLegalAgent
from balatro_gym.env.envs import BalatroEnv

VIEWER_HTML = Path(__file__).parent / "viewer.html"

SUIT_NAMES = ["spades", "hearts", "clubs", "diamonds"]
RANK_LABELS = {11: "J", 12: "Q", 13: "K", 14: "A"}


class ModelAgent:
    def __init__(self, path: str):
        from sb3_contrib import MaskablePPO

        self.model = MaskablePPO.load(path)

    def act(self, env, obs, mask) -> int:
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)


def make_agent(name: str, model_path: str | None):
    if name == "planner":
        from balatro_gym.env.planner import PlannerAgent

        return PlannerAgent()
    if name == "strategist":
        from balatro_gym.env.baselines import StrategistAgent

        return StrategistAgent()
    if name == "model":
        if not model_path:
            raise SystemExit("--agent model requires --model <path.zip>")
        return ModelAgent(model_path)
    if name == "random":
        return RandomLegalAgent()
    return HeuristicAgent()


class Session:
    """Holds one env + agent; steps on demand from HTTP requests."""

    def __init__(self, agent_name: str, model_path: str | None, seed: int,
                 deck: str = "d_red", stake: str = "white"):
        self.lock = threading.Lock()
        self.agent_name = agent_name
        self.agent = make_agent(agent_name, model_path)
        self.env = BalatroEnv(render_mode="ansi", deck=deck, stake=stake)
        self.seed = seed
        self.step_count = 0
        self.total_reward = 0.0
        self.last_action: str | None = None
        self.last_selected: list[int] = []
        self.feed: list[dict] = []
        self.terminal = False
        self.episode_stats: dict | None = None
        self.obs, _ = self.env.reset(seed=seed)

    # ------------------------------------------------------------------ steps
    def reset(self, seed: int | None = None):
        with self.lock:
            if seed is not None:
                self.seed = seed
            else:
                self.seed += 1
            self.obs, _ = self.env.reset(seed=self.seed)
            self.step_count = 0
            self.total_reward = 0.0
            self.last_action = None
            self.last_selected = []
            self.feed = []
            self.terminal = False
            self.episode_stats = None

    def step(self):
        """Advance one step with the agent choosing."""
        with self.lock:
            if self.terminal:
                return
            mask = self.env.action_masks()
            legal = np.flatnonzero(mask)
            if len(legal) == 0:
                self.terminal = True
                return
            action = int(self.agent.act(self.env, self.obs, mask))
            if not mask[action]:
                action = int(legal[0])
            self._apply(action)

    def act(self, action: int) -> str | None:
        """Apply a human-chosen action. Returns an error string if illegal."""
        with self.lock:
            if self.terminal:
                return "run is over — start a new run"
            mask = self.env.action_masks()
            if not (0 <= action < len(mask)) or not mask[action]:
                return f"illegal action: {codec.to_string(action)}"
            self._apply(action)
            return None

    def _apply(self, action: int):
        kind, payload = codec.decode_kind(action)
        self.last_action = codec.to_string(action)
        self.last_selected = (
            list(payload) if kind in ("play", "discard", "select_targets") else []
        )
        self.obs, r, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        self.total_reward += r
        for e in info.get("events", []):
            self._feed_event(e)
        if terminated or truncated:
            self.terminal = True
            self.episode_stats = info.get("episode_stats")

    def _feed_event(self, e):
        d: dict | None = None
        if isinstance(e, HandScored):
            res = e.result
            d = {
                "kind": "hand",
                "text": f"{res.hand_result.hand_type.name.replace('_', ' ').title()}: "
                f"{res.chips:.0f} x {res.mult:g} = {res.score:.0f}",
                "breakdown": [
                    f"{ev.source_kind}:{ev.source_id} {ev.op} {ev.amount:g}"
                    for ev in res.events
                    if ev.op in ("+c", "+m", "xm", "+$")
                ][:14],
            }
        elif isinstance(e, BlindCleared):
            d = {"kind": "clear", "text": f"Blind cleared! (ante {e.ante})"}
        elif isinstance(e, RoundPayout):
            d = {
                "kind": "money",
                "text": f"Cash out: ${e.total} (${e.blind_reward} blind, "
                f"${e.unused_hands} hands, ${e.interest} interest, ${e.other} other)",
            }
        elif isinstance(e, RunWon):
            d = {"kind": "win", "text": "RUN WON — ante 8 boss defeated!"}
        elif isinstance(e, RunLost):
            d = {
                "kind": "loss",
                "text": f"Run lost at ante {e.ante} "
                f"({e.chips_scored:.0f}/{e.chips_needed:.0f})",
            }
        elif isinstance(e, ItemBought):
            d = {"kind": "shop", "text": f"Bought {e.item_id or e.kind} (${e.cost})"}
        elif isinstance(e, JokerSold):
            d = {"kind": "shop", "text": f"Sold {e.joker_id} (+${e.value})"}
        elif isinstance(e, BlindSkipped):
            d = {"kind": "shop", "text": f"Skipped blind → tag {e.tag_id}"}
        elif isinstance(e, Discarded):
            d = {"kind": "info", "text": f"Discarded {e.count} cards"}
        if d:
            self.feed.append(d)
            self.feed = self.feed[-30:]

    # ------------------------------------------------------------------ state
    def _card(self, card, boss, run, face_down_uids, selected_slot=None) -> dict:
        return {
            "rank": RANK_LABELS.get(int(card.rank), str(int(card.rank))),
            "suit": SUIT_NAMES[int(card.suit)],
            "stone": card.is_stone,
            "enhancement": str(card.enhancement) if str(card.enhancement) != "none" else None,
            "edition": str(card.edition) if str(card.edition) != "none" else None,
            "seal": str(card.seal) if str(card.seal) != "none" else None,
            "debuffed": bool(boss and boss.card_debuffed(run, card)),
            "face_down": card.uid in face_down_uids,
            "selected": selected_slot in self.last_selected if selected_slot is not None else False,
        }

    def state(self) -> dict:
        with self.lock:
            env = self.env
            run = env.run
            rnd = run.round
            from balatro_gym.engine import cardops

            boss = (
                boss_def(rnd.boss_id).effect
                if (rnd and rnd.boss_id and not rnd.boss_state.get("disabled")) else None
            )
            face_down = set(rnd.face_down_uids) if rnd else set()

            hand = []
            visible = cardops.current_hand(run)
            if visible:
                for slot, i in enumerate(env._hand_order):
                    hand.append(self._card(visible[i], boss, run, face_down, slot))

            shop = None
            if run.shop is not None or run.open_pack is not None:
                s = run.shop
                shop = {
                    "cards": [
                        {"kind": it.kind, "id": it.item_id or repr(it.card), "cost": it.cost,
                         "edition": str(it.edition) if str(it.edition) != "none" else None,
                         "stickers": list(it.stickers)}
                        for it in (s.card_slots if s else [])
                    ],
                    "packs": [{"id": p.pack_id, "cost": p.cost} for p in (s.pack_slots if s else [])],
                    "vouchers": list(s.vouchers) if s else [],
                    "reroll_cost": (0 if s.free_rerolls else s.reroll_cost) if s else 0,
                    "open_pack": None,
                }
                if run.open_pack is not None:
                    p = run.open_pack
                    shop["open_pack"] = {
                        "id": p.pack_id,
                        "picks_left": p.picks_left,
                        "options": [
                            {"kind": p.option_kinds[i],
                             "id": p.option_ids[i],
                             "edition": p.option_editions[i] if i < len(p.option_editions) else "none",
                             "card": self._card(p.option_cards[i], None, run, set())
                             if p.option_cards[i] else None}
                            for i in range(len(p.option_kinds))
                        ],
                    }

            mask = (
                self.env.action_masks()
                if not self.terminal
                else np.zeros(codec.N_ACTIONS, dtype=bool)
            )
            return {
                "agent": self.agent_name,
                "legal": [int(i) for i in np.flatnonzero(mask)],
                "seed": self.seed,
                "step": self.step_count,
                "reward": round(self.total_reward, 3),
                "terminal": self.terminal,
                "episode_stats": self.episode_stats,
                "phase": env._env_phase(),
                "ante": run.ante,
                "blind_index": run.blind_index,
                "boss": boss_def(rnd.boss_id).name if (rnd and rnd.boss_id) else None,
                "boss_disabled": bool(rnd and rnd.boss_state.get("disabled")),
                "ante_boss": boss_def(run.ante_boss).name if run.ante_boss else None,
                "blind_tags": [t.id if t else None for t in run.blind_tags],
                "stake": str(run.stake),
                "deck": run.deck_id,
                "codec": {
                    "hand_slots": codec.HAND_SLOTS,
                    "SELECT_BLIND": codec.SELECT_BLIND, "SKIP_BLIND": codec.SKIP_BLIND,
                    "REROLL_BOSS": codec.REROLL_BOSS_IDX, "PLAY": codec.PLAY_BASE,
                    "DISCARD": codec.DISCARD_BASE, "TARGET": codec.TARGET_BASE,
                    "USE_CONSUMABLE": codec.USE_CONSUMABLE_BASE,
                    "SELL_CONSUMABLE": codec.SELL_CONSUMABLE_BASE,
                    "SELL_JOKER": codec.SELL_JOKER_BASE, "SWAP_JOKERS": codec.SWAP_JOKER_BASE,
                    "BUY_CARD": codec.BUY_CARD_BASE, "BUY_PACK": codec.BUY_PACK_BASE,
                    "VOUCHER": codec.BUY_VOUCHER_BASE, "REROLL": codec.REROLL_IDX,
                    "LEAVE": codec.LEAVE_SHOP_IDX, "PICK_PACK": codec.PICK_PACK_BASE,
                    "SKIP_PACK": codec.SKIP_PACK_IDX,
                },
                "money": run.money,
                "chips_scored": rnd.chips_scored if rnd else 0,
                "chips_needed": rnd.chips_needed if rnd else 0,
                "hands_left": rnd.hands_left if rnd else 0,
                "discards_left": rnd.discards_left if rnd else 0,
                "deck_remaining": len(rnd.deck_remaining) if rnd else len(run.full_deck),
                "deck_total": len(run.full_deck),
                "hand": hand,
                "jokers": [
                    {"id": "???" if (rnd and rnd.boss_state.get("jokers_hidden")) else j.id,
                     "sell": j.sell_value,
                     "edition": str(j.edition) if str(j.edition) != "none" else None,
                     "stickers": sorted(j.stickers),
                     "debuffed": j.debuffed or bool(rnd and rnd.boss_state.get("disabled_joker_uid") == j.uid),
                     "state": {k: v for k, v in j.state.items()
                               if isinstance(v, (int, float)) and not isinstance(v, bool)}}
                    for j in run.jokers
                ],
                "joker_slots": run.joker_slots,
                "consumables": [
                    {"id": c.id, "kind": str(c.kind), "negative": c.edition == Edition.NEGATIVE,
                     "sell": c.sell_value}
                    for c in run.consumables
                ],
                "consumable_slots": run.consumable_slots,
                "tags": [t.id for t in run.tags],
                "vouchers": sorted(run.vouchers),
                "shop": shop,
                "last_action": self.last_action,
                "feed": list(reversed(self.feed)),
            }


class Handler(BaseHTTPRequestHandler):
    session: Session = None  # set at startup

    def log_message(self, *args):  # quiet
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = VIEWER_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            self._json(self.session.state())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/step":
            self.session.step()
            self._json(self.session.state())
        elif self.path == "/api/act":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            error = self.session.act(int(body.get("action", -1)))
            state = self.session.state()
            if error:
                state["error"] = error
            self._json(state)
        elif self.path == "/api/reset":
            self.session.reset()
            self._json(self.session.state())
        else:
            self.send_error(404)


def main():
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["heuristic", "strategist", "planner", "random", "model"],
                        default="planner")
    parser.add_argument("--deck", default="d_red", help="deck id, e.g. d_red, d_plasma")
    parser.add_argument("--stake", default="white", help="white|red|green|black|blue|purple|orange|gold")
    parser.add_argument("--model", default=None, help="path to MaskablePPO .zip")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", 8712)),
        help="defaults to $PORT if set, else 8712",
    )
    args = parser.parse_args()

    Handler.session = Session(args.agent, args.model, args.seed, args.deck, args.stake)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"watching {args.agent} agent at http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
