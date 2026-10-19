"""Env-level coverage for v2: new actions, pack target micro-phase, 12-slot
hands, strategist end-to-end, deck/stake options, obs bounds."""

from __future__ import annotations

import numpy as np

from balatro_gym.engine import BlindKind, TagInst, cardops
from balatro_gym.engine.round import acquire_tag
from balatro_gym.env import action_codec as codec
from balatro_gym.env.baselines import StrategistAgent
from balatro_gym.env.envs import BalatroBlindEnv, BalatroEnv, BalatroScriptedShopEnv, ScenarioConfig


def test_new_actions_are_maskable():
    env = BalatroEnv(stake="gold", deck="d_black")
    obs, info = env.reset(seed=1)
    run = env.run
    from tests.helpers import joker

    run.jokers = [joker("j_joker"), joker("j_jolly")]
    cardops.create_consumable(run, "c_mercury")
    cardops.grant_voucher(run, "v_directors_cut")
    run.money = 30
    env._mask_cache = None
    mask = env.action_masks()
    assert mask[codec.SWAP_JOKER_BASE] and mask[codec.SELL_CONSUMABLE_BASE]
    assert mask[codec.REROLL_BOSS_IDX]
    obs, r, term, trunc, info = env.step(codec.SWAP_JOKER_BASE)
    assert [j.id for j in run.jokers] == ["j_jolly", "j_joker"] and "illegal_action" not in info
    obs, r, term, trunc, info = env.step(codec.SELL_CONSUMABLE_BASE)
    assert run.consumables == [] and run.money == 31
    old = run.ante_boss
    obs, r, term, trunc, info = env.step(codec.REROLL_BOSS_IDX)
    assert run.ante_boss != old and run.money == 21


def test_pack_target_micro_phase():
    env = BalatroEnv()
    obs, info = env.reset(seed=2)
    run = env.run
    acquire_tag(run, TagInst("t_charm"))  # opens a mega arcana pack right away
    env._mask_cache = None
    env._refresh_hand_order()
    assert env._env_phase() == "pack_open"
    assert len(env._hand_order) == 8  # dealt pack hand
    pack = run.open_pack
    targeted = [i for i, cid in enumerate(pack.option_ids)
                if cid in ("c_magician", "c_empress", "c_hierophant", "c_lovers", "c_chariot",
                           "c_justice", "c_devil", "c_tower", "c_star", "c_sun", "c_moon", "c_world",
                           "c_strength", "c_death", "c_hanged_man")]
    if not targeted:
        return  # seed gave only untargeted tarots; nothing more to check
    i = targeted[0]
    mask = env.action_masks()
    assert mask[codec.PICK_PACK_BASE + i]
    obs, r, term, trunc, info = env.step(codec.PICK_PACK_BASE + i)
    assert info["phase"] == "target" and env._pending[0] == "pack"
    mask = info["action_mask"]
    assert mask.any() and all(codec.TARGET_BASE <= k < codec.USE_CONSUMABLE_BASE for k in np.flatnonzero(mask))
    obs, r, term, trunc, info = env.step(int(np.flatnonzero(mask)[0]))
    assert "illegal_action" not in info and env._pending is None
    assert run.consumables == []  # used, not stored


def test_twelve_slot_hand_is_playable():
    env = BalatroBlindEnv(scenario=ScenarioConfig(joker_ids=("j_juggler", "j_troubadour")))
    obs, info = env.reset(seed=3)
    assert len(env.run.round.hand) == 11
    mask = info["action_mask"]
    # a play using the 11th canonical slot (index 10) must be legal
    subset = (10,)
    assert mask[codec.PLAY_BASE + codec._SUBSET5_INDEX[subset]]
    obs, r, term, trunc, info = env.step(codec.PLAY_BASE + codec._SUBSET5_INDEX[subset])
    assert "illegal_action" not in info


def test_strategist_full_runs_are_legal_and_in_bounds():
    env = BalatroEnv(stake="red", deck="d_plasma")
    agent = StrategistAgent()
    space = env.observation_space
    for seed in range(2):
        obs, info = env.reset(seed=seed)
        assert space.contains(obs)
        done = False
        steps = 0
        while not done and steps < 600:
            a = agent.act(env, obs, info["action_mask"])
            assert info["action_mask"][a]
            obs, r, term, trunc, info = env.step(a)
            assert "illegal_action" not in info
            done = term
            steps += 1
            if steps % 50 == 0:
                assert space.contains(obs)


def test_scripted_shop_handles_pack_targets():
    env = BalatroScriptedShopEnv()
    for seed in range(3):
        obs, info = env.reset(seed=seed)
        assert info["phase"] in ("playing", "target", "terminal")
        done = False
        steps = 0
        agent = StrategistAgent()
        while not done and steps < 300:
            a = agent.act(env, obs, info["action_mask"])
            obs, r, term, trunc, info = env.step(a)
            assert info["phase"] in ("playing", "target", "terminal")
            done = term
            steps += 1


def test_blind_env_boss_scenarios():
    env = BalatroBlindEnv(scenario=ScenarioConfig(ante=2, blind_kind=BlindKind.BOSS, boss_id="b_needle"))
    obs, info = env.reset(seed=4)
    assert env.run.round.boss_id == "b_needle" and env.run.round.hands_left == 1
    assert obs["boss_id"][0] > 0
