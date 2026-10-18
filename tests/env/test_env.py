"""Gym layer: codec bijectivity, mask==legality, full episodes via gym API."""

import gymnasium as gym
import numpy as np
import pytest

import balatro_gym.env  # noqa: F401  (registers envs, loads content)
import tests.helpers  # noqa: F401
from balatro_gym.env import action_codec as codec
from balatro_gym.env.envs import BalatroBlindEnv, BalatroEnv, BalatroScriptedShopEnv


class TestCodec:
    def test_subset_counts(self):
        from math import comb

        n = codec.HAND_SLOTS
        assert len(codec.SUBSETS_5) == sum(comb(n, k) for k in range(1, 6))
        assert len(codec.SUBSETS_3) == sum(comb(n, k) for k in range(1, 4))
        assert codec.N_ACTIONS == codec.SKIP_PACK_IDX + 1

    def test_decode_covers_all_indices(self):
        kinds = [codec.decode_kind(i)[0] for i in range(codec.N_ACTIONS)]
        assert kinds.count("reserved") == 0
        assert kinds[0] == "select_blind"
        assert kinds[codec.PLAY_BASE] == "play"
        assert kinds[codec.SKIP_PACK_IDX] == "skip_pack"

    def test_encode_decode_roundtrip(self):
        from balatro_gym.engine import Discard, PlayHand

        for subset in codec.SUBSETS_5[:20] + codec.SUBSETS_5[-20:]:
            idx = codec.encode(PlayHand(subset))
            assert codec.decode_kind(idx) == ("play", subset)
            idx = codec.encode(Discard(subset))
            assert codec.decode_kind(idx) == ("discard", subset)

    def test_to_string(self):
        assert codec.to_string(0) == "select_blind"
        assert "play(" in codec.to_string(codec.PLAY_BASE)


class TestMasking:
    def test_mask_matches_legality_along_episode(self):
        """Every masked-legal action must be applicable; sampled per step."""
        env = BalatroEnv()
        env.reset(seed=7)
        rng = np.random.default_rng(7)
        for _ in range(200):
            mask = env.action_masks()
            legal = np.flatnonzero(mask)
            if len(legal) == 0:
                break
            action = int(rng.choice(legal))
            _, _, terminated, _, info = env.step(action)
            assert "illegal_action" not in info, (
                f"masked-legal action {codec.to_string(action)} rejected: "
                f"{info['illegal_action']}"
            )
            if terminated:
                break

    def test_illegal_action_is_noop_with_flag(self):
        env = BalatroEnv()
        env.reset(seed=3)
        mask = env.action_masks()
        illegal = int(np.flatnonzero(~mask)[0])
        _, reward, _, _, info = env.step(illegal)
        assert info.get("illegal_action")
        assert reward == 0.0

    def test_strict_masking_raises(self):
        from balatro_gym.engine import IllegalActionError

        env = BalatroEnv(strict_masking=True)
        env.reset(seed=3)
        mask = env.action_masks()
        illegal = int(np.flatnonzero(~mask)[0])
        with pytest.raises(IllegalActionError):
            env.step(illegal)


class TestEpisodes:
    def _rollout(self, env, seed, max_steps=3000):
        obs, info = env.reset(seed=seed)
        space = env.observation_space
        assert space.contains(obs), "reset obs out of space"
        rng = np.random.default_rng(seed)
        total = 0.0
        for step in range(max_steps):
            legal = np.flatnonzero(env.action_masks())
            assert len(legal) > 0, f"no legal actions at step {step}"
            obs, r, terminated, truncated, info = env.step(int(rng.choice(legal)))
            total += r
            if terminated or truncated:
                return total, step + 1, info
        pytest.fail("episode did not terminate")

    def test_full_env_random_episodes(self):
        env = BalatroEnv()
        for seed in [0, 1, 2]:
            total, steps, info = self._rollout(env, seed)
            assert "episode_stats" in info
            assert info["episode_stats"]["ante_reached"] >= 1

    def test_blind_env_short_episodes(self):
        env = BalatroBlindEnv()
        for seed in range(5):
            total, steps, info = self._rollout(env, seed, max_steps=100)
            assert steps <= 40

    def test_scripted_shop_only_play_phases(self):
        env = BalatroScriptedShopEnv()
        obs, info = env.reset(seed=11)
        rng = np.random.default_rng(11)
        for _ in range(500):
            assert info["phase"] in ("playing", "target", "terminal")
            legal = np.flatnonzero(env.action_masks())
            if len(legal) == 0:
                break
            obs, r, terminated, truncated, info = env.step(int(rng.choice(legal)))
            if terminated:
                break

    def test_gym_make_registration(self):
        env = gym.make("balatro_gym/BalatroBlind-v0")
        obs, info = env.reset(seed=0)
        assert "action_mask" in info


class TestDeterminism:
    def test_same_seed_same_episode(self):
        def run(seed):
            env = BalatroEnv()
            obs, _ = env.reset(seed=seed)
            rng = np.random.default_rng(99)
            trace = []
            for _ in range(300):
                legal = np.flatnonzero(env.action_masks())
                if len(legal) == 0:
                    break
                a = int(legal[rng.integers(len(legal))])
                obs, r, term, trunc, info = env.step(a)
                trace.append((a, r))
                if term or trunc:
                    break
            return trace

        assert run(42) == run(42)

    def test_reward_modes(self):
        from balatro_gym.env import RewardConfig

        env = BalatroBlindEnv(reward_config=RewardConfig(mode="shaped"))
        obs, info = env.reset(seed=5)
        rng = np.random.default_rng(5)
        rewards = []
        for _ in range(50):
            legal = np.flatnonzero(env.action_masks())
            if len(legal) == 0:
                break
            obs, r, term, trunc, info = env.step(int(rng.choice(legal)))
            rewards.append(r)
            if term:
                break
        # Shaped mode should produce some nonzero intermediate reward on plays.
        assert any(r != 0 for r in rewards)
