"""Gymnasium layer. Importing this module registers the envs:

    balatro_gym/Balatro-v0
    balatro_gym/BalatroBlind-v0
    balatro_gym/BalatroScriptedShop-v0
"""

from gymnasium.envs.registration import register

from .envs import BalatroBlindEnv, BalatroEnv, BalatroScriptedShopEnv, ScenarioConfig
from .rewards import RewardConfig

register(
    id="balatro_gym/Balatro-v0",
    entry_point="balatro_gym.env.envs:BalatroEnv",
    max_episode_steps=2000,
)
register(
    id="balatro_gym/BalatroBlind-v0",
    entry_point="balatro_gym.env.envs:BalatroBlindEnv",
    max_episode_steps=100,
)
register(
    id="balatro_gym/BalatroScriptedShop-v0",
    entry_point="balatro_gym.env.envs:BalatroScriptedShopEnv",
    max_episode_steps=2000,
)

__all__ = [
    "BalatroBlindEnv",
    "BalatroEnv",
    "BalatroScriptedShopEnv",
    "RewardConfig",
    "ScenarioConfig",
]
