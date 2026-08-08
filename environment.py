"""Mario environment creation and observation preprocessing."""

from __future__ import annotations

import gymnasium as gym
import gym_super_mario_bros
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class SkipFrame(gym.Wrapper):
    """Repeat an action for several frames and accumulate its reward."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return observation, total_reward, terminated, truncated, info


def create_mario_env(
    game: str = "SuperMarioBros-1-1-v0",
    frame_skip: int = 4,
    frame_stack: int = 4,
    resize: int = 84,
    render_mode: str | None = None,
) -> gym.Env:
    """Create Mario with discrete actions and stacked 84x84 grayscale frames.

    The returned observation is a stack of ``frame_stack`` recent frames.  With
    the default settings it is compatible with a DQN expecting ``(4, 84, 84)``.
    """
    env = gym.make(game, render_mode=render_mode)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=frame_skip)
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=(resize, resize))
    env = FrameStackObservation(env, stack_size=frame_stack)
    return env
