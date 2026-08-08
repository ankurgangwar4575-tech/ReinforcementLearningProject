"""Run a trained Double DQN Mario agent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
import torch
import yaml

from dqn import DQN
from environment import create_mario_env


def reset_env(env):
    """Support both Gym reset API variants."""
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    """Support both Gym step API variants and return one done flag."""
    result = env.step(action)
    if len(result) == 5:
        state, reward, terminated, truncated, info = result
        return state, reward, terminated or truncated, info
    return result


def load_model(model, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # agent.py may save either a plain state dictionary or a training checkpoint.
    state_dict = checkpoint.get("online_net_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)


def main():
    with open("parameters.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    checkpoint_path = config["training"]["checkpoint_path"]
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"No model checkpoint at '{checkpoint_path}'. Train the agent first."
        )

    env_config = config["environment"]
    # ``pyglet`` human rendering is incompatible with some Python 3.13
    # Windows installs. RGB frames displayed by OpenCV avoid that renderer.
    env = create_mario_env(**env_config, render_mode="rgb_array")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(action_dim=env.action_space.n).to(device)
    load_model(model, checkpoint_path, device)
    model.eval()

    state = reset_env(env)
    total_reward = 0.0

    try:
        while True:
            frame = env.render()
            if frame is not None:
                cv2.imshow("Mario Double DQN (press Q to quit)", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            state_tensor = torch.as_tensor(np.array(state), device=device).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax(dim=1).item()

            state, reward, done, _ = step_env(env, action)
            total_reward += reward
            if done:
                print(f"Episode reward: {total_reward:.1f}")
                break
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
