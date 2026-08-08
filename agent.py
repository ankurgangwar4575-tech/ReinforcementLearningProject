"""Train a Mario-playing agent with Double Deep Q-Learning."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from dqn import DQN
from environment import create_mario_env
from experience_replay import ReplayMemory


class TrainingLogger:
    """TensorBoard logger that never stops training when a log file is locked."""

    def __init__(self, log_dir: Path):
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.disabled = False

    def add_scalar(self, tag, value, step):
        if self.disabled:
            return
        try:
            self.writer.add_scalar(tag, value, step)
        except OSError as error:
            self.disabled = True
            print(f"TensorBoard logging disabled: {error}")

    def close(self):
        try:
            self.writer.close()
        except OSError:
            pass


def reset_env(env):
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        state, reward, terminated, truncated, info = result
        return state, reward, terminated or truncated, info
    return result


def epsilon_at(step: int, start: float, end: float, decay_steps: int) -> float:
    fraction = min(step / decay_steps, 1.0)
    return start + fraction * (end - start)


def state_tensor(state, device):
    return torch.as_tensor(np.asarray(state), device=device).unsqueeze(0)


def learn(online_net, target_net, optimizer, replay, batch_size, gamma, device):
    states, actions, rewards, next_states, dones = zip(*replay.sample(batch_size))
    states = torch.as_tensor(np.asarray(states), device=device)
    next_states = torch.as_tensor(np.asarray(next_states), device=device)
    actions = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=device)

    current_q = online_net(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        # Double DQN: online network selects, target network evaluates.
        next_actions = online_net(next_states).argmax(dim=1, keepdim=True)
        next_q = target_net(next_states).gather(1, next_actions).squeeze(1)
        target_q = rewards + gamma * (1.0 - dones) * next_q

    loss = functional.smooth_l1_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item()


def main():
    with open("parameters.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    train_config = config["training"]
    exploration = config["exploration"]
    seed = train_config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    env = create_mario_env(**config["environment"])
    env.action_space.seed(seed)

    online_net = DQN(env.action_space.n).to(device)
    target_net = DQN(env.action_space.n).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()
    optimizer = Adam(online_net.parameters(), lr=train_config["learning_rate"])
    replay = ReplayMemory(train_config["replay_memory_size"], seed=seed)

    checkpoint_path = Path(train_config["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # OneDrive can lock files inside the project directory. Store the optional
    # TensorBoard logs in the local application-data folder instead.
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.cwd()))
    log_dir = local_app_data / "MarioDoubleDQN" / "tensorboard" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    writer = TrainingLogger(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    state = reset_env(env)
    episode_reward = 0.0
    episode_number = 1

    try:
        for step in range(1, train_config["total_steps"] + 1):
            epsilon = epsilon_at(
                step,
                exploration["epsilon_start"],
                exploration["epsilon_end"],
                exploration["epsilon_decay_steps"],
            )
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = online_net(state_tensor(state, device)).argmax(dim=1).item()

            next_state, reward, done, _ = step_env(env, action)
            # Copy observations so a wrapper cannot mutate a stored experience.
            replay.append((np.asarray(state).copy(), action, reward, np.asarray(next_state).copy(), done))
            state = next_state
            episode_reward += reward
            writer.add_scalar("training/epsilon", epsilon, step)

            if (
                step >= train_config["learning_starts"]
                and step % train_config["train_frequency"] == 0
                and len(replay) >= train_config["batch_size"]
            ):
                loss = learn(
                    online_net, target_net, optimizer, replay,
                    train_config["batch_size"], train_config["gamma"], device,
                )
                writer.add_scalar("training/loss", loss, step)

            if step % train_config["target_update_frequency"] == 0:
                target_net.load_state_dict(online_net.state_dict())

            if step % train_config["checkpoint_frequency"] == 0:
                torch.save({"online_net_state_dict": online_net.state_dict(), "step": step}, checkpoint_path)
                print(f"Saved checkpoint at step {step}")

            if done:
                writer.add_scalar("episode/reward", episode_reward, episode_number)
                print(f"Episode {episode_number}: reward={episode_reward:.1f}, step={step}")
                state = reset_env(env)
                episode_reward = 0.0
                episode_number += 1

        torch.save({"online_net_state_dict": online_net.state_dict(), "step": train_config["total_steps"]}, checkpoint_path)
        print(f"Training complete. Model saved to {checkpoint_path}")
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
