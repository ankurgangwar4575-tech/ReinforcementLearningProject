"""Fixed-size replay buffer used by the Double DQN agent."""

from collections import deque
import random


class ReplayMemory:
    def __init__(self, max_len: int, seed: int | None = None):
        self.memory = deque(maxlen=max_len)
        self.random = random.Random(seed)

    def append(self, experience):
        """Store ``(state, action, reward, next_state, done)``."""
        self.memory.append(experience)

    def sample(self, sample_size: int):
        return self.random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
