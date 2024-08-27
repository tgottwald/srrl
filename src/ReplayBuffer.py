from typing import Dict, List, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np

# Based on https://github.com/medipixel/rl_algorithms/blob/master/rl_algorithms/common/buffer/replay_buffer.py


class BaseReplayBuffer:
    def __init__(self, buffer_size: int = 100000, seed: int = None):
        self.max_buffer_size = buffer_size
        self.curr_buffer_size = 0

        self.rng = np.random.default_rng(seed)

        self.observation_buffer: np.ndarray
        self.action_buffer: np.ndarray
        self.reward_buffer: np.ndarray
        self.next_observation_buffer: np.ndarray
        self.done_buffer: np.ndarray

        self.buffer_idx = 0

    def _initialize_buffers(
        self, exemplary_state: np.ndarray, exemplary_action: np.ndarray
    ) -> None:
        self.observation_buffer = np.zeros(
            ([self.max_buffer_size] + list(exemplary_state.shape)),
            dtype=np.float32,
        )
        self.action_buffer = np.zeros(
            ([self.max_buffer_size] + list(exemplary_action.shape)),
            dtype=np.float32,
        )
        self.reward_buffer = np.zeros((self.max_buffer_size), dtype=np.float32)
        self.next_observation_buffer = np.zeros(
            ([self.max_buffer_size] + list(exemplary_state.shape)),
            dtype=np.float32,
        )
        self.done_buffer = np.zeros((self.max_buffer_size), dtype=np.int32)

    def add(
        self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]
    ) -> None:
        assert len(transition) == 5

        state, action, reward, next_state, done = transition

        if self.curr_buffer_size == 0:
            self._initialize_buffers(state, action)

        self.observation_buffer[self.buffer_idx] = state
        self.action_buffer[self.buffer_idx] = action
        self.reward_buffer[self.buffer_idx] = reward
        self.next_observation_buffer[self.buffer_idx] = next_state
        self.done_buffer[self.buffer_idx] = done

        self.buffer_idx = (self.buffer_idx + 1) % self.max_buffer_size
        self.curr_buffer_size = min(self.curr_buffer_size + 1, self.max_buffer_size)

    def sample(self, batch_size: int = 64) -> Dict[str, jnp.ndarray]:
        assert (self.curr_buffer_size) >= batch_size

        idx = self.rng.integers(self.curr_buffer_size, size=batch_size)

        return (
            self.observation_buffer[idx],
            self.action_buffer[idx],
            self.reward_buffer[idx],
            self.next_observation_buffer[idx],
            self.done_buffer[idx],
        )


class ExtendedReplayBuffer(BaseReplayBuffer):
    """Regular replay buffer with additional data field after reward"""

    def __init__(self, buffer_size: int = 100000, seed: int = None):
        super().__init__(buffer_size, seed)

        self.extra_data: np.ndarray

    def _initialize_buffers(
        self, exemplary_state: np.ndarray, exemplary_action: np.ndarray
    ) -> None:
        super()._initialize_buffers(exemplary_state, exemplary_action)
        self.extra_data = np.zeros((self.max_buffer_size), dtype=np.float32)

    def add(
        self,
        transition: Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, bool],
    ) -> None:
        assert len(transition) == 6

        state, action, reward, extra_value, next_state, done = transition

        if self.curr_buffer_size == 0:
            self._initialize_buffers(state, action)

        self.observation_buffer[self.buffer_idx] = state
        self.action_buffer[self.buffer_idx] = action
        self.reward_buffer[self.buffer_idx] = reward
        self.extra_data[self.buffer_idx] = extra_value
        self.next_observation_buffer[self.buffer_idx] = next_state
        self.done_buffer[self.buffer_idx] = done

        self.buffer_idx = (self.buffer_idx + 1) % self.max_buffer_size
        self.curr_buffer_size = min(self.curr_buffer_size + 1, self.max_buffer_size)

    def sample(self, batch_size: int = 64) -> Dict[str, jnp.ndarray]:
        assert (self.curr_buffer_size) >= batch_size

        idx = self.rng.integers(self.curr_buffer_size, size=batch_size)

        return (
            self.observation_buffer[idx],
            self.action_buffer[idx],
            self.reward_buffer[idx],
            self.extra_data[idx],
            self.next_observation_buffer[idx],
            self.done_buffer[idx],
        )
