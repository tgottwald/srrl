from typing import Dict

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

# Based on https://github.com/tinkoff-ai/sac-rnd/blob/main/offline_sac/utils/running_moments.py


class RNDArchitecture(nn.Module):
    embedding_dim: int = 256

    @nn.compact
    def __call__(self, state, action):
        network = nn.Sequential(
            [
                nn.Dense(
                    512,
                    kernel_init=nn.initializers.kaiming_uniform(),
                ),
                nn.relu,
                nn.Dense(
                    256,
                    kernel_init=nn.initializers.kaiming_uniform(),
                ),
                nn.relu,
                nn.Dense(
                    self.embedding_dim,
                    kernel_init=nn.initializers.kaiming_uniform(),
                ),
            ]
        )
        state_action = jnp.hstack([state, action])
        out = network(state_action)  # .squeeze(-1)
        return out


class RND(nn.Module):
    embedding_dim: int = 256
    init_features: jnp.ndarray = None

    def setup(self):
        self.target_network = RNDArchitecture(self.embedding_dim)
        self.predictor_network = RNDArchitecture(self.embedding_dim)

    def __call__(self, state, action):
        target = self.target_network(state, action)
        prediction = self.predictor_network(state, action)

        return prediction, jax.lax.stop_gradient(target)


@chex.dataclass(frozen=True)
class RNDTrainDict:
    state: Dict[str, jnp.ndarray]

    @staticmethod
    def init():
        state = {
            "count": jnp.array([1e-4]),
            "mean": jnp.array([0.0]),
            "squared_diff_sum": jnp.array([0.0]),
            "var": jnp.array([1.0]),
        }
        return RNDTrainDict(state=state)


class RNDTrainState(TrainState):
    rnd_state: RNDTrainDict
    enabled: bool = flax.struct.field(pytree_node=False)
