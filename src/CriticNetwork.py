from typing import Callable, Tuple

import flax
import flax.linen as nn
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


class SoftQNetwork(nn.Module):
    @nn.compact
    def __call__(self, state, action):
        network = nn.Sequential(
            [
                nn.Dense(
                    256,
                    kernel_init=nn.initializers.lecun_uniform(),
                ),
                nn.relu,
                nn.Dense(
                    256,
                    kernel_init=nn.initializers.lecun_uniform(),
                ),
                nn.relu,
                nn.Dense(
                    1,
                    kernel_init=nn.initializers.lecun_uniform(),
                ),
            ]
        )
        state_action = jnp.hstack([state, action])
        out = network(state_action).squeeze(-1)
        return out


class SoftQNetworkEnsemble(nn.Module):
    # hidden_dim: int = 256
    fe_constructor_fn: Callable[[], nn.Module]
    ensemble_size: int = 2

    @nn.compact
    def __call__(self, state, action):
        ensemble = nn.vmap(
            target=SoftQNetwork,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.ensemble_size,
        )
        # q_values = ensemble(self.hidden_dim)(state, action)
        # Shared feature extractor for all critics
        features = self.fe_constructor_fn()(state)
        q_values = ensemble()(features, action)
        return q_values


class CriticTrainState(TrainState):
    target_params: flax.core.FrozenDict
    ensemble_sample_size: Tuple[int,] = flax.struct.field(pytree_node=False)
    gamma: float = 0.99
    tau: float = 0.005

    def soft_update(self):
        new_target_params = optax.incremental_update(
            self.params, self.target_params, self.tau
        )
        return self.replace(target_params=new_target_params)


class ExampleBasedCriticTrainState(TrainState):
    target_params: flax.core.FrozenDict
    ensemble_sample_size: Tuple[int,] = flax.struct.field(pytree_node=False)
    gamma: float = 0.99
    tau: float = 0.005
    beta: float = 1.0
    success_target_scaling: float = 1.0

    def soft_update(self):
        new_target_params = optax.incremental_update(
            self.params, self.target_params, self.tau
        )
        return self.replace(target_params=new_target_params)
