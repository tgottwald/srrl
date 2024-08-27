from typing import Callable

import distrax
import flax
import flax.linen as nn
import jax.numpy as jnp
from flax.training.train_state import TrainState

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())


class Actor(nn.Module):
    fe_constructor_fn: Callable[[], nn.Module]
    action_dim: int

    @nn.compact
    def __call__(self, state):
        net = nn.Sequential(
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
            ]
        )
        log_std_net = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.lecun_uniform(),
        )
        mean_net = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.lecun_uniform(),
        )

        features = self.fe_constructor_fn()(state)
        trunk = net(features)
        mean = mean_net(trunk)
        log_std = log_std_net(trunk)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

        # IMPORTANT! Only compatible with actions within [-1, 1]
        dist = TanhNormal(mean, jnp.exp(log_std))
        return dist


class ActorTrainState(TrainState):
    use_mean: bool = flax.struct.field(pytree_node=False)
    damp_scale: float = 0.0
    cvar_std_coeff: float = 1.0
