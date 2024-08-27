import jax
import jax.numpy as jnp
import flax.linen as nn
import flax
from flax.training.train_state import TrainState
import optax
from typing import Callable, Tuple
from .CriticNetwork import SoftQNetworkEnsemble

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class PyTorchDense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]
        bound = 1.0 / jnp.sqrt(fan_in)

        kernel_init = jax.nn.initializers.kaiming_uniform()

        def bias_init(key, shape, dtype):
            dtype = jax.dtypes.canonicalize_dtype(dtype)
            return jax.random.uniform(key, shape, dtype, -bound, bound)

        return nn.Dense(
            features=self.features, kernel_init=kernel_init, bias_init=bias_init
        )(x)


class IQNDistributionalCriticNetwork(nn.Module):
    # See https://arxiv.org/pdf/1806.06923.pdf with exemplary implementation https://github.com/toshikwa/fqf-iqn-qrdqn.pytorch/blob/master/fqf_iqn_qrdqn/model/iqn.py (PyTorch)
    fe_constructor_fn: Callable[[], nn.Module]
    num_quantiles: int = 32
    embedding_dim: int = 512

    @nn.compact
    def __call__(self, state, action, iota):
        base_net = nn.Sequential(
            [
                PyTorchDense(256),
                nn.relu,
                PyTorchDense(self.embedding_dim),
                nn.relu,
            ]
        )
        embedding_net = nn.Sequential(
            [
                PyTorchDense(self.embedding_dim),
                nn.sigmoid,
            ]
        )
        merge_net = nn.Sequential(
            [
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
        cos_factors = jnp.arange(1, self.embedding_dim + 1, dtype=jnp.float32) * jnp.pi

        features = self.fe_constructor_fn()(state)

        state_action = jnp.hstack([features, action])
        transformed_state_action = base_net(state_action)  # (batch_size, embedding_dim)

        # Apply cosine embedding to iota
        cosine_iota = jnp.cos(
            cos_factors * iota[..., None]
        )  # (batch_size, num_quantiles, embedding_dim)

        iota_embedded = embedding_net(
            cosine_iota
        )  # (batch_size, num_quantiles, embedding_dim)

        trunk = (
            transformed_state_action[..., None, :] * iota_embedded
        )  # (batch_size, num_quantiles, embedding_dim)
        z = merge_net(trunk).squeeze(-1)  # (batch_size, num_quantiles)

        return z


class WCSACIQNNetwork(nn.Module):
    fe_constructor_fn: Callable[[], nn.Module]
    num_reward_critics: int = 2
    num_quantiles: int = 32
    embedding_dim: int = 512

    @nn.compact
    def __call__(self, state, action, iota):
        reward_critic = SoftQNetworkEnsemble(
            self.fe_constructor_fn, self.num_reward_critics
        )
        safety_critic = IQNDistributionalCriticNetwork(
            self.fe_constructor_fn, self.num_quantiles, self.embedding_dim
        )
        q = reward_critic(state, action)
        z = safety_critic(state, action, iota)
        return q, z


class WCSACCriticTrainState(TrainState):
    target_params: flax.core.FrozenDict
    risk_level: float  # TODO: Move to actor?
    ensemble_sample_size: Tuple[int,] = flax.struct.field(pytree_node=False)
    num_iota_samples: int = flax.struct.field(pytree_node=False)
    huber_kappa: float = 1.0
    gamma: float = 0.99
    tau: float = 0.005

    def soft_update(self):
        new_target_params = optax.incremental_update(
            self.params, self.target_params, self.tau
        )
        return self.replace(target_params=new_target_params)


class DistributionalQNetworkEnsemble(nn.Module):
    # hidden_dim: int = 256
    fe_constructor_fn: Callable[[], nn.Module]
    ensemble_size: int = 2
    num_quantiles: int = 32
    embedding_dim: int = 512

    @nn.compact
    def __call__(self, state, action, iota):
        ensemble = nn.vmap(
            target=IQNDistributionalCriticNetwork,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.ensemble_size,
        )
        # Shared feature extractor for all critics
        q_dist = ensemble(
            self.fe_constructor_fn, self.num_quantiles, self.embedding_dim
        )(state, action, iota)
        return q_dist


class DistributionalCriticTrainState(TrainState):
    target_params: flax.core.FrozenDict
    ensemble_sample_size: Tuple[int,] = flax.struct.field(pytree_node=False)
    confidence_level: float
    num_iota_samples: int = flax.struct.field(pytree_node=False)
    huber_kappa: float = 1.0
    gamma: float = 0.99
    tau: float = 0.005
    beta: float = 1.0
    success_target_scaling: float = 1.0

    def soft_update(self):
        new_target_params = optax.incremental_update(
            self.params, self.target_params, self.tau
        )
        return self.replace(target_params=new_target_params)
