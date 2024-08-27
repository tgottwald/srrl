import flax.linen as nn
import jax.numpy as jnp


class DummyFeatureExtractor(nn.Module):

    @nn.compact
    def __call__(self, state):
        return state
