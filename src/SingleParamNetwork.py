import flax
import flax.linen as nn
import jax.numpy as jnp
from flax.training.train_state import TrainState


class SingleParamNetwork(nn.Module):
    init_value: float = 1.0
    param_name: str = "log_param"

    @nn.compact
    def __call__(self):
        log_param = self.param(
            self.param_name, lambda key: jnp.array([jnp.log(self.init_value)])
        )
        return jnp.exp(log_param)


class SingleParamTrainState(TrainState):
    target: float
    fixed: bool = flax.struct.field(
        pytree_node=False
    )  # Indicates whether param should be updated during the training
