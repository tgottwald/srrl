import numpy as np
from typing import Tuple
from numba import jit


class AbstractSteeringModel:
    @property
    def beta(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def integrate(self, control, dt):
        raise NotImplementedError


class DirectSteeringModel(AbstractSteeringModel):
    def __init__(self, beta_max: float):
        self.beta_max = beta_max
        self._beta = 0.

    @property
    def beta(self):
        return self._beta

    def reset(self):
        self._beta = 0.

    def integrate(self, control, dt):
        self._beta = np.clip(control * self.beta_max, -self.beta_max, self.beta_max)


@jit(nopython=True)
def _rate_limited_integrate(beta_max, beta_rate, beta_old, control, dt):
    beta_target = min(max(control * beta_max, -beta_max), beta_max)
    d_beta = np.sign(beta_target - beta_old) * beta_rate
    new_beta = min(max(beta_old + dt * d_beta, -beta_max), beta_max)
    zero_crossing = np.sign(beta_target - beta_old) != np.sign(beta_target - new_beta)
    if zero_crossing:
        new_beta = beta_target
    return new_beta


class RateLimitedSteeringModel(AbstractSteeringModel):
    __slots__ = ("beta_max", "beta_rate", "_beta")

    def __init__(self, beta_max: float, beta_rate: float):
        self.beta_max = beta_max
        self.beta_rate = beta_rate
        self._beta = 0.

    @property
    def beta(self):
        return self._beta

    def reset(self):
        self._beta = 0.

    def integrate(self, control, dt):
        assert np.shape(control) == ()
        self._beta = _rate_limited_integrate(self.beta_max, self.beta_rate, self._beta, control, dt)
