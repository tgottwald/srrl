import gym
import numpy as np

from .ParallelParkingProblem import ParallelParkingProblem


class TerminatableParallelParkingProblem(ParallelParkingProblem):
    def __init__(self, *args, **kwargs):
        super(TerminatableParallelParkingProblem, self).__init__(*args, **kwargs)

    @property
    def state_observation_space(self):
        return gym.spaces.Box(-1, 1, (3,))

    def observe_state(self, env):
        # Observe the remaining time
        time = env.time / self.max_time
        return np.concatenate([super(TerminatableParallelParkingProblem, self).observe_state(env), [time]])
