from typing import Tuple

from .RacingProblem import RacingProblem
from ..BatchedDucks import BatchedDucks


class RacingDuckProblem(RacingProblem):
    def configure_env(self, env, rng=None) -> Tuple[float, float, float]:
        result = super().configure_env(env, rng)

        env.objects['ducks'] = BatchedDucks.from_track_dict(self.track_dict, env.np_random)

        return result
