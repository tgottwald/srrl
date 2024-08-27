import gymnasium as gym
import numpy as np

from .Sensor import Sensor
from .BatchedCones import BatchedCones


class SensorDuckMap(Sensor):
    def __init__(self, env, max_objects=100, observe_heading=False, normalize=True, bbox=(-30, 30, -30, 30)):
        super(SensorDuckMap, self).__init__(env)
        self._observe_heading = observe_heading
        self._max_objects = max_objects
        self._normalize = normalize
        self._bbox = bbox

    @property
    def bbox(self):
        return tuple(self._bbox)

    @property
    def observation_space(self) -> gym.Space:
        dims = 5 if self._observe_heading else 3
        return gym.spaces.Box(-np.inf, np.inf, shape=(self._max_objects, dims))

    @property
    def view_normalizer(self):
        return max((abs(x) for x in self._bbox))

    def observe(self, env):
        ducks = env.objects['ducks'].transformed(env.ego_transform).filtered_aabb(*self._bbox)

        vis_pos = ducks.data[:, :2]
        thetas = ducks.data[:, 2]

        # Normalization
        if self._normalize:
            vis_pos = vis_pos / self.view_normalizer

        count = vis_pos.shape[0]
        enc_count = min(count, self._max_objects)

        if enc_count < count:
            import warnings
            warnings.warn(f"Discarding {count - enc_count} objects because {self._max_objects = } is too low.")

        result = np.zeros((self._max_objects, 4 + 1), dtype=np.float32)
        result[:enc_count, 0] = 1
        result[:enc_count, 1:3] = vis_pos[:enc_count]

        if self._observe_heading:
            result[:enc_count, 3] = np.cos(thetas[:enc_count])
            result[:enc_count, 4] = np.sin(thetas[:enc_count])
        else:
            result = result[:3]

        return result
