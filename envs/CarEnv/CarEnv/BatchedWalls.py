import cairo
import numpy as np

from .BatchedObjects import BatchedObjects
from .Collision import intersection_aabb_lines, intersection_distance_aabb_lines


class BatchedWalls(BatchedObjects):
    def __init__(
        self,
        data,
        hit_count=0,
        soft_collision_distance=None,
        soft_collision_max_penalty=0.2,
    ):
        assert len(data.shape) == 2
        assert data.shape[-1] == 4

        # Features are x_start, y_start, x_end, y_end
        self._data = data
        self._cached_renderer = None
        self.hit_count = hit_count
        self.soft_collision_distance = soft_collision_distance
        self.soft_collision_max_penalty = soft_collision_max_penalty

    def __with_new_data(self, data):
        return BatchedWalls(
            data,
            hit_count=self.hit_count,
            soft_collision_distance=self.soft_collision_distance,
            soft_collision_max_penalty=self.soft_collision_max_penalty,
        )

    def draw(self, ctx: cairo.Context):
        if self._cached_renderer is None:
            from .Rendering.WallRenderer import WallRenderer

            self._cached_renderer = WallRenderer()

        # types = np.argmax(self.data[:, 2:], axis=-1) + 1
        self._cached_renderer.render(ctx, self.data, self.thickness)

    @property
    def thickness(self):
        return 0.1

    @property
    def starting_point(self):
        return self._data[:, :2]

    @property
    def end_point(self):
        return self._data[:, 3:]

    @property
    def data(self):
        return self._data

    def transformed(self, transform) -> "BatchedWalls":
        start_pos_hom = np.concatenate(
            [self._data[:, :2], np.ones_like(self._data[:, :1])], axis=-1
        )
        new_start_pos = np.squeeze(transform @ start_pos_hom[..., None], -1)

        end_pos_hom = np.concatenate(
            [self._data[:, 2:], np.ones_like(self._data[:, :1])], axis=-1
        )
        new_end_pos = np.squeeze(transform @ end_pos_hom[..., None], -1)

        new_data = np.concatenate([new_start_pos[:, :2], new_end_pos[:, :2]], axis=-1)
        return self.__with_new_data(new_data)

    def filtered_aabb(self, *args) -> "BatchedWalls":
        mask = BatchedObjects.filter_mask_aabb(
            self._data[:, :2], *args
        )  # TODO: Never used? Still needs to be adapted
        return self.__with_new_data(self._data[mask])

    # def update(self, env):
    #     us_in_local = self.transformed(env.ego_transform)
    #     intersected = intersection_aabb_lines(
    #         env.collision_bb, self.thickness, us_in_local.data
    #     )

    #     hits = np.sum(intersected)
    #     self.hit_count += hits
    #     env.add_to_reward(hits * -0.2)  # TODO Terminate episode if wall hit
    #     env.metrics["walls_hit"] = env.metrics.get("walls_hit", 0) + hits

    def update(self, env):
        us_in_local = self.transformed(env.ego_transform)

        if self.soft_collision_distance is None:
            intersected = intersection_aabb_lines(
                env.collision_bb, self.thickness, us_in_local.data
            )
        else:
            distances = intersection_distance_aabb_lines(
                env.collision_bb,
                self.thickness + self.soft_collision_distance,
                us_in_local.data,
            )
            distances[~np.isfinite(distances)] = (
                self.thickness + self.soft_collision_distance
            )

            intersected = distances < self.thickness
            soft_intersected = (
                distances < self.thickness + self.soft_collision_distance
            ) & (~intersected)
            soft_intersection_penalties = (
                (1.0 - (distances - self.thickness) / self.soft_collision_distance)
                * soft_intersected
                * self.soft_collision_max_penalty
            )
            soft_collision_penalty = np.sum(soft_intersection_penalties) * -1
            env.add_to_reward(soft_collision_penalty)

        # hits = np.sum(intersected)
        self.hit_count = np.any(intersected)
        # env.add_to_reward(hits * -0.2)
        env.metrics["walls_hit"] = self.hit_count

        # self._data = self._data[~intersected]
