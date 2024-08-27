from typing import Tuple

import gymnasium as gym
import numpy as np
from shapely.geometry import Polygon
from dataclasses import dataclass

from .Problem import Problem
from ..BatchedCones import BatchedCones


@dataclass
class PosRewardInfo:
    delta_x: float
    delta_y: float
    delta_angle: float
    reward: float


def angle_difference(alpha, beta):
    """
    Return difference of radian angles alpha and beta in radian in range (-pi, pi]
    """

    diff = (alpha - beta) % (2 * np.pi)
    assert diff >= 0

    if diff > np.pi:
        return diff - 2 * np.pi
    else:
        return diff


class ParallelParkingProblem(Problem):
    def __init__(self, start="before", k_continuous=.05, max_time=None, soft_collisions=False, reward_term="lffds"):
        self.target_pose = None
        self.k_continuous = k_continuous

        if max_time is None:
            self.max_time = 15. if start == 'before' else 7.
        else:
            self.max_time = max_time

        self.start = start
        self.reward_term = reward_term
        self.help_pos = None
        self.track_dict = None
        self.soft_collisions = soft_collisions

    @property
    def render_hints(self) -> dict:
        return {
            'scale': 30,
            'from_ego': False,
        }

    @property
    def state_observation_space(self):
        return gym.spaces.Box(-1, 1, (2,))

    def observe_state(self, env):
        return np.array([
            # TODO: Better normalization
            env.vehicle_last_speed / 10.,
            env.steering_history[-1],
        ], dtype=np.float32)

    def _make_problem(self, env, rng=None):
        gap_start = rng.integers(5, 10) * 2
        gap_size = 8
        gap_stop = gap_start + gap_size
        gap_width = 2
        street_width = 5
        car_width = env.collision_bb[3] - env.collision_bb[2]
        street_pad = 1.  # For visualization

        start_x = 0. if self.start == 'before' else 20 + gap_size + 5 + rng.uniform()

        # 2m spacing on longitudinal
        pos0 = np.stack([
            np.linspace(0, 50, 26),
            np.ones(26) * -street_width,
        ], axis=-1)
        pos1 = np.stack([
            np.linspace(0, gap_start, gap_start // 2 + 1),
            np.zeros(gap_start // 2 + 1),
        ], axis=-1)

        # 1m spacing on lateral
        pos2 = np.stack([
            np.ones(gap_width - 1) * gap_start,
            np.linspace(1, gap_width - 1, gap_width - 1),
        ], axis=-1)

        # 2m spacing on longitudinal
        pos3 = np.stack([
            np.linspace(gap_start, gap_start + gap_size, gap_size // 2 + 1),
            np.ones(gap_size // 2 + 1) * gap_width
        ], axis=-1)

        # 1m spacing on lateral
        pos4 = np.stack([
            np.ones(gap_width - 1) * gap_stop,
            np.linspace(1, gap_width - 1, gap_width - 1),
        ], axis=-1)

        # 2m spacing on longitudinal
        pos5 = np.stack([
            np.linspace(gap_stop, 50, (50 - gap_stop) // 2 + 1),
            np.zeros((50 - gap_stop) // 2 + 1),
        ], axis=-1)

        # Zero mean on x
        cones_pos = np.concatenate([pos0, pos1, pos2, pos3, pos4, pos5]) + np.array([-25, street_width / 2])

        # Random angle [-10°, 10°)
        start_angle = (rng.uniform() - .5) * 20 / 180 * np.pi

        start_pose = start_x - 25, rng.uniform() - .5, start_angle

        target_pose = np.array([gap_start + gap_size / 2 - 25, street_width / 2 + car_width / 2, 0.])

        track_dict = {
            'centerline': np.array([[-25 - street_pad, 0.], [25. + street_pad, 0.]]),
            'poly': Polygon(np.array([[-25 - street_pad, -street_width / 2. - street_pad],
                                      [25 + street_pad, -street_width / 2 - street_pad],
                                      [25. + street_pad, street_width / 2 + street_pad],
                                      [-25. + gap_start + gap_size + street_pad, street_width / 2 + street_pad],
                                      [-25. + gap_start + gap_size + street_pad, street_width / 2 + gap_width + street_pad],
                                      [-25. + gap_start - street_pad, street_width / 2 + gap_width + street_pad],
                                      [-25. + gap_start - street_pad, street_width / 2 + street_pad],
                                      [-25 - street_pad, street_width / 2 + street_pad]]))
        }

        return start_pose, target_pose, cones_pos, track_dict

    def configure_env(self, env, rng=None) -> Tuple[float, float, float]:
        start_pose, target_pose, cones_pos, track_dict = self._make_problem(env, rng)

        self.target_pose = target_pose
        self.track_dict = track_dict

        cones = np.concatenate([
            cones_pos,
            np.ones_like(cones_pos[:, :1]),
            np.zeros_like(cones_pos[:, :1]),
        ], axis=-1)

        env.objects = {
            'cones': BatchedCones(cones, soft_collision_distance=.15 if self.soft_collisions else None)
        }

        return start_pose

    def render(self, ctx, env):
        import cairo
        from ..Rendering.Rendering import stroke_fill
        ctx: cairo.Context = ctx

        ctx.arc(*self.target_pose[:2], .1, 0, 2 * np.pi)
        ctx.close_path()
        stroke_fill(ctx, (0., 0., 0.), None)

        veh_pose = env.ego_pose
        ctx.arc(*veh_pose[:2], .075, 0, 2 * np.pi)
        ctx.close_path()
        stroke_fill(ctx, (0., 0., 0.), None)

    def pose_reward(self, current_pose, target_pose) -> PosRewardInfo:
        current_xy, current_alpha, target_xy, target_alpha = current_pose[:2], current_pose[2], target_pose[:2], target_pose[2]
        abs_delta_x, abs_delta_y = np.abs(np.array(current_xy) - np.array(target_xy))

        delta_angle = current_alpha - target_alpha

        if self.reward_term == "lffds":
            reward_x = np.maximum(0., 20 - abs_delta_x) / 20
            reward_y = np.maximum(0., 5 - abs_delta_y) / 5
            reward_angle = np.maximum(0., np.cos(delta_angle))
            reward = reward_x * reward_y * reward_angle
        elif self.reward_term == "lffds2":
            reward_x = np.maximum(0., 20 - abs_delta_x) / 20
            reward_y = np.maximum(0., 5 - abs_delta_y) / 5
            reward_angle = np.maximum(0., np.pi / 2 - np.abs(delta_angle)) / (np.pi / 2)
            reward = reward_x * reward_y * reward_angle
        elif self.reward_term == "matlab":
            # Modified from:
            # https://de.mathworks.com/help/reinforcement-learning/ug/train-ppo-agent-for-automatic-parking-valet.html
            reward = .8 * np.exp(-0.1 * abs_delta_x ** 2 - 0.2 * abs_delta_y ** 2) + .2 * np.exp(-40. * delta_angle ** 2)
        else:
            raise ValueError(f"{self.reward_term = }")

        return PosRewardInfo(abs_delta_x, abs_delta_y, delta_angle, reward)

    def is_out_of_bounds(self, pose):
        return pose[0] < -25 or pose[0] > 25

    def update(self, env, dt: float) -> Tuple[bool, bool]:
        if env.objects['cones'].hit_count > 0:
            env.set_reward(-1)
            return True, False

        pose = env.ego_pose

        if self.is_out_of_bounds(pose):
            env.set_reward(-1)
            return True, False

        reward_info = self.pose_reward(pose, self.target_pose)

        env.add_info('FinalMetric.AbsDeltaX', abs(reward_info.delta_x))
        env.add_info('FinalMetric.AbsDeltaY', abs(reward_info.delta_y))
        env.add_info('FinalMetric.AbsDeltaAngle', abs(reward_info.delta_angle))
        env.add_info('FinalMetric.PoseReward', reward_info.reward)
        env.add_to_reward(reward_info.reward * self.k_continuous)

        truncated = env.time >= self.max_time

        return False, truncated
