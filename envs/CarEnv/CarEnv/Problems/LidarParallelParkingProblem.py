from typing import Tuple

import gymnasium as gym
import numpy as np
from shapely.geometry import Polygon
from dataclasses import dataclass

from .Problem import Problem
from ..BatchedWalls import BatchedWalls


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


class LidarParallelParkingProblem(Problem):
    def __init__(
        self,
        start="before",
        k_continuous=0.05,
        max_time=None,
        soft_collisions=False,
        reward_term="lffds",
    ):
        self.target_pose = None
        self.k_continuous = k_continuous

        if max_time is None:
            self.max_time = 15.0 if start == "before" else 7.0
        else:
            self.max_time = max_time

        self.start = start
        self.reward_term = reward_term
        self.help_pos = None
        self.track_dict = None
        self.soft_collisions = soft_collisions

        if self.start == "before":
            self.start_x_interval = 0.0
            self.start_y_interval = np.array([[-0.5], [0.5]])
            self.start_angle_interval = np.array([[-0.5], [0.5]])
        elif self.start == "diverse_after":
            # Single starting zone within x=[30, 40], y=[-1, 1], angle=[-20, 20]
            self.start_x_interval = np.array([[4.0], [12.0]])
            self.start_y_interval = np.array([[-1.0], [1.0]])
            self.start_angle_interval = np.array([[-1.0], [1.0]])
        elif self.start == "diverse":
            # Second starting zone within x=[2, 8], y=[-1, 1], angle=[-20, 20]
            self.start_x_interval = np.array([[2.0, -10 - 8 - 8], [12.0, -10 - 8 - 2]])
            self.start_y_interval = np.array([[-1.0, -1.0], [1.0, 1.0]])
            self.start_angle_interval = np.array([[-1.0, -1.0], [1.0, 1.0]])
        else:
            # Single starting zone within x=[33, 34], y=[-0.5, 0.5], angle=[-10, 10]
            self.start_x_interval = np.array([[5.0], [6.0]])
            self.start_y_interval = np.array([[-0.5], [0.5]])
            self.start_angle_interval = np.array([[-0.5], [0.5]])

    @property
    def render_hints(self) -> dict:
        return {
            "scale": 30,
            "from_ego": False,
        }

    @property
    def state_observation_space(self):
        return gym.spaces.Box(-1, 1, (2,))

    def observe_state(self, env):
        return np.array(
            [
                # TODO: Better normalization
                env.vehicle_last_speed / 10.0,
                env.steering_history[-1],
            ],
            dtype=np.float32,
        )

    def _make_problem(self, env, rng=None):
        gap_start = rng.uniform(10.0, 20.0)
        gap_size = 8
        gap_stop = gap_start + gap_size
        gap_width = 2
        street_width = 5
        street_length = 50
        car_width = env.collision_bb[3] - env.collision_bb[2]
        car_height = env.collision_bb[1] - env.collision_bb[0]
        street_pad = 0.0  # 1.0  # For visualization

        # Create the walls simulating the street and parking bay limits

        # ------------------------wall0-----------------------
        #
        #
        # -------wall1-------              -------wall5-------
        #                   |              |
        #                 wall2          wall4
        #                   |              |
        #                   -----wall3------

        wall0 = np.array([0.0, -street_width, street_length, -street_width])
        wall1 = np.array([0.0, 0.0, gap_start, 0.0])
        wall2 = np.array([gap_start, 0.0, gap_start, gap_width])
        wall3 = np.array([gap_start, gap_width, gap_stop, gap_width])
        wall4 = np.array([gap_stop, gap_width, gap_stop, 0.0])
        wall5 = np.array([gap_stop, 0.0, street_length, 0.0])

        # Zero mean on x
        wall_pos = np.vstack([wall0, wall1, wall2, wall3, wall4, wall5])

        wall_pos += np.array(
            [-street_length / 2, street_width / 2, -street_length / 2, street_width / 2]
        )

        # Calculate the starting pose based on the interval set in the contructor
        start_x = (
            0.0
            if self.start == "before"
            else 20 + gap_size + rng.uniform(*self.start_x_interval)
        )
        start_y = rng.uniform(*self.start_y_interval)
        # Random angle [-10°, 10°) ("before" and "after")
        start_angle = rng.uniform(*self.start_angle_interval) * 20 / 180 * np.pi

        # Select one of the starting zones at random
        starting_zone_idx = rng.integers(start_x.shape[-1])

        start_pose = np.array(
            [
                start_x[starting_zone_idx] - street_length / 2,
                start_y[starting_zone_idx],
                start_angle[starting_zone_idx],
            ]
        )

        self.start_pose = start_pose

        # x position counts as valid for reset if it is at least 2m from the beginning/end of the gap and 3m from start/end of the street
        if self.start == "diverse":
            valid_reset_x_interval = np.array(
                [
                    [gap_stop + 2 - street_length / 2, 0 + 3 - street_length / 2],
                    [
                        street_length - 3 - street_length / 2,
                        gap_start - 2 - street_length / 2,
                    ],
                ]
            )
        else:
            # FIXME: Should be interval [gap_stop + 5 - street_length / 2, gap_stop + 16 - street_length / 2] if relative coords of starting pos to gap stop are considered
            # valid_reset_x_interval = np.array(
            #     [
            #         [gap_stop + 4 - street_length / 2],
            #         [street_length - 4 - street_length / 2],
            #     ]
            # )
            valid_reset_x_interval = np.array(
                [
                    [gap_stop + 5 - street_length / 2],
                    [gap_stop + 16 - street_length / 2],
                ]
            )

        # Calculate the centers of the pose intervall in which the car counts as reset
        # Dimension: (# Reset zones, 3)
        self.valid_reset_interval_center = np.array(
            [
                np.mean(valid_reset_x_interval, axis=0),
                np.mean(self.start_y_interval, axis=0),
                np.mean(self.start_angle_interval, axis=0) * 20 / 180 * np.pi,
            ]
        ).T

        # Calculate the tolerances of the pose intervall with respect to the center in which the car counts as reset
        self.valid_reset_interval_tolerances = np.array(
            [
                np.max(valid_reset_x_interval, axis=0)
                - np.mean(valid_reset_x_interval, axis=0),
                np.max(self.start_y_interval, axis=0)
                - np.mean(self.start_y_interval, axis=0),
                (
                    np.max(self.start_angle_interval, axis=0)
                    - np.mean(self.start_angle_interval, axis=0)
                )
                * 20
                / 180
                * np.pi,
            ]
        ).T

        target_pose = np.array(
            [
                gap_start + gap_size / 2 - street_length / 2,
                street_width / 2 + car_width / 2,
                0.0,
            ]
        )

        # Safety zone markings (purley for visualization)
        safety_line0 = np.array(
            [
                0.0,
                -street_width + car_width / 2,
                street_length,
                -street_width + car_width / 2,
            ]
        )
        safety_line1 = np.array(
            [0.0, 0.0 - car_width / 2, gap_start + car_width / 2, 0.0 - car_width / 2]
        )
        safety_line2 = np.array(
            [
                gap_start + car_width / 2,
                0.0 - car_width / 2,
                gap_start + car_width / 2,
                gap_width - car_width / 2,
            ]
        )
        safety_line3 = np.array(
            [
                gap_start + car_width / 2,
                gap_width - car_width / 2,
                gap_stop - car_width / 2,
                gap_width - car_width / 2,
            ]
        )
        safety_line4 = np.array(
            [
                gap_stop - car_width / 2,
                gap_width - car_width / 2,
                gap_stop - car_width / 2,
                0.0 - car_width / 2,
            ]
        )
        safety_line5 = np.array(
            [
                gap_stop - car_width / 2,
                0.0 - car_width / 2,
                street_length,
                0.0 - car_width / 2,
            ]
        )
        safety_line_pos = np.vstack(
            [
                safety_line0,
                safety_line1,
                safety_line2,
                safety_line3,
                safety_line4,
                safety_line5,
            ]
        )
        safety_line_pos += np.array(
            [-street_length / 2, street_width / 2, -street_length / 2, street_width / 2]
        )
        safety_line_pos[2, 3] = target_pose[1] + car_width / 2
        safety_line_pos[3, [1, 3]] = target_pose[1] + car_width / 2
        safety_line_pos[4, 1] = target_pose[1] + car_width / 2

        track_dict = {
            "safety_zone": safety_line_pos,
            # "safety_zone": np.array(
            #     [
            #         # Drivable area boundaries
            #         [
            #             -street_length / 2,
            #             -street_width / 2 + car_width / 2,
            #             street_length / 2,
            #             -street_width / 2 + car_width / 2,
            #         ],
            #         [
            #             -street_length / 2,
            #             street_width / 2 - car_width / 2,
            #             street_length / 2,
            #             street_width / 2 - car_width / 2,
            #         ],
            #     ]
            # ),
            "target_zone": np.array(
                [
                    [
                        target_pose[0] - car_height / 2,
                        target_pose[1] - car_width / 2,
                        target_pose[0] - car_height / 2,
                        target_pose[1] + car_width / 2,
                    ],
                    [
                        target_pose[0] - car_height / 2,
                        target_pose[1] + car_width / 2,
                        target_pose[0] + car_height / 2,
                        target_pose[1] + car_width / 2,
                    ],
                    [
                        target_pose[0] + car_height / 2,
                        target_pose[1] + car_width / 2,
                        target_pose[0] + car_height / 2,
                        target_pose[1] - car_width / 2,
                    ],
                    [
                        target_pose[0] + car_height / 2,
                        target_pose[1] - car_width / 2,
                        target_pose[0] - car_height / 2,
                        target_pose[1] - car_width / 2,
                    ],
                    [
                        target_pose[0] - car_height / 2 + 0.25,
                        target_pose[1] - car_width / 2 + 0.25,
                        target_pose[0] + car_height / 2 - 0.25,
                        target_pose[1] + car_width / 2 - 0.25,
                    ],
                    [
                        target_pose[0] + car_height / 2 - 0.25,
                        target_pose[1] - car_width / 2 + 0.25,
                        target_pose[0] - car_height / 2 + 0.25,
                        target_pose[1] + car_width / 2 - 0.25,
                    ],
                ]
            ),
            "centerline": np.array(
                [
                    [-street_length / 2 - street_pad, 0.0],
                    [street_length / 2 + street_pad, 0.0],
                ]
            ),
            "poly": Polygon(
                np.array(
                    [
                        [
                            -street_length / 2 - street_pad,
                            -street_width / 2.0 - street_pad,
                        ],
                        [
                            street_length / 2 + street_pad,
                            -street_width / 2 - street_pad,
                        ],
                        [street_length / 2 + street_pad, street_width / 2 + street_pad],
                        [
                            -street_length / 2 + gap_start + gap_size + street_pad,
                            street_width / 2 + street_pad,
                        ],
                        [
                            -street_length / 2 + gap_start + gap_size + street_pad,
                            street_width / 2 + gap_width + street_pad,
                        ],
                        [
                            -street_length / 2 + gap_start - street_pad,
                            street_width / 2 + gap_width + street_pad,
                        ],
                        [
                            -street_length / 2 + gap_start - street_pad,
                            street_width / 2 + street_pad,
                        ],
                        [
                            -street_length / 2 - street_pad,
                            street_width / 2 + street_pad,
                        ],
                    ]
                )
            ),
        }

        return start_pose, target_pose, wall_pos, track_dict

    def configure_env(self, env, rng=None) -> Tuple[float, float, float]:
        start_pose, target_pose, wall_pos, track_dict = self._make_problem(env, rng)

        self.target_pose = target_pose
        self.track_dict = track_dict

        env.objects = {
            "walls": BatchedWalls(
                wall_pos, soft_collision_distance=0.15 if self.soft_collisions else None
            )
        }

        return start_pose

    def render(self, ctx, env):
        import cairo
        from ..Rendering.Rendering import stroke_fill

        ctx: cairo.Context = ctx

        ctx.arc(*self.target_pose[:2], 0.1, 0, 2 * np.pi)
        ctx.close_path()
        stroke_fill(ctx, (0.0, 0.0, 0.0), None)

        veh_pose = env.ego_pose
        ctx.arc(*veh_pose[:2], 0.075, 0, 2 * np.pi)
        ctx.close_path()
        stroke_fill(ctx, (0.0, 0.0, 0.0), None)

    def pose_reward(self, current_pose, target_pose) -> PosRewardInfo:
        current_xy, current_alpha, target_xy, target_alpha = (
            current_pose[:2],
            current_pose[2],
            target_pose[:2],
            target_pose[2],
        )
        abs_delta_x, abs_delta_y = np.abs(np.array(current_xy) - np.array(target_xy))

        delta_angle = current_alpha - target_alpha

        if self.reward_term == "lffds":
            reward_x = np.maximum(0.0, 20 - abs_delta_x) / 20
            reward_y = np.maximum(0.0, 5 - abs_delta_y) / 5
            reward_angle = np.maximum(0.0, np.cos(delta_angle))
            reward = reward_x * reward_y * reward_angle
        elif self.reward_term == "lffds2":
            reward_x = np.maximum(0.0, 20 - abs_delta_x) / 20
            reward_y = np.maximum(0.0, 5 - abs_delta_y) / 5
            reward_angle = np.maximum(0.0, np.pi / 2 - np.abs(delta_angle)) / (
                np.pi / 2
            )
            reward = reward_x * reward_y * reward_angle
        elif self.reward_term == "matlab":
            # Modified from:
            # https://de.mathworks.com/help/reinforcement-learning/ug/train-ppo-agent-for-automatic-parking-valet.html
            reward = 0.8 * np.exp(
                -0.1 * abs_delta_x**2 - 0.2 * abs_delta_y**2
            ) + 0.2 * np.exp(-40.0 * delta_angle**2)
        else:
            raise ValueError(f"{self.reward_term = }")

        return PosRewardInfo(abs_delta_x, abs_delta_y, delta_angle, reward)

    def is_out_of_bounds(self, pose):
        return pose[0] < (-50 / 2) or pose[0] > (50 / 2)

    def update(self, env, dt: float) -> Tuple[bool, bool]:
        lidar_scan = env.last_observations[..., 2:]
        if env.objects["walls"].hit_count > 0:
            # Crashed because car frame hit wall
            env.set_reward(-1)
            # print("Hit Wall!")
            return True, False
        elif np.any(
            lidar_scan[lidar_scan >= 0.0]
            - env.polar_vehicle_boundary[lidar_scan >= 0.0]
            <= 0
        ):
            # Crashed because noise caused lidar to report value which would be a collision
            # Added this to simulation because in real world car would be stopped as soon has lidar reports a collision
            env.set_reward(-1)
            # print("Lidar noise collision!")
            return True, False

        pose = env.ego_pose

        if self.is_out_of_bounds(pose):
            env.set_reward(-1)
            # print("Out of bounds!")

            return True, False

        reward_info = self.pose_reward(pose, self.target_pose)

        env.add_info("FinalMetric.AbsDeltaX", abs(reward_info.delta_x))
        env.add_info("FinalMetric.AbsDeltaY", abs(reward_info.delta_y))
        env.add_info("FinalMetric.AbsDeltaAngle", abs(reward_info.delta_angle))
        env.add_info("FinalMetric.PoseReward", reward_info.reward)
        env.add_to_reward(reward_info.reward * self.k_continuous)

        truncated = env.time >= self.max_time

        return False, truncated