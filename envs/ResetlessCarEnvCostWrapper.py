import gymnasium as gym
import numpy as np
from PIL import Image
from enum import Enum
from copy import deepcopy


class DynamicsMode(Enum):
    FORWARD = 1
    RESET = 2


class ResetlessCarEnvCostWrapper(gym.core.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        inverse_problem: bool = False,
        cost_mode: str = "exponential_v3",
        cost_buffer_radius: float = 0.0,
        reset_reward_type: str = "default",
        debug: bool = False,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            inverse_problem=inverse_problem,
            cost_mode=cost_mode,
            cost_buffer_radius=cost_buffer_radius,
            reset_reward_type=reset_reward_type,
            debug=debug,
        )
        gym.Wrapper.__init__(self, env)
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if self.is_vector_env:
            raise ValueError(
                "ResetlessCarEnvCostWrapper does not support vectorized environments."
            )
        self.episode_length = 0
        self.cumulative_reward = 0
        self.cumulative_cost = 0
        self.env.metadata["render_fps"] = int(np.ceil(1 / self.env.unwrapped.dt))
        self.lidar_max_range = float(
            self.env.unwrapped.sensors["lidar_points"]._range_max
        )

        self.inverse_problem = inverse_problem
        self.cost_mode = cost_mode
        self.cost_buffer_radius = cost_buffer_radius
        self.reset_reward_type = reset_reward_type
        self.debug = debug

        if debug:
            self.success_count = 0

    def step(self, action):
        # Trick env into thinking it does not have to be reset after a crash
        self.env.unwrapped._reset_required = False

        obs, forward_rews, terminateds, truncateds, infos = self.env.step(action)

        x, y, theta = self.env.unwrapped.ego_pose
        self.trajectory = np.append(self.trajectory, [[x, y, theta]], axis=0)
        self.has_collided = terminateds

        cost = self._calculate_costs(forward_rews, obs, terminateds)

        # Cost is stored in info field analogous to SafetyGym
        infos["cost"] = cost

        # Inverse reward function for the reset agent
        reset_rews = self._calculate_reset_reward(forward_rews)
        infos["reset_rewards"] = reset_rews

        rews = forward_rews if self.mode == DynamicsMode.FORWARD else reset_rews

        self.episode_length += 1
        self.cumulative_reward += rews
        self.cumulative_cost += cost

        # Set truncation flag depending on the currently active agent
        if (
            self.mode == DynamicsMode.FORWARD
            and self.episode_length >= self.max_forward_steps
        ):
            truncateds = True
            infos["active_agent"] = "forward"
        elif (
            self.mode == DynamicsMode.RESET
            and self.episode_length >= self.max_reset_steps
        ):
            truncateds = True
            infos["active_agent"] = "reset"

        if self.mode == DynamicsMode.FORWARD and terminateds:
            # Vehicle has crashed in forward mode and therefore looses all momentum
            # Reset the env to the last state before the collision
            self.unwrapped.vehicle_state = self.last_save_vehicle_state
            self.unwrapped.vehicle_model.state_ = self.last_save_vehicle_state
            self.unwrapped.steering_history = self.last_save_steering_history
            self.unwrapped.vehicle_last_speed = self.last_save_vehicle_last_speed
            self.unwrapped.traveled_distance = self.last_save_traveled_distance

            # Set all velocities to zero
            self.env.unwrapped.vehicle_model.state_[3] = 0.0
            self.env.unwrapped.vehicle_state[3] = 0.0
            self.env.unwrapped.vehicle_last_speed = 0.0

            # Restore corresponding observation and reset its velocity to zero
            obs = self.last_save_obs
            obs[0] = 0.0

            # Clear crash counter
            self.env.unwrapped.objects["walls"].hit_count = 0

        if terminateds or truncateds:
            infos["episode"] = {
                "r": self.cumulative_reward,
                "l": self.episode_length,
                "c": self.cumulative_cost,
            }

        if terminateds:
            infos["real_termination"] = True
            infos["collision"] = True

        if truncateds:
            # Can be set to obs for CarEnv since the returned observation is always valid
            infos["final_observation"] = obs

        # Store environment data so that the last state before a collision can be restored
        if not terminateds:
            self.last_save_vehicle_state = self.unwrapped.vehicle_state
            self.last_save_steering_history = self.unwrapped.steering_history
            self.last_save_vehicle_last_speed = self.unwrapped.vehicle_last_speed
            self.last_save_traveled_distance = self.unwrapped.traveled_distance
            self.last_save_obs = obs

        return deepcopy((obs, forward_rews, terminateds, truncateds, infos))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        x, y, theta = self.env.unwrapped.ego_pose
        self.trajectory = np.array([[x, y, theta]])
        self.episode_length = 0
        self.cumulative_reward = 0
        self.cumulative_cost = 0
        self.consecutive_steps_in_reset_states = 0

        self.mode = DynamicsMode.FORWARD
        self.env.unwrapped.show_agent_mode_indicator(False)

        return obs, info

    def reset_to_pose(self, pose, **kwargs):
        _, info = self.env.reset(**kwargs)

        start_pose = np.array([pose[0], pose[1], pose[2]])
        self.env.unwrapped.vehicle_model.set_pose(start_pose)
        obs = self.env.unwrapped._make_obs()

        self.mode = DynamicsMode.FORWARD

        x, y, theta = self.env.unwrapped.ego_pose
        self.trajectory = np.array([[x, y, theta]])
        self.episode_length = 0
        self.cumulative_reward = 0
        self.cumulative_cost = 0
        self.consecutive_steps_in_reset_states = 0

        return obs, info

    def start_forward_episode(self) -> None:
        self.episode_length = 0
        self.cumulative_reward = 0
        self.cumulative_cost = 0
        self.consecutive_steps_in_reset_states = 0

        x, y, theta = self.env.unwrapped.ego_pose
        self.trajectory = np.array([[x, y, theta]])

        self.mode = DynamicsMode.FORWARD
        self.env.unwrapped.show_agent_mode_indicator(False)

    def soft_reset(self) -> None:
        self.episode_length = 0
        self.cumulative_reward = 0
        self.cumulative_cost = 0
        self.consecutive_steps_in_reset_states = 0

        x, y, theta = self.env.unwrapped.ego_pose
        self.trajectory = np.array([[x, y, theta]])

        self.mode = DynamicsMode.RESET
        self.env.unwrapped.show_agent_mode_indicator(True)

    def set_step_limits(self, max_forward_steps: int, max_reset_steps: int) -> None:
        self.max_forward_steps = max_forward_steps
        self.max_reset_steps = max_reset_steps

    def _calculate_reset_reward(self, rews: float):
        problem = self.env.unwrapped.problem
        if self.reset_reward_type == "lnt":
            # Reward is calculated based on the distance to the center of the initial state distribution
            reward = problem.pose_reward(
                self.env.unwrapped.ego_pose, problem.valid_reset_interval_center[0]
            ).reward

        elif self.reset_reward_type == "lnt_sparse":
            reward = self.is_reset() * 1.0

        else:
            # Reward is calculated as above but maxes out when entering area in which start poses can be generated in
            # Also adds reward for near zero velocity at target pose
            current_xy, current_alpha, target_xy, target_alpha = (
                self.env.unwrapped.ego_pose[:2],
                self.env.unwrapped.ego_pose[2],
                problem.valid_reset_interval_center[:, :2],
                problem.valid_reset_interval_center[:, 2],
            )

            # Find the closest starting zone and use it to calculate the reward
            delta_xy = np.abs(current_xy - target_xy)
            closest_starting_zone_idx = np.argmin(np.linalg.norm(delta_xy, axis=1))

            abs_delta_x, abs_delta_y = delta_xy[closest_starting_zone_idx]
            target_alpha = target_alpha[closest_starting_zone_idx]

            delta_angle = current_alpha - target_alpha

            abs_delta_vel = np.abs(self.env.unwrapped.vehicle_model.state_[3])

            # Calculate deviations with respect to the general start state area
            abs_delta_x = np.maximum(
                problem.valid_reset_interval_tolerances[closest_starting_zone_idx, 0],
                abs_delta_x,
            )
            abs_delta_y = np.maximum(
                problem.valid_reset_interval_tolerances[closest_starting_zone_idx, 1],
                abs_delta_y,
            )
            delta_angle = np.maximum(
                problem.valid_reset_interval_tolerances[closest_starting_zone_idx, 2],
                abs(delta_angle),
            )
            abs_delta_vel = np.maximum(0.05, abs_delta_vel)

            reward_x = np.maximum(0.0, 20 - abs_delta_x) / 20
            reward_y = np.maximum(0.0, 5 - abs_delta_y) / 5
            reward_angle = np.maximum(0.0, np.cos(delta_angle))
            reward_vel = (
                np.maximum(
                    0.0,
                    self.env.unwrapped.vehicle_model.velocity_controller.top_speed
                    - abs_delta_vel,
                )
                / self.env.unwrapped.vehicle_model.velocity_controller.top_speed
            )
            reward = reward_x * reward_y * reward_angle * reward_vel
        return reward * problem.k_continuous if rews >= 0.0 else -1.0

    def _calculate_costs(self, reward, next_obs, termination):
        if self.cost_mode == "buffer_zone_v2":
            dist_buffer = self.cost_buffer_radius / self.lidar_max_range
            if np.any(
                (
                    next_obs[..., 2:]
                    - self.env.unwrapped.polar_vehicle_boundary
                    - dist_buffer
                )[next_obs[..., 2:] >= 0]
                <= 0
            ):
                cost = 1
            else:
                cost = 0
        elif self.cost_mode == "inverse_distance":
            dist_vehicle_to_wall = (
                next_obs[..., 2:] - self.env.unwrapped.polar_vehicle_boundary
            )[next_obs[..., 2:] >= 0].min()
            cost = 1 - dist_vehicle_to_wall
        elif self.cost_mode == "exponential_v2":
            dist_buffer = self.cost_buffer_radius / self.lidar_max_range
            dist_vehicle_to_wall = (
                next_obs[..., 2:]
                - self.env.unwrapped.polar_vehicle_boundary
                - dist_buffer
            )[next_obs[..., 2:] >= 0].min()
            cost = np.exp(-dist_vehicle_to_wall)
            cost = np.clip(cost, 0.0, 1.0)
            cost /= 300
            if dist_vehicle_to_wall <= 0:
                cost = 1
        elif self.cost_mode == "exponential_v3":
            dist_buffer = self.cost_buffer_radius / self.lidar_max_range
            dist_vehicle_to_wall = (
                next_obs[..., 2:]
                - self.env.unwrapped.polar_vehicle_boundary
                - dist_buffer
            )[next_obs[..., 2:] >= 0].min()
            cost = np.exp(-dist_vehicle_to_wall)
            cost = np.clip(cost, 0.0, 1.0)
            cost /= 300
            if reward < 0.0:  # and dist_vehicle_to_wall <= 0
                cost = 1
        else:
            print("Invalid cost mode. Using default cost function.")
            cost = float(reward < 0.0)
        return cost

    def get_success_examples(self, mode: str = "forward"):
        if mode == "forward":
            self.reset()
            rng = self.env.unwrapped._np_random

            target_pose = self.env.unwrapped.problem.target_pose

            example_pose = np.array(
                [
                    rng.uniform(target_pose[0] - 1.0, target_pose[0] + 1.0),
                    rng.uniform(target_pose[1] - 0.15, target_pose[1] + 0.15),
                    rng.uniform(target_pose[2] - 0.1, target_pose[2] + 0.1),
                ]
            )

            self.env.unwrapped.vehicle_model.set_pose(example_pose)

            if self.debug:
                self.step(np.array([0.0, 0.0]))
                img = self.render()
                img = Image.fromarray(img)
                img.save(
                    "./debug/success_examples/example_"
                    + str(self.success_count)
                    + ".jpg"
                )
                print("Example No. ", self.success_count)
                self.success_count += 1

            obs = self.env.unwrapped._make_obs()
            return obs
        elif mode == "reset":
            obs = self.reset()[0]
            if self.debug:
                self.step(np.array([0.0, 0.0]))
                img = self.render()
                img = Image.fromarray(img)
                img.save(
                    "./debug/success_examples/example_"
                    + str(self.success_count)
                    + ".jpg"
                )
                print("Example No. ", self.success_count)
                self.success_count += 1
            return obs
        else:
            raise ValueError(mode, "is a invalid value for mode")

    def is_reset(self):
        # Tolerances are taken from the distribution used for creating the start position
        is_currently_reset = np.isclose(
            self.env.unwrapped.vehicle_model.get_pose(),
            self.env.unwrapped.problem.valid_reset_interval_center,
            atol=self.env.unwrapped.problem.valid_reset_interval_tolerances,
        ).all(axis=1).any() and np.isclose(
            self.env.unwrapped.vehicle_model.state_[3], 0.0, atol=0.05
        )
        if is_currently_reset:
            self.consecutive_steps_in_reset_states += 1
        elif not is_currently_reset and self.consecutive_steps_in_reset_states > 0:
            self.consecutive_steps_in_reset_states = 0

        if self.consecutive_steps_in_reset_states >= 1:
            # Set all velocities to zero since actually reaching zero velocity is quite hard for the agent
            self.env.unwrapped.vehicle_model.state_[3] = 0.0
            self.env.unwrapped.vehicle_state[3] = 0.0
            self.env.unwrapped.vehicle_last_speed = 0.0
            return True
        else:
            return False

    def get_delta_x_goal(self):
        return np.abs(
            self.env.unwrapped.problem.target_pose[0]
            - self.env.unwrapped.vehicle_model.get_pose()[0]
        )

    def get_delta_y_goal(self):
        return np.abs(
            self.env.unwrapped.problem.target_pose[1]
            - self.env.unwrapped.vehicle_model.get_pose()[1]
        )

    def get_delta_theta_goal(self):
        delta_abs = np.abs(
            self.env.unwrapped.problem.target_pose[2]
            - self.env.unwrapped.vehicle_model.get_pose()[2]
        )
        if delta_abs > np.pi:
            delta_abs = 2 * np.pi - delta_abs
        return delta_abs

    def get_delta_x_start(self):
        # if self.reset_reward_type == "lnt":
        #     return np.abs(
        #         self.env.unwrapped.problem.start_pose[0]
        #         - self.env.unwrapped.vehicle_model.get_pose()[0]
        #     )
        # else:
        # Calculate distance to the closest starting area
        curr_x = self.env.unwrapped.vehicle_model.get_pose()[0]
        delta_x = np.abs(
            curr_x - self.env.unwrapped.problem.valid_reset_interval_center[:, 0]
        )
        closest_starting_zone_idx = np.argmin(delta_x)
        tolerance = self.env.unwrapped.problem.valid_reset_interval_tolerances[
            closest_starting_zone_idx, 0
        ]
        delta_x = delta_x[closest_starting_zone_idx]
        adjusted_delta_x = (delta_x - tolerance) * (delta_x > tolerance)
        return np.abs(adjusted_delta_x)

    def get_delta_y_start(self):
        # if self.reset_reward_type == "lnt":
        #     return np.abs(
        #         self.env.unwrapped.problem.start_pose[1]
        #         - self.env.unwrapped.vehicle_model.get_pose()[1]
        #     )
        # else:
        # Calculate distance to the closest starting area
        curr_y = self.env.unwrapped.vehicle_model.get_pose()[1]
        delta_y = np.abs(
            curr_y - self.env.unwrapped.problem.valid_reset_interval_center[:, 1]
        )
        closest_starting_zone_idx = np.argmin(delta_y)
        tolerance = self.env.unwrapped.problem.valid_reset_interval_tolerances[
            closest_starting_zone_idx, 1
        ]
        delta_y = delta_y[closest_starting_zone_idx]
        adjusted_delta_y = (delta_y - tolerance) * (delta_y > tolerance)
        return np.abs(adjusted_delta_y)

    def get_delta_theta_start(self):
        # if self.reset_reward_type == "lnt":
        #     delta_abs = np.abs(
        #         self.env.unwrapped.problem.start_pose[2]
        #         - self.env.unwrapped.vehicle_model.get_pose()[2]
        #     )
        #     if delta_abs > np.pi:
        #         delta_abs = 2 * np.pi - delta_abs
        #     return delta_abs
        # else:
        curr_theta = self.env.unwrapped.vehicle_model.get_pose()[2]
        delta_theta = np.abs(
            curr_theta - self.env.unwrapped.problem.valid_reset_interval_center[:, 2]
        )
        if delta_theta > np.pi:
            delta_theta = 2 * np.pi - delta_theta
        closest_starting_zone_idx = np.argmin(delta_theta)
        tolerance = self.env.unwrapped.problem.valid_reset_interval_tolerances[
            closest_starting_zone_idx, 2
        ]
        delta_theta = delta_theta[closest_starting_zone_idx]
        adjusted_delta_theta = (delta_theta - tolerance) * (delta_theta > tolerance)
        return np.abs(adjusted_delta_theta)

    def get_delta_v(self):
        return np.abs(self.unwrapped.vehicle_last_speed)

    def get_init_trajectory_data(
        self, max_episode_steps: int, trajectory_count: int = 100
    ):
        if self.inverse_problem:
            env_obstacles = self.get_obstacles().copy()
            env_obstacles[:, :2] -= self.get_stat_without_noise()
            env_obstacles[:, 2:] -= self.get_stat_without_noise()
        else:
            # Store obstacles (after normalizing them to the goal position) for visualization (see trajectory plot)
            env_obstacles = self.get_obstacles().copy()
            env_obstacles[:, :2] -= self.get_goal()
            env_obstacles[:, 2:] -= self.get_goal()

        return (
            np.zeros((trajectory_count, max_episode_steps + 1, 2)),
            np.zeros((trajectory_count, 2)),
            np.zeros((trajectory_count, 2)),
            np.zeros((trajectory_count, 2)),
            env_obstacles,
            np.zeros((trajectory_count,)),
        )

    def get_trajectory(self):
        # (Target (x,y), Trajectory (x, y, theta), has_collided)
        return (
            self.get_stat_without_noise() if self.inverse_problem else self.get_goal(),
            self.get_goal(),
            self.trajectory,
            self.has_collided,
        )

    def get_episode_info(self):
        return {
            "r": self.cumulative_reward,
            "l": self.episode_length,
            "c": self.cumulative_cost,
        }

    def get_goal(self):
        return self.env.unwrapped.problem.target_pose[:2]

    def get_start(self):
        return self.env.unwrapped.problem.start_pose[:2]

    def get_stat_without_noise(self):
        return self.env.unwrapped.problem.orig_start_pose[:2]

    def get_obstacles(self):
        return self.env.unwrapped.objects["walls"].data
