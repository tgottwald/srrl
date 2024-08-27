import gymnasium as gym
import numpy as np
from PIL import Image
from copy import deepcopy


class CarEnvCostWrapper(gym.core.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        inverse_problem: bool = False,
        cost_mode: str = "buffer_zone_v2",
        cost_buffer_radius: float = 0.0,
        debug: bool = False,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            inverse_problem=inverse_problem,
            cost_mode=cost_mode,
            cost_buffer_radius=cost_buffer_radius,
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
                "CarEnvCostWrapper does not support vectorized environments."
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
        self.debug = debug

        if debug:
            self.success_count = 0

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)

        x, y, theta = self.env.unwrapped.ego_pose
        self.trajectory = np.append(self.trajectory, [[x, y, theta]], axis=0)
        self.has_collided = terminateds

        cost = self._calculate_costs(rews, obs, terminateds)

        # Cost is stored in info field analogous to SafetyGym
        infos["cost"] = cost

        self.episode_length += 1
        self.cumulative_reward += rews
        self.cumulative_cost += cost

        if terminateds or truncateds:
            infos["episode"] = {
                "r": self.cumulative_reward,
                "l": self.episode_length,
                "c": self.cumulative_cost,
            }

        if terminateds:
            infos["real_termination"] = True

        if truncateds:
            # Can be set to next_obs for CarEnv since the returned observation is always valid
            infos["final_observation"] = obs

        return deepcopy((obs, rews, terminateds, truncateds, infos))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        x, y, theta = self.env.unwrapped.ego_pose
        self.trajectory = np.array([[x, y, theta]])
        self.episode_length = 0
        self.cumulative_reward = 0
        self.cumulative_cost = 0

        return obs, info

    def reset_to_pose(self, pose, **kwargs):
        _, info = self.env.reset(**kwargs)

        start_pose = np.array([pose[0], pose[1], pose[2]])
        self.env.unwrapped.vehicle_model.set_pose(start_pose)
        obs = self.env.unwrapped._make_obs()

        x, y, theta = self.env.unwrapped.ego_pose
        self.trajectory = np.array([[x, y, theta]])
        self.episode_length = 0
        self.cumulative_reward = 0
        self.cumulative_cost = 0

        return obs, info

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

    def get_goal(self):
        return self.env.unwrapped.problem.target_pose[:2]

    def get_start(self):
        return self.env.unwrapped.problem.start_pose[:2]

    def get_stat_without_noise(self):
        return self.env.unwrapped.problem.orig_start_pose[:2]

    def get_obstacles(self):
        return self.env.unwrapped.objects["walls"].data
