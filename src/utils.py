import CarEnv
import gymnasium as gym
import numpy as np
import wandb
from omegaconf import DictConfig

from envs.CarEnvCostWrapper import CarEnvCostWrapper
from envs.ResetlessCarEnvCostWrapper import ResetlessCarEnvCostWrapper

from .DREDQ import DREDQ
from .ExampleBasedSAC import ExampleBasedSAC
from .SAC import SAC
from .WCSAC_IQN import WCSAC_IQN


def make_agent(agent_class, cfg, **kwargs):
    if agent_class == "SAC":
        return SAC(cfg, **kwargs)
    elif agent_class == "DREDQ":
        return DREDQ(cfg, **kwargs)
    elif agent_class == "WCSAC_IQN":
        return WCSAC_IQN(cfg, **kwargs)
    elif agent_class == "ExampleBasedSAC":
        return ExampleBasedSAC(cfg, **kwargs)
    else:
        raise ValueError(f"Unknown agent class: {agent_class}")


def make_env(
    cfg: DictConfig,
    eval: bool = False,
):
    if eval:
        render_mode = "rgb_array"
    else:
        render_mode = None
    env = gym.make(cfg.env_id, render_mode=render_mode)
    env = CarEnvCostWrapper(
        env,
        inverse_problem="Inverse" in cfg.env_id,
        cost_mode=cfg.cost_mode,
        cost_buffer_radius=cfg.cost_buffer_radius,
    )
    env.action_space.seed(cfg.seed)
    env.reset(seed=cfg.seed)
    return env


def make_resetless_env(
    cfg: DictConfig,
    eval: bool = False,
):
    if eval:
        render_mode = "rgb_array"
    else:
        render_mode = None
    env = gym.make(cfg.env_id, render_mode=render_mode)
    env = ResetlessCarEnvCostWrapper(
        env,
        inverse_problem="Inverse" in cfg.env_id,
        cost_mode=cfg.cost_mode,
        cost_buffer_radius=cfg.cost_buffer_radius,
        reset_reward_type=cfg.reset_reward_type,
    )
    env.action_space.seed(cfg.seed)
    env.reset(seed=cfg.seed)
    return env


def evaluate_env(
    eval_env,
    key,
    act_fn,
    writer,
    curr_timestep,
    cfg,
):
    episode_rewards = []
    episode_costs = []
    episode_steps = []
    episode_delta_x = []
    episode_delta_y = []
    episode_delta_theta = []
    episode_delta_v = []
    bitmaps = []
    for i in range(cfg.eval_tries):
        obs, _ = eval_env.reset(seed=i)
        if i == 0:
            bitmaps.append(eval_env.render())
        done = False
        acc_rew = 0.0
        acc_cost = 0.0
        step = 0
        while not done:
            act = act_fn(
                key, obs, deterministic=True
            )  # NOTE: Key can be used here without splitting because deterministic mode is enabled and the key is therefore not used.

            obs, rew, terminated, truncated, info = eval_env.step(act)
            done = terminated or truncated
            if i == 0:
                bitmaps.append(eval_env.render())
            acc_rew += rew
            acc_cost += info["cost"]
            step += 1

        episode_rewards.append(acc_rew)
        episode_costs.append(acc_cost)
        episode_steps.append(step)
        episode_delta_x.append(eval_env.get_delta_x_goal())
        episode_delta_y.append(eval_env.get_delta_y_goal())
        episode_delta_theta.append(eval_env.get_delta_theta_goal())
        episode_delta_v.append(eval_env.get_delta_v())

    bitmaps = np.transpose(np.stack(bitmaps), (0, 3, 1, 2))
    dt = eval_env.unwrapped.dt if isinstance(eval_env, gym.Wrapper) else eval_env.dt
    fps = int(np.ceil(1 / dt))
    if cfg.experiment_meta.track and not cfg.env_id.startswith("Safety"):
        wandb.log({"eval/video": wandb.Video(bitmaps, fps=fps)}, step=curr_timestep)
    writer.add_scalar("eval/ep_rew_mean", np.mean(episode_rewards), curr_timestep)
    writer.add_scalar("eval/ep_cost_mean", np.mean(episode_costs), curr_timestep)
    writer.add_scalar("eval/ep_len_mean", np.mean(episode_steps), curr_timestep)
    writer.add_scalar(
        "eval/ep_delta_x_goal_mean",
        np.mean(episode_delta_x),
        curr_timestep,
    )
    writer.add_scalar(
        "eval/ep_delta_y_goal_mean",
        np.mean(episode_delta_y),
        curr_timestep,
    )
    writer.add_scalar(
        "eval/ep_delta_theta_goal_mean",
        np.mean(episode_delta_theta),
        curr_timestep,
    )
    writer.add_scalar(
        "eval/ep_delta_v_goal_mean",
        np.mean(episode_delta_v),
        curr_timestep,
    )


def final_evaluation(
    eval_env,
    key,
    rng,
    act_fn,
    writer,
    curr_timestep,
    cfg,
):
    eval_env.reset(seed=0)

    # Starting point grid
    x_space = np.arange(-22.0, 23.0, 1.0)
    y_space = np.arange(-1, 2, 1.0)

    from src.logging_util import (
        TrajectoryData,
        draw_start_position_episode_reward_heatmap,
        update_trajectory_data,
    )

    trajectory_data = TrajectoryData(
        *eval_env.get_init_trajectory_data(
            cfg.max_episode_steps, x_space.size * y_space.size
        )
    )

    for x in np.nditer(x_space):
        for y in np.nditer(y_space):
            x_start = x
            y_start = y
            theta_start = rng.uniform(-0.5, 0.5) * 20 / 180 * np.pi

            starting_pose = np.array([x_start, y_start, theta_start])
            obs, _ = eval_env.reset_to_pose(starting_pose, seed=0)
            done = False
            step = 0
            while not done:
                act = act_fn(key, obs, deterministic=True)

                obs, rew, terminated, truncated, info = eval_env.step(act)
                done = terminated or truncated
                step += 1

            trajectory_data = update_trajectory_data(
                trajectory_data, eval_env.get_trajectory(), cfg.max_episode_steps, info
            )

    if cfg.experiment_meta.track:
        draw_start_position_episode_reward_heatmap(
            curr_timestep,
            [trajectory_data],
            (-1.0, 15.0),
            "post_training/episode_reward_heatmap",
            eval_env.unwrapped.problem.valid_reset_interval_center,
            eval_env.unwrapped.problem.valid_reset_interval_tolerances,
        )
