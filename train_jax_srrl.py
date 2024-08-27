import os
import numpy as np
import jax
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import gymnasium as gym

from src.utils import make_resetless_env
from src.SRRL import SRRL

# https://jax.readthedocs.io/en/latest/gpu_performance_tips.html
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_async_collectives=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)


def evaluate_env(
    eval_env,
    key,
    forward_act_fn,
    reset_act_fn,
    evaluate_reset_critic_fn,
    is_reset_fn,
    writer,
    curr_timestep,
    cfg,
):
    episode_rewards = []
    episode_costs = []
    episode_forward_steps = []
    episode_reset_steps = []
    episode_delta_x_goal = []
    episode_delta_y_goal = []
    episode_delta_theta_goal = []
    episode_delta_x_start = []
    episode_delta_y_start = []
    episode_delta_theta_start = []
    episode_delta_v_start = []
    episode_delta_v_goal = []
    bitmaps = []
    for i in range(cfg.eval_tries):
        obs, _ = eval_env.reset(seed=i)
        if i == 0:
            bitmaps.append(eval_env.render())

        # Use forward agent
        done = False
        acc_rew = 0.0
        acc_cost = 0.0
        step = 0
        while not done:
            action = forward_act_fn(
                key, obs, deterministic=True
            )  # Note: Key can be used here without splitting because deterministic mode is enabled and the key is therefore not used.

            obs, rew, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            if i == 0:
                bitmaps.append(eval_env.render())
            acc_rew += rew
            acc_cost += info["cost"]
            step += 1

        episode_rewards.append(acc_rew)
        episode_costs.append(acc_cost)
        episode_forward_steps.append(step)
        episode_delta_x_goal.append(eval_env.get_delta_x_goal())
        episode_delta_y_goal.append(eval_env.get_delta_y_goal())
        episode_delta_theta_goal.append(eval_env.get_delta_theta_goal())
        episode_delta_v_goal.append(eval_env.get_delta_v())

        # Trigger soft reset
        eval_env.soft_reset()

        # Use reset agent
        done = False
        step = 0
        while not done:
            action = reset_act_fn(
                key, obs, deterministic=True
            )  # Note: Key can be used here without splitting because deterministic mode is enabled and the key is therefore not used.

            obs, rew, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated or is_reset_fn(obs, action, eval=True)
            if i == 0:
                bitmaps.append(eval_env.render())
            step += 1

        episode_reset_steps.append(step)
        episode_delta_x_start.append(eval_env.get_delta_x_start())
        episode_delta_y_start.append(eval_env.get_delta_y_start())
        episode_delta_theta_start.append(eval_env.get_delta_theta_start())
        episode_delta_v_start.append(eval_env.get_delta_v())

    bitmaps = np.transpose(np.stack(bitmaps), (0, 3, 1, 2))
    dt = eval_env.unwrapped.dt if isinstance(eval_env, gym.Wrapper) else eval_env.dt
    fps = int(np.ceil(1 / dt))
    if cfg.experiment_meta.track:
        wandb.log({"eval/video": wandb.Video(bitmaps, fps=fps)}, step=curr_timestep)
    writer.add_scalar("eval/ep_rew_mean", np.mean(episode_rewards), curr_timestep)
    writer.add_scalar("eval/ep_cost_mean", np.mean(episode_costs), curr_timestep)
    writer.add_scalar(
        "eval/ep_len_forward_mean", np.mean(episode_forward_steps), curr_timestep
    )
    writer.add_scalar(
        "eval/ep_len_reset_mean", np.mean(episode_reset_steps), curr_timestep
    )
    writer.add_scalar(
        "eval/ep_delta_x_goal_mean",
        np.mean(episode_delta_x_goal),
        curr_timestep,
    )
    writer.add_scalar(
        "eval/ep_delta_y_goal_mean",
        np.mean(episode_delta_y_goal),
        curr_timestep,
    )
    writer.add_scalar(
        "eval/ep_delta_theta_goal_mean",
        np.mean(episode_delta_theta_goal),
        curr_timestep,
    )
    writer.add_scalar(
        "eval/ep_delta_v_goal_mean",
        np.mean(episode_delta_v_goal),
        curr_timestep,
    )
    writer.add_scalar(
        "eval/ep_delta_x_start_mean",
        np.mean(episode_delta_x_start),
        curr_timestep,
    )
    writer.add_scalar(
        "eval/ep_delta_y_start_mean",
        np.mean(episode_delta_y_start),
        curr_timestep,
    )
    writer.add_scalar(
        "eval/ep_delta_theta_start_mean",
        np.mean(episode_delta_theta_start),
        curr_timestep,
    )
    writer.add_scalar(
        "eval/ep_delta_v_start_mean",
        np.mean(episode_delta_v_start),
        curr_timestep,
    )


def final_evaluation(
    eval_env,
    key,
    rng,
    forward_act_fn,
    reset_act_fn,
    evaluate_reset_critic_fn,
    is_reset_fn,
    writer,
    curr_timestep,
    cfg,
):
    if not cfg.env_id.startswith("CarEnv"):
        return

    eval_env.reset(seed=0)

    # Starting point grid
    x_space = np.arange(-22.0, 23.0, 1.0)
    y_space = np.arange(-1, 2, 1.0)

    from src.logging_util import (
        TrajectoryData,
        update_trajectory_data,
        draw_start_position_episode_reward_heatmap,
    )

    forward_trajectory_data = TrajectoryData(
        *eval_env.get_init_trajectory_data(
            cfg.forward_agent.max_episode_steps, x_space.size * y_space.size
        )
    )
    reset_trajectory_data = TrajectoryData(
        *eval_env.get_init_trajectory_data(
            cfg.reset_agent.max_episode_steps, x_space.size * y_space.size
        )
    )

    for x in np.nditer(x_space):
        for y in np.nditer(y_space):
            x_start = x
            y_start = y
            theta_start = rng.uniform(-0.5, 0.5) * 20 / 180 * np.pi

            starting_pose = np.array([x_start, y_start, theta_start])
            obs, _ = eval_env.reset_to_pose(starting_pose, seed=0)
            info = None
            done = False
            step = 0
            while not done:
                act = forward_act_fn(key, obs, deterministic=True)
                # if evaluate_reset_critic_fn(key, obs, act) < cfg.reset_thresh:
                #     break
                obs, rew, terminated, truncated, info = eval_env.step(act)
                done = terminated or truncated
                step += 1

            if info is not None:
                if "episode" not in info:
                    info["episode"] = eval_env.get_episode_info()
                forward_trajectory_data = update_trajectory_data(
                    forward_trajectory_data,
                    eval_env.get_trajectory(),
                    cfg.forward_agent.max_episode_steps,
                    info,
                )

            # Trigger soft reset
            eval_env.soft_reset()
            done = False
            step = 0
            while not done:
                act = reset_act_fn(key, obs, deterministic=True)
                obs, rew, terminated, truncated, info = eval_env.step(act)
                done = terminated or truncated or is_reset_fn(obs, act, eval=True)
                step += 1

            if "episode" not in info:
                info["episode"] = eval_env.get_episode_info()
            reset_trajectory_data = update_trajectory_data(
                reset_trajectory_data,
                eval_env.get_trajectory(),
                cfg.reset_agent.max_episode_steps,
                info,
            )

    if cfg.experiment_meta.track:
        draw_start_position_episode_reward_heatmap(
            curr_timestep,
            [forward_trajectory_data],
            (-1.0, 15.0),
            "post_training/forward_episode_reward_heatmap",
            eval_env.unwrapped.problem.valid_reset_interval_center,
            eval_env.unwrapped.problem.valid_reset_interval_tolerances,
        )
        draw_start_position_episode_reward_heatmap(
            curr_timestep,
            [reset_trajectory_data],
            (-1.0, 15.0),
            "post_training/reset_episode_reward_heatmap",
            eval_env.unwrapped.problem.valid_reset_interval_center,
            eval_env.unwrapped.problem.valid_reset_interval_tolerances,
        )


@hydra.main(version_base=None, config_path="cfg", config_name="srrl.yaml")
def main(cfg: DictConfig) -> None:
    if jax.default_backend() == "cpu":
        raise RuntimeError(
            "Not able to run on GPU. Aborting as CPU would be used instead..."
        )

    if "seed" not in cfg:
        OmegaConf.update(cfg, "seed", np.random.randint(2**32 - 1), force_add=True)

    env = make_resetless_env(cfg)
    eval_env = make_resetless_env(cfg, eval=True)

    algo = SRRL(cfg, env, eval_env)
    algo.learn(evaluate_env, final_evaluation)


if __name__ == "__main__":
    main()
