import random
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .logging_util import *
from .utils import make_agent


class SRRL:
    def __init__(
        self,
        cfg: DictConfig,
        env: Union[gym.Env, gym.Wrapper],
        eval_env: Union[gym.Env, gym.Wrapper],
    ):
        self.cfg = cfg
        self.seed = cfg.seed

        run_name = (
            f"{cfg.experiment_meta.experiment_name}_seed={self.seed}_{int(time.time())}"
        )

        if cfg.experiment_meta.track:
            # Initialize Weights and Biases
            import wandb

            wandb.init(
                project=cfg.experiment_meta.wandb_project_name,
                group=cfg.experiment_meta.wandb_group_name,
                sync_tensorboard=True,
                config=cfg,
                name=run_name,
                monitor_gym=True,
                dir="tmp",
            )

        self.writer = SummaryWriter(f"tmp/runs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()])),
        )

        # Seeding
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.key = jax.random.PRNGKey(seed=self.seed)

        self.env = env
        self.eval_env = eval_env

        # Inform EnvWrapper about step limits for each agent
        self.env.set_step_limits(
            cfg.forward_agent.max_episode_steps, cfg.reset_agent.max_episode_steps
        )
        self.eval_env.set_step_limits(
            cfg.forward_agent.max_episode_steps, cfg.reset_agent.max_episode_steps
        )

        assert isinstance(
            self.env.action_space, gym.spaces.Box
        ), "only continuous action space is supported"
        assert isinstance(
            self.eval_env.action_space, gym.spaces.Box
        ), "only continuous action space is supported"

        # Create forward and backward agent instances
        self.forward_agent = make_agent(
            cfg.forward_agent.algorithm,
            cfg.forward_agent,
            env=self.env,
            eval_env=self.eval_env,
            slave=True,
            seed=cfg.seed,
            writer=self.writer,
            logging_prefix="forward/",
        )
        self.reset_agent = make_agent(
            cfg.reset_agent.algorithm,
            cfg.reset_agent,
            env=self.env,
            eval_env=self.eval_env,
            slave=True,
            seed=cfg.seed,
            writer=self.writer,
            logging_prefix="reset/",
        )

        # Store configs params in seperate variables to increase performance
        self.total_timesteps = cfg.total_timesteps
        self.learning_starts = cfg.learning_starts
        self.logging_frequency = cfg.logging_frequency
        self.eval_episode_frequency = cfg.eval_episode_frequency
        self.reset_thresh = cfg.reset_thresh
        self.n_step_violation_causality = cfg.n_step_violation_causality
        self.min_forward_step_per_episode = cfg.min_forward_step_per_episode

    def is_forward_agent(self, agent: Any) -> bool:
        return agent.prefix == "forward/"

    def is_reset_agent(self, agent: Any) -> bool:
        return agent.prefix == "reset/"

    def start_forward_episode(self) -> None:
        """Starts a fresh forward episode by selecting the forward agent"""
        self.switch_agents = True
        self.env.start_forward_episode()
        self.forward_episodes += 1

    def soft_reset(self) -> None:
        """Triggers a soft reset of the environment by selecting the reset agent"""
        self.switch_agents = True
        self.env.soft_reset()
        self.reset_episodes += 1

    def hard_reset(self, agent: Any) -> np.ndarray:
        # Only switch active agent if hard reset was triggered in reset episode
        self.switch_agents = self.is_reset_agent(agent)
        obs, _ = self.env.reset()
        if self.is_reset_agent(agent):
            self.forward_episodes += 1
        else:
            self.reset_episodes += 1
        return obs

    def add_to_buffer(
        self,
        agent: Any,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: float,
        real_next_obs: np.ndarray,
        dones: bool,
        infos: Dict[str, Any],
    ) -> None:

        # Check if termination was caused by a safety constraint violation
        violation_termination = "collision" in infos

        # Add transition to currently selected agents buffer
        if self.is_forward_agent(agent):
            self.forward_agent.add_to_buffer(
                obs, actions, rewards, real_next_obs, dones, infos
            )
        elif self.is_reset_agent(agent):
            self.reset_agent.add_to_buffer(
                obs, actions, infos["reset_rewards"], real_next_obs, dones, infos
            )
            if (
                violation_termination
                and self.env.episode_length < self.n_step_violation_causality
            ):
                # Termination was probably due to actions of the forward agent
                self.inevitable_violation_count += 1

    def learn(
        self, eval_callback: Callable, training_finished_callback: Callable
    ) -> None:
        split_keys = jax.jit(lambda k: jax.random.split(k, 5))

        obs, _ = self.env.reset(seed=self.seed)
        self.selected_agent = self.forward_agent
        self.switch_agents = False

        # Collect initial trajectory data (only used for evaluation purposes)
        self.eval_env.reset(seed=0)
        self.forward_agent.trajectory_data = TrajectoryData(
            *self.eval_env.get_init_trajectory_data(self.reset_agent.max_episode_steps)
        )
        self.eval_env.reset(seed=0)
        self.reset_agent.trajectory_data = TrajectoryData(
            *self.eval_env.get_init_trajectory_data(self.reset_agent.max_episode_steps)
        )

        latest_soft_resets_successful = np.zeros((100,), dtype=bool)
        total_failed_soft_resets = 0
        total_succ_soft_resets = 0
        forward_action_step = 0
        reset_action_step = 0
        rejected_action_step = 0
        consecutively_rejected_steps = 0
        self.forward_episodes = 0
        self.reset_episodes = 0
        self.forward_agent.violation_count = 0
        self.reset_agent.violation_count = 0
        self.forward_agent.hard_reset_count = 0
        self.reset_agent.hard_reset_count = 0
        self.inevitable_violation_count = 0
        self.forward_agent.total_cumulative_cost = 0.0
        self.reset_agent.total_cumulative_cost = 0.0

        pre_abort_state = None
        pre_abort_action = None

        start_time = time.time()

        # Training loop
        for global_step in tqdm(
            range(self.total_timesteps),
            desc="\033[1mTraining Progress",
            colour="green",
        ):
            self.key, act_key, act_noise_key, check_key, train_key = split_keys(
                self.key
            )

            # Select corresponding step count for each agent (necessary for correctly triggering the)
            if self.is_forward_agent(self.selected_agent):
                selected_agent_step = forward_action_step
            elif self.is_reset_agent(self.selected_agent):
                selected_agent_step = reset_action_step

            if (
                self.is_forward_agent(self.selected_agent)
                and selected_agent_step < self.learning_starts
            ):
                actions = self.env.action_space.sample()
            else:
                actions = self.selected_agent.act(act_key, obs)

                # Add noise to the action if it would be rejected
                # Results in exploration around areas deemed unresettable by the rest agent -> Increases information for reset agent
                if (
                    self.is_forward_agent(self.selected_agent)
                    and reset_action_step > self.learning_starts
                    and not self.reset_agent.is_reset(obs, actions)
                    and self.env.episode_length > self.min_forward_step_per_episode
                    and self.reset_agent.evaluate_critic(check_key, obs, actions)
                    < self.reset_thresh
                ):
                    actions += jax.random.normal(act_noise_key, actions.shape) * 0.1
                    actions = jnp.clip(actions, -1.0, 1.0)

            # Forward agent would execute a inresettable action
            # Reset agents critic evaluation is only valid after it has started learning
            if (
                self.is_forward_agent(self.selected_agent)
                and reset_action_step > self.learning_starts
                and not self.reset_agent.is_reset(obs, actions)
                and self.env.episode_length > self.min_forward_step_per_episode
                and self.reset_agent.evaluate_critic(check_key, obs, actions)
                < self.reset_thresh
            ):
                # Handle rejection as if the agent had a fatal termination
                rejected_action_step += 1
                consecutively_rejected_steps += 1
                # Manually retrieve the info dict which would be returned by step
                infos["episode"] = {
                    "r": self.env.cumulative_reward,
                    "l": self.env.episode_length,
                    "c": self.env.cumulative_cost,
                }
                infos = deepcopy(infos)
                trajectory = deepcopy(self.env.get_trajectory())

                # Store the state and action which triggered the abort
                pre_abort_state = obs
                pre_abort_action = actions

                # Forward agent tried to execute a inresettable action -> soft reset
                self.soft_reset()

            else:
                next_obs, rewards, terminations, truncations, infos = self.env.step(
                    actions
                )

                # Check if termination was caused by a safety constraint violation
                violation_termination = "collision" in infos

                # Update step counter of respspective agent
                if self.is_forward_agent(self.selected_agent):
                    forward_action_step += 1
                elif self.is_reset_agent(self.selected_agent):
                    reset_action_step += 1

                # Record rewards for plotting purposes
                if "episode" in infos:
                    trajectory = deepcopy(self.env.get_trajectory())

                real_next_obs = next_obs.copy()

                if truncations:
                    real_next_obs = infos["final_observation"]

                # Add transition to replay buffer
                self.add_to_buffer(
                    self.selected_agent,
                    obs,
                    actions,
                    rewards,
                    real_next_obs,
                    terminations,
                    infos,
                )

                # Handle successfull reset
                if (
                    not (violation_termination or truncations)
                    and self.is_reset_agent(self.selected_agent)
                    and self.reset_agent.is_reset(obs, actions)
                ):
                    # Reset agent successfully reset environment and forward agent may start again
                    total_succ_soft_resets += 1
                    latest_soft_resets_successful[0] = True
                    # Manually retrieve the info dict which would be returned by step
                    infos["episode"] = {
                        "r": self.env.cumulative_reward,
                        "l": self.env.episode_length,
                        "c": self.env.cumulative_cost,
                    }
                    infos = deepcopy(infos)
                    trajectory = deepcopy(self.env.get_trajectory())

                    # If reset was triggered by a abort, build forward transition from state and action
                    # causing the abort and the reset state and reward of reset episode
                    if pre_abort_state is not None and pre_abort_action is not None:
                        self.add_to_buffer(
                            self.forward_agent,
                            pre_abort_state,
                            pre_abort_action,
                            rewards,
                            real_next_obs,
                            False,
                            infos,
                        )
                        pre_abort_state = None
                        pre_abort_action = None

                    # Reset agent managed to reach target -> Let forward agent take over again
                    self.start_forward_episode()

                # Handle termination and truncation and setting obs
                if self.is_reset_agent(self.selected_agent) and (
                    violation_termination or truncations
                ):
                    total_failed_soft_resets += 1
                    self.reset_agent.hard_reset_count += 1
                    latest_soft_resets_successful[0] = False
                    # Reset agent did not manage to reset in time or terminated -> hard reset
                    obs = self.hard_reset(self.selected_agent)

                    # If reset was triggered by a abort, build forward transition from state and action
                    # causing the abort and the reset state and reward of reset episode
                    if pre_abort_state is not None and pre_abort_action is not None:
                        self.add_to_buffer(
                            self.forward_agent,
                            pre_abort_state,
                            pre_abort_action,
                            rewards,
                            obs,
                            False,
                            infos,
                        )
                        pre_abort_state = None
                        pre_abort_action = None

                else:
                    obs = next_obs
                    if self.is_forward_agent(self.selected_agent) and (
                        violation_termination or truncations
                    ):
                        consecutively_rejected_steps = 0
                        if violation_termination:
                            # Forward agent terminated due to a collision -> hard reset
                            self.forward_agent.hard_reset_count += 1
                            obs = self.hard_reset(self.selected_agent)
                        elif truncations:
                            # Forward agent terminated due to a truncation -> soft reset
                            self.soft_reset()

            if "episode" in infos:
                # Episode has terminated or was truncated
                self.writer.add_scalar(
                    "charts/forward_steps",
                    forward_action_step,
                    global_step,
                )
                self.writer.add_scalar(
                    "charts/reset_steps",
                    reset_action_step,
                    global_step,
                )
                (
                    self.selected_agent.violation_count,
                    self.selected_agent.total_cumulative_cost,
                    self.selected_agent.trajectory_data,
                ) = log_end_of_episode(
                    global_step,
                    self.selected_agent.prefix,
                    self.writer,
                    infos,
                    self.selected_agent.violation_count,
                    self.selected_agent.hard_reset_count,
                    self.selected_agent.trajectory_data,
                    trajectory,
                    self.selected_agent.max_episode_steps,
                    self.selected_agent.total_cumulative_cost,
                )
                # Update soft reset success ratio
                if self.is_reset_agent(self.selected_agent):
                    latest_soft_resets_successful = np.roll(
                        latest_soft_resets_successful, shift=1
                    )

            if selected_agent_step > self.learning_starts:
                # TRAINING
                metrics = self.selected_agent.train_step(train_key)

                if global_step % self.logging_frequency == 0:

                    # Expand metrics with resetless specific information
                    metrics["forward_steps"] = forward_action_step
                    metrics["reset_steps"] = reset_action_step
                    metrics["rejected_steps"] = rejected_action_step
                    metrics["consecutively_rejected_steps"] = (
                        consecutively_rejected_steps
                    )
                    metrics["inevitable_violation_count"] = (
                        self.inevitable_violation_count
                    )
                    metrics["hard_resets"] = total_failed_soft_resets
                    soft_reset_success_ratio = (
                        total_succ_soft_resets
                        / (total_succ_soft_resets + total_failed_soft_resets)
                        if total_succ_soft_resets + total_failed_soft_resets > 0
                        else 0
                    )
                    metrics["soft_reset_success_ratio"] = soft_reset_success_ratio
                    metrics["latest_soft_reset_success_ratio"] = (
                        latest_soft_resets_successful.mean()
                    )
                    metrics["elapsed_time"] = time.time() - start_time
                    log_metrics(
                        global_step,
                        self.selected_agent.prefix,
                        self.writer,
                        metrics,
                        [log_framework_metrics, self.selected_agent.specific_log_fn],
                    )

                if (
                    global_step % self.eval_episode_frequency == 0
                    or (global_step + 1) == self.total_timesteps
                ) and reset_action_step > self.learning_starts:
                    # EVALUATION

                    # Plot the last N trajectories
                    if self.cfg.experiment_meta.track:
                        draw_trajectories(
                            global_step,
                            [
                                self.forward_agent.trajectory_data,
                                self.reset_agent.trajectory_data,
                            ],
                            ["blue", "orange"],
                            ["green", "cyan"],
                            ["red", "purple"],
                        )
                    # Log the cumulative cost at evaluation time
                    self.writer.add_scalar(
                        prefix_string(self.forward_agent.prefix)
                        + "data/total_cumulative_cost",
                        self.forward_agent.total_cumulative_cost,
                        global_step,
                    )
                    self.writer.add_scalar(
                        prefix_string(self.reset_agent.prefix)
                        + "data/total_cumulative_cost",
                        self.reset_agent.total_cumulative_cost,
                        global_step,
                    )
                    self.writer.add_scalar(
                        prefix_string(self.forward_agent.prefix) + "charts/violations",
                        self.forward_agent.violation_count,
                        global_step,
                    )
                    self.writer.add_scalar(
                        prefix_string(self.reset_agent.prefix) + "charts/violations",
                        self.reset_agent.violation_count,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/forward_violations",
                        self.forward_agent.violation_count,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/reset_violations",
                        self.reset_agent.violation_count,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/total_violations",
                        self.forward_agent.violation_count
                        + self.reset_agent.violation_count,
                        global_step,
                    )
                    self.writer.add_scalar(
                        prefix_string(self.forward_agent.prefix) + "charts/hard_resets",
                        self.forward_agent.hard_reset_count,
                        global_step,
                    )
                    self.writer.add_scalar(
                        prefix_string(self.reset_agent.prefix) + "charts/hard_resets",
                        self.reset_agent.hard_reset_count,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/hard_resets",
                        self.forward_agent.hard_reset_count
                        + self.reset_agent.hard_reset_count,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/forward_episodes",
                        self.forward_episodes,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/reset_episodes",
                        self.reset_episodes,
                        global_step,
                    )

                    soft_reset_success_ratio = (
                        total_succ_soft_resets
                        / (total_succ_soft_resets + total_failed_soft_resets)
                        if total_succ_soft_resets + total_failed_soft_resets > 0
                        else 0
                    )
                    self.writer.add_scalar(
                        "eval/soft_reset_success_ratio",
                        soft_reset_success_ratio,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/latest_soft_reset_success_ratio",
                        latest_soft_resets_successful.mean(),
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/forward_steps",
                        forward_action_step,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/reset_steps",
                        reset_action_step,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/rejected_steps",
                        rejected_action_step,
                        global_step,
                    )

                    if reset_action_step > self.learning_starts:
                        # Start an evaluation episode
                        self.key, eval_key = jax.random.split(self.key, 2)
                        eval_callback(
                            self.eval_env,
                            eval_key,
                            self.forward_agent.act,
                            self.reset_agent.act,
                            self.reset_agent.evaluate_critic,
                            self.reset_agent.is_reset,
                            self.writer,
                            global_step,
                            self.cfg,
                        )
                    else:
                        print(
                            "Skipping evaluation as reset agent has not started learning yet"
                        )

            # Switch agents (if triggered) after performing the agents optimization step
            if self.switch_agents:
                self.switch_agents = False
                if self.is_forward_agent(self.selected_agent):
                    self.selected_agent = self.reset_agent
                elif self.is_reset_agent(self.selected_agent):
                    self.selected_agent = self.forward_agent

        training_finished_callback(
            self.eval_env,
            self.key,
            self.rng,
            self.forward_agent.act,
            self.reset_agent.act,
            self.reset_agent.evaluate_critic,
            self.reset_agent.is_reset,
            self.writer,
            global_step,
            self.cfg,
        )

        self.env.close()
        self.eval_env.close()
        self.writer.close()


def log_framework_metrics(
    global_step: int, prefix: str, writer: SummaryWriter, metrics: Dict[str, Any]
) -> None:
    writer.add_scalar(
        "charts/forward_steps",
        metrics["forward_steps"],
        global_step,
    )
    writer.add_scalar(
        "charts/reset_steps",
        metrics["reset_steps"],
        global_step,
    )
    writer.add_scalar(
        "charts/rejected_steps",
        metrics["rejected_steps"],
        global_step,
    )
    writer.add_scalar(
        "charts/consecutively_rejected_steps",
        metrics["consecutively_rejected_steps"],
        global_step,
    )
    writer.add_scalar(
        "charts/soft_reset_success_ratio",
        metrics["soft_reset_success_ratio"],
        global_step,
    )
    writer.add_scalar(
        "charts/latest_soft_reset_success_ratio",
        metrics["latest_soft_reset_success_ratio"],
        global_step,
    )
    writer.add_scalar("charts/hard_resets", metrics["hard_resets"], global_step)
    writer.add_scalar(
        "reset/charts/inevitable_violation_count",
        metrics["inevitable_violation_count"],
        global_step,
    )
