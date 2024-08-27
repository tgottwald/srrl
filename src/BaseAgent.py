import random
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Union

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from .logging_util import *


class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(
        self,
        cfg: DictConfig,
        env: Union[gym.Env, gym.Wrapper],
        eval_env: Union[gym.Env, gym.Wrapper],
        slave: bool = False,
        seed: int = None,
        writer: SummaryWriter = None,
        logging_prefix: str = None,
    ):
        self.cfg = cfg
        self.prefix = logging_prefix

        # Store configs params in seperate variables to increase performance
        self.max_episode_steps = cfg.max_episode_steps
        self.batch_size = cfg.batch_size
        self.utd_ratio = cfg.update_to_data_ratio

        # Agent specific logging function. Set/Replace for each agent individually.
        self.specific_log_fn = None

        self.env = env
        self.eval_env = eval_env

        assert isinstance(
            self.env.action_space, gym.spaces.Box
        ), "only continuous action space is supported"
        assert isinstance(
            self.eval_env.action_space, gym.spaces.Box
        ), "only continuous action space is supported"

        # Use config seed if seed not specified in parameters
        if seed is None:
            self.seed = cfg.seed
        else:
            self.seed = seed

        if slave:
            # Start as part of another algorithm (e.g. Resetless RL framework)
            run_name = f"seed={self.seed}_{int(time.time())}"

            assert writer is not None, "Need to specify writer if running as slave"
            self.writer = writer
        else:
            # Start as standalone algorithm
            run_name = f"{cfg.experiment_meta.experiment_name}_seed={self.seed}_{int(time.time())}"

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

            self.writer = SummaryWriter(
                "tmp/runs/" + prefix_string(logging_prefix) + f"{run_name}"
            )
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s"
                % ("\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()])),
            )

            # Seeding
            random.seed(self.seed)
            np.random.seed(self.seed)
            self.rng = np.random.default_rng(seed)

            self.total_timesteps = cfg.total_timesteps
            self.learning_starts = cfg.learning_starts
            self.logging_frequency = cfg.logging_frequency
            self.eval_episode_frequency = cfg.eval_episode_frequency

    @abstractmethod
    def add_to_buffer(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: float,
        real_next_obs: np.ndarray,
        dones: bool,
        infos: Dict[str, Any],
    ) -> None:
        """Adds a transition to the agents replay buffer

        Args:
            obs (np.ndarray): s
            actions (np.ndarray): a
            rewards (float): r
            real_next_obs (np.ndarray): s'
            dones (bool): Whether the episode has terminated or was truncated
            infos (Dict[str, Any]): Additional information from the environment

        Raises:
            NotImplementedError: Function was not implemented
        """
        raise NotImplementedError

    @abstractmethod
    def act(
        self,
        key: jax.random.PRNGKey,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        """Selects an action based on the current observation

        Args:
            key (jax.random.PRNGKey): JAX random key
            observation (np.ndarray): Current observation
            deterministic (bool, optional): If action should be selected using deterministic behavior. Defaults to False.

        Raises:
            NotImplementedError: Function was not implemented

        Returns:
            jnp.ndarray: Action
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(self, key: jax.random.PRNGKey) -> Dict[str, Any]:
        """Performs a single training step

        Args:
            key (jax.random.PRNGKey): JAX random key

        Raises:
            NotImplementedError: Function was not implemented

        Returns:
            Dict[str, Any]: Dictionary containing the loss and other metrics
        """
        raise NotImplementedError

    @abstractmethod
    def learn(
        self, eval_callback: Callable, training_finished_callback: Callable
    ) -> None:
        """Starts the main training loop

        Args:
            eval_callback (Callable): Callback for handling of evaluation
            training_finished_callback (Callable): Callback for handling final evaluation after training has finished

        Raises:
            NotImplementedError: Function was not implemented
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_critic(
        self, key: jax.random.PRNGKey, observation: np.ndarray, action: np.ndarray
    ) -> float:
        """Evaluates the critic for a given state-action tuple and returns a scalar indicating whether the SRRL framework should trigger a reset

        Args:
            key (jax.random.PRNGKey): JAX random key
            observation (np.ndarray): Current observation
            action (np.ndarray): Action

        Raises:
            NotImplementedError: Function was not implemented

        Returns:
            float: Abstract scalar indicating the quality of the state-action tuple
        """
        raise NotImplementedError

    def is_reset(
        self, observation: np.ndarray, action: np.array, eval: bool = False
    ) -> bool:
        """Returns whether the environment has been reset according to the observation and action

        Args:
            observation (np.ndarray): Current observation
            action (np.array): Selected action
            eval (bool, optional): If called in eval mode. Defaults to False.

        Returns:
            bool: True = environment has been reset, False = environment has not been reset
        """
        # NOTE: This method currently relies on signals from the environment. Please see future work section of the paper.
        if eval:
            return self.eval_env.is_reset()
        else:
            return self.env.is_reset()
