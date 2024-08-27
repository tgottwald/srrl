import random
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .ActorNetwork import Actor, ActorTrainState
from .BaseAgent import BaseAgent
from .FeatureExtractors import DummyFeatureExtractor
from .logging_util import *
from .ReplayBuffer import BaseReplayBuffer
from .DistributionalCriticNetwork import (
    DistributionalCriticTrainState,
    DistributionalQNetworkEnsemble,
)
from .SingleParamNetwork import SingleParamNetwork, SingleParamTrainState


@jax.jit
def _evaluate_critic(
    key: jax.random.PRNGKey,
    critic: TrainState,
    observation: jnp.ndarray,
    action: jnp.ndarray,
) -> float:
    observation = observation[None, ...]
    action = action[None, ...]
    _, iota_hat, iota_weights = _sample_iota(
        key, observation.shape[0], critic.num_iota_samples, critic.confidence_level
    )
    q = critic.apply_fn(critic.target_params, observation, action, iota_hat).mean(0)

    q_value_cvar = jnp.sum(iota_weights * q, axis=-1)

    # Max cumulative reward = max_episode_steps * dt * max_reward_single_step
    # 15 = 300 * 0.05 * 1
    return q_value_cvar  # (q_value_cvar + 1) / (15 + 1)


@partial(
    jax.jit,
    static_argnums=(1, 2),
)
def _sample_iota(
    key: jax.random.PRNGKey,
    batch_size: int,
    num_quantiles: int,
    confidence_level: float = 1.0,
) -> jnp.ndarray:
    presum_iota = jax.random.uniform(key, (batch_size, num_quantiles)) + 0.1
    presum_iota /= presum_iota.sum(axis=-1, keepdims=True)

    iota = jnp.cumsum(presum_iota, axis=-1)  # (batch_size, num_quantiles)

    # Calculate quantile midpoints (iota_hat) by avergaging adjacent quantiles
    iota_hat = jnp.zeros_like(iota)
    iota_hat = iota_hat.at[:, 0].set(iota[:, 0] / 2)
    iota_hat = iota_hat.at[:, 1:].set((iota[:, :-1] + iota[:, 1:]) / 2)

    # Move quantiles to relevant range for risk level
    # Note: Different from WCSAC as we want to look at the lower tail of the distribution
    iota = confidence_level * iota
    iota_hat = confidence_level * iota_hat

    return iota, iota_hat, presum_iota


@jax.jit
def _train_step(
    key: jax.random.PRNGKey,
    actor: TrainState,
    critic: TrainState,
    alpha: TrainState,
    batches: List[Tuple[np.array, np.array, np.array, np.array, np.array]],
) -> Tuple[TrainState, TrainState, TrainState, TrainState, Dict[str, Any]]:
    for batch in batches:
        (
            key,
            policy_key,
            ensemble_sample_key,
            critic_iota_key1,
            critic_iota_key2,
            actor_key,
            actor_iota_key,
        ) = jax.random.split(key, 7)
        observations, actions, rewards, next_observations, dones = batch

        dist = actor.apply_fn(actor.params, next_observations)
        next_pi, next_log_pi = dist.sample_and_log_prob(seed=policy_key)

        # Sample random iotas for quantile regression at current and  next time step
        _, iota_hat, _ = _sample_iota(
            critic_iota_key1, next_observations.shape[0], critic.num_iota_samples
        )
        _, next_iota_hat, next_iota_weights = _sample_iota(
            critic_iota_key2, next_observations.shape[0], critic.num_iota_samples
        )

        next_q_candidates = critic.apply_fn(
            critic.target_params, next_observations, next_pi, next_iota_hat
        )

        # Used for DREDQ implementation: Is equal to vanilla SAC for ensemble_size = ensemble_sample_size = 2
        sampled_critics_indices = jax.random.choice(
            ensemble_sample_key,
            next_q_candidates.shape[0],
            critic.ensemble_sample_size,
            replace=False,
        )

        next_q = next_q_candidates[sampled_critics_indices].min(
            0
        )  # min operation from Eq. 3

        entropy_reward = -alpha.apply_fn(alpha.params) * next_log_pi.sum(-1)
        next_q = next_q + entropy_reward[..., None]

        target_q = rewards[..., None] + (1 - dones[..., None]) * critic.gamma * next_q

        def critic_loss_fn(critic_params):
            q = critic.apply_fn(critic_params, observations, actions, iota_hat)

            quantile_values = q[..., None]
            target_quantile_values = target_q[..., None, :]
            # Calculate the element-wise huber loss for every quantile-target_quantile combination for a sample
            quantile_loss = optax.huber_loss(
                quantile_values, target_quantile_values, critic.huber_kappa
            )  # (batch_size, num_quantiles(quantile_values), num_quantiles(target_quantile_values)) Rest of Eq. 3
            heaviside = jnp.heaviside(quantile_values - target_quantile_values, 0.5)
            rho = (
                jnp.abs(iota_hat[..., None] - heaviside)
                * (quantile_loss / critic.huber_kappa)
                * next_iota_weights[..., None, :]
            )

            critic_loss = rho.sum(axis=-1).mean()  # Eq. 4

            return critic_loss, q

        (critic_loss, critic_q_values), grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(critic.params)
        critic = critic.apply_gradients(grads=grads)

        # Actor update

        # Sample quantiles in range [confidence_level, 1] (CVar intevall) for quantile regression
        _, iota, iota_weights = _sample_iota(
            actor_iota_key,
            observations.shape[0],
            critic.num_iota_samples,
            critic.confidence_level,
        )

        def actor_loss_fn(actor_params):
            dist = actor.apply_fn(actor_params, observations)
            pi, log_pi = dist.sample_and_log_prob(seed=actor_key)

            q_values = critic.apply_fn(
                critic.params,
                observations,
                pi,
                iota,
            )

            # Calculate CVaR (Eq. 6)
            if actor.use_mean:
                q_values = q_values.mean(0)
            else:
                q_values = q_values.min(0)
            q_value_cvar = jnp.sum(iota_weights * q_values, axis=-1)

            actor_loss = (
                alpha.apply_fn(alpha.params) * log_pi.sum(-1) - q_value_cvar
            ).mean()  # Eq. 5

            batch_entropy = -log_pi.sum(-1).mean()
            return actor_loss, (batch_entropy, q_value_cvar)

        (
            actor_loss,
            (batch_entropy, q_value_cvar),
        ), grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(actor.params)
        actor = actor.apply_gradients(grads=grads)

        # Entropy coefficient update
        if not alpha.fixed:

            def alpha_loss_fn(alpha_params):
                alpha_value = alpha.apply_fn(alpha_params)
                alpha_loss = (alpha_value * (batch_entropy - alpha.target)).mean()
                return alpha_loss, alpha_value

            (alpha_loss, alpha_value), grads = jax.value_and_grad(
                alpha_loss_fn, has_aux=True
            )(alpha.params)
            alpha = alpha.apply_gradients(grads=grads)
        else:
            alpha_loss = 0.0
            alpha_value = alpha.apply_fn(alpha.params)

        # Polyak averaging for target networks
        critic = critic.soft_update()

        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": alpha_value,
            "batch_entropy": batch_entropy,
            "entropy_reward": entropy_reward.mean(),
            "q1_mean": critic_q_values[0].mean(),
            "q1_std": critic_q_values[0].std(),
            "cvar": q_value_cvar,
        }

    return (
        actor,
        critic,
        alpha,
        metrics,
    )


@partial(
    jax.jit,
    static_argnums=(3,),
)
def _act(
    key: jax.random.PRNGKey,
    actor: TrainState,
    observation: jnp.ndarray,
    deterministic: bool = False,
) -> jnp.ndarray:
    dist = actor.apply_fn(actor.params, observation)
    actions = dist.mean() if deterministic else dist.sample(seed=key)
    return actions


class DREDQ(BaseAgent):
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
        super().__init__(cfg, env, eval_env, slave, seed, writer, logging_prefix)

        key = jax.random.PRNGKey(seed=self.seed)
        (
            self.key,
            actor_key,
            critic_key,
            alpha_key,
        ) = jax.random.split(key, 4)

        # Determine whether DSAC or DREDQ is used by looking at the ensemble size -> Use mean or min operation for combining critics in actor update accordingly
        if cfg.critic.ensemble_size > 2 or cfg.critic.ensemble_sample_size > 2:
            self.actor_use_mean = True
        else:
            self.actor_use_mean = False

        init_state = jnp.asarray(self.env.observation_space.sample()[None, ...])
        init_action = jnp.asarray(self.env.action_space.sample()[None, ...])

        self.num_iota_samples = cfg.critic.num_iota_samples
        self.embedding_dim = cfg.critic.embedding_dim
        init_iota = jnp.zeros((self.num_iota_samples,))

        # Actor init
        actor_module = Actor(
            fe_constructor_fn=DummyFeatureExtractor,
            action_dim=np.prod(self.env.action_space.shape),
        )
        self.actor = ActorTrainState.create(
            apply_fn=actor_module.apply,
            params=actor_module.init(actor_key, init_state),
            use_mean=self.actor_use_mean,
            tx=optax.adabelief(learning_rate=cfg.actor.policy_lr),
        )

        # Critic init
        critic_module = DistributionalQNetworkEnsemble(
            fe_constructor_fn=DummyFeatureExtractor,
            ensemble_size=cfg.critic.ensemble_size,
            num_quantiles=self.num_iota_samples,
            embedding_dim=self.embedding_dim,
        )
        self.critic = DistributionalCriticTrainState.create(
            apply_fn=critic_module.apply,
            params=critic_module.init(critic_key, init_state, init_action, init_iota),
            target_params=critic_module.init(
                critic_key, init_state, init_action, init_iota
            ),
            ensemble_sample_size=(cfg.critic.ensemble_sample_size,),
            gamma=cfg.gamma,
            tau=cfg.critic.tau,
            num_iota_samples=self.num_iota_samples,
            huber_kappa=cfg.critic.quantile_huber_loss_kappa,
            confidence_level=cfg.confidence_level,
            tx=optax.adabelief(learning_rate=cfg.critic.q_lr),
        )

        # Entropy tuning
        if cfg.entropy_coeff == "auto":
            alpha_module = SingleParamNetwork(init_value=1.0, param_name="log_alpha")
        else:
            alpha_module = SingleParamNetwork(
                init_value=cfg.entropy_coeff, param_name="log_alpha"
            )
        self.alpha = SingleParamTrainState.create(
            apply_fn=alpha_module.apply,
            params=alpha_module.init(alpha_key),
            target=-np.prod(self.env.action_space.shape),
            fixed=(cfg.entropy_coeff != "auto"),
            tx=optax.adam(learning_rate=cfg.critic.q_lr),
        )

        self.replay_buffer = BaseReplayBuffer(cfg.buffer_size, self.seed)

        self.specific_log_fn = log_dsac_metrics

    def add_to_buffer(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: float,
        real_next_obs: np.ndarray,
        dones: bool,
        infos: Dict[str, Any],
    ) -> None:
        self.replay_buffer.add(
            (
                obs,
                actions,
                np.array(rewards),
                real_next_obs,
                np.array(dones),
            )
        )

    def act(
        self,
        key: jax.random.PRNGKey,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        return _act(key, self.actor, observation, deterministic)

    def train_step(self, key: jax.random.PRNGKey) -> Dict[str, Any]:
        batches = [
            self.replay_buffer.sample(self.batch_size) for _ in range(self.utd_ratio)
        ]
        self.actor, self.critic, self.alpha, metrics = _train_step(
            key, self.actor, self.critic, self.alpha, batches
        )
        return metrics

    def evaluate_critic(
        self, key: jax.random.PRNGKey, observation: np.ndarray, action: np.ndarray
    ) -> float:
        return _evaluate_critic(key, self.critic, observation, action)

    def learn(
        self, eval_callback: Callable, training_finished_callback: Callable
    ) -> None:
        split_keys = jax.jit(lambda k: jax.random.split(k, 3))

        obs, _ = self.env.reset(seed=self.seed)

        violation_count = 0
        hard_reset_count = 0
        total_cumulative_cost = 0.0

        self.eval_env.reset(seed=0)
        trajectory_data = TrajectoryData(
            *self.eval_env.get_init_trajectory_data(self.max_episode_steps)
        )

        start_time = time.time()

        # DSAC/DREDQ Training loop
        for global_step in tqdm(
            range(self.total_timesteps),
            desc="\033[1mTraining Progress",
            colour="green",
        ):
            self.key, act_key, train_key = split_keys(self.key)

            if global_step < self.learning_starts:
                # Select random actions during warmup phase
                actions = self.env.action_space.sample()
            else:
                actions = self.act(act_key, obs)

            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            costs = infos["cost"]

            if "episode" in infos:
                # Episode has terminated or was truncated
                violation_count, total_cumulative_cost, trajectory_data = (
                    log_end_of_episode(
                        global_step,
                        self.prefix,
                        self.writer,
                        infos,
                        violation_count,
                        hard_reset_count,
                        trajectory_data,
                        self.env.get_trajectory(),
                        self.max_episode_steps,
                        total_cumulative_cost,
                    )
                )

            # Handle next observation after truncation
            real_next_obs = next_obs.copy()
            if truncations:
                real_next_obs = infos["final_observation"]
            self.add_to_buffer(
                obs, actions, rewards, real_next_obs, terminations, infos
            )

            # Set previous obs for next step and reset env if necessary
            if terminations or truncations:
                hard_reset_count += 1
                obs, _ = self.env.reset()
            else:
                obs = next_obs

            if global_step > self.learning_starts:
                # TRAINING
                metrics = self.train_step(train_key)

                if global_step % self.logging_frequency == 0:
                    metrics["elapsed_time"] = time.time() - start_time
                    log_metrics(
                        global_step,
                        self.prefix,
                        self.writer,
                        metrics,
                        [self.specific_log_fn],
                    )

                if (
                    global_step % self.eval_episode_frequency == 0
                    or (global_step + 1) == self.total_timesteps
                ):
                    # EVALUATION

                    # Plot the last N trajectories
                    if self.cfg.experiment_meta.track:
                        draw_trajectories(global_step, [trajectory_data])
                    # Log the cumulative cost at evaluation time
                    self.writer.add_scalar(
                        prefix_string(self.prefix) + "data/total_cumulative_cost",
                        total_cumulative_cost,
                        global_step,
                    )
                    self.writer.add_scalar(
                        prefix_string(self.prefix) + "charts/violations",
                        violation_count,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/forward_violations",
                        violation_count,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/reset_violations",
                        0,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/total_violations",
                        violation_count,
                        global_step,
                    )

                    self.writer.add_scalar(
                        "eval/hard_resets",
                        hard_reset_count,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/forward_episodes",
                        hard_reset_count,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/reset_episodes",
                        0,
                        global_step,
                    )

                    self.writer.add_scalar(
                        "eval/soft_reset_success_ratio",
                        0,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/latest_soft_reset_success_ratio",
                        0,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/forward_steps",
                        global_step,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/reset_steps",
                        0,
                        global_step,
                    )
                    self.writer.add_scalar(
                        "eval/rejected_steps",
                        0,
                        global_step,
                    )
                    # Start an evaluation episode
                    self.key, eval_key = jax.random.split(self.key, 2)
                    eval_callback(
                        self.eval_env,
                        eval_key,
                        self.act,
                        self.writer,
                        global_step,
                        self.cfg,
                    )

        training_finished_callback(
            self.eval_env,
            self.key,
            self.rng,
            self.act,
            self.writer,
            global_step,
            self.cfg,
        )

        self.env.close()
        self.eval_env.close()
        self.writer.close()


def log_dsac_metrics(
    global_step: int, prefix: str, writer: SummaryWriter, metrics: Dict[str, Any]
) -> None:
    # Logging: Losses
    writer.add_scalar(
        prefix + "network_logging/cvar",
        metrics["cvar"].mean().item(),
        global_step,
    )
