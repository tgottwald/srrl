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
from .ReplayBuffer import ExtendedReplayBuffer
from .DistributionalCriticNetwork import WCSACCriticTrainState, WCSACIQNNetwork
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
        key, observation.shape[0], critic.num_iota_samples, critic.risk_level
    )
    q, z = critic.apply_fn(critic.target_params, observation, action, iota_hat)
    q = q.mean(0)

    # Max cumulative reward = max_episode_steps * dt * max_reward_single_step
    # 15 = 300 * 0.05 * 1
    return (q + 1) / (15 + 1)


@partial(
    jax.jit,
    static_argnums=(1, 2),
)
def _sample_iota(
    key: jax.random.PRNGKey,
    batch_size: int,
    num_quantiles: int,
    risk_level: float = 1.0,
) -> jnp.ndarray:
    presum_iota = jax.random.uniform(key, (batch_size, num_quantiles)) + 0.1
    presum_iota /= presum_iota.sum(axis=-1, keepdims=True)

    iota = jnp.cumsum(presum_iota, axis=-1)  # (batch_size, num_quantiles)

    # Calculate quantile midpoints (iota_hat) by avergaging adjacent quantiles
    # See Dabney et al. 2017 Lemma 2
    iota_hat = jnp.zeros_like(iota)
    iota_hat = iota_hat.at[:, 0].set(iota[:, 0] / 2)
    iota_hat = iota_hat.at[:, 1:].set((iota[:, :-1] + iota[:, 1:]) / 2)

    # Move quantiles to relevant range for risk level
    # E.g. risk_level=0.5 -> All iota(_hat) values are in [0.5, 1]
    iota = 1 - risk_level + risk_level * iota
    iota_hat = 1 - risk_level + risk_level * iota_hat
    # presum_iota can be used as a weight for the quantiles as it indicates the size of the interval
    return iota, iota_hat, presum_iota


@jax.jit
def _train_step(
    key: jax.random.PRNGKey,
    actor: TrainState,
    critic: TrainState,
    alpha: TrainState,
    omega: TrainState,
    batches: List[Tuple[np.array, np.array, np.array, np.array, np.array, np.array]],
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
        observations, actions, rewards, costs, next_observations, dones = batch

        dist = actor.apply_fn(actor.params, next_observations)
        next_pi, next_log_pi = dist.sample_and_log_prob(seed=policy_key)

        # Sample random iotas for quantile regression at next time step
        # τ = iota_hat; τ' = next_iota_hat in Yang et al. 2022
        _, iota_hat, _ = _sample_iota(
            critic_iota_key1, next_observations.shape[0], critic.num_iota_samples
        )
        _, next_iota_hat, next_iota_weights = _sample_iota(
            critic_iota_key2, next_observations.shape[0], critic.num_iota_samples
        )

        next_q_candidates, next_z_candidates = critic.apply_fn(
            critic.target_params, next_observations, next_pi, next_iota_hat
        )
        # Used for REDQ implementation: Is equal to vanilla SAC for ensemble_size = ensemble_sample_size = 2
        sampled_critics_indices = jax.random.choice(
            ensemble_sample_key,
            next_q_candidates.shape[0],
            critic.ensemble_sample_size,
            replace=False,
        )
        next_q = next_q_candidates[sampled_critics_indices].min(0)
        next_z = next_z_candidates

        entropy_reward = -alpha.apply_fn(alpha.params) * next_log_pi.sum(-1)
        next_q = next_q + entropy_reward

        target_q = rewards + (1 - dones) * critic.gamma * next_q
        target_z = costs[..., None] + (1 - dones[..., None]) * critic.gamma * next_z

        def critic_loss_fn(critic_params):
            q, z = critic.apply_fn(critic_params, observations, actions, iota_hat)
            # Reward critic loss
            critic_q_loss = optax.huber_loss(q, target_q[None, ...]).mean(1).mean(0)

            # Safety critic loss
            quantile_values = z[..., None]
            target_quantile_values = target_z[..., None, :]
            # Calculate the element-wise huber loss for every quantile-target_quantile combination for a sample
            quantile_loss = optax.huber_loss(
                quantile_values, target_quantile_values, critic.huber_kappa
            )  # (batch_size, num_quantiles(quantile_values), num_quantiles(target_quantile_values))
            heaviside = jnp.heaviside(quantile_values - target_quantile_values, 0.5)
            rho = (
                jnp.abs(iota_hat[..., None] - heaviside)
                * (quantile_loss / critic.huber_kappa)
                * next_iota_weights[..., None, :]
            )  # Quantile Huber Loss, Yang et al. 2022, Eq. 7

            critic_z_loss = rho.sum(axis=-1).mean()  # Dabney et al. 2017 Eq. 3

            critic_loss = critic_q_loss + critic_z_loss
            return critic_loss, (critic_q_loss, critic_z_loss, q, z)

        (
            critic_loss,
            (critic_q_loss, critic_z_loss, critic_q_values, critic_z_values),
        ), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic.params)
        critic = critic.apply_gradients(grads=grads)

        # Actor update

        # Sample quantiles in range [risk_level, 1] (CVar intevall) for quantile regression
        _, iota, iota_weights = _sample_iota(
            actor_iota_key,
            observations.shape[0],
            critic.num_iota_samples,
            critic.risk_level,
        )

        def actor_loss_fn(actor_params):
            dist = actor.apply_fn(actor_params, observations)
            pi, log_pi = dist.sample_and_log_prob(seed=actor_key)

            q_values, z = critic.apply_fn(critic.params, observations, pi, iota)
            if actor.use_mean:
                q_values = q_values.mean(0)
            else:
                q_values = q_values.min(0)

            # Use actions stored in buffer for cvar calculation
            _, buffer_z = critic.apply_fn(critic.params, observations, actions, iota)
            # buffer_z = jnp.clip(buffer_z, a_min=1e-8, a_max=1e8)

            # Calculate CVar (weighted mean of quantile values)
            # See Yang et al. 2022, Eq. 23
            cvar_policy_actions = jnp.sum(iota_weights * z, axis=-1)
            cvar_buffer_actions = jnp.sum(iota_weights * buffer_z, axis=-1)

            omega_value = jnp.clip(omega.apply_fn(omega.params), -1e2, 1e2)
            damp = actor.damp_scale * (omega.target - cvar_buffer_actions).mean()

            actor_loss = (
                alpha.apply_fn(alpha.params) * log_pi.sum(-1)
                - q_values
                + (omega_value - damp) * cvar_policy_actions
            ).mean()

            safety_reward = ((omega_value - damp) * cvar_policy_actions).mean()

            batch_entropy = -log_pi.sum(-1).mean()
            return actor_loss, (
                batch_entropy,
                cvar_buffer_actions,
                cvar_policy_actions,
                safety_reward,
                damp,
            )

        (
            actor_loss,
            (
                batch_entropy,
                cvar_buffer_actions,
                cvar_policy_actions,
                safety_reward,
                omega_dampening,
            ),
        ), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor.params)
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

        # Safety weight update
        def omega_loss_fn(omega_params):
            omega_value = omega.apply_fn(omega_params)
            # target_cost > cvar: Safe -> decrease omega; target_cost < cvar: Unsafe -> increase omega -> increase importance of safety
            loss = (omega_value * (omega.target - cvar_buffer_actions)).mean()
            return loss, omega_value

        (omega_loss, omega_value), grads = jax.value_and_grad(
            omega_loss_fn, has_aux=True
        )(omega.params)
        omega = omega.apply_gradients(grads=grads)

        # Polyak averaging for target networks
        critic = critic.soft_update()

        metrics = {
            "critic_loss": critic_loss,
            "reward_critic_loss": critic_q_loss,
            "safety_critic_loss": critic_z_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "omega_loss": omega_loss,
            "alpha": alpha_value,
            "omega": omega_value,
            "omega_dampening": omega_dampening,
            "batch_entropy": batch_entropy,
            "entropy_reward": entropy_reward.mean(),
            "safety_reward": safety_reward,
            "q1_mean": critic_q_values[0].mean(),
            "q1_std": critic_q_values[0].std(),
            "z1_mean": critic_z_values.mean(),
            "z1_std": critic_z_values.std(),
            "cvar": cvar_buffer_actions,
            "cvar_diff": cvar_policy_actions - cvar_buffer_actions,
        }

    return (
        actor,
        critic,
        alpha,
        omega,
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


class WCSAC_IQN(BaseAgent):
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
            omega_key,
        ) = jax.random.split(key, 5)

        # Determine whether SAC or REDQ is used by looking at the ensemble size -> Use mean or min operation for combining critics in actor update accordingly
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
            damp_scale=cfg.damp_scale,
            tx=optax.adabelief(learning_rate=cfg.actor.policy_lr),
        )

        # Critic init
        wcsac_critics_module = WCSACIQNNetwork(
            fe_constructor_fn=DummyFeatureExtractor,
            num_reward_critics=cfg.critic.ensemble_size,
            num_quantiles=self.num_iota_samples,
            embedding_dim=self.embedding_dim,
        )
        self.wcsac_critics = WCSACCriticTrainState.create(
            apply_fn=wcsac_critics_module.apply,
            params=wcsac_critics_module.init(
                critic_key, init_state, init_action, init_iota
            ),
            target_params=wcsac_critics_module.init(
                critic_key, init_state, init_action, init_iota
            ),
            ensemble_sample_size=(cfg.critic.ensemble_sample_size,),
            gamma=cfg.gamma,
            tau=cfg.critic.tau,
            num_iota_samples=self.num_iota_samples,
            huber_kappa=cfg.critic.quantile_huber_loss_kappa,
            risk_level=cfg.risk_level,
            tx=optax.adabelief(learning_rate=cfg.critic.q_lr),
        )

        # Entropy init
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

        # Automatic safety weight tuning
        target_cost = (
            cfg.cost_limit
            * (1 - cfg.gamma**cfg.max_episode_steps)
            / (cfg.max_episode_steps * (1 - cfg.gamma))
        )
        print("Target cost:", target_cost)
        omega_module = SingleParamNetwork(init_value=0.5, param_name="log_omega")
        self.omega = SingleParamTrainState.create(
            apply_fn=omega_module.apply,
            params=omega_module.init(omega_key),
            target=target_cost,
            fixed=False,
            tx=optax.adam(learning_rate=cfg.omega.lr),
        )

        self.replay_buffer = ExtendedReplayBuffer(cfg.buffer_size, self.seed)

        self.specific_log_fn = log_wcsac_metrics

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
                infos["cost"],
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
        self.actor, self.wcsac_critics, self.alpha, self.omega, metrics = _train_step(
            key, self.actor, self.wcsac_critics, self.alpha, self.omega, batches
        )
        return metrics

    def evaluate_critic(
        self, key: jax.random.PRNGKey, observation: np.ndarray, action: np.ndarray
    ) -> float:
        return _evaluate_critic(key, self.wcsac_critics, observation, action)

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

        # Training loop
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
                        [log_wcsac_metrics],
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
                        prefix_string(self.prefix) + "eval/violations",
                        violation_count,
                        global_step,
                    )
                    self.writer.add_scalar(
                        prefix_string(self.prefix) + "charts/hard_resets",
                        hard_reset_count,
                        global_step,
                    )
                    self.writer.add_scalar(
                        prefix_string(self.prefix) + "eval/hard_resets",
                        hard_reset_count,
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


def log_wcsac_metrics(
    global_step: int, prefix: str, writer: SummaryWriter, metrics: Dict[str, Any]
) -> None:
    # Logging: Losses
    writer.add_scalar(
        prefix + "losses/reward_critic_loss",
        metrics["reward_critic_loss"].item(),
        global_step,
    )
    writer.add_scalar(
        prefix + "losses/safety_critic_loss",
        metrics["safety_critic_loss"].item(),
        global_step,
    )
    writer.add_scalar(
        prefix + "losses/omega_loss",
        metrics["omega_loss"].item(),
        global_step,
    )

    # Logging Network logging
    writer.add_scalar(
        prefix + "network_logging/z1_value_mean",
        metrics["z1_mean"].item(),
        global_step,
    )
    writer.add_scalar(
        prefix + "network_logging/z1_value_std",
        metrics["z1_std"].item(),
        global_step,
    )
    writer.add_scalar(
        prefix + "network_logging/cvar",
        metrics["cvar"].mean().item(),
        global_step,
    )
    writer.add_scalar(
        prefix + "network_logging/cvar_diff",
        metrics["cvar_diff"].mean().item(),
        global_step,
    )
    # Logging: Rewards
    writer.add_scalar(
        prefix + "rewards/omega",
        metrics["omega"].item(),
        global_step,
    )
    writer.add_scalar(
        prefix + "rewards/omega_dampening",
        metrics["omega_dampening"].item(),
        global_step,
    )
    writer.add_scalar(
        prefix + "rewards/safety_reward",
        metrics["safety_reward"].item(),
        global_step,
    )
