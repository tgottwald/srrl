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
from .CriticNetwork import ExampleBasedCriticTrainState, SoftQNetworkEnsemble
from .FeatureExtractors import DummyFeatureExtractor
from .logging_util import *
from .ReplayBuffer import BaseReplayBuffer, ExtendedReplayBuffer
from .RND import RND, RNDTrainDict, RNDTrainState
from .SingleParamNetwork import SingleParamNetwork, SingleParamTrainState


@jax.jit
def _get_intrinsic_reward(
    rnd: TrainState, state: jnp.ndarray, action: jnp.ndarray
) -> float:
    def rnd_loss_fn(rnd_params):
        prediction, target = rnd.apply_fn(rnd_params, state, action)
        # Do not reduce loss over state dimensions to get loss for each individual state dimension
        loss = (prediction - target) ** 2
        return loss

    intrinsic_reward = jax.lax.stop_gradient(rnd_loss_fn(rnd.params))

    return (intrinsic_reward / jnp.sqrt(rnd.rnd_state.state["var"])).mean(-1)


@jax.jit
def _update_rnd_reward_normalization(
    rnd: TrainState, loss_per_dim: jnp.ndarray
) -> TrainState:
    # Update running standard deviation
    # RMS implementation from  https://github.com/openai/random-network-distillation/blob/master/mpi_util.py
    batch_mean = loss_per_dim.mean(axis=0)
    batch_var = loss_per_dim.var(axis=0)
    batch_count = loss_per_dim.shape[0]

    delta = batch_mean - rnd.rnd_state.state["mean"]
    tot_count = rnd.rnd_state.state["count"] + batch_count

    new_mean = rnd.rnd_state.state["mean"] + delta * batch_count / tot_count
    m_a = rnd.rnd_state.state["var"] * (rnd.rnd_state.state["count"])
    m_b = batch_var * (batch_count)
    M2 = (
        m_a
        + m_b
        + jnp.square(delta)
        * rnd.rnd_state.state["count"]
        * batch_count
        / (rnd.rnd_state.state["count"] + batch_count)
    )
    new_var = M2 / (rnd.rnd_state.state["count"] + batch_count)

    new_count = batch_count + rnd.rnd_state.state["count"]

    new_rnd_state = rnd.rnd_state.replace(
        state={"mean": new_mean, "var": new_var, "count": new_count}
    )
    rnd = rnd.replace(rnd_state=new_rnd_state)

    return rnd


@jax.jit
def _update_rnd(
    rnd: TrainState, state_batch: jnp.ndarray, action_batch: jnp.ndarray
) -> Tuple[TrainState, Dict[str, Any]]:
    def rnd_loss_fn(rnd_params):
        prediction, target = rnd.apply_fn(rnd_params, state_batch, action_batch)
        # Do not reduce loss over state dimensions to get loss for each individual state dimension
        loss = (prediction - target) ** 2
        return loss.mean(), loss

    (loss, loss_per_dim), grads = jax.value_and_grad(rnd_loss_fn, has_aux=True)(
        rnd.params
    )
    new_rnd = rnd.apply_gradients(grads=grads)

    new_rnd = _update_rnd_reward_normalization(new_rnd, loss_per_dim)

    # Logging
    info = {"rnd_loss": loss}
    return new_rnd, info


@jax.jit
def _evaluate_critic(
    critic: TrainState,
    observation: jnp.ndarray,
    action: jnp.ndarray,
) -> float:
    q = critic.apply_fn(critic.target_params, observation, action).mean(0)

    return (q + 1) / (critic.success_target_scaling + 1)


@jax.jit
def _train_step(
    key: jax.random.PRNGKey,
    actor: TrainState,
    critic: TrainState,
    alpha: TrainState,
    rnd: TrainState,
    batches: List[Tuple[np.array, np.array, np.array, np.array, np.array, np.array]],
    example_batches: List[Tuple[np.array, np.array, np.array, np.array, np.array]],
) -> Tuple[TrainState, TrainState, TrainState, TrainState, Dict[str, Any]]:
    for batch, example_batch in zip(batches, example_batches):
        key, policy_key, example_policy_key, ensemble_sample_key, actor_key = (
            jax.random.split(key, 5)
        )
        observations, actions, rewards, intrinsic_rewards, next_observations, dones = (
            batch
        )
        example_observations, _, _, _, _ = example_batch

        dist = actor.apply_fn(actor.params, next_observations)
        next_pi, next_log_pi = dist.sample_and_log_prob(seed=policy_key)

        next_q_candidates = critic.apply_fn(
            critic.target_params, next_observations, next_pi
        )
        # Used for REDQ implementation: Is equal to vanilla SAC for ensemble_size = ensemble_sample_size = 2
        sampled_critics_indices = jax.random.choice(
            ensemble_sample_key,
            next_q_candidates.shape[0],
            critic.ensemble_sample_size,
            replace=False,
        )
        next_q = next_q_candidates[sampled_critics_indices].min(0)

        entropy_reward = -alpha.apply_fn(alpha.params) * next_log_pi.sum(-1)

        if rnd.enabled:
            exploration_reward = critic.beta * _get_intrinsic_reward(
                rnd, observations, actions
            )
        else:
            exploration_reward = jnp.array([0.0])

        next_q = next_q + entropy_reward

        # Only access extrinsic reward to get information about violations
        violation_indicator = jnp.where(rewards < 0.0, -1.0, 0.0)

        # Note: done may only be set to indicate a termination not a truncation
        target_q = (
            exploration_reward
            + violation_indicator
            + (1 - dones) * critic.gamma * next_q
        )  # No extrinsic rewards used!

        # Choose action according to current policy for example_observation
        example_dist = actor.apply_fn(actor.params, example_observations)
        example_pi, _ = example_dist.sample_and_log_prob(seed=example_policy_key)

        example_target_q = critic.success_target_scaling * jnp.ones(example_pi.shape[0])

        # Combine batch and success examples into single batch for critic update
        combined_observations = jnp.vstack([observations, example_observations])
        combined_actions = jnp.vstack([actions, example_pi])
        combined_target_q = jnp.hstack([target_q, example_target_q])

        def critic_loss_fn(critic_params):
            # [N, batch_size] - [1, batch_size]
            combined_q = critic.apply_fn(
                critic_params, combined_observations, combined_actions
            )
            critic_loss = (
                optax.huber_loss(combined_q, combined_target_q[None, ...])
                .mean(1)
                .mean(0)
            )
            return critic_loss, combined_q

        # Calc grad with respect to the critics and the feature extractors params
        (critic_loss, q_values), grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(critic.params)
        critic = critic.apply_gradients(grads=grads)

        # Actor update

        def actor_loss_fn(actor_params):
            dist = actor.apply_fn(actor_params, observations)
            pi, log_pi = dist.sample_and_log_prob(seed=actor_key)

            if actor.use_mean:
                q_pi = critic.apply_fn(critic.params, observations, pi).mean(0)
            else:
                q_pi = critic.apply_fn(critic.params, observations, pi).min(0)
            actor_loss = (alpha.apply_fn(alpha.params) * log_pi.sum(-1) - q_pi).mean()

            batch_entropy = -log_pi.sum(-1).mean()
            return actor_loss, batch_entropy

        (actor_loss, batch_entropy), grads = jax.value_and_grad(
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
            "q1_mean": q_values[0, : dones.shape[0]].mean(),
            "q1_std": q_values[0, : dones.shape[0]].std(),
            "q1_example_mean": q_values[0, dones.shape[0] :].mean(),
            "q1_example_std": q_values[0, dones.shape[0] :].std(),
            "exploration_reward": exploration_reward.mean(),
            "exploration_reward_min": exploration_reward.min(),
            "exploration_reward_max": exploration_reward.max(),
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


class ExampleBasedSAC(BaseAgent):
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
            rnd_key,
        ) = jax.random.split(key, 5)

        # Determine whether SAC or REDQ is used by looking at the ensemble size -> Use mean or min operation for combining critics in actor update accordingly
        if cfg.critic.ensemble_size > 2 or cfg.critic.ensemble_sample_size > 2:
            self.actor_use_mean = True
        else:
            self.actor_use_mean = False

        init_state = jnp.asarray(self.env.observation_space.sample()[None, ...])
        init_action = jnp.asarray(self.env.action_space.sample()[None, ...])

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

        # Critic and TargetCritic init
        # CriticTrainState holds both parameter sets for the regular and the target critic ensemble
        critic_module = SoftQNetworkEnsemble(
            fe_constructor_fn=DummyFeatureExtractor,
            ensemble_size=cfg.critic.ensemble_size,
        )
        self.critic = ExampleBasedCriticTrainState.create(
            apply_fn=critic_module.apply,
            params=critic_module.init(critic_key, init_state, init_action),
            target_params=critic_module.init(critic_key, init_state, init_action),
            ensemble_sample_size=(cfg.critic.ensemble_sample_size,),
            gamma=cfg.gamma,
            tau=cfg.critic.tau,
            beta=cfg.rnd.beta,
            success_target_scaling=cfg.critic.success_target_scaling,
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

        # Random Network Distillation
        self.rnd_enabled = cfg.rnd.beta > 0.0
        # if self.rnd_enabled:
        rnd_module = RND(embedding_dim=cfg.rnd.embedding_dim, init_features=init_state)
        self.rnd = RNDTrainState.create(
            apply_fn=rnd_module.apply,
            params=rnd_module.init(rnd_key, init_state, init_action),
            tx=optax.adam(learning_rate=cfg.rnd.predictor_lr),
            rnd_state=RNDTrainDict.init(),
            enabled=self.rnd_enabled,
        )
        self.rnd_state_batch = []
        self.rnd_action_batch = []
        self.rnd_batch_size = cfg.rnd.batch_size

        self.replay_buffer = ExtendedReplayBuffer(cfg.buffer_size, self.seed)
        # Additional buffer for storing examples
        self.example_buffer = BaseReplayBuffer(cfg.example_count, self.seed)

        self.specific_log_fn = log_example_based_metrics

        # Collect the success examples used in the critic update
        if logging_prefix is not None:
            self.collect_success_examples(mode=logging_prefix[:-1])
        else:
            self.collect_success_examples(mode="forward")

    def collect_success_examples(self, mode: str = "forward") -> None:
        for _ in range(self.cfg.example_count):
            example_obs = self.env.get_success_examples(mode)
            self.example_buffer.add(
                (
                    example_obs,
                    np.zeros(self.env.action_space.shape, dtype=np.float32),
                    np.array([1.0]),
                    np.zeros_like(example_obs),
                    np.array([False], dtype=bool),
                )
            )

    def add_to_buffer(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: float,
        real_next_obs: np.ndarray,
        dones: bool,
        infos: Dict[str, Any],
    ) -> None:
        if self.rnd_enabled:
            # TODO: Previous implementation accidentally updated after every step -> Check impact
            intrinsic_rewards = _get_intrinsic_reward(self.rnd, obs, actions)
            # Add observation to RND batch and update RND accordingly
            self.rnd_state_batch.append(obs)
            self.rnd_action_batch.append(actions)

            # Only update the RNDs predictor network if batch is full
            if len(self.rnd_state_batch) >= self.rnd_batch_size:
                # Update the RND network with the last batch
                self.rnd, info = _update_rnd(
                    self.rnd,
                    np.array(self.rnd_state_batch),
                    np.array(self.rnd_action_batch),
                )
                self.rnd_state_batch = []
                self.rnd_action_batch = []
        else:
            intrinsic_rewards = np.zeros_like(rewards)
        self.replay_buffer.add(
            (
                obs,
                actions,
                np.array(rewards),
                intrinsic_rewards,
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
        example_batches = [
            self.example_buffer.sample(self.batch_size) for _ in range(self.utd_ratio)
        ]
        self.actor, self.critic, self.alpha, metrics = _train_step(
            key, self.actor, self.critic, self.alpha, self.rnd, batches, example_batches
        )
        return metrics

    def evaluate_critic(
        self, key: jax.random.PRNGKey, observation: np.ndarray, action: np.ndarray
    ) -> float:
        return _evaluate_critic(self.critic, observation, action)

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
                obs, actions, rewards, real_next_obs, terminations or truncations, infos
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


def log_example_based_metrics(
    global_step: int, prefix: str, writer: SummaryWriter, metrics: Dict[str, Any]
) -> None:
    writer.add_scalar(
        prefix_string(prefix) + "network_logging/q1_example_mean",
        metrics["q1_example_mean"].item(),
        global_step,
    )
    writer.add_scalar(
        prefix_string(prefix) + "network_logging/q1_example_std",
        metrics["q1_example_std"].item(),
        global_step,
    )
    writer.add_scalar(
        prefix_string(prefix) + "rewards/exploration_reward",
        metrics["exploration_reward"].item(),
        global_step,
    )
    writer.add_scalar(
        prefix_string(prefix) + "rewards/exploration_reward_min",
        metrics["exploration_reward_min"].item(),
        global_step,
    )
    writer.add_scalar(
        prefix_string(prefix) + "rewards/exploration_reward_max",
        metrics["exploration_reward_max"].item(),
        global_step,
    )
