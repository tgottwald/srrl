from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata
from torch.utils.tensorboard import SummaryWriter

CM = 1 / 2.54  # centimeters in inches
TEXT_WIDTH = 15.74776 * CM
HALF_TEXT_WIDTH = TEXT_WIDTH / 2
FONTSIZE = 12


def prefix_string(s):
    if s is None:
        return ""
    return str(s)


def log_metrics(
    global_step: int,
    logging_prefix: Any,
    writer: SummaryWriter,
    metrics: Dict[str, Any],
    log_additional_metrics_fn: List[
        Callable[[int, str, SummaryWriter, Dict[str, Any]], None]
    ] = None,
) -> None:
    # Deduce logging prefix if it is set
    prefix = prefix_string(logging_prefix)

    writer.add_scalar(
        "charts/SPS",
        int(global_step / metrics["elapsed_time"]),
        global_step,
    )

    # Logging: Losses
    writer.add_scalar(
        prefix + "losses/critic_loss",
        metrics["critic_loss"].item(),
        global_step,
    )
    writer.add_scalar(
        prefix + "losses/actor_loss",
        metrics["actor_loss"].item(),
        global_step,
    )
    writer.add_scalar(
        prefix + "losses/alpha_loss",
        metrics["alpha_loss"].item(),
        global_step,
    )

    # Logging Network logging
    writer.add_scalar(
        prefix + "network_logging/qf1_value_mean",
        metrics["q1_mean"].item(),
        global_step,
    )
    writer.add_scalar(
        prefix + "network_logging/qf1_value_std",
        metrics["q1_std"].item(),
        global_step,
    )

    # Logging: Rewards
    writer.add_scalar(
        prefix + "rewards/alpha",
        metrics["alpha"].item(),
        global_step,
    )
    writer.add_scalar(
        prefix + "rewards/entropy",
        metrics["entropy_reward"].item(),
        global_step,
    )
    writer.add_scalar(
        prefix + "rewards/pure_entropy",
        metrics["batch_entropy"].item(),
        global_step,
    )

    # Log all metrics not covered above
    if log_additional_metrics_fn is not None:
        for fn in log_additional_metrics_fn:
            if fn is not None:
                fn(global_step, prefix, writer, metrics)


@dataclass
class TrajectoryData:
    """Dataclass to store trajectory data for visualization"""

    trajectories: np.ndarray
    normalization_pos: np.ndarray
    target_pos: np.ndarray
    collision_pos: np.ndarray
    obstacles: np.ndarray
    cumulative_reward: np.ndarray


def update_trajectory_data(
    trajectory_data: TrajectoryData,
    curr_trajectory_infos: Tuple[np.ndarray, np.ndarray, bool],
    max_episode_steps: int,
    env_infos: Dict[str, Any],
) -> TrajectoryData:

    # Store the latest full trajectory for visualization
    trajectories = np.roll(trajectory_data.trajectories, shift=1, axis=0)
    normalization_pos = np.roll(trajectory_data.normalization_pos, shift=1, axis=0)
    target_pos = np.roll(trajectory_data.target_pos, shift=1, axis=0)
    collision_pos = np.roll(trajectory_data.collision_pos, shift=1, axis=0)
    cumulative_reward = np.roll(trajectory_data.cumulative_reward, shift=1, axis=0)

    trajectory_normalization_pos, trajectory_target_pos, trajectory, has_collided = (
        curr_trajectory_infos
    )
    trajectory = trajectory[..., :2]
    # Ensure all trajectories have the same length
    if trajectory.shape[0] < (max_episode_steps + 1):
        trajectory = np.concatenate(
            (
                trajectory,
                np.full(
                    (max_episode_steps + 1 - trajectory.shape[0], 2),
                    trajectory[-1],
                ),
            ),
            axis=0,
        )
    normalization_pos[0] = trajectory_normalization_pos
    target_pos[0] = trajectory_target_pos
    trajectories[0] = trajectory
    if has_collided:
        collision_pos[0] = trajectory[-1]
    else:
        collision_pos[0] = np.zeros(2)
    cumulative_reward[0] = env_infos["episode"]["r"]

    trajectory_data.trajectories = trajectories
    trajectory_data.normalization_pos = normalization_pos
    trajectory_data.target_pos = target_pos
    trajectory_data.collision_pos = collision_pos
    trajectory_data.cumulative_reward = cumulative_reward

    return trajectory_data


def log_end_of_episode(
    global_step: int,
    logging_prefix: Any,
    writer: SummaryWriter,
    env_infos: Dict[str, Any],
    violation_count: int,
    hard_reset_count: int,
    trajectory_data: TrajectoryData,
    curr_trajectory_infos: Tuple[np.ndarray, np.ndarray, np.ndarray, bool],
    max_episode_steps: int,
    total_cumulative_cost: float = None,
) -> Tuple[int, float, TrajectoryData]:
    writer.add_scalar(
        prefix_string(logging_prefix) + "charts/episodic_return",
        env_infos["episode"]["r"],
        global_step,
    )
    writer.add_scalar(
        prefix_string(logging_prefix) + "charts/episodic_length",
        env_infos["episode"]["l"],
        global_step,
    )
    if "real_termination" in env_infos:
        violation_count += 1

        writer.add_scalar(
            prefix_string(logging_prefix) + "data/collision_x",
            curr_trajectory_infos[2][-1, 0] - curr_trajectory_infos[0][0],
            global_step,
        )
        writer.add_scalar(
            prefix_string(logging_prefix) + "data/collision_y",
            curr_trajectory_infos[2][-1, 1] - curr_trajectory_infos[0][1],
            global_step,
        )

    writer.add_scalar(
        prefix_string(logging_prefix) + "charts/violations",
        violation_count,
        global_step,
    )

    writer.add_scalar(
        prefix_string(logging_prefix) + "charts/hard_resets",
        hard_reset_count,
        global_step,
    )

    if "c" in env_infos["episode"]:
        writer.add_scalar(
            prefix_string(logging_prefix) + "charts/episodic_cost",
            env_infos["episode"]["c"],
            global_step,
        )
        total_cumulative_cost += env_infos["episode"]["c"]
        writer.add_scalar(
            prefix_string(logging_prefix) + "data/total_cumulative_cost",
            total_cumulative_cost,
            global_step,
        )

    trajectory_data = update_trajectory_data(
        trajectory_data, curr_trajectory_infos, max_episode_steps, env_infos
    )

    return violation_count, total_cumulative_cost, trajectory_data


def draw_trajectories(
    global_step: int,
    trajectory_data: List[TrajectoryData],
    trajectory_color: List[str] = ["blue"],
    start_marker_color: List[str] = ["green"],
    collision_marker_color: List[str] = ["red"],
) -> None:
    assert len(trajectory_data) == len(trajectory_color)
    assert len(trajectory_data) == len(start_marker_color)
    assert len(trajectory_data) == len(collision_marker_color)

    dpi = 100
    fig, ax = plt.subplots(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)
    ax.axis("off")
    ax.set_aspect(9 / 16)
    ax.set_xlim(-21, 30)
    ax.set_ylim(-2, 7)

    # Plot trajectories + start and collision marker
    for data, traj_c, start_c, coll_c in zip(
        trajectory_data, trajectory_color, start_marker_color, collision_marker_color
    ):

        trajectories, normalization_pos, target_pos, collision_pos, obstacles = (
            data.trajectories,
            data.normalization_pos,
            data.target_pos,
            data.collision_pos,
            data.obstacles,
        )

        normalized_trajectories = trajectories - normalization_pos[:, None, :]
        normalized_collision_pos = (collision_pos - normalization_pos)[
            np.logical_or(collision_pos[..., 0] != 0, collision_pos[..., 1] != 0)
        ]

        # Flip all y coords to match the video
        normalized_trajectories[..., 1] *= -1
        normalized_collision_pos[..., 1] *= -1

        # Plot all trajectories
        for i in range(normalized_trajectories.shape[0]):
            ax.plot(
                normalized_trajectories[i, :, 0],
                normalized_trajectories[i, :, 1],
                c=traj_c,
                alpha=10 / normalized_trajectories.shape[0],
            )

        # Plot start, target and collision positions
        ax.scatter(
            target_pos[:, 0] - normalization_pos[:, 0],
            -1 * (target_pos[:, 1] - normalization_pos[:, 1]),
            c="gold",
            edgecolors="orange",
            s=512,
            marker="*",
        )
        ax.scatter(
            normalized_trajectories[:, 0, 0],
            normalized_trajectories[:, 0, 1],
            facecolors="none",
            edgecolors=start_c,
            marker="o",
        )
        ax.scatter(
            normalized_collision_pos[:, 0],
            normalized_collision_pos[:, 1],
            c=coll_c,
            marker="x",
        )

    # Plot obstacles
    for i in range(obstacles.shape[0]):
        ax.plot(
            [obstacles[i, 0], obstacles[i, 2]],
            [-obstacles[i, 1], -obstacles[i, 3]],
            color="black",
            linewidth=2,
        )

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())

    wandb.log({"eval/trajectories": wandb.Image(img, mode="RGBA")}, step=global_step)


def draw_heatmap(
    global_step: int,
    name: str,
    coordinates: np.ndarray,
    z_data: float,
    obstacles: np.ndarray,
    z_limits: Tuple[float, float],
    patches: Union[Polygon, Rectangle] = None,
) -> None:
    x = coordinates[:, 0]
    y = -coordinates[:, 1]
    xi, yi = np.linspace(x.min(), x.max(), 2000), np.linspace(y.min(), y.max(), 2000)

    zi = griddata(
        (x, y),
        z_data,
        (xi[None, :], yi[:, None]),
        method="linear",
        # fill_value=z_limits[0],
    )
    # Replace out of range interpolated values with None
    zi[zi < z_limits[0]] = z_limits[0]
    zi[zi > z_limits[1]] = z_limits[1]

    dpi = 100
    fig, ax = plt.subplots(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)
    ax.axis("off")
    ax.set_aspect(9 / 16)
    ax.set_xlim(-21, 30)
    ax.set_ylim(-3, 7)

    heatmap = ax.contourf(
        xi,
        yi,
        zi,
        cmap="viridis",
        levels=np.arange(z_limits[0], z_limits[1] + 1),
        vmin=z_limits[0],
        vmax=z_limits[1],
    )

    axins = inset_axes(
        ax,
        width="75%",
        height="5%",
        loc="lower center",
    )
    axins.xaxis.set_ticks_position("bottom")
    cbar = fig.colorbar(
        cax=axins,
        mappable=heatmap,
        orientation="horizontal",
        boundaries=np.linspace(z_limits[0], z_limits[1], 100),
        ticks=np.arange(z_limits[0], z_limits[1] + 1),
    )

    cbar.set_label(r"$\text{Episodic Return}$", fontsize=FONTSIZE)

    # Plot obstacles
    for i in range(obstacles.shape[0]):
        ax.plot(
            [obstacles[i, 0], obstacles[i, 2]],
            [-obstacles[i, 1], -obstacles[i, 3]],
            color="black",
            linewidth=2,
        )

    if patches is not None:
        ax.add_patch(patches)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())

    # Free memory
    xi, yi, zi, heatmap = None, None, None, None

    wandb.log({name: wandb.Image(img, mode="RGBA")}, step=global_step)


def draw_start_position_episode_reward_heatmap(
    global_step: int,
    trajectory_data: List[TrajectoryData],
    limits: Tuple[float, float],
    name: str = "eval/start_position_episode_reward",
    starting_poses_center: np.ndarray = None,
    starting_poses_tolerances: np.ndarray = None,
) -> None:
    for data in trajectory_data:
        coordinates = data.trajectories[:, 0, :] - data.target_pos
        z_data = data.cumulative_reward

        # Set mean cumulative reward for coordinates if coords appear multiple times
        unique_xy, _ = np.unique(coordinates, axis=0, return_index=True)

        deduplicated_data = np.zeros((unique_xy.shape[0], 3))

        for i, (x, y) in enumerate(unique_xy):
            duplicate_indices = np.where(
                np.logical_and(coordinates[:, 0] == x, coordinates[:, 1] == y)
            )[0]
            mean_z = np.mean(z_data[duplicate_indices])
            deduplicated_data[i] = [x, y, mean_z]

        coordinates = deduplicated_data[:, :2]
        z_data = deduplicated_data[:, 2]

        starting_pose_patch = None
        if starting_poses_center is not None and starting_poses_tolerances is not None:
            start_x = (
                starting_poses_center[..., 0]
                - starting_poses_tolerances[..., 0]
                - data.target_pos[0, 0]
            )
            start_y = -1.0 * (
                starting_poses_center[..., 1]
                + starting_poses_tolerances[..., 1]
                - data.target_pos[0, 1]
            )
            len_x = 2.0 * starting_poses_tolerances[..., 0]
            len_y = 2.0 * starting_poses_tolerances[..., 1]
            starting_pose_patch = Rectangle(
                (start_x, start_y),
                len_x,
                len_y,
                edgecolor="r",
                facecolor="none",
                linewidth=1,
            )

        draw_heatmap(
            global_step,
            name,
            coordinates,
            z_data,
            data.obstacles,
            limits,
            starting_pose_patch,
        )
