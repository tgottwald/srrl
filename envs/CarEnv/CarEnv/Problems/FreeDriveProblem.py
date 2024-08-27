import gymnasium as gym
import numpy as np

from .Problem import Problem
from typing import Tuple, Optional
from shapely.geometry import Point, LinearRing
from ..BatchedCones import BatchedCones
from ..Track.Generator import make_full_environment
import sys


class DirLookup:
    """
    A very fast approximation of finding the forward direction along a LinearRing for a given interpolation.
    Avoids using shapely after initialization.
    """
    def __init__(self, lr: LinearRing):
        # Construct segment lengths and offsets from LinearRing
        coords = np.asarray(lr.coords)
        offsets = np.roll(coords, -1, axis=0) - coords
        lengths = np.linalg.norm(offsets, axis=-1)

        # May have zero length segments if following points equal, filter out
        mask = lengths > 0.
        offsets = offsets[mask]
        lengths = lengths[mask]

        # Calculate normalized directions
        self._cum_lengths = np.cumsum(lengths)
        self._dirs = offsets / lengths[:, None]

    def __call__(self, interp):
        # Could interpolate here, but just return closest
        idx = np.searchsorted(self._cum_lengths, interp)
        return self._dirs[idx]


class FreeDriveProblem(Problem):
    def __init__(self, track_width=6., cone_width=5., k_center=0., k_base=.05, k_forwards=0., extend=100.,
                 lap_limit=None, time_limit=None):
        if lap_limit is None and time_limit is None:
            raise ValueError("At least one of lap_limit and time_limit must be set")

        self.track_width = track_width
        self.cone_width = cone_width
        self.extend = extend
        self.k_center = k_center
        self.k_forwards = k_forwards
        self.k_base = k_base
        self.time_limit = time_limit
        self.lap_limit = lap_limit
        self.track_dict = None
        self.lr: Optional[LinearRing] = None
        self.old_pose_xy = None
        self.old_projection = None
        self.idle_time = 0.
        self.track_progress = 0.
        self._dir_lookup = None

    @property
    def state_observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (2,))

    def observe_state(self, env):
        return np.array([
            # TODO: Normalize and not actually required
            env.vehicle_last_speed,
            env.steering_history[-1],
        ], dtype=np.float32)

    def configure_env(self, env, rng=None) -> Tuple[float, float, float]:
        from ..Track.Util import shapely_safe_buffer

        self.track_dict = make_full_environment(width=self.track_width, extends=(self.extend, self.extend),
                                                cone_width=self.cone_width, rng=rng)
        self.lr = LinearRing(self.track_dict['centerline'])
        # Add outline to track
        self.track_dict['poly'] = shapely_safe_buffer(self.lr, self.track_dict['width'] / 2 + .3)
        self.idle_time = 0.
        self.track_progress = 0.
        self._dir_lookup = DirLookup(self.lr)

        env.objects = {
            'cones': BatchedCones.from_track_dict(self.track_dict),
        }

        x, y = self.track_dict['start_xy']
        theta = self.track_dict['start_theta']
        self.old_pose_xy = np.array([x, y])
        self.old_projection = self.lr.project(Point(self.old_pose_xy))
        return x, y, theta

    def calculate_forward_vel_coeff(self, env, projection=None):
        pose_xy = env.ego_pose[:2]

        if projection is None:
            projection = self.lr.project(Point(pose_xy))

        forward_dir = self._dir_lookup(projection)
        t_forward = np.dot(env.vehicle_model.v_, forward_dir).item()
        return t_forward

    def update(self, env, dt) -> Tuple[bool, bool]:
        env.add_to_reward(self.k_base)

        pose_xy = env.ego_pose[:2]
        new_projection = self.lr.project(Point(pose_xy))

        # Forward is negative along LinearRing
        forward_v = -self.calculate_forward_vel_coeff(env, projection=new_projection)
        env.metrics['forward_velocity'] = forward_v
        env.add_to_reward(forward_v * dt * self.k_forwards)

        # Check if moving less than 0.1 m/s
        moved_distance = np.linalg.norm(pose_xy - self.old_pose_xy, axis=-1)
        if moved_distance / dt < .1:
            self.idle_time += dt
        else:
            self.idle_time = .0

        # Generators puts us backward on track, take module when wrapping
        moved = (self.old_projection - new_projection) % self.lr.length

        if moved > self.lr.length / 2:
            # Should be negative, moving backwards
            moved = moved - self.lr.length

        self.track_progress += moved

        distance_from_center = self.lr.distance(Point(pose_xy))
        env.add_to_reward(-distance_from_center * dt * self.k_center)

        terminated = False
        truncated = False

        if distance_from_center > self.track_width / 2:
            env.set_reward(-1)
            env.add_info('Done.Reason', 'LeftTrack')
            terminated = True
        elif self.lap_limit is not None and self.track_progress > self.lr.length * self.lap_limit:
            print("Probably closed track, ending episode", file=sys.stderr)
            env.add_info('Done.Reason', 'CompletedTrack')
            truncated = True
        elif self.time_limit is not None and env.time >= self.time_limit:
            print("Time limit exceeded", file=sys.stderr)
            env.add_info('Done.Reason', 'MaxTime')
            truncated = True
        elif self.idle_time > 5.:
            print("Truncated due to idling", file=sys.stderr)
            env.add_info('Done.Reason', 'Idling')
            truncated = True

        self.old_pose_xy = pose_xy
        self.old_projection = new_projection

        return terminated, truncated
