import gymnasium as gym
import numpy as np

from .Sensor import Sensor
from .BatchedWalls import BatchedWalls
import numba


def toCartesian(p: np.ndarray) -> np.ndarray:
    """Converts polar coordinate input the cartesian coordinates

    Args:
        p (np.ndarray): Nx2 polar coordinate array with first component being the distance and the second one being the angle in radians

    Returns:
        np.ndarray: Input converted to cartesian coordinates (x,y)
    """
    x = p[..., 0] * np.cos(p[..., 1])
    y = p[..., 0] * np.sin(p[..., 1])
    return np.stack((x, y), axis=-1)


def toPolar(c: np.ndarray) -> np.ndarray:
    """Converts cartesian coordinate input the polar coordinates

    Args:
        c (np.ndarray): Nx2 cartesian coordinate array (x,y)

    Returns:
        np.ndarray: Input converted to polar coordinates (r,theta)
    """
    r = np.sqrt(c[..., 0] ** 2 + c[..., 1] ** 2)
    theta = np.arctan2(c[..., 1], c[..., 0])
    return np.stack((r, theta), axis=-1)


def calculate_intersection(l0: np.ndarray, l1: np.ndarray) -> np.ndarray:
    """Calculates the intersection points from a single line (l0) to multiple lines (l1)

    Args:
        l0 (np.ndarray): (line start x, line start y, line end x, line end y)
        l1 (np.ndarray): Nx4 Array with each row specifying a new line using the same format as l0

    Returns:
        np.ndarray: Nx2 Array containing the intersection points (x,y) between l0 and l1. If no intersections is found (0,0) will be returned for the line.
    """
    assert len(l0.shape) == 1
    assert l0.shape[0] == 4
    assert len(l1.shape) == 2
    assert l1.shape[1] == 4
    l0_start = l0[:2]
    l0_end = l0[2:]
    l1_start = l1[:, :2]
    l1_end = l1[:, 2:]

    # See https://en.wikipedia.org/wiki/l1%E2%80%93l1_intersection#Given_two_points_on_each_line_segment

    T_numerator = (l0_start[0] - l1_start[:, 0]) * (l1_start[:, 1] - l1_end[:, 1]) - (
        l0_start[1] - l1_start[:, 1]
    ) * (l1_start[:, 0] - l1_end[:, 0])
    T_denominator = (
        (l0_start[0] - l0_end[0]) * (l1_start[:, 1] - l1_end[:, 1])
        - (l0_start[1] - l0_end[1]) * (l1_start[:, 0] - l1_end[:, 0])
        + 0.00001
    )

    U_numerator = (l0_start[0] - l0_end[0]) * (l0_start[1] - l1_start[:, 1]) - (
        l0_start[1] - l0_end[1]
    ) * (l0_start[0] - l1_start[:, 0])
    U_denominator = (
        (l0_start[0] - l0_end[0]) * (l1_start[:, 1] - l1_end[:, 1])
        - (l0_start[1] - l0_end[1]) * (l1_start[:, 0] - l1_end[:, 0])
        + 0.00001
    )

    t = np.divide(T_numerator, T_denominator)
    u = np.divide(U_numerator, U_denominator)

    intersection_points = l0_start + np.multiply(t[:, None], l0_end - l0_start)

    # There exists an intersection if 0 <= t,u <= 1
    # In the following 0 < t,u <= 1 will be used to account for inaccuries created by the + 0.00001 (this prevents division by 0)
    # Therefore any intersection at (0,0) will be disregarded (also it should not be physically possible...)
    invalid_intersections = np.logical_or(
        np.logical_or(t <= 0, t > 1), np.logical_or(u <= 0, u > 1)
    )
    intersection_points[invalid_intersections] = 0
    return intersection_points


class SensorLidar(Sensor):
    def __init__(
        self,
        env,
        angle_min,
        angle_max,
        angle_increment,
        range_max,
        measurement_noise_stdev,
        normalize=True,
    ):
        super(SensorLidar, self).__init__(env)
        angle_count = np.ceil((angle_max - angle_min) / angle_increment) + 1
        self._angles = np.linspace(angle_min, angle_max, angle_count.astype(int))
        self._range_max = range_max
        self._normalize = normalize
        self._measurement_noise_stdev = measurement_noise_stdev
        self.first = True

        # Calculate the lidars range values for the vehicles outer hull
        x1, x2, y1, y2 = env.collision_bb
        vehicle_boundary_lines = np.array(
            [[x1, y1, x1, y2], [x1, y2, x2, y2], [x2, y2, x2, y1], [x2, y1, x1, y1]]
        )
        vehicle_boundary = self.get_intersections(self._angles, vehicle_boundary_lines)
        polar_vehicle_boundary = toPolar(vehicle_boundary)
        # Set value for points with no intersection arbitarily high
        polar_vehicle_boundary[polar_vehicle_boundary[..., 0] == 0, 0] = (
            2 * self._range_max
        )
        # Find echo from closest intersected wall
        measurements_idx = np.argmin(polar_vehicle_boundary[..., 0], axis=1)
        polar_vehicle_boundary = np.squeeze(
            polar_vehicle_boundary[
                np.arange(measurements_idx.shape[0]), measurements_idx, :
            ]
        )
        # Debug: Draw the vehicle boundary
        # cartesian_vehicle_boundary = toCartesian(polar_vehicle_boundary)
        # self.env.objects["lidar_points"] = cartesian_vehicle_boundary

        # Leave only the distance, discard angle
        polar_vehicle_boundary = polar_vehicle_boundary[:, 0]
        if self._normalize:
            # Normalize all measurements
            polar_vehicle_boundary /= self._range_max

        self.env.polar_vehicle_boundary = polar_vehicle_boundary

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(-1.0, 1.0, shape=(self._angles.shape[0],))

    def get_intersections(self, angles: np.ndarray, walls: np.ndarray) -> np.ndarray:
        """Calculates the intersections points for each laser beam with each wall

        Args:
            angles (np.ndarray): Nx1 Array holding all angles [radians] emitting a ray
            walls (np.ndarray): Nx4 Array holding all walls used in the current environment

        Returns:
            np.ndarray: Nx2 Array containing the intersection points (x,y) between rays and walls. If no intersections is found (0,0) will be returned for the ray-wall tuple.
        """
        # Calculate the start and end point of each laser beam based on its angle
        # Use beam length of 1.5 * self._range_max to account for measurements outside of range specified by manufacturer
        beams = np.hstack(
            (
                np.zeros((angles.shape[0], 2)),
                toCartesian(
                    np.vstack((np.ones_like(angles) * 1.5 * self._range_max, angles)).T
                ),
            )
        )
        # Calculate the intersections
        intersections = np.apply_along_axis(
            calculate_intersection, -1, arr=walls, l1=beams
        )
        intersections = np.swapaxes(intersections, 0, 1)

        return intersections

    def apply_noise(self, measurements: np.ndarray, sigma: float) -> np.ndarray:
        noise = self.env._np_random.normal(0.0, sigma, measurements.shape[0])
        noise[measurements[:, 0] == -1] = 0.0
        measurements[:, 0] += noise
        return measurements

    def observe(self, env):
        if "walls" not in env.objects:
            return None

        walls = env.objects["walls"].transformed(env.ego_transform)

        intersections = self.get_intersections(self._angles, walls.data)
        polar_intersections = toPolar(intersections)

        # Set value for points with no intersection arbitarily high
        polar_intersections[polar_intersections[..., 0] == 0, 0] = 2 * self._range_max
        # Find echo from closest intersected wall
        measurements_idx = np.argmin(polar_intersections[..., 0], axis=1)
        measurements = np.squeeze(
            polar_intersections[
                np.arange(measurements_idx.shape[0]), measurements_idx, :
            ]
        )

        # Mark echoes with no intersection or a distance outside the valid range as invalid
        measurements[measurements[..., 0] > self._range_max, 0] = -1
        noisy_measurements = self.apply_noise(
            measurements, self._measurement_noise_stdev
        )
        # Ensure valid value range
        noisy_measurements[:, 0] = np.clip(
            noisy_measurements[:, 0], -1.0, self._range_max
        )

        # Make all valid measurements available for rendering in cartesian space
        noisy_measurements_cartesian = toCartesian(noisy_measurements)
        self.env.objects["lidar_points"] = noisy_measurements_cartesian[
            noisy_measurements[..., 0] != -1
        ]

        # Remove theta component from measurements and leave only the distance
        noisy_measurements = noisy_measurements[:, 0]

        if self._normalize:
            # Normalize all valid measurements
            noisy_measurements[noisy_measurements != -1] /= self._range_max

        return noisy_measurements.astype(np.float32)
