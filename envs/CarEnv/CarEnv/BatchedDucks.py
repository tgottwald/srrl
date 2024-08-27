import cairo
import numpy as np

from .BatchedObjects import BatchedObjects
from .Collision import intersections_aabb_circles
from shapely.geometry import LinearRing, Polygon, Point, LineString, MultiLineString
from .Track.Util import shapely_safe_buffer


def rotate_angles(transform, angles):
    x, y, _ = transform @ np.array([1., 0., 0.])
    rotation = np.arctan2(y, x)

    return angles + rotation


class BatchedDucks(BatchedObjects):
    def __init__(self, data, track_lr, track_poly, t=0.):
        # Data is n x 4 (x, y, theta, v, dist, max_dist)
        self._data = data
        # Only used for drawing animation
        self._t = t
        # Used for spawning
        self._track_lr = track_lr
        self._track_poly = track_poly

    def __with_new_data(self, data):
        return BatchedDucks(data, track_lr=self._track_lr, track_poly=self._track_poly, t=self._t)

    @staticmethod
    def put_ducks(lr, poly, n, rng: np.random.Generator):
        if not isinstance(poly, Polygon):
            raise TypeError(type(poly))

        choices = [poly.exterior] + list(poly.interiors)

        data = np.zeros((n, 6))

        for i in range(n):
            which = choices[rng.integers(0, len(choices))]
            data[i, :2] = np.asarray(which.interpolate(rng.uniform(0., which.length)).coords)[0]
            center = np.asarray(lr.interpolate(lr.project(Point(data[i, :2]))).coords)[0]
            data[i, 2] = np.arctan2(*(center - data[i, :2])[::-1]) + rng.uniform(-.2, .2)

            # Determining the length the duck has to walk to cross the road. Only compute this geometry once
            # for performance reasons.
            offset = np.array([np.cos(data[i, 2]), np.sin(data[i, 2])])
            path = poly.intersection(LineString(np.stack([data[i, :2], data[i, :2] + offset * 1000])))

            if isinstance(path, LineString):
                data[i, 5] = path.length
            elif isinstance(path, MultiLineString):
                data[i, 5] = path.geoms[0].length
            else:
                raise TypeError(type(path))

        data[:, 3] = rng.uniform(1.5, 2.5, size=(len(data)))

        return data

    def get_bad_ducks(self):
        return self._data[:, 4] >= self._data[:, 5]

    @staticmethod
    def from_track_dict(track_dict, rng: np.random.Generator, n_ducks=20):
        lr = LinearRing(track_dict['centerline'])
        poly = shapely_safe_buffer(lr, track_dict['width'] / 2. + 3.)
        return BatchedDucks(BatchedDucks.put_ducks(lr, poly, n_ducks, rng), track_lr=lr, track_poly=poly)

    @property
    def data(self):
        return self._data

    def draw(self, ctx: cairo.Context):
        from .Rendering.Rendering import draw_duck

        for x, y, theta, *_ in self._data:
            draw_duck(ctx, x, y, theta, t=self._t * 10)

    def transformed(self, transform) -> 'BatchedDucks':
        pos_hom = np.concatenate([self._data[:, :2], np.ones_like(self._data[:, :1])], axis=-1)
        new_pos = np.squeeze(transform @ pos_hom[..., None], -1)
        new_angles = rotate_angles(transform, self._data[:, 2])
        new_data = np.concatenate([new_pos[:, :2], new_angles[:, None], self._data[:, 3:]], axis=-1)

        return self.__with_new_data(new_data)

    def filtered_aabb(self, *args) -> 'BatchedDucks':
        mask = BatchedObjects.filter_mask_aabb(self._data[:, :2], *args)
        return self.__with_new_data(self._data[mask])

    def update(self, env):
        dt = env.dt
        self._t += dt

        x, y, theta, v, dist, max_dist = self._data.T

        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        dist += v * dt

        self._data = np.stack([x, y, theta, v, dist, max_dist], axis=-1)

        bad_mask = self.get_bad_ducks()
        n_bad = np.sum(bad_mask)

        if n_bad > 0:
            self._data[bad_mask] = BatchedDucks.put_ducks(self._track_lr, self._track_poly, n_bad, env.np_random)

        # Check Intersections
        us_in_local = self.transformed(env.ego_transform)
        intersected = intersections_aabb_circles(env.collision_bb, .5, us_in_local.data[:, :2])

        if np.any(intersected):
            env.set_reward(-1)
            env.add_info('Done.Reason', 'HitDuck')
            env.set_terminated(True)
