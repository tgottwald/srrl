import cairo
import numpy as np
from .Rendering import stroke_fill


class LidarRenderer:
    def __init__(self):
        pass

    def reset(self):
        pass

    def transformed(self, lidar_measurements, transform):
        lidar_measurements_hom = np.concatenate(
            [lidar_measurements, np.ones_like(lidar_measurements[:, :1])], axis=-1
        )
        lidar_measurements_global_frame = np.squeeze(
            transform @ lidar_measurements_hom[..., None], -1
        )

        return lidar_measurements_global_frame[:, :2]

    def drawMeasurement(self, ctx: cairo.Context, x, y):
        ctx.arc(x, y, 0.125, 0, 2 * np.pi)
        stroke_fill(ctx, (0.6, 0.0, 0.0), (0.6, 0.0, 0.0))

    def render(self, ctx: cairo.Context, lidar_measurements, ego_transform):
        ctx.set_fill_rule(cairo.FILL_RULE_WINDING)

        lidar_measurements = self.transformed(
            lidar_measurements, np.linalg.inv(ego_transform)
        )

        for measurement_idx in range(lidar_measurements.shape[0]):
            self.drawMeasurement(
                ctx,
                lidar_measurements[measurement_idx, 0],
                lidar_measurements[measurement_idx, 1],
            )
