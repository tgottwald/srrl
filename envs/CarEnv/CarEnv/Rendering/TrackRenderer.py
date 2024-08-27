import cairo
from .Rendering import stroke_fill
from shapely.geometry import Polygon


def _trace(ctx: cairo.Context, path):
    ctx.move_to(*path[0])

    for el in path[1:]:
        ctx.line_to(*el)

    ctx.close_path()


def draw_zone(ctx: cairo.Context, zone_marking, color=(1.0, 0.6, 0.0)):
    ctx.move_to(zone_marking[0], zone_marking[1])

    ctx.line_to(zone_marking[2], zone_marking[3])
    stroke_fill(ctx, color, None, 3)

    ctx.close_path()


class TrackRenderer:
    def __init__(self):
        self._track_path = None
        self._center_path = None

    def reset(self):
        self._track_path = None
        self._center_path = None

    def render(
        self,
        ctx: cairo.Context,
        centerline,
        poly: Polygon,
        target_zone=None,
        safety_zone=None,
    ):
        if self._track_path is None:
            _trace(ctx, poly.exterior.coords)
            for interior in poly.interiors:
                _trace(ctx, interior.coords)

            self._track_path = ctx.copy_path()
        else:
            ctx.append_path(self._track_path)

        stroke_fill(ctx, (0.0, 0.0, 0.0), (0.6, 0.6, 0.6))

        if target_zone is not None:
            for zone_marking in target_zone:
                draw_zone(ctx, zone_marking, color=(1.0, 1.0, 1.0))

        if safety_zone is not None:
            for zone_marking in safety_zone:
                draw_zone(ctx, zone_marking)

        # Do not draw centerline
        # if self._center_path is None:
        #     _trace(ctx, centerline)
        #     self._center_path = ctx.copy_path()
        # else:
        #     ctx.append_path(self._center_path)
        # stroke_fill(ctx, (1.0, 0.6, 0.0), None, line_width=2)
