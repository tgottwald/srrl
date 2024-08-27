import cairo
import numpy as np
from .Rendering import stroke_fill
from ..BatchedWalls import BatchedWalls


def draw_wall(ctx: cairo.Context, wall):
    ctx.move_to(wall[0], wall[1])

    ctx.line_to(wall[2], wall[3])
    stroke_fill(ctx, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 3.0)

    ctx.close_path()


class WallRenderer:
    def __init__(self):
        self._n = None
        self._t = None
        self._caches = {}

    def reset(self):
        self._n = None
        self._caches = {}

    def render(self, ctx: cairo.Context, wall_pos, thickness):
        ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
        for i in range(wall_pos.shape[0]):
            draw_wall(ctx, wall_pos[i])
            # TODO: Use cache for better performance?
        # if thickness != self._t:
        #     self._caches = {}
        #     self._t = thickness
        #     for which in (1, 2):
        #         mask = cone_types == which
        #         for cp in wall_pos[mask]:
        #             x, y = cp
        #             stencil_cone(ctx, x, y, self._r)
        #         self._caches[which] = ctx.copy_path()
        #         stroke_fill(ctx, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        # else:
        #     for which, path in self._caches.items():
        #         ctx.append_path(path)
        #         stroke_fill(ctx, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
