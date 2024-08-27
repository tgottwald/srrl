from cairo import Context
import numpy as np
from .Rendering import stroke_fill


class AgentModeIndicator:
    def __init__(self):
        self.pos_x = 30
        self.pos_y = 30
        self.radius = 20.0

    def draw(self, ctx: Context, show):
        if not show:
            return
        ctx.arc(self.pos_x, self.pos_y, self.radius, 0.0, 2 * np.pi)
        stroke_fill(ctx, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
