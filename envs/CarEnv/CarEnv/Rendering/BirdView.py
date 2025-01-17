import cairo
import numpy as np
from .Rendering import BitmapRenderer, draw_vehicle_proxy, draw_vehicle_state
from .ConeRenderer import ConeRenderer
from .TrackRenderer import TrackRenderer
from .LidarRenderer import LidarRenderer
from .WallRenderer import WallRenderer


class BirdViewRenderer:
    def __init__(
        self,
        width,
        height,
        scale=15.0,
        orient_forward=False,
        draw_physics=False,
        draw_sensors=False,
        hud=True,
        forward_center=0.8,
        draw_grid=True,
    ):
        self._bvr = BitmapRenderer(width, height)
        self._bvr.open()
        self.width = width
        self.height = height
        self.scale = scale
        self.orient_forward = orient_forward
        self.forward_center = forward_center
        self.last_transform = None
        self.slip_lines = []
        self.draw_physics = draw_physics
        self.draw_sensors = draw_sensors
        self.draw_hud = hud
        self.draw_grid = draw_grid
        self.start_lights = None
        self.ghosts = None
        self.show_track = True
        self.show_digital_tachometer = False
        self._cone_renderer = ConeRenderer()
        self._track_renderer = TrackRenderer()
        self._lidar_renderer = LidarRenderer()
        self._wall_renderer = WallRenderer()
        self._bg_grid_path = None
        self.draw_agent_mode_indicator = False

    def _transform_for_view(self, ctx: cairo.Context, x, y, theta):
        if self.orient_forward:
            ctx.translate(self.width / 2, self.height * self.forward_center)
            ctx.rotate(-theta - np.pi / 2)
        else:
            ctx.translate(self.width / 2, self.height / 2)
        ctx.scale(self.scale, self.scale)
        ctx.translate(-x, -y)

    def _render_ctx(self, env, ctx):
        from .Gauge import Gauge
        from .Colors import BACKGROUND_GRID
        from .Rendering import stroke_fill
        from .AgentModeIndicator import AgentModeIndicator

        pose = env.ego_pose

        # Tried to do this with patterns but unbearably slow
        if self._bg_grid_path is None:
            s = 1000
            for t in np.linspace(-s, s, 41):
                ctx.move_to(-s, t)
                ctx.line_to(s, t)
                ctx.move_to(t, -s)
                ctx.line_to(t, s)
            self._bg_grid_path = ctx.copy_path_flat()
            ctx.new_path()

        # Since the background grid doesn't repeat endlessly, snap to nearest origin
        if self.draw_grid:
            gx, gy, gtheta = pose
            self._transform_for_view(ctx, gx % 50, gy % 50, gtheta)
            ctx.append_path(self._bg_grid_path)
            stroke_fill(ctx, BACKGROUND_GRID, None)

        ctx.identity_matrix()
        self._transform_for_view(ctx, *pose)

        if hasattr(env.problem, "track_dict") and self.show_track:
            env.problem.track_dict["target_zone"] = (
                None
                if "target_zone" not in env.problem.track_dict
                else env.problem.track_dict["target_zone"]
            )
            env.problem.track_dict["safety_zone"] = (
                None
                if "safety_zone" not in env.problem.track_dict
                else env.problem.track_dict["safety_zone"]
            )

            self._track_renderer.render(
                ctx,
                env.problem.track_dict["centerline"],
                env.problem.track_dict["poly"],
                env.problem.track_dict["target_zone"],
                env.problem.track_dict["safety_zone"],
            )

        self.render_tire_slip(ctx, env)

        for obj in env.objects.values():
            if hasattr(obj, "draw"):
                obj.draw(ctx)

        if "lidar_points" in env.objects:
            lidar_points = env.objects["lidar_points"]
            self._lidar_renderer.render(ctx, lidar_points, env.ego_transform)

        self.render_ghosts(ctx, env)
        draw_vehicle_proxy(ctx, env)

        if self.draw_hud:
            env.problem.render(ctx, env)

            if self.draw_sensors:
                for s in env.sensors.values():
                    s.draw(ctx, env)

            if self.draw_physics:
                ctx.identity_matrix()
                ctx.translate(100, self.height - 100)
                draw_vehicle_state(ctx, env)

            ctx.identity_matrix()
            ctx.translate(self.width - 80, self.height - 80)
            Gauge(0, 100).draw(ctx, abs(env.vehicle_last_speed) * 3.6)
            self.render_pedals(ctx, env)

            AgentModeIndicator().draw(ctx, self.draw_agent_mode_indicator)

            if self.show_digital_tachometer:
                self.render_digital_tachometer(ctx, env)

            env.action.render(ctx, self.width, self.height)

            self.render_start_lights(ctx)

    def render(self, env):
        ctx = self._bvr.clear()
        self._render_ctx(env, ctx)
        return self._bvr.get_data()

    def render_pdf(self, env, path):
        import cairo

        surface = cairo.PDFSurface(path, self.width, self.height)

        try:
            ctx = cairo.Context(surface)
            ctx.set_source_rgb(0.8, 0.8, 0.8)
            ctx.fill()
            self._render_ctx(env, ctx)
        finally:
            surface.finish()

    def reset(self):
        self.last_transform = None
        self.ghosts = None
        self._track_renderer.reset()
        self.slip_lines = []
        self._cone_renderer.reset()

    def _add_slip_line(self, t1, t2, x, y):
        vec = np.array([x, y, 1])

        p1 = (t1 @ vec)[:2]
        p2 = (t2 @ vec)[:2]
        self.slip_lines.append((p1, p2))
        self.slip_lines = self.slip_lines[-500:]  # Limit to not overwhelm

    def render_tire_slip(self, ctx: cairo.Context, env):
        from .Rendering import stroke_fill

        if not hasattr(env.vehicle_model, "front_slip_"):
            # Does not appear to be correct vehicle model
            return

        transform = np.linalg.inv(env.ego_transform)

        if self.last_transform is None:
            self.last_transform = transform
            return

        # Return if no physics set yet
        if env.vehicle_model.front_slip_ is None:
            return

        front_slip = env.vehicle_model.front_slip_
        rear_slip = env.vehicle_model.rear_slip_

        h_wb = env.vehicle_model.wheelbase / 2
        h_w = env.collision_bb[-1]

        if front_slip:
            self._add_slip_line(transform, self.last_transform, h_wb, -h_w)
            self._add_slip_line(transform, self.last_transform, h_wb, h_w)
        if rear_slip:
            self._add_slip_line(transform, self.last_transform, -h_wb, -h_w)
            self._add_slip_line(transform, self.last_transform, -h_wb, h_w)

        self.last_transform = transform

        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for p1, p2 in self.slip_lines:
            ctx.move_to(*p1)
            ctx.line_to(*p2)
        stroke_fill(ctx, (0.0, 0.0, 0.0), None, 3.0)

    def render_pedals(self, ctx: cairo.Context, env):
        from .Rendering import stroke_fill

        if not hasattr(env.action, "throttle_position_"):
            return

        ctx.identity_matrix()
        ctx.translate(self.width - 200, self.height - 120)

        ctx.rectangle(40, 0, 20, 80)
        stroke_fill(ctx, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        bar_size = 80 * env.action.throttle_position_
        ctx.rectangle(40, 80 - bar_size, 20, bar_size)
        stroke_fill(ctx, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))

        ctx.rectangle(0, 0, 20, 80)
        stroke_fill(ctx, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        bar_size = 80 * env.action.brake_position_
        ctx.rectangle(0, 80 - bar_size, 20, bar_size)
        stroke_fill(ctx, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    def render_digital_tachometer(self, ctx: cairo.Context, env):
        from .Rendering import stroke_fill

        v = f"{int(env.vehicle_last_speed * 3.6)}"
        ctx.select_font_face(
            "Latin Modern Mono", cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD
        )
        advance = ctx.text_extents("0").x_advance
        height = ctx.text_extents("0").height
        ctx.set_font_size(100)
        ctx.identity_matrix()

        ctx.move_to(self.width / 2 - 250, self.height)
        ctx.line_to(self.width / 2 - 130, self.height - 120)
        ctx.line_to(self.width / 2 + 130, self.height - 120)
        ctx.line_to(self.width / 2 + 250, self.height)
        ctx.close_path()
        stroke_fill(ctx, (0.0, 0.0, 0.0), (0.3, 0.3, 0.3))

        for i, k in enumerate(v.rjust(3, " ")):
            ctx.move_to(
                self.width / 2 - advance * 3 / 2 + i * advance + 30,
                self.height - 60 + height / 2,
            )
            ctx.text_path(k)
            stroke_fill(ctx, (0.0, 0.0, 0.0), (1.0, 0.3, 0.3))

        if (
            hasattr(env.vehicle_model, "front_slip_")
            and env.vehicle_model.front_slip_ is not None
            and env.vehicle_model.front_slip_[0]
        ):
            ctx.translate(0, self.height - 60 + height / 2)
            ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
            ctx.arc(self.width / 2 - 100, -30, 20, 0, 2 * np.pi)
            ctx.new_sub_path()
            ctx.arc(self.width / 2 - 100, -30, 10, 0, 2 * np.pi)
            ctx.new_sub_path()
            ctx.rectangle(self.width / 2 - 120, -10, 40, 10)
            stroke_fill(ctx, (0.0, 0.0, 0.0), (1.0, 0.3, 0.3))

    def render_ghosts(self, ctx: cairo.Context, env):
        if self.ghosts is None:
            return

        for color, pose in self.ghosts:
            draw_vehicle_proxy(ctx, env, pose=pose, query_env=False, color=color)

    def render_start_lights(self, ctx: cairo.Context):
        from .Rendering import stroke_fill

        r = 20

        if self.start_lights is None:
            return

        def draw(on):
            ctx.rectangle(-1.5 * r, -4 * r, 3 * r, 8 * r)
            stroke_fill(ctx, (0.0, 0.0, 0.0), (0.2, 0.2, 0.2))

            ctx.arc(0.0, 0.0, r, 0.0, 2 * np.pi)
            ctx.new_sub_path()
            ctx.arc(0.0, -2.5 * r, r, 0.0, 2 * np.pi)
            ctx.new_sub_path()
            ctx.arc(0.0, 2.5 * r, r, 0.0, 2 * np.pi)
            stroke_fill(
                ctx, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0) if on else (0.4, 0.2, 0.2)
            )

        ctx.identity_matrix()
        ctx.translate(self.width * 0.5 - 3.5 * r, self.height * 0.2)
        draw(self.start_lights >= 1)
        ctx.identity_matrix()
        ctx.translate(self.width * 0.5 + 0.0 * r, self.height * 0.2)
        draw(self.start_lights >= 2)
        ctx.identity_matrix()
        ctx.translate(self.width * 0.5 + 3.5 * r, self.height * 0.2)
        draw(self.start_lights >= 3)

    def close(self):
        self._bvr.close()
