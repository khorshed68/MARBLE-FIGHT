"""Orbit camera with smooth motion, safe limits and automatic arena framing.

``fit_arena`` computes, from the real lens projection, the camera distance and a
vertical image shift (lens film offset) that place the whole circular arena
inside a given screen rectangle — the free area between the HUD panels. It is
re-run whenever the window's aspect ratio changes, so the full playground is
visible on any window size (16:9, 16:10, ultra-wide, 4:3, maximised windows...).
"""

from __future__ import annotations

import math

from panda3d.core import NodePath, Point2, Point3


class CameraRig:
    PITCH_MIN, PITCH_MAX = 28.0, 82.0
    DIST_MIN, DIST_MAX = 13.0, 36.0

    def __init__(self, base, heading: float = 0.0, pitch: float = 54.0, dist: float = 30.0) -> None:
        self.base = base
        self.h = self.th = heading
        self.p = self.tp = pitch
        self.d = self.td = dist
        self.target = Point3(0, -1.0, 0)
        self.dragging = False
        self.last_mouse = None
        self.enabled = True
        self.auto_orbit = 0.0      # deg/s (menu background)
        self.dist_max = self.DIST_MAX
        self.home_dist = dist
        self.home_pitch = pitch
        self.home_heading = heading
        self._fit_args = None
        self.aspect_listeners: list = []      # called (before re-framing) when the window shape changes
        base.camLens.setFov(48)
        base.camLens.setNearFar(0.5, 400)
        base.accept("aspectRatioChanged", self._refit)
        self.apply()

    def set_view(self, heading=None, pitch=None, dist=None, snap=False) -> None:
        if heading is not None:
            self.th = heading
        if pitch is not None:
            self.tp = max(self.PITCH_MIN, min(self.PITCH_MAX, pitch))
        if dist is not None:
            self.td = max(self.DIST_MIN, min(self.dist_max, dist))
        if snap:
            self.h, self.p, self.d = self.th, self.tp, self.td
            self.apply()

    def orbit(self, dh: float, dp: float) -> None:
        self.th += dh
        self.tp = max(self.PITCH_MIN, min(self.PITCH_MAX, self.tp + dp))

    def zoom(self, factor: float) -> None:
        self.td = max(self.DIST_MIN, min(self.dist_max, self.td * factor))

    def reset_zoom(self) -> None:
        self.td = self.home_dist

    def reset_view(self) -> None:
        """Back to the framed home view: distance, tilt and heading (the C key)."""
        self.td = self.home_dist
        self.tp = self.home_pitch
        self.th = self.home_heading

    def begin_drag(self) -> None:
        if self.base.mouseWatcherNode.hasMouse():
            m = self.base.mouseWatcherNode.getMouse()
            self.last_mouse = (m.x, m.y)
            self.dragging = True

    def end_drag(self) -> None:
        self.dragging = False

    def update(self, dt: float) -> None:
        if self.dragging and self.base.mouseWatcherNode.hasMouse():
            m = self.base.mouseWatcherNode.getMouse()
            if self.last_mouse:
                self.orbit(-(m.x - self.last_mouse[0]) * 120, (m.y - self.last_mouse[1]) * 60)
            self.last_mouse = (m.x, m.y)
        if self.auto_orbit:
            self.th += self.auto_orbit * dt
        k = 1.0 - math.exp(-dt * 7.0)
        self.h += (self.th - self.h) * k
        self.p += (self.tp - self.p) * k
        self.d += (self.td - self.d) * k
        self.apply()

    def apply(self) -> None:
        h, p = math.radians(self.h), math.radians(self.p)
        x = self.target.x + self.d * math.cos(p) * math.sin(h)
        y = self.target.y - self.d * math.cos(p) * math.cos(h)
        z = self.target.z + self.d * math.sin(p)
        self.base.camera.setPos(x, y, z)
        self.base.camera.lookAt(self.target)

    # ------------------------------------------------------------------ automatic framing
    def _ndc_points(self, radius: float, pitch: float, dist: float) -> list[tuple[float, float]] | None:
        """Screen positions (NDC, no film offset) of the arena rim, on the floor and at marble height."""
        lens = self.base.camLens
        probe = NodePath("fit-probe")
        p = math.radians(pitch)
        probe.setPos(0, -dist * math.cos(p), dist * math.sin(p))
        probe.lookAt(Point3(0, 0, 0))
        world = NodePath()
        pts = []
        for k in range(96):
            a = 2 * math.pi * k / 96
            for zz in (0.0, 0.9):           # floor ring and the top of a marble sitting on the rim
                q = probe.getRelativePoint(world, Point3(radius * math.cos(a), radius * math.sin(a), zz))
                out = Point2()
                if not lens.project(q, out):
                    return None
                pts.append((out.x, out.y))
        return pts

    @staticmethod
    def _shift_that_fits(pts, rects, edge: float = 0.985, pad: float = 0.012) -> tuple[float, float] | None:
        """Smallest image shift (sx, sy) that keeps every point on screen and outside every HUD rectangle.

        Horizontal shifts let the arena use free space beside a side panel on narrow windows.
        """
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        if max(xs) - min(xs) > 2 * edge or max(ys) - min(ys) > 2 * edge:
            return None
        sxs = sorted((k / 20.0 for k in range(-8, 9)), key=abs)     # -0.4 .. 0.4
        sys_ = sorted((k / 50.0 for k in range(-35, 36)), key=abs)  # -0.7 .. 0.7
        for sx in sxs:
            if max(xs) + sx > edge or min(xs) + sx < -edge:
                continue
            for sy in sys_:
                if max(ys) + sy > edge or min(ys) + sy < -edge:
                    continue
                ok = True
                for x, y in pts:
                    xx, yy = x + sx, y + sy
                    for x0, x1, y0, y1 in rects:
                        if x0 - pad < xx < x1 + pad and y0 - pad < yy < y1 + pad:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    return sx, sy
        return None

    def fit_arena(self, radius: float, obstacles, pitches=(46.0, 50.0, 54.0, 58.0), snap: bool = True) -> float:
        """Frame the circle of ``radius`` as large as possible without touching any HUD panel.

        ``obstacles`` is a list (or a callable returning a list, re-evaluated on window
        resize) of HUD rectangles ``(x0, x1, y0, y1)`` in normalised screen coordinates.
        For each candidate camera pitch the smallest camera distance is found by binary
        search such that some vertical image shift (lens film offset) keeps the whole rim
        on screen and clear of the panels; the pitch giving the largest arena wins.
        """
        self._fit_args = (radius, obstacles, pitches)
        rects = obstacles() if callable(obstacles) else obstacles
        lens = self.base.camLens
        lens.setFilmOffset(0, 0)
        best = None
        for pitch in pitches:
            lo, hi, found = 8.0, 250.0, None
            for _ in range(26):
                mid = (lo + hi) / 2
                pts = self._ndc_points(radius, pitch, mid)
                sh = self._shift_that_fits(pts, rects) if pts else None
                if sh is not None:
                    hi, found = mid, (mid, sh, pts)
                else:
                    lo = mid
            if found is None:
                continue
            d, sh, pts = found
            xs, ys = [x for x, _ in pts], [y for _, y in pts]
            size = (max(xs) - min(xs)) * (max(ys) - min(ys))   # on-screen area of the arena
            if best is None or size > best[0]:
                best = (size, pitch, d, sh)
        if best is None:                     # pathological window: fall back to a safe default
            best = (0.0, 54.0, 40.0, (0.0, 0.0))
        _, pitch, dist, (sx, sy) = best
        fs = lens.getFilmSize()
        # a film offset of +f moves the image by -f / (film_size / 2) in NDC (left / down)
        lens.setFilmOffset(-sx * fs.x / 2, -sy * fs.y / 2)
        self.target = Point3(0, 0, 0)
        self.home_dist = dist
        self.home_pitch = pitch
        self.home_heading = self.th
        self.dist_max = max(self.DIST_MAX, dist * 1.6)
        self.set_view(pitch=pitch, dist=dist, snap=snap)
        return dist

    def clear_fit(self) -> None:
        """Back to free framing (menus): no film offset, default target."""
        self._fit_args = None
        self.base.camLens.setFilmOffset(0, 0)
        self.target = Point3(0, -1.0, 0)
        self.dist_max = self.DIST_MAX

    def _refit(self) -> None:
        for fn in list(self.aspect_listeners):
            fn()
        if self._fit_args is not None:
            r, obstacles, pitches = self._fit_args
            self.fit_arena(r, obstacles, pitches, snap=False)

    def mouse_to_floor(self) -> tuple[float, float] | None:
        """Intersect the mouse ray with the floor plane z = 0."""
        if not self.base.mouseWatcherNode.hasMouse():
            return None
        return self.screen_to_floor(self.base.mouseWatcherNode.getMouse())

    def screen_to_floor(self, m) -> tuple[float, float] | None:
        """Floor point under normalised screen position ``m`` (film offset included)."""
        near, far = Point3(), Point3()
        self.base.camLens.extrude(m, near, far)
        rel = self.base.render.getRelativePoint
        n, f = rel(self.base.camera, near), rel(self.base.camera, far)
        dz = f.z - n.z
        if abs(dz) < 1e-9:
            return None
        t = -n.z / dz
        if t < 0:
            return None
        return n.x + (f.x - n.x) * t, n.y + (f.y - n.y) * t
