"""Procedural meshes and textures (no external assets needed)."""

from __future__ import annotations

import math
import random

from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter,
                          NodePath, PNMImage, SamplerState, Texture)


def _writers(name: str, colors: bool = True, uvs: bool = True):
    fmt = GeomVertexFormat.getV3n3c4t2() if (colors and uvs) else (
        GeomVertexFormat.getV3n3c4() if colors else GeomVertexFormat.getV3n3t2())
    vd = GeomVertexData(name, fmt, Geom.UHStatic)
    w = {"v": GeomVertexWriter(vd, "vertex"), "n": GeomVertexWriter(vd, "normal")}
    if colors:
        w["c"] = GeomVertexWriter(vd, "color")
    if uvs:
        w["t"] = GeomVertexWriter(vd, "texcoord")
    return vd, w


def _node(name: str, vd: GeomVertexData, tris: GeomTriangles) -> NodePath:
    g = Geom(vd)
    g.addPrimitive(tris)
    gn = GeomNode(name)
    gn.addGeom(g)
    return NodePath(gn)


def sphere(radius: float = 1.0, rings: int = 18, segments: int = 28, color=(1, 1, 1, 1)) -> NodePath:
    vd, w = _writers("sphere")
    for i in range(rings + 1):
        th = math.pi * i / rings
        for j in range(segments + 1):
            ph = 2 * math.pi * j / segments
            x, y, z = math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)
            w["v"].addData3(x * radius, y * radius, z * radius)
            w["n"].addData3(x, y, z)
            w["c"].addData4(*color)
            w["t"].addData2(j / segments, 1 - i / rings)
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(rings):
        for j in range(segments):
            a = i * (segments + 1) + j
            b = a + segments + 1
            tris.addVertices(a, b, a + 1)
            tris.addVertices(a + 1, b, b + 1)
    return _node("sphere", vd, tris)


def disc(radius: float, segments: int = 96, color=(1, 1, 1, 1), rings_color=None, z: float = 0.0,
         uv_scale: float = 1.0) -> NodePath:
    """Flat disc in the XY plane (UVs map world xy -> texture)."""
    vd, w = _writers("disc")
    w["v"].addData3(0, 0, z)
    w["n"].addData3(0, 0, 1)
    w["c"].addData4(*color)
    w["t"].addData2(0.5, 0.5)
    for j in range(segments + 1):
        a = 2 * math.pi * j / segments
        x, y = radius * math.cos(a), radius * math.sin(a)
        w["v"].addData3(x, y, z)
        w["n"].addData3(0, 0, 1)
        w["c"].addData4(*(rings_color or color))
        w["t"].addData2(0.5 + 0.5 * math.cos(a) * uv_scale, 0.5 + 0.5 * math.sin(a) * uv_scale)
    tris = GeomTriangles(Geom.UHStatic)
    for j in range(segments):
        tris.addVertices(0, j + 1, j + 2)
    return _node("disc", vd, tris)


def annulus(r_in: float, r_out: float, segments: int = 128, color=(1, 1, 1, 1), color_out=None,
            z: float = 0.0, z_out: float | None = None, arc: float = 2 * math.pi, start: float = 0.0) -> NodePath:
    """Ring (optionally sloped: inner edge at z, outer edge at z_out)."""
    vd, w = _writers("annulus")
    zo = z if z_out is None else z_out
    co = color if color_out is None else color_out
    slope = math.atan2(z - zo, r_out - r_in)
    nz, nr = math.cos(slope), math.sin(slope)
    for j in range(segments + 1):
        a = start + arc * j / segments
        c, s = math.cos(a), math.sin(a)
        for r, zz, col, v in ((r_in, z, color, 0.0), (r_out, zo, co, 1.0)):
            w["v"].addData3(r * c, r * s, zz)
            w["n"].addData3(nr * c, nr * s, nz)
            w["c"].addData4(*col)
            w["t"].addData2(j / segments * 8.0, v)
    tris = GeomTriangles(Geom.UHStatic)
    for j in range(segments):
        a = 2 * j
        tris.addVertices(a, a + 1, a + 2)
        tris.addVertices(a + 1, a + 3, a + 2)
    return _node("annulus", vd, tris)


def cylinder(radius: float, height: float, segments: int = 40, color=(1, 1, 1, 1), top_color=None,
             capped: bool = True) -> NodePath:
    vd, w = _writers("cyl")
    for j in range(segments + 1):
        a = 2 * math.pi * j / segments
        c, s = math.cos(a), math.sin(a)
        for zz in (0.0, height):
            w["v"].addData3(radius * c, radius * s, zz)
            w["n"].addData3(c, s, 0)
            w["c"].addData4(*color)
            w["t"].addData2(j / segments, zz / max(height, 1e-6))
    tris = GeomTriangles(Geom.UHStatic)
    for j in range(segments):
        a = 2 * j
        tris.addVertices(a, a + 2, a + 1)
        tris.addVertices(a + 1, a + 2, a + 3)
    base_idx = 2 * (segments + 1)
    if capped:
        tc = top_color or color
        w["v"].addData3(0, 0, height)
        w["n"].addData3(0, 0, 1)
        w["c"].addData4(*tc)
        w["t"].addData2(0.5, 0.5)
        for j in range(segments + 1):
            a = 2 * math.pi * j / segments
            w["v"].addData3(radius * math.cos(a), radius * math.sin(a), height)
            w["n"].addData3(0, 0, 1)
            w["c"].addData4(*tc)
            w["t"].addData2(0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.sin(a))
        for j in range(segments):
            tris.addVertices(base_idx, base_idx + j + 1, base_idx + j + 2)
    return _node("cylinder", vd, tris)


def card(w: float, h: float, color=(1, 1, 1, 1), z: float = 0.0) -> NodePath:
    """Horizontal quad centred at the origin."""
    vd, wr = _writers("card")
    for x, y, u, v in ((-w / 2, -h / 2, 0, 0), (w / 2, -h / 2, 1, 0), (w / 2, h / 2, 1, 1), (-w / 2, h / 2, 0, 1)):
        wr["v"].addData3(x, y, z)
        wr["n"].addData3(0, 0, 1)
        wr["c"].addData4(*color)
        wr["t"].addData2(u, v)
    tris = GeomTriangles(Geom.UHStatic)
    tris.addVertices(0, 1, 2)
    tris.addVertices(0, 2, 3)
    return _node("card", vd, tris)


# ----------------------------------------------------------------------------- textures
def _tex(img: PNMImage, name: str, mip: bool = True) -> Texture:
    t = Texture(name)
    t.load(img)
    if mip:
        t.setMinfilter(SamplerState.FT_linear_mipmap_linear)
        t.setMagfilter(SamplerState.FT_linear)
    t.setAnisotropicDegree(4)
    return t


def radial_gradient(size: int = 128, inner=(1, 1, 1, 1), outer=(1, 1, 1, 0), power: float = 1.6) -> Texture:
    img = PNMImage(size, size, 4)
    c = (size - 1) / 2
    for y in range(size):
        for x in range(size):
            d = min(1.0, math.hypot(x - c, y - c) / c)
            k = d ** power
            img.setXelA(x, y, *[inner[i] + (outer[i] - inner[i]) * k for i in range(4)])
    t = _tex(img, "radial", mip=False)
    t.setWrapU(SamplerState.WM_clamp)
    t.setWrapV(SamplerState.WM_clamp)
    return t


def swirl_texture(base, accent, seed: int = 0, size: int = 128) -> Texture:
    """Twisted-ribbon pattern for the glowing core inside a glass marble."""
    rng = random.Random(seed)
    img = PNMImage(size, size // 2, 4)
    bands = rng.choice([2, 3, 3, 4])
    twist = rng.uniform(1.5, 3.0)
    phase = rng.uniform(0, 2 * math.pi)
    for y in range(size // 2):
        v = y / (size // 2 - 1)
        for x in range(size):
            u = x / size
            s = math.sin(2 * math.pi * (bands * u + twist * v) + phase)
            k = max(0.0, s) ** 3
            g = 0.82 + 0.18 * math.sin(2 * math.pi * (u * 2 + v))
            col = [(base[i] * (1 - k) + accent[i] * k) * g for i in range(3)]
            img.setXelA(x, y, min(1, col[0]), min(1, col[1]), min(1, col[2]), 1.0)
    return _tex(img, "swirl")


def floor_texture(base, alt, ring, size: int = 512, style: str = "classic", seed: int = 3) -> Texture:
    """Arena floor pattern: concentric rings, sector lines and surface noise."""
    rng = random.Random(seed)
    img = PNMImage(size, size, 4)
    c = (size - 1) / 2
    noise = [[rng.random() for _ in range(64)] for _ in range(64)]

    def n2(x, y):
        gx, gy = x * 63 / size, y * 63 / size
        x0, y0 = int(gx), int(gy)
        fx, fy = gx - x0, gy - y0
        x1, y1 = min(63, x0 + 1), min(63, y0 + 1)
        a = noise[y0][x0] * (1 - fx) + noise[y0][x1] * fx
        b = noise[y1][x0] * (1 - fx) + noise[y1][x1] * fx
        return a * (1 - fy) + b * fy

    for y in range(size):
        for x in range(size):
            dx, dy = (x - c) / c, (y - c) / c
            r = math.hypot(dx, dy)
            ang = math.atan2(dy, dx)
            nz = n2(x, y) * 0.6 + n2((x * 4) % size, (y * 4) % size) * 0.4
            if style == "classic":
                k = 0.5 + 0.5 * math.sin(r * 40 + nz * 3)
                col = [base[i] * (0.85 + 0.25 * nz) * (0.94 + 0.06 * k) for i in range(3)]
            elif style == "neon":
                tile = (int((dx + 1) * 12) + int((dy + 1) * 12)) % 2
                col = [(base[i] if tile else alt[i]) * (0.9 + 0.2 * nz) for i in range(3)]
                if abs(math.sin(ang * 6)) < 0.012 * (1 / max(r, 0.15)):
                    col = [col[i] * 0.4 + ring[i] * 0.35 for i in range(3)]
            else:  # magnetic: hexagonal-ish plates with hazard veins
                hx = math.sin(dx * 30) * math.sin(dy * 30 + dx * 15)
                col = [(alt[i] if hx > 0 else base[i]) * (0.85 + 0.3 * nz) for i in range(3)]
                vein = abs(math.sin(r * 22 + ang * 3 + nz * 4))
                if vein < 0.04:
                    col = [col[i] + ring[i] * 0.25 for i in range(3)]
            img.setXelA(x, y, min(1, col[0]), min(1, col[1]), min(1, col[2]), 1.0)
    return _tex(img, "floor")


def stripe_texture(c1, c2, size: int = 64) -> Texture:
    img = PNMImage(size, size, 4)
    for y in range(size):
        for x in range(size):
            k = ((x + y) // (size // 4)) % 2
            c = c1 if k else c2
            img.setXelA(x, y, *c)
    return _tex(img, "stripes")


def spiral_texture(color, size: int = 128) -> Texture:
    img = PNMImage(size, size, 4)
    c = (size - 1) / 2
    for y in range(size):
        for x in range(size):
            dx, dy = (x - c) / c, (y - c) / c
            r = math.hypot(dx, dy)
            a = math.atan2(dy, dx)
            s = 0.5 + 0.5 * math.sin(a * 3 + r * 14)
            fade = max(0.0, 1 - r) ** 0.7
            img.setXelA(x, y, color[0], color[1], color[2], min(1.0, s * fade * 0.9))
    t = _tex(img, "spiral", mip=False)
    t.setWrapU(SamplerState.WM_clamp)
    t.setWrapV(SamplerState.WM_clamp)
    return t


def ribbon(points: list[tuple[float, float]], width: float, color=(1, 1, 1, 1), z: float = 0.05,
           color_end=None) -> NodePath:
    """Flat thick polyline on the floor (driver-independent alternative to line thickness)."""
    vd, w = _writers("ribbon")
    n = len(points)
    ce = color if color_end is None else color_end
    hw = width / 2
    for i, (x, y) in enumerate(points):
        if i == 0:
            dx, dy = points[1][0] - x, points[1][1] - y
        elif i == n - 1:
            dx, dy = x - points[i - 1][0], y - points[i - 1][1]
        else:
            dx, dy = points[i + 1][0] - points[i - 1][0], points[i + 1][1] - points[i - 1][1]
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L * hw, dx / L * hw
        u = i / max(1, n - 1)
        col = [color[k] + (ce[k] - color[k]) * u for k in range(4)]
        for sx, v in ((1, 0.0), (-1, 1.0)):
            w["v"].addData3(x + nx * sx, y + ny * sx, z)
            w["n"].addData3(0, 0, 1)
            w["c"].addData4(*col)
            w["t"].addData2(u, v)
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(n - 1):
        a = 2 * i
        tris.addVertices(a, a + 1, a + 2)
        tris.addVertices(a + 1, a + 3, a + 2)
    return _node("ribbon", vd, tris)
