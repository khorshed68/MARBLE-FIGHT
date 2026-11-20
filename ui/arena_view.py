"""3-D presentation of an arena and its marbles.

The view only *displays* positions that come from the physics engine (live
playback frames or a settled state). Rolling, trails, sparks and elimination
bursts are purely cosmetic and never feed back into the simulation.
"""

from __future__ import annotations

import math
import os
import random
from collections import deque

from panda3d.core import (AmbientLight, AntialiasAttrib, ColorBlendAttrib, DirectionalLight, LineSegs, Material,
                          NodePath, PNMImage, PointLight, Quat, TextNode, Texture, TextureStage, TransparencyAttrib,
                          Vec3)

from config import DATA_DIR, GameConfig
from game.arena import Arena
from game.state import GameState
from ui import geometry as G
from ui.theme import PLAYER_COLORS, PLAYER_DEEP, font

GEN_DIR = os.path.join(os.path.dirname(DATA_DIR), "assets", "generated")

# per-marble visual variations (cosmetic only; physics is identical for all marbles)
P1_SHADES = [((0.10, 0.75, 1.00), (0.85, 1.0, 1.0)), ((0.15, 0.45, 1.00), (0.55, 0.95, 1.0)),
             ((0.05, 0.90, 0.85), (0.80, 1.0, 0.9)), ((0.30, 0.60, 1.00), (0.95, 0.95, 1.0))]
P2_SHADES = [((1.00, 0.50, 0.10), (1.0, 0.9, 0.5)), ((1.00, 0.20, 0.55), (1.0, 0.75, 0.9)),
             ((1.00, 0.70, 0.15), (1.0, 0.35, 0.2)), ((0.95, 0.30, 0.25), (1.0, 0.85, 0.6))]


def _glow(np: NodePath, additive: bool = True) -> NodePath:
    np.setLightOff(1)
    np.setShaderOff(1)
    np.setTransparency(TransparencyAttrib.MAlpha)
    np.setDepthWrite(False)
    if additive:
        np.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.MAdd, ColorBlendAttrib.OIncomingAlpha,
                                           ColorBlendAttrib.OOne))
    np.setBin("fixed", 10)
    return np


def _cached_texture(name: str, build) -> Texture:
    os.makedirs(GEN_DIR, exist_ok=True)
    path = os.path.join(GEN_DIR, name)
    if os.path.exists(path):
        img = PNMImage()
        if img.read(path):
            t = Texture(name)
            t.load(img)
            t.setMinfilter(Texture.FT_linear_mipmap_linear)
            t.setAnisotropicDegree(4)
            return t
    tex = build()
    img = PNMImage()
    if tex.store(img):
        try:
            img.write(path)
        except Exception:
            pass
    return tex


class MarbleVisual:
    def __init__(self, parent: NodePath, mid: int, owner: int, radius: float, variant: int, shared: dict,
                 detail: int = 2, trails: bool = True) -> None:
        self.mid, self.owner, self.r = mid, owner, radius
        self.root = parent.attachNewNode(f"marble-{mid}")
        self.spin = self.root.attachNewNode("spin")
        self.spin.setZ(radius)
        base, accent = (P1_SHADES if owner == 0 else P2_SHADES)[variant % 4]
        rings, segs = ((8, 12), (12, 18), (16, 24))[detail]
        core = G.sphere(radius * 0.80, rings, segs)
        core.reparentTo(self.spin)
        core.setTexture(G.swirl_texture(base, accent, seed=mid * 17 + owner), 1)
        mc = Material()
        mc.setDiffuse((1, 1, 1, 1))
        mc.setEmission((base[0] * 0.35, base[1] * 0.35, base[2] * 0.35, 1))
        mc.setSpecular((0.6, 0.6, 0.6, 1))
        mc.setShininess(40)
        core.setMaterial(mc, 1)
        shell = G.sphere(radius, rings + 4, segs + 8, color=(base[0] * 0.6 + 0.4, base[1] * 0.6 + 0.4, base[2] * 0.6 + 0.4, 0.33))
        shell.reparentTo(self.spin)
        shell.setTransparency(TransparencyAttrib.MAlpha)
        ms = Material()
        ms.setDiffuse((0.9, 0.95, 1.0, 0.33))
        ms.setSpecular((1, 1, 1, 1))
        ms.setShininess(110)
        shell.setMaterial(ms, 1)
        shell.setDepthWrite(False)
        shell.setBin("transparent", 20)
        # player-colour ring (indicator) and blob shadow live on the floor, they do not roll
        col = PLAYER_COLORS[owner]
        ring = G.annulus(radius * 1.05, radius * 1.32, 48, color=(col[0], col[1], col[2], 0.85))
        ring.reparentTo(self.root)
        ring.setZ(0.012)
        _glow(ring)
        self.ring = ring
        shadow = G.card(radius * 3.0, radius * 3.0, color=(0, 0, 0, 0.75), z=0.008)
        shadow.reparentTo(self.root)
        shadow.setTexture(shared["shadow_tex"], 1)
        shadow.setTransparency(TransparencyAttrib.MAlpha)
        shadow.setLightOff(1)
        shadow.setShaderOff(1)
        shadow.setDepthWrite(False)
        shadow.setPos(0.12, 0.10, 0)
        self.shadow = shadow
        # identity tag "M<id>" above the marble (does not roll; hidden until a match turns tags on)
        tn = TextNode(f"tag-{mid}")
        tn.setText(f"M{mid}")
        tn.setAlign(TextNode.ACenter)
        tn.setTextColor(col[0], col[1], col[2], 1)
        tn.setShadow(0.05, 0.05)
        tn.setShadowColor(0, 0, 0, 1)
        tn.setCardColor(0.02, 0.03, 0.07, 0.8)
        tn.setCardAsMargin(0.18, 0.18, 0.06, 0.02)
        f = font("title")
        if f is not None:
            tn.setFont(f)
        tag = self.root.attachNewNode(tn)
        tag.setScale(0.5)
        tag.setPos(0, 0, radius * 2 + 0.35)
        tag.setBillboardPointEye()
        tag.setDepthTest(False)
        tag.setDepthWrite(False)
        tag.setBin("fixed", 55)
        tag.setLightOff()
        tag.hide()
        self.tag = tag
        # trail ghosts
        self.trail = []
        for k in range(6 if trails else 0):
            g = G.sphere(radius * (0.85 - 0.11 * k), 8, 12, color=(col[0], col[1], col[2], 0.28 - 0.04 * k))
            g.reparentTo(parent)
            _glow(g)
            g.hide()
            self.trail.append(g)
        self.history: deque = deque(maxlen=18)
        self.alive = True
        self.falling = None   # elimination animation state

    def place(self, x: float, y: float) -> None:
        self.root.setPos(x, y, 0)

    def roll(self, vx: float, vy: float, dt: float) -> None:
        speed = math.hypot(vx, vy)
        if speed < 1e-4 or dt <= 0:
            return
        axis = Vec3(-vy / speed, vx / speed, 0)          # up x velocity
        ang = math.degrees(speed * dt / self.r)
        q = Quat()
        q.setFromAxisAngle(ang, axis)
        self.spin.setQuat(self.spin.getQuat() * q)

    def update_trail(self, speed: float) -> None:
        self.history.appendleft(self.root.getPos())
        show = speed > 3.0 and self.alive
        for k, g in enumerate(self.trail):
            idx = 2 + k * 3
            if show and idx < len(self.history):
                p = self.history[idx]
                g.setPos(p.x, p.y, self.r)
                g.show()
            else:
                g.hide()

    def destroy(self) -> None:
        for g in self.trail:
            g.removeNode()
        self.root.removeNode()


class ArenaView:
    QUALITY_DETAIL = {"low": 0, "medium": 1, "high": 2}
    EFFECT_SCALE = {"off": 0.0, "reduced": 0.4, "full": 1.0}

    def __init__(self, base, arena: Arena, cfg: GameConfig, parent: NodePath | None = None,
                 quality: str = "high", effects: str = "full") -> None:
        self.base = base
        self.quality = quality if quality in self.QUALITY_DETAIL else "high"
        self.effects_level = effects if effects in self.EFFECT_SCALE else "full"
        self.detail = self.QUALITY_DETAIL[self.quality]
        self.fx = self.EFFECT_SCALE[self.effects_level]
        self.trails = self.effects_level == "full" and self.quality != "low"
        self.arena = arena
        self.cfg = cfg
        self.r = cfg.physics.marble_radius
        self.root = (parent or base.render).attachNewNode(f"arena-{arena.arena_id}")
        self.root.setShaderAuto()
        self.root.setAntialias(AntialiasAttrib.MNone if self.quality == "low" else AntialiasAttrib.MMultisample)
        self.pal = arena.palette
        self.shared = {"shadow_tex": G.radial_gradient(64, (0, 0, 0, 0.8), (0, 0, 0, 0), 1.3)}
        self.marbles: dict[int, MarbleVisual] = {}
        self.show_tags = False
        self.effects: list[dict] = []
        self.paths_np: NodePath | None = None
        self.aim_np: NodePath | None = None
        self.preview_np: NodePath | None = None
        self.select_np: NodePath | None = None
        self.selected: int | None = None
        self.time = 0.0
        self.rng = random.Random(1234)   # cosmetic randomness only
        self._build_lights()
        self._build_floor()
        self._build_features()
        self._build_select_ring()

    # ------------------------------------------------------------------ scene construction
    def _build_lights(self) -> None:
        amb = AmbientLight("amb")
        amb.setColor((0.32, 0.34, 0.42, 1))
        key = DirectionalLight("key")
        key.setColor((0.95, 0.92, 0.86, 1))
        key.setSpecularColor((1, 1, 1, 1))
        fill = DirectionalLight("fill")
        accent = self.pal.get("accent", (1, 0.6, 0.2))
        fill.setColor((accent[0] * 0.25, accent[1] * 0.25, accent[2] * 0.3, 1))
        self.amb_np = self.root.attachNewNode(amb)
        self.key_np = self.root.attachNewNode(key)
        self.key_np.setHpr(-30, -58, 0)
        self.fill_np = self.root.attachNewNode(fill)
        self.fill_np.setHpr(150, -35, 0)
        ring = self.pal.get("ring", (0.3, 0.9, 1))
        pl = PointLight("rimglow")
        pl.setColor((ring[0] * 0.6, ring[1] * 0.6, ring[2] * 0.6, 1))
        pl.setAttenuation((1, 0, 0.02))
        self.pl_np = self.root.attachNewNode(pl)
        self.pl_np.setPos(0, 0, 6)
        lights = [self.amb_np, self.key_np, self.fill_np] + ([self.pl_np] if self.quality == "high" else [])
        for np in lights:
            self.root.setLight(np)

    def _build_floor(self) -> None:
        a, pal, R = self.arena, self.pal, self.arena.radius
        base = pal.get("floor", (0.5, 0.4, 0.3))
        alt = pal.get("floor_alt", base)
        ring = pal.get("ring", (1, 1, 1))
        rim = pal.get("rim", (0.2, 0.2, 0.3))
        style = a.arena_id if a.arena_id in ("classic", "neon") else "magnetic"
        tex = _cached_texture(f"floor_{a.arena_id}_v2.png",
                              lambda: G.floor_texture(base, alt, ring, 384, style, seed=len(a.arena_id)))
        floor = G.disc(R + 0.02, 160)
        floor.reparentTo(self.root)
        floor.setTexture(tex, 1)
        mf = Material()
        mf.setDiffuse((1, 1, 1, 1))
        mf.setSpecular((0.25, 0.25, 0.3, 1) if style != "classic" else (0.08, 0.08, 0.08, 1))
        mf.setShininess(30)
        floor.setMaterial(mf, 1)
        # sloped lip beyond the boundary: marbles that cross it are out
        lip = G.annulus(R + 0.02, R + 1.4, 160, color=(rim[0], rim[1], rim[2], 1),
                        color_out=(rim[0] * 0.5, rim[1] * 0.5, rim[2] * 0.5, 1), z=0.0, z_out=-0.9)
        lip.reparentTo(self.root)
        plinth = G.cylinder(R + 1.4, 1.6, 160, color=(rim[0] * 0.45, rim[1] * 0.45, rim[2] * 0.5, 1), capped=False)
        plinth.reparentTo(self.root)
        plinth.setZ(-2.5)
        # glowing boundary line + halo
        line = G.annulus(R - 0.07, R + 0.07, 200, color=(ring[0], ring[1], ring[2], 1.0), z=0.015)
        line.reparentTo(self.root)
        _glow(line)
        halo = G.annulus(R - 0.6, R + 0.05, 200, color=(ring[0], ring[1], ring[2], 0.0),
                         color_out=(ring[0], ring[1], ring[2], 0.35), z=0.013)
        halo.reparentTo(self.root)
        _glow(halo)
        self.boundary = line
        # danger band hint (last 1.5 units): faint red wash
        danger = G.annulus(R - 1.5, R - 0.07, 160, color=(1, 0.2, 0.2, 0.0), color_out=(1, 0.25, 0.2, 0.12), z=0.011)
        danger.reparentTo(self.root)
        _glow(danger)
        # crisp concentric court markings (geometry, not texture, so they never alias)
        for rr, al in ((3.0, 0.55), (6.0, 0.55)):
            m = G.annulus(rr - 0.035, rr + 0.035, 160, color=(ring[0], ring[1], ring[2], al), z=0.01)
            m.reparentTo(self.root)
            _glow(m, additive=False)
        # centre emblem
        emb = G.annulus(0.35, 0.45, 64, color=(ring[0], ring[1], ring[2], 0.55), z=0.014)
        emb.reparentTo(self.root)
        _glow(emb)
        # surrounding stage
        stage = G.disc(60, 96, color=(0.025, 0.03, 0.06, 1), z=-2.5)
        stage.reparentTo(self.root)
        for rr, al in ((R + 4, 0.25), (R + 8, 0.12), (R + 14, 0.06)):
            s = G.annulus(rr - 0.04, rr + 0.04, 160, color=(ring[0], ring[1], ring[2], al), z=-2.48)
            s.reparentTo(self.root)
            _glow(s)

    def _floor_label(self, txt: str, x: float, y: float, color, scale: float = 0.42) -> None:
        tn = TextNode("lbl")
        tn.setText(txt)
        tn.setAlign(TextNode.ACenter)
        tn.setTextColor(*color)
        f = font("body")
        if f is not None:
            tn.setFont(f)
        np = self.root.attachNewNode(tn)
        np.setPos(x, y - 0.15, 0.02)
        np.setP(-90)
        np.setScale(scale)
        _glow(np, additive=False)

    def _build_features(self) -> None:
        a = self.arena
        accent = self.pal.get("accent", (1, 0.6, 0.2))
        ring = self.pal.get("ring", (0.3, 0.9, 1))
        self.dynamic_bumpers: list[tuple[NodePath, object]] = []
        self.magnet_nps: list[NodePath] = []
        zone_cols = {"sticky": (0.3, 1.0, 0.5), "glide": (0.55, 0.9, 1.0), "slip": (1.0, 0.3, 0.25)}
        stripe = G.stripe_texture((1, 0.3, 0.2, 0.55), (0.1, 0.05, 0.05, 0.25))
        for z in a.zones:
            c = zone_cols.get(z.kind, (1, 1, 1))
            d = G.disc(z.r, 72, color=(c[0], c[1], c[2], 0.22), z=0.006)
            d.reparentTo(self.root)
            d.setPos(z.x, z.y, 0)
            if z.kind == "slip":
                d.setTexture(stripe, 1)
                d.setColor(1, 1, 1, 1)
                d.setTexScale(TextureStage.getDefault(), 3, 3)
            _glow(d, additive=False)
            edge = G.annulus(z.r - 0.05, z.r + 0.03, 72, color=(c[0], c[1], c[2], 0.9), z=0.012)
            edge.reparentTo(self.root)
            edge.setPos(z.x, z.y, 0)
            _glow(edge)
            mult = f"x{z.friction:g} friction"
            self._floor_label(f"{z.label or z.kind}\n{mult}", z.x, z.y, (c[0], c[1], c[2], 0.85), 0.32)
        for b in a.bumpers:
            self._make_bumper(b.x, b.y, b.r, accent, ring)
        for rb in a.rotating_bumpers:
            track = G.annulus(rb.orbit - 0.03, rb.orbit + 0.03, 160, color=(accent[0], accent[1], accent[2], 0.35), z=0.011)
            track.reparentTo(self.root)
            _glow(track)
        for rb in a.rotating_bumpers:
            np = self._make_bumper(*rb.position(0.0), rb.r, accent, ring, moving=True)
            self.dynamic_bumpers.append((np, rb))
        spiral = G.spiral_texture((accent[0], accent[1], accent[2]))
        for m in a.magnets:
            d = G.disc(m.r, 72, z=0.009)
            d.reparentTo(self.root)
            d.setPos(m.x, m.y, 0)
            d.setTexture(spiral, 1)
            _glow(d)
            self.magnet_nps.append(d)
            edge = G.annulus(m.r - 0.04, m.r + 0.04, 72, color=(accent[0], accent[1], accent[2], 0.8), z=0.012)
            edge.reparentTo(self.root)
            edge.setPos(m.x, m.y, 0)
            _glow(edge)
            self._floor_label("Magnetic Vortex", m.x, m.y - m.r - 0.45, (accent[0], accent[1], accent[2], 0.9), 0.3)

    def _make_bumper(self, x, y, r, accent, ring, moving=False) -> NodePath:
        np = self.root.attachNewNode("bumper")
        body = G.cylinder(r, 0.75, 48, color=(0.22, 0.24, 0.3, 1), top_color=(0.12, 0.13, 0.17, 1))
        body.reparentTo(np)
        mb = Material()
        mb.setDiffuse((0.22, 0.24, 0.3, 1))
        mb.setSpecular((0.9, 0.9, 1, 1))
        mb.setShininess(70)
        body.setMaterial(mb, 1)
        c = accent if moving else ring
        cap = G.annulus(r * 0.55, r * 0.95, 48, color=(c[0], c[1], c[2], 0.95), z=0.76)
        cap.reparentTo(np)
        _glow(cap)
        base_glow = G.annulus(r, r + 0.35, 48, color=(c[0], c[1], c[2], 0.5), color_out=(c[0], c[1], c[2], 0.0), z=0.012)
        base_glow.reparentTo(np)
        _glow(base_glow)
        np.setPos(x, y, 0)
        return np

    def _build_select_ring(self) -> None:
        self.select_np = G.annulus(self.r * 1.45, self.r * 1.75, 64, color=(1, 1, 1, 0.9), z=0.016)
        self.select_np.reparentTo(self.root)
        _glow(self.select_np)
        self.select_np.hide()

    # ------------------------------------------------------------------ marbles
    def set_state(self, state: GameState) -> None:
        """Snap the view to a settled authoritative state."""
        if not self.marbles:
            counts = [0, 0]
            for m in state.marbles:
                self.marbles[m.mid] = MarbleVisual(self.root, m.mid, m.owner, self.r, counts[m.owner], self.shared,
                                                   self.detail, self.trails)
                counts[m.owner] += 1
            self.set_tags_visible(self.show_tags)
        for m in state.marbles:
            mv = self.marbles[m.mid]
            if mv.falling:
                continue
            mv.alive = m.alive
            if m.alive:
                mv.place(m.x, m.y)
                mv.root.show()
                mv.root.setAlphaScale(1)
                mv.root.setScale(1)
            else:
                mv.root.hide()
            mv.update_trail(0.0)
        self.set_clock(state.clock)

    def set_tags_visible(self, visible: bool) -> None:
        """Show the "M<id>" identity tag above every marble still in play."""
        self.show_tags = visible
        for mv in self.marbles.values():
            (mv.tag.show if visible and not mv.falling else mv.tag.hide)()

    def reset_marbles(self) -> None:
        for mv in self.marbles.values():
            mv.destroy()
        self.marbles.clear()
        for fx in self.effects:
            fx["np"].removeNode()
        self.effects.clear()

    def set_clock(self, clock: float) -> None:
        for np, rb in self.dynamic_bumpers:
            x, y = rb.position(clock)
            np.setPos(x, y, 0)

    def apply_sample(self, pos, alive, vel, dt: float) -> None:
        for mid, mv in self.marbles.items():
            if mv.falling:
                continue
            if not alive[mid] and not mv.alive:
                continue
            x, y = pos[mid]
            mv.place(x, y)
            vx, vy = vel[mid]
            mv.roll(vx, vy, dt)
            mv.update_trail(math.hypot(vx, vy))

    # ------------------------------------------------------------------ effects
    def spawn_contact(self, x: float, y: float, strength: float, color=(1, 1, 1)) -> None:
        if self.fx <= 0.0:
            return
        k = max(0.2, min(1.5, strength / 4.0))
        ring = G.annulus(0.15, 0.3, 40, color=(color[0], color[1], color[2], 0.9), z=0.3)
        ring.reparentTo(self.root)
        ring.setPos(x, y, 0)
        _glow(ring)
        self.effects.append({"np": ring, "t": 0.0, "life": 0.35, "kind": "ring", "k": 1.0 + 3.0 * k})
        for _ in range(int((4 + 6 * k) * self.fx) if self.fx >= 1.0 else 0):
            p = G.sphere(0.05, 4, 6, color=(1, 0.95, 0.8, 1))
            p.reparentTo(self.root)
            p.setPos(x, y, 0.35)
            _glow(p)
            a = self.rng.uniform(0, 2 * math.pi)
            sp = self.rng.uniform(1.5, 4.5) * k
            self.effects.append({"np": p, "t": 0.0, "life": self.rng.uniform(0.25, 0.5), "kind": "spark",
                                 "v": [math.cos(a) * sp, math.sin(a) * sp, self.rng.uniform(1, 3)]})

    def spawn_elimination(self, mid: int, vx: float, vy: float) -> None:
        mv = self.marbles.get(mid)
        if mv is None:
            return
        mv.alive = False
        sp = math.hypot(vx, vy)
        if sp < 1.0:
            p = mv.root.getPos()
            n = math.hypot(p.x, p.y) or 1.0
            vx, vy = p.x / n * 2.0, p.y / n * 2.0
        mv.falling = {"t": 0.0, "v": [vx, vy, 0.0]}
        mv.ring.hide()
        mv.shadow.hide()
        mv.tag.hide()
        for g in mv.trail:
            g.hide()
        col = PLAYER_COLORS[mv.owner]
        deep = PLAYER_DEEP[mv.owner]
        p = mv.root.getPos()
        flash = G.annulus(0.2, 0.5, 48, color=(col[0], col[1], col[2], 1.0), z=0.2)
        flash.reparentTo(self.root)
        flash.setPos(p.x, p.y, 0)
        _glow(flash)
        self.effects.append({"np": flash, "t": 0.0, "life": 0.7, "kind": "ring", "k": 7.0})
        for i in range(int(26 * self.fx)):
            c = col if i % 2 else deep
            q = G.sphere(0.07, 4, 6, color=(c[0], c[1], c[2], 1))
            q.reparentTo(self.root)
            q.setPos(p.x, p.y, 0.4)
            _glow(q)
            a = self.rng.uniform(0, 2 * math.pi)
            s = self.rng.uniform(2, 6)
            self.effects.append({"np": q, "t": 0.0, "life": self.rng.uniform(0.6, 1.1), "kind": "spark",
                                 "v": [math.cos(a) * s + vx * 0.3, math.sin(a) * s + vy * 0.3, self.rng.uniform(2, 6)]})

    def update(self, dt: float) -> None:
        self.time += dt
        # pulse boundary + selection
        pulse = 0.75 + 0.25 * math.sin(self.time * 3.0)
        self.boundary.setAlphaScale(pulse)
        for d in self.magnet_nps:
            d.setH(d.getH() - 50 * dt)
        if self.selected is not None and self.selected in self.marbles and self.marbles[self.selected].alive:
            p = self.marbles[self.selected].root.getPos()
            self.select_np.setPos(p.x, p.y, 0)
            self.select_np.setScale(1.0 + 0.12 * math.sin(self.time * 6))
            self.select_np.show()
        else:
            self.select_np.hide()
        # falling marbles
        for mv in self.marbles.values():
            f = mv.falling
            if not f:
                continue
            f["t"] += dt
            f["v"][2] -= 18 * dt
            p = mv.root.getPos()
            mv.root.setPos(p.x + f["v"][0] * dt * 0.6, p.y + f["v"][1] * dt * 0.6, p.z + f["v"][2] * dt)
            k = max(0.0, 1.0 - f["t"] / 0.9)
            mv.root.setScale(max(0.05, k))
            if f["t"] > 0.9:
                mv.root.hide()
                mv.falling = None
        # particles / rings
        keep = []
        for fx in self.effects:
            fx["t"] += dt
            u = fx["t"] / fx["life"]
            if u >= 1.0:
                fx["np"].removeNode()
                continue
            if fx["kind"] == "ring":
                fx["np"].setScale(1 + fx["k"] * u)
                fx["np"].setAlphaScale(1 - u)
            else:
                v = fx["v"]
                v[2] -= 12 * dt
                p = fx["np"].getPos()
                fx["np"].setPos(p.x + v[0] * dt, p.y + v[1] * dt, max(0.02, p.z + v[2] * dt))
                fx["np"].setAlphaScale(1 - u)
            keep.append(fx)
        self.effects = keep

    @property
    def busy(self) -> bool:
        return any(mv.falling for mv in self.marbles.values())

    # ------------------------------------------------------------------ guides
    def set_selected(self, mid: int | None, color=(1, 1, 1)) -> None:
        self.selected = mid
        if self.select_np is not None:
            self.select_np.setColor(color[0], color[1], color[2], 1)

    def show_aim(self, x: float, y: float, angle: float, power: float, color) -> None:
        """Arrow from the selected marble; length and colour follow the power."""
        self.clear_aim()
        root = self.root.attachNewNode("aim")
        length = 1.0 + 4.5 * power
        hot = (1.0, 0.85 - 0.55 * power, 0.25 - 0.2 * power, 1.0)     # power tint toward red
        c0 = (color[0], color[1], color[2], 0.95)
        c1 = (hot[0] * power + color[0] * (1 - power), hot[1] * power + color[1] * (1 - power),
              hot[2] * power + color[2] * (1 - power), 0.95)
        ca, sa = math.cos(angle), math.sin(angle)
        start = self.r * 1.25
        pts = [(x + ca * (start + length * k / 12), y + sa * (start + length * k / 12)) for k in range(13)]
        G.ribbon(pts, 0.16, c0, 0.06, c1).reparentTo(root)
        tipx, tipy = pts[-1]
        head = []
        for s_ in (-1, 1):
            head.append([(tipx, tipy), (tipx - math.cos(angle + s_ * 0.5) * 0.6, tipy - math.sin(angle + s_ * 0.5) * 0.6)])
        for h in head:
            G.ribbon(h, 0.14, c1, 0.06).reparentTo(root)
        _glow(root, additive=False)
        self.aim_np = root

    def clear_aim(self) -> None:
        if self.aim_np is not None:
            self.aim_np.removeNode()
            self.aim_np = None

    def show_paths(self, paths: list[tuple[list[tuple[float, float]], tuple, str]],
                   labels: list[tuple[float, float, str, tuple]] | None = None) -> None:
        """paths: (points, rgba, style) with style in {"solid", "thin", "dashed"}.
        labels: optional (x, y, text, rgba) tags drawn above the floor, always facing the camera."""
        self.clear_paths()
        root = self.root.attachNewNode("paths")
        for pts, col, style in paths:
            if len(pts) < 2:
                continue
            width = {"solid": 0.14, "thin": 0.06, "dashed": 0.09}[style]
            fade = (col[0], col[1], col[2], col[3] * 0.35)
            if style == "dashed":
                for i in range(0, len(pts) - 1, 2):
                    G.ribbon([pts[i], pts[i + 1]], width, col, 0.05).reparentTo(root)
            else:
                G.ribbon(pts, width, col, 0.05, fade).reparentTo(root)
            end = G.annulus(0.2, 0.3, 24, color=col, z=0.05)
            end.reparentTo(root)
            end.setPos(pts[-1][0], pts[-1][1], 0)
        _glow(root, additive=False)
        self.paths_np = root
        if labels:
            lab_root = self.root.attachNewNode("path-labels")
            f = font("title")
            for i, (x, y, txt, col) in enumerate(labels):
                tn = TextNode("tag")
                tn.setText(txt)
                tn.setAlign(TextNode.ACenter)
                tn.setTextColor(*col)
                tn.setShadow(0.06, 0.06)
                tn.setShadowColor(0, 0, 0, 0.9)
                tn.setCardColor(0.02, 0.03, 0.07, 0.75)
                tn.setCardAsMargin(0.15, 0.15, 0.08, 0.05)
                if f is not None:
                    tn.setFont(f)
                np = lab_root.attachNewNode(tn)
                np.setScale(0.55)
                np.setPos(x, y, 0.9 + 0.8 * i)          # stacked: tags never cover each other
                np.setBillboardPointEye()
                np.setDepthTest(False)
                np.setDepthWrite(False)
                np.setBin("fixed", 60)
                np.setLightOff()
            self.labels_np = lab_root

    def clear_paths(self) -> None:
        if self.paths_np is not None:
            self.paths_np.removeNode()
            self.paths_np = None
        if getattr(self, "labels_np", None) is not None:
            self.labels_np.removeNode()
            self.labels_np = None

    def pick(self, x: float, y: float, owner: int | None = None) -> int | None:
        best, bd = None, 1e9
        for mid, mv in self.marbles.items():
            if not mv.alive or (owner is not None and mv.owner != owner):
                continue
            p = mv.root.getPos()
            d = math.hypot(p.x - x, p.y - y)
            if d < self.r * 2.2 and d < bd:
                best, bd = mid, d
        return best

    def destroy(self) -> None:
        self.clear_aim()
        self.clear_paths()
        self.reset_marbles()
        self.root.clearLight()
        self.root.removeNode()
