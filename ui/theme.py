"""Visual identity: colours, fonts and DirectGUI widget helpers."""

from __future__ import annotations

import os
from typing import Any, Callable

from direct.gui.DirectGui import DGG, DirectButton, DirectFrame, DirectLabel, DirectOptionMenu
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import Filename, TextNode

# Player colours: P1 cyan/blue, P2 orange/magenta.
P1 = (0.15, 0.85, 1.0, 1.0)
P1_DEEP = (0.05, 0.35, 0.95, 1.0)
P2 = (1.0, 0.55, 0.12, 1.0)
P2_DEEP = (0.95, 0.15, 0.55, 1.0)
PLAYER_COLORS = (P1, P2)
PLAYER_DEEP = (P1_DEEP, P2_DEEP)

BG = (0.03, 0.04, 0.09, 1.0)
PANEL = (0.05, 0.07, 0.13, 0.88)
PANEL_LIGHT = (0.09, 0.12, 0.20, 0.92)
EDGE = (0.25, 0.75, 1.0, 0.9)
TEXT = (0.92, 0.95, 1.0, 1.0)
MUTED = (0.62, 0.68, 0.80, 1.0)
GOOD = (0.35, 1.0, 0.6, 1.0)
WARN = (1.0, 0.78, 0.25, 1.0)
DANGER = (1.0, 0.3, 0.35, 1.0)
GOLD = (1.0, 0.82, 0.3, 1.0)

BTN = (0.08, 0.13, 0.24, 0.95)
BTN_HOVER = (0.12, 0.42, 0.72, 1.0)
BTN_PRESS = (0.05, 0.6, 0.85, 1.0)
BTN_DISABLED = (0.08, 0.09, 0.12, 0.7)
BTN_ACTIVE = (0.07, 0.5, 0.75, 1.0)

_fonts: dict[str, Any] = {}


def font(kind: str = "body"):
    """Bahnschrift (title/UI) or Segoe UI, falling back to Panda's default font."""
    if kind in _fonts:
        return _fonts[kind]
    from direct.showbase import ShowBaseGlobal
    loader = ShowBaseGlobal.base.loader
    candidates = {
        "title": ["bahnschrift.ttf", "segoeuib.ttf", "arialbd.ttf"],
        "body": ["bahnschrift.ttf", "segoeui.ttf", "arial.ttf"],
        "mono": ["consola.ttf", "cour.ttf"],
    }[kind]
    f = None
    for name in candidates:
        path = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", name)
        if os.path.exists(path):
            try:
                f = loader.loadFont(Filename.fromOsSpecific(path))
                if f is None or not f.isValid():
                    f = None
                    continue
                f.setPixelsPerUnit(72)
                break
            except Exception:
                f = None
    _fonts[kind] = f
    return f


def text(parent, txt: str, pos=(0, 0), scale=0.05, fg=TEXT, align="left", kind="body", shadow=True,
         wordwrap=None, mayChange=True) -> OnscreenText:
    al = {"left": TextNode.ALeft, "center": TextNode.ACenter, "right": TextNode.ARight}[align]
    kw = dict(text=txt, pos=pos, scale=scale, fg=fg, align=al, parent=parent, mayChange=mayChange)
    f = font(kind)
    if f is not None:
        kw["font"] = f
    if shadow:
        kw["shadow"] = (0, 0, 0, 0.75)
        kw["shadowOffset"] = (0.06, 0.06)
    if wordwrap:
        kw["wordwrap"] = wordwrap
    return OnscreenText(**kw)


def panel(parent, frame=(-0.5, 0.5, -0.5, 0.5), pos=(0, 0, 0), color=PANEL, border=True) -> DirectFrame:
    f = DirectFrame(parent=parent, frameColor=color, frameSize=frame, pos=pos)
    if border:
        l, r, b, t = frame
        w = 0.004
        for fs in ((l, r, t - w, t), (l, r, b, b + w), (l, l + w, b, t), (r - w, r, b, t)):
            DirectFrame(parent=f, frameColor=(EDGE[0], EDGE[1], EDGE[2], 0.35), frameSize=fs)
    return f


def button(parent, label: str, command: Callable, pos=(0, 0, 0), size=(0.42, 0.065), scale=1.0,
           text_scale=0.045, extra_args=None, color=BTN, enabled=True, accent=None) -> DirectButton:
    w, h = size
    c = color if enabled else BTN_DISABLED
    kw = dict(
        parent=parent, text=label, command=command, extraArgs=extra_args or [], pos=pos, scale=scale,
        frameSize=(-w, w, -h, h), frameColor=(c, BTN_PRESS, BTN_HOVER, BTN_DISABLED),
        text_fg=TEXT if enabled else MUTED, text_scale=text_scale, text_pos=(0, -text_scale * 0.35),
        relief=DGG.FLAT, pressEffect=1, state=DGG.NORMAL if enabled else DGG.DISABLED,
    )
    f = font("body")
    if f is not None:
        kw["text_font"] = f
    b = DirectButton(**kw)
    stripe = accent or EDGE
    DirectFrame(parent=b, frameColor=(stripe[0], stripe[1], stripe[2], 0.9), frameSize=(-w, -w + 0.008, -h, h))
    return b


def set_button_active(b: DirectButton, active: bool) -> None:
    c = BTN_ACTIVE if active else BTN
    b["frameColor"] = (c, BTN_PRESS, BTN_HOVER, BTN_DISABLED)


def label(parent, txt: str, pos=(0, 0, 0), scale=0.045, fg=TEXT, align="left") -> DirectLabel:
    al = {"left": TextNode.ALeft, "center": TextNode.ACenter, "right": TextNode.ARight}[align]
    kw = dict(parent=parent, text=txt, pos=pos, text_scale=scale, text_fg=fg, text_align=al, frameColor=(0, 0, 0, 0))
    f = font("body")
    if f is not None:
        kw["text_font"] = f
    return DirectLabel(**kw)


def option_menu(parent, items: list[str], initial: int, command: Callable, pos=(0, 0, 0), scale=0.045,
                width=10) -> DirectOptionMenu:
    kw = dict(parent=parent, items=items, initialitem=initial, command=command, pos=pos, scale=scale,
              frameColor=BTN, text_fg=TEXT, highlightColor=BTN_HOVER, item_frameColor=PANEL_LIGHT,
              item_text_fg=TEXT, popupMarker_frameColor=EDGE, relief=DGG.FLAT, text_align=TextNode.ALeft,
              frameSize=(-0.2, width, -0.45, 0.95))
    f = font("body")
    if f is not None:
        kw["text_font"] = f
        kw["item_text_font"] = f
    return DirectOptionMenu(**kw)


def fmt(v: Any, nd: int = 2, unit: str = "") -> str:
    """Format a measured value; None means the metric was not measured (shown as n/a, never as 0)."""
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}{unit}"
    if isinstance(v, int):
        return f"{v:,}{unit}"
    return f"{v}{unit}"
