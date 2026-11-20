"""Reusable DirectGUI widgets: option cycler, bar chart, table, tabs."""

from __future__ import annotations

from typing import Any, Callable

from direct.gui.DirectGui import DGG, DirectButton, DirectFrame

from ui import theme as T


class Cycler:
    """``Label      < value >`` selector that cycles through options."""

    def __init__(self, parent, label: str, options: list[tuple[Any, str]], index: int, pos, on_change=None,
                 width: float = 1.2, label_w: float = 0.55, scale: float = 0.042) -> None:
        self.options = options
        self.index = max(0, min(index, len(options) - 1))
        self.on_change = on_change
        x, z = pos
        self.frame = DirectFrame(parent=parent, frameColor=(0.07, 0.1, 0.18, 0.85),
                                 frameSize=(0, width, -0.045, 0.045), pos=(x, 0, z))
        T.text(self.frame, label, (0.02, -0.015), scale * 0.9, T.MUTED)
        vx = label_w + (width - label_w) / 2
        self.value = T.text(self.frame, "", (vx, -0.015), scale, T.TEXT, "center")
        for d, bx in ((-1, label_w + 0.04), (1, width - 0.04)):
            DirectButton(parent=self.frame, text="<" if d < 0 else ">", command=self.step, extraArgs=[d],
                         pos=(bx, 0, 0), frameSize=(-0.032, 0.032, -0.035, 0.035), frameColor=(T.BTN, T.BTN_PRESS, T.BTN_HOVER, T.BTN_DISABLED),
                         text_fg=T.EDGE, text_scale=0.045, text_pos=(0, -0.015), relief=DGG.FLAT)
        self._render()

    def step(self, d: int) -> None:
        self.index = (self.index + d) % len(self.options)
        self._render()
        if self.on_change:
            self.on_change(self.get())

    def _render(self) -> None:
        self.value.setText(self.options[self.index][1])

    def get(self) -> Any:
        return self.options[self.index][0]

    def set(self, value: Any) -> None:
        for i, (v, _) in enumerate(self.options):
            if v == value:
                self.index = i
                self._render()
                return

    def destroy(self) -> None:
        self.frame.destroy()


def index_of(options: list[tuple[Any, str]], value: Any, default: int = 0) -> int:
    for i, (v, _) in enumerate(options):
        if v == value:
            return i
    return default


class BarChart:
    """Horizontal bar chart. ``rows`` = [(label, value or None, colour)]. None -> 'n/a'."""

    def __init__(self, parent, title: str, rows: list[tuple[str, float | None, tuple]], pos, width=1.0,
                 row_h=0.075, fmt: Callable[[float], str] = lambda v: f"{v:.2f}", max_value: float | None = None) -> None:
        x, z = pos
        h = 0.08 + row_h * len(rows)
        self.frame = T.panel(parent, (0, width, -h, 0), (x, 0, z), color=T.PANEL_LIGHT)
        T.text(self.frame, title, (0.025, -0.045), 0.034, T.EDGE, kind="title")
        vals = [v for _, v, _ in rows if v is not None]
        top = max_value if max_value is not None else (max(vals) if vals else 1.0)
        top = top if top and top > 0 else 1.0
        lw = width * 0.28
        bw = width * 0.5
        for i, (lab, v, col) in enumerate(rows):
            y = -0.1 - i * row_h
            T.text(self.frame, lab, (0.025, y - 0.012), 0.03, T.TEXT)
            DirectFrame(parent=self.frame, frameColor=(0.12, 0.14, 0.22, 1), frameSize=(lw, lw + bw, y - 0.022, y + 0.018))
            if v is not None:
                w = bw * max(0.0, min(1.0, v / top))
                DirectFrame(parent=self.frame, frameColor=col, frameSize=(lw, lw + max(0.004, w), y - 0.022, y + 0.018))
            T.text(self.frame, fmt(v) if v is not None else "—", (width - 0.025, y - 0.012), 0.03, T.TEXT, "right")
        self.height = h

    def destroy(self) -> None:
        self.frame.destroy()


class Table:
    def __init__(self, parent, headers: list[str], rows: list[list[str]], pos, col_w: list[float], row_h=0.05,
                 scale=0.028, title: str | None = None, colors: list[tuple] | None = None) -> None:
        x, z = pos
        width = sum(col_w) + 0.04
        h = (0.06 if title else 0.0) + row_h * (len(rows) + 1) + 0.03
        self.frame = T.panel(parent, (0, width, -h, 0), (x, 0, z), color=T.PANEL_LIGHT)
        y = -0.045
        if title:
            T.text(self.frame, title, (0.02, y), 0.034, T.EDGE, kind="title")
            y -= 0.06
        cx = 0.02
        for hname, w in zip(headers, col_w):
            T.text(self.frame, hname, (cx, y), scale, T.MUTED)
            cx += w
        for r, row in enumerate(rows):
            y -= row_h
            cx = 0.02
            for c, (cell, w) in enumerate(zip(row, col_w)):
                fg = colors[r] if (colors and c == 0) else T.TEXT
                T.text(self.frame, str(cell), (cx, y), scale, fg)
                cx += w
        self.height = h

    def destroy(self) -> None:
        self.frame.destroy()
