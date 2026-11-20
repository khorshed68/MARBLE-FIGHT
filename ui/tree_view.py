"""Search-tree drawing shared by the Algorithm Lab and the in-match SHOW TREE popup.

The class is a mixin: the host provides ``_mcts_step(n)`` and ``_mcts_reset()`` for the
MCTS step buttons. Coordinates are aspect2d units laid out for a 16:9 window
(the tree occupies x = -0.7 .. 1.5).
"""

from __future__ import annotations

import math

from direct.gui.DirectGui import DirectFrame
from panda3d.core import LineSegs

from ui import theme as T

MCTS_ITERS = 40


class TreeDrawing:
    TREE_X = (-0.7, 1.5)        # horizontal extent of the tree (aspect2d units)
    TREE_BIG = 1.0              # text / box size factor

    @staticmethod
    def _short_label(a) -> str:
        from ai.search_trace import action_label
        return action_label(a)

    @staticmethod
    def _two_lines(a) -> tuple[str, str]:
        from ai.search_trace import action_label
        lab = action_label(a)
        if " | " in lab:
            head, tail = lab.split(" | ", 1)
            return head, tail
        head, _, power = lab.rpartition(" ")
        return head, f"Normal {power}"

    def _draw_mcts(self, parent, st) -> None:
        """Top two levels of the real MCTS tree, with N, Q, UCT and the path of the last iteration."""
        x0, x1 = self.TREE_X
        s = self.TREE_BIG
        root = st.root
        rp = st.player
        if s > 1:
            rows = {0: 0.6, 1: 0.17, 2: -0.25}
            heights = {0: 0.17 * s, 1: 0.29 * s, 2: 0.22 * s}
        else:
            rows = {0: 0.56, 1: 0.13, 2: -0.27}
            heights = {0: 0.17, 1: 0.29, 2: 0.22}
        untried_root = list(root.untried or [])
        level1 = [(c, False) for c in root.children] + [(a, True) for a in untried_root]
        # a shot that ends the match has no replies: still give it a readable (2-slot) box
        units = [(len(c.children) or 2) if not ghost else 1 for c, ghost in level1]
        slot = (x1 - x0) / max(1, sum(units))
        last = st.events[-1] if st.events else None
        on_path = set(id(n) for n in last.path) if last else set()
        new = id(last.expanded) if (last and last.expanded is not None) else None
        best = st.best()
        pos = {}
        acc = x0
        for (c, ghost), u in zip(level1, units):
            pos[id(c)] = acc + u * slot / 2
            if not ghost:
                for j, g in enumerate(c.children):
                    pos[id(g)] = acc + (j + 0.5) * slot
            acc += u * slot
        pos[id(root)] = (x0 + x1) / 2
        # edges: brightness and line count grow with the share of visits
        ls = LineSegs()
        ls.setThickness(2)

        def edge(parent_node, child, ply, ghost=False):
            px, pz = pos[id(parent_node)], rows[ply - 1] - heights[ply - 1] / 2
            cx, cz = pos[id(child)], rows[ply] + heights[ply] / 2
            if ghost:
                ls.setColor(0.5, 0.5, 0.55, 0.45)
                for k in range(0, 11, 2):
                    f0, f1 = k / 11, (k + 1) / 11
                    ls.moveTo(px + (cx - px) * f0, 0, pz + (cz - pz) * f0)
                    ls.drawTo(px + (cx - px) * f1, 0, pz + (cz - pz) * f1)
                return
            share = child.visits / max(1, parent_node.visits)
            if id(child) in on_path and id(parent_node) in on_path:
                col = T.GOOD
            elif parent_node is root and child is best:
                col = T.GOLD
            else:
                col = (0.55, 0.62, 0.78, 1.0)
            ls.setColor(col[0], col[1], col[2], 0.3 + 0.7 * share)
            for k in range(1 + int(round(3 * share))):
                off = (k - 0.5 * int(round(3 * share))) * 0.004
                ls.moveTo(px + off, 0, pz)
                ls.drawTo(cx + off, 0, cz)

        for c, ghost in level1:
            edge(root, c, 1, ghost)
            if not ghost:
                for g in c.children:
                    edge(c, g, 2)
        parent.attachNewNode(ls.create())

        def box(node, ply, w, ghost=False):
            x, z, h = pos[id(node)], rows[ply], heights[ply]
            if ply == 1 and not ghost and node is best:
                DirectFrame(parent=parent, frameColor=T.GOLD, frameSize=(-w / 2 - 0.008, w / 2 + 0.008,
                                                                       -h / 2 - 0.008, h / 2 + 0.008), pos=(x, 0, z))
            elif not ghost and id(node) in on_path:
                DirectFrame(parent=parent, frameColor=T.GOOD, frameSize=(-w / 2 - 0.006, w / 2 + 0.006,
                                                                       -h / 2 - 0.006, h / 2 + 0.006), pos=(x, 0, z))
            colr = (0.07, 0.07, 0.09, 0.95) if ghost else T.PANEL_LIGHT
            b = T.panel(parent, (-w / 2, w / 2, -h / 2, h / 2), (x, 0, z), color=colr)
            if not ghost:
                mover = rp if node is root else node.mover
                bar = T.PLAYER_COLORS[mover]
                DirectFrame(parent=b, frameColor=bar, frameSize=(-w / 2, w / 2, h / 2 - 0.008, h / 2))
            return b

        def write(b, h, rows_txt, k=1.0):
            y = h / 2 - 0.034 * k
            for txt, c, sc in rows_txt:
                sc *= k
                T.text(b, txt, (0, y), sc, c, "center")
                y -= sc + 0.011

        # root
        lim = st.widening_limit(root)
        b = box(root, 0, 0.66 * s)
        write(b, heights[0], [(f"ROOT | Player {rp + 1} to move", T.PLAYER_COLORS[rp], 0.028),
                              (f"N = {root.visits} iterations", T.GOLD, 0.032),
                              (f"children {len(root.children)} / widening limit {lim} | untried {len(untried_root)}",
                               T.MUTED, 0.021),
                              ("plays the MOST VISITED child", T.MUTED, 0.021)], s)
        for (c, ghost), u in zip(level1, units):
            w = min(0.5 * s, u * slot - 0.02)
            k1 = s if w >= 0.3 else 1.0            # larger text only where the box is wide enough
            if ghost:
                l1, l2 = self._two_lines(c)
                bb = box(c, 1, w, ghost=True)
                write(bb, heights[1], [("NOT EXPANDED YET", T.MUTED, 0.022), (l1, T.MUTED, 0.021),
                                       (l2, T.MUTED, 0.02), ("progressive widening", T.MUTED, 0.019),
                                       ("opens it when N grows", T.MUTED, 0.019)], k1)
                continue
            l1, l2 = self._two_lines(c.action)
            bb = box(c, 1, w)
            head = ("NEW  " if id(c) == new else "") + f"P{c.mover + 1}: {l1}"
            write(bb, heights[1], [
                (head, T.GOOD if id(c) == new else T.PLAYER_COLORS[c.mover], 0.025), (l2, T.TEXT, 0.023),
                (f"N {c.visits}   Q {c.mean:.3f}", T.GOLD if c is best else T.TEXT, 0.027),
                (f"UCT {st.uct(c):.3f}" if c.visits else "UCT +inf", T.TEXT, 0.022),
                (f"risk edge {c.edge:+.1f}", T.MUTED, 0.021),
                self._ends(c, rp) or (f"{len(c.children)} replies | {len(c.untried or [])} untried", T.MUTED,
                                      0.021)], k1)
            for g in c.children:
                gw = slot - 0.012
                gb = box(g, 2, gw)
                gl1, gl2 = self._two_lines(g.action)
                deeper = sum(1 for _ in g.children)
                big = 1.0 if slot < 0.2 else 1.25          # few replies -> wide boxes -> larger text
                rows_g = [("NEW" if id(g) == new else f"P{g.mover + 1} reply",
                           T.GOOD if id(g) == new else T.PLAYER_COLORS[g.mover], 0.018 * big),
                          (gl1.split(" ", 1)[-1] if slot < 0.2 else gl1, T.TEXT, 0.018 * big),
                          (gl2.replace(" 100%", "").replace("Normal", "Norm") if slot < 0.2 else gl2, T.TEXT,
                           0.017 * big),
                          (f"N {g.visits}", T.GOLD, 0.024 * big), (f"Q {g.mean:.2f}", T.TEXT, 0.024 * big)]
                if deeper:
                    rows_g.append((f"+{deeper} deeper", T.MUTED, 0.017 * big))
                end = self._ends(g, rp, short=True)
                if end:
                    rows_g = [r for r in rows_g if "deeper" not in r[0]] + [end]
                if s > 1 and slot < 0.22:          # popup, narrow boxes: fewer, larger lines
                    ab = gl2.split(" ")[0].replace("Normal", "Norm").replace("Precision", "Prec")
                    rows_g = [(gl1, T.GOOD if id(g) == new else T.PLAYER_COLORS[g.mover], 0.023),
                              (ab, T.TEXT, 0.022), (f"N {g.visits}", T.GOLD, 0.03), (f"Q {g.mean:.2f}", T.TEXT, 0.027)]
                    if end:
                        rows_g.append(end)
                    elif deeper:
                        rows_g.append((f"+{deeper} deeper", T.MUTED, 0.02))
                write(gb, heights[2], rows_g, s if slot >= 0.22 else 1.0)
        # ---- the last iteration, phase by phase
        y = -0.47 if s == 1 else -0.46
        ps = 1.0 if s == 1 else 1.15
        if last is None:
            T.text(parent, "Press NEXT ITERATION: each press runs one full MCTS iteration and shows its 4 phases.",
                   ((x0 + x1) / 2, y), 0.027 * ps, T.WARN, "center")
        else:
            names = []
            for n in last.path[1:]:
                l1, l2 = self._two_lines(n.action)
                names.append(f"P{n.mover + 1} {l1} ({l2})")
            sel = " > ".join(["root"] + names[:-1] if last.expanded is not None else ["root"] + names)
            if len(last.path) > 3:
                sel += f"   [reached depth {len(last.path) - 1}; deeper levels are not drawn]"
            exp = names[-1] if last.expanded is not None else "nothing (terminal or fully expanded)"
            own = last.reward
            texts = [
                (f"ITERATION {last.index}", T.GOOD, 0.03),
                (f"1 Selection (highest UCT at each level): {sel}", T.TEXT, 0.023),
                (f"2 Expansion: {exp}", T.GOOD, 0.023),
                (("3 Terminal position: exact result " if last.terminal else
                  f"3 Rollout: {last.rollout_turns} quick turn(s), then evaluation -> ") +
                 f"reward {own:.3f} for Player {rp + 1}", T.TEXT, 0.023),
                (f"4 Backpropagation: every node on the path gets N + 1 and W + {own:.3f} (Player {rp + 1}'s shots) "
                 f"or W + {1 - own:.3f} (Player {2 - rp}'s shots)", T.TEXT, 0.023),
            ]
            for txt, c, sc in texts:
                sc *= ps
                o = T.text(parent, txt, (x0, y), sc, c, "left", wordwrap=(x1 - x0) / sc)
                y -= (sc + 0.012) * self._nrows(o) + 0.005
        bx = [(-0.45, "NEXT ITERATION", lambda: self._mcts_step(1)), (0.05, "+10", lambda: self._mcts_step(10)),
              (0.5, "RUN ALL", lambda: self._mcts_step(MCTS_ITERS)), (0.95, "RESET", self._mcts_reset)]
        for x, lab, cmd in bx:
            btn = T.button(parent, lab, cmd, pos=(x0 + 0.25 + (x + 0.45) * (x1 - x0) / 2.2, 0, -0.8),
                           size=(0.2, 0.035), text_scale=0.026 * ps,
                           enabled=not (st.done and lab != "RESET"))
        T.text(parent, "green = path of the last iteration   gold = current choice   grey dashed = not expanded yet"
                       "   brighter / more lines = more visits", ((x0 + x1) / 2, -0.875), 0.021 * ps, T.MUTED,
               "center")

    @staticmethod
    def _draw_minimap(parent, state, played, best, cx, cz, half, label_scale=0.022, owner_labels=False):
        """Top view of the position (same coordinates as the physics): rim, obstacles, marbles with
        their M ids, a white line for the shot played and a gold line for the tree's choice."""
        from game.arena import get_arena
        arena = get_arena(state.arena_id)
        s = (half - 0.03) / arena.radius
        mp = T.panel(parent, (-half, half, -half, half), (cx, 0, cz), color=T.PANEL_LIGHT)
        ls = LineSegs()
        ls.setThickness(2)
        ls.setColor(T.EDGE[0], T.EDGE[1], T.EDGE[2], 0.9)
        for k in range(73):
            a = 2 * math.pi * k / 72
            (ls.moveTo if k == 0 else ls.drawTo)(arena.radius * s * math.cos(a), 0, arena.radius * s * math.sin(a))
        for b in arena.obstacle_positions(state.clock):
            bx, by, br = b
            ls.setColor(1.0, 0.75, 0.3, 0.8)
            for k in range(17):
                a = 2 * math.pi * k / 16
                (ls.moveTo if k == 0 else ls.drawTo)((bx + br * math.cos(a)) * s, 0, (by + br * math.sin(a)) * s)
        mp.attachNewNode(ls.create())
        # played shot: wide white line underneath, tree choice: thin gold line on top, so both stay
        # visible when they are the same shot
        for act, col, th in ((played, (1, 1, 1, 0.95), 6 if owner_labels else 2), (best, T.GOLD, 2)):
            if act is not None:
                m = state.marbles[act.marble_id]
                sl = LineSegs()
                sl.setThickness(th)
                sl.setColor(col[0], col[1], col[2], col[3])
                sl.moveTo(m.x * s, 0, m.y * s)
                sl.drawTo((m.x + 4.0 * math.cos(act.angle)) * s, 0, (m.y + 4.0 * math.sin(act.angle)) * s)
                mp.attachNewNode(sl.create())
        for m in state.marbles:
            if m.alive:
                r = max(0.012, 0.45 * s * 1.3)
                DirectFrame(parent=mp, frameColor=T.PLAYER_COLORS[m.owner], frameSize=(-r, r, -r, r),
                            pos=(m.x * s, 0, m.y * s))
                if owner_labels:          # popup: centred above the marble, in the owner's colour
                    T.text(mp, f"M{m.mid}", (m.x * s, m.y * s + r + 0.008), label_scale, T.PLAYER_COLORS[m.owner],
                           "center")
                else:
                    T.text(mp, f"M{m.mid}", (m.x * s + 0.02, m.y * s + 0.015), label_scale, T.TEXT)
        return mp

    @staticmethod
    def _ends(node, rp, short=False):
        """Text row for a shot that ends the match (no replies follow), else None."""
        w = node.state.winner if node.state is not None else None
        if w is None:
            return None
        if w < 0:
            return ("DRAW" if short else "MATCH ENDS: draw", T.WARN, 0.022)
        col = T.GOOD if w == rp else T.DANGER
        return (f"P{w + 1} WINS" if short else f"MATCH ENDS: Player {w + 1} wins", col, 0.025 if short else 0.022)

    @staticmethod
    def _nrows(o) -> int:
        try:
            return max(1, int(o.textNode.getNumRows()))
        except Exception:
            return 1

    def _draw_tree(self, parent, tr) -> None:
        nodes = tr.nodes
        x0, x1 = self.TREE_X
        s = self.TREE_BIG
        if s > 1:
            rows = {0: 0.6, 1: 0.13 if tr.depth == 2 else -0.05, 2: -0.37}
        else:
            rows = {0: 0.56, 1: 0.12 if tr.depth == 2 else -0.05, 2: -0.36}
        units: dict[int, int] = {}

        def count(n):
            ch = nodes[n].children
            units[n] = max(1, sum(count(c) for c in ch)) if ch else 1
            return units[n]

        count(0)
        slot = (x1 - x0) / units[0]
        pos: dict[int, float] = {}

        def place(n, left):
            pos[n] = left + units[n] * slot / 2
            acc = left
            for c in nodes[n].children:
                place(c, acc)
                acc += units[c] * slot

        place(0, x0)
        pv = set(tr.principal_variation())
        rp = tr.root_player
        heights = {0: 0.185, 1: 0.27 if tr.depth == 2 else 0.25, 2: 0.25}
        if s > 1:
            heights = {0: 0.235, 1: 0.38 if tr.depth == 2 else 0.32, 2: 0.33}
        best = tr.best_child
        # how far the chosen shot is ahead of the next root shot (values are shown rounded,
        # so a near-tie must be said explicitly)
        others = [nodes[c].backed_up for c in nodes[0].children
                  if c != best and not nodes[c].pruned and nodes[c].backed_up is not None]
        lead = (nodes[best].backed_up - max(others)) if (best is not None and others) else None
        # edges first (drawn under the boxes)
        ls = LineSegs()
        ls.setThickness(2)
        for n in nodes:
            if n.parent is None:
                continue
            px, pz = pos[n.parent], rows[nodes[n.parent].ply] - heights[nodes[n.parent].ply] / 2
            cx, cz = pos[n.nid], rows[n.ply] + heights[n.ply] / 2
            if n.pruned:
                ls.setColor(T.DANGER[0], T.DANGER[1], T.DANGER[2], 0.55)
                for k in range(0, 11, 2):                      # dashed
                    f0, f1 = k / 11, (k + 1) / 11
                    ls.moveTo(px + (cx - px) * f0, 0, pz + (cz - pz) * f0)
                    ls.drawTo(px + (cx - px) * f1, 0, pz + (cz - pz) * f1)
                continue
            col = T.GOLD if (n.nid in pv and n.parent in pv) else (0.55, 0.62, 0.78, 0.7)
            ls.setColor(col[0], col[1], col[2], col[3])
            ls.moveTo(px, 0, pz)
            ls.drawTo(cx, 0, cz)
        parent.attachNewNode(ls.create())

        def fv(v):
            if v is None:
                return "-"
            if v == float("inf"):
                return "+inf"
            if v == float("-inf"):
                return "-inf"
            return f"{v:+.2f}"

        def bound(n):
            """A node with a cutoff returned a bound, not an exact value (fail-soft alpha-beta)."""
            if not n.cutoff:
                return ""
            return "<= " if not n.is_max else ">= "

        for n in nodes:
            x, z, h = pos[n.nid], rows[n.ply], heights[n.ply]
            w = min((0.62 if n.ply == 0 else 0.5) * (1.15 if s > 1 else 1.0), units[n.nid] * slot - 0.02)
            if n.nid in pv:
                DirectFrame(parent=parent, frameColor=T.GOLD, frameSize=(-w / 2 - 0.008, w / 2 + 0.008,
                                                                       -h / 2 - 0.008, h / 2 + 0.008), pos=(x, 0, z))
            mover = nodes[n.parent].state.to_move if n.parent is not None else None
            if n.pruned:
                box = T.panel(parent, (-w / 2, w / 2, -h / 2, h / 2), (x, 0, z), color=(0.07, 0.07, 0.09, 0.95))
                l1, l2 = self._two_lines(n.action)
                rows_txt = [("PRUNED", T.DANGER, 0.027), (l1, T.MUTED, 0.021), (l2, T.MUTED, 0.021),
                            ("not simulated", T.MUTED, 0.02)]
            else:
                col = T.PLAYER_COLORS[rp if mover is None else mover]      # bar = who took this shot
                box = T.panel(parent, (-w / 2, w / 2, -h / 2, h / 2), (x, 0, z), color=T.PANEL_LIGHT)
                DirectFrame(parent=box, frameColor=col, frameSize=(-w / 2, w / 2, h / 2 - 0.008, h / 2))
                if n.parent is None:
                    rows_txt = [(f"ROOT | MAX | Player {rp + 1} to move", T.PLAYER_COLORS[rp], 0.03),
                                (f"value {fv(n.value)}", T.GOLD, 0.038),
                                ("takes the LARGEST child value", T.MUTED, 0.022)]
                    if lead is not None:
                        rows_txt.append((f"near tie: ahead of the next shot by only {lead:.3f}" if lead < 0.01 else
                                         f"ahead of the next shot by {lead:.2f}", T.WARN if lead < 0.01 else T.MUTED,
                                         0.021))
                elif n.leaf:
                    l1, l2 = self._two_lines(n.action)
                    head = f"P{mover + 1}: {l1}" if s == 1 else l1           # popup: the colour bar shows the player
                    rows_txt = [(head, T.PLAYER_COLORS[mover], 0.022), (l2, T.TEXT, 0.021),
                                (f"eval {fv(n.value)}", T.TEXT, 0.022),
                                (f"risk {fv(n.edge)}" if s > 1 else f"risk edge {fv(n.edge)}", T.MUTED, 0.022 if s > 1 else 0.02),
                                (f"= {fv(n.backed_up)}", T.GOLD if n.nid in pv else T.TEXT, 0.028)]
                    if n.state.winner is not None:
                        rows_txt[2] = ("WIN" if n.state.winner == rp else ("DRAW" if n.state.winner < 0 else "LOSS"),
                                       T.GOOD if n.state.winner == rp else T.DANGER, 0.02)
                else:
                    l1, l2 = self._two_lines(n.action)
                    kind = "MAX" if n.is_max else "MIN"
                    b = bound(n)
                    rows_txt = [(f"P{mover + 1}: {l1}", T.PLAYER_COLORS[mover], 0.026), (l2, T.TEXT, 0.024),
                                (f"{kind} node, window ({fv(n.alpha)}, {fv(n.beta)})", T.MUTED, 0.021),
                                (f"best reply {b}{fv(n.value)}  +  risk {fv(n.edge)}", T.TEXT, 0.023),
                                (f"= {b}{fv(n.backed_up)}", T.GOLD if n.nid in pv else T.TEXT, 0.03)]
                    if n.cutoff:
                        rows_txt.append((f"CUTOFF: {n.pruned_count} skipped", T.DANGER, 0.024))
            grow = 1.3 if (tr.depth == 1 and n.ply == 1) else 1.0     # wide boxes when only one ply is drawn
            if s > 1:
                grow = max(grow, (1.45 if n.ply else s) if w >= 0.3 else 1.3)
            y = h / 2 - 0.03 - 0.005 * grow
            for txt, c, sc in rows_txt:
                sc *= grow
                T.text(box, txt, (0, y), sc, c, "center")
                y -= sc + 0.011
        # legend
        ls_ = 0.025 if s == 1 else 0.029
        ly = -0.58 if s == 1 else -0.62
        T.text(parent, "gold = principal variation (best line for both players)    red dashed = pruned by alpha-beta"
                       "    value = risk + child value", ((x0 + x1) / 2, ly), ls_, T.MUTED, "center",
               wordwrap=(x1 - x0) / ls_)
        T.text(parent, "A cutoff happens when alpha >= beta: the opponent already has a reply that makes this shot "
                       "worse than one we found earlier, so its other replies cannot change the decision. "
                       "'<=' marks such a node: its true value is at most that number.",
               ((x0 + x1) / 2, ly - 0.06), ls_, T.MUTED, "center", wordwrap=(x1 - x0) / ls_)


class TreePopup(TreeDrawing):
    """SHOW TREE during a match: the decision tree for the most recent AI decision.

    The tree is recomputed for exactly that position with the Lab's small display
    budget (the match itself searched with the full difficulty budget), so the
    popup always states whether the tree's choice equals the shot that was played.
    """

    TREE_X = (-0.74, 1.72)
    TREE_BIG = 1.3

    def __init__(self, app, state, decision, use_fuzzy: bool, on_close) -> None:
        from ai.search_trace import MCTSStepper, trace_decision
        self.app = app
        self.state = state
        self.decision = decision
        self.use_fuzzy = use_fuzzy
        self.on_close = on_close
        self.mcts = "MCTS" in decision.algorithm
        self.root = DirectFrame(parent=app.aspect2d, frameColor=(0.02, 0.03, 0.06, 0.99), frameSize=(-3, 3, -1, 1))
        try:
            aspect = float(app.getAspectRatio())
        except Exception:
            aspect = 16 / 9
        self.root.setScale(min(1.0, max(0.6, (aspect - 0.03) / 1.62)))
        self.body = None
        if self.mcts:
            self.stepper = MCTSStepper(state, iterations=MCTS_ITERS, candidate_width=4, rollout_depth=1, seed=7,
                                       use_fuzzy=use_fuzzy, game_cfg=app.game_cfg)
            self.stepper.run(MCTS_ITERS)
        else:
            self.trace = trace_decision(state, player=decision.player, depth=2, root_width=4, inner_width=3,
                                        use_fuzzy=use_fuzzy, game_cfg=app.game_cfg)
        self._draw()

    # --- MCTS step buttons (called by TreeDrawing._draw_mcts)
    def _mcts_step(self, n: int) -> None:
        self.stepper.run(n)
        self._draw()

    def _mcts_reset(self) -> None:
        from ai.search_trace import MCTSStepper
        self.stepper = MCTSStepper(self.state, iterations=MCTS_ITERS, candidate_width=4, rollout_depth=1, seed=7,
                                   use_fuzzy=self.use_fuzzy, game_cfg=self.app.game_cfg)
        self._draw()

    def _obstacle_key(self) -> tuple:
        from game.arena import get_arena
        if get_arena(self.state.arena_id).obstacle_positions(self.state.clock):
            return (("orange ring:", (1.0, 0.75, 0.3, 1)), ("obstacle", (1.0, 0.75, 0.3, 1)))
        return ()

    def _first_p2(self) -> int:
        return next((m.mid for m in self.state.marbles if m.owner == 1), 0)

    def tree_choice(self):
        if self.mcts:
            b = self.stepper.best()
            return b.action if b is not None else None
        t = self.trace
        return t.nodes[t.best_child].action if t.best_child is not None else None

    def _draw(self) -> None:
        from ai.explain import explain
        from ai.search_trace import action_label, same_shot
        if self.body is not None:
            self.body.destroy()
        b = DirectFrame(parent=self.root, frameColor=(0, 0, 0, 0))
        self.body = b
        d = self.decision
        p = d.player
        algo = "MCTS (UCT)" if self.mcts else "Minimax + Alpha-Beta"
        T.text(b, "DECISION TREE FOR THIS MOMENT", (0, 0.88), 0.06, T.TEXT, "center", "title")
        T.text(b, f"Turn {self.state.turn + 1}  |  Player {p + 1} ({algo}) to move  |  the match is paused",
               (0, 0.82), 0.03, T.PLAYER_COLORS[p], "center")
        played, mine = d.action, self.tree_choice()
        same = same_shot(played, mine)
        X, W = -1.72, 0.9                       # left column: x and width
        y = 0.7

        def put(txt, col, sc, kind="body"):
            nonlocal y
            o = T.text(b, txt, (X, y), sc, col, kind=kind, wordwrap=W / sc)
            y -= (sc + 0.012) * self._nrows(o) + 0.006
            return o

        put("In the match the AI played:", T.MUTED, 0.032)
        put(action_label(played), (1, 1, 1, 1), 0.038)
        put("This tree chooses:", T.MUTED, 0.032)
        put(action_label(mine) if mine is not None else "run iterations first", T.GOLD, 0.038)
        put("SAME SHOT" if same else "DIFFERENT SHOT", T.GOOD if same else T.WARN, 0.042, "title")
        if not same and mine is not None:
            put("Not an error: this tree is much smaller than the match search (see SEARCH SIZE), "
                "so on a close decision it can prefer another shot.", T.WARN, 0.026)
        try:
            ex = explain(self.state, d, cfg=self.app.game_cfg)
        except Exception:
            ex = None
        if ex is not None:
            y -= 0.02
            put("WHY THE MATCH AI CHOSE IT", T.GOLD, 0.034, "title")
            mark = {"good": ("+", T.GOOD), "bad": ("-", T.DANGER), "neutral": ("~", T.WARN)}
            for r in ex.reasons[:5]:
                sym, col = mark.get(r.kind, ("~", T.TEXT))
                put(f"{sym} {r.text}", col, 0.03)
        # search size: the match search versus this (small) tree
        s = d.stats or {}
        y -= 0.02
        put("SEARCH SIZE", T.EDGE, 0.034, "title")
        if self.mcts:
            put(f"Match: {T.fmt(s.get('iterations'))} iterations, chosen shot visited "
                f"{T.fmt(s.get('selected_visits'))} times", T.TEXT, 0.029)
            put(f"This tree: {self.stepper.root.visits} of {MCTS_ITERS} iterations, 4 candidate shots", T.TEXT, 0.029)
            put("Fewer iterations, so visit counts are smaller; compare the chosen shot.", T.MUTED, 0.027)
        else:
            shown = sum(1 for n in self.trace.nodes if not n.pruned)
            put(f"Match: {T.fmt(s.get('depth_reached'))} turns ahead, {T.fmt(s.get('root_candidates'))} shots, "
                f"{T.fmt(s.get('nodes_visited'))} positions, value {T.fmt(s.get('evaluation'), 1)}", T.TEXT, 0.029)
            put(f"This tree: 2 turns ahead (my shot + reply), 4 shots x 3 replies, {shown} positions, "
                f"value {self.trace.value:+.2f}", T.TEXT, 0.029)
            put("The tree stops one turn earlier, so its values are different numbers; compare the chosen shot.",
                T.MUTED, 0.027)
        # live position of the marbles at this moment (the 3-D board is behind the popup)
        top, bottom = min(y - 0.04, -0.06), -0.8
        half = max(0.17, min(0.3, (top - bottom) / 2))
        mx, mz = X + half, top - half
        T.text(b, "BOARD NOW (top view)", (X, top + 0.012), 0.028, T.EDGE, kind="title")
        self._draw_minimap(b, self.state, played, mine, mx, mz, half, 0.03, owner_labels=True)
        lx = X + 2 * half + 0.04
        ky = mz + half - 0.04
        for txt, col in (("white line:", (1, 1, 1, 1)), ("shot played", (1, 1, 1, 1)), ("", None),
                         ("gold line:", T.GOLD), ("tree's choice", T.GOLD), ("", None),
                         (f"M0 to M{self._first_p2() - 1}:", T.PLAYER_COLORS[0]), ("Player 1", T.PLAYER_COLORS[0]),
                         ("", None),
                         (f"M{self._first_p2()} to M{len(self.state.marbles) - 1}:", T.PLAYER_COLORS[1]),
                         ("Player 2", T.PLAYER_COLORS[1]), ("", None)) + self._obstacle_key():
            if txt:
                T.text(b, txt, (lx, ky), 0.025, col)
            ky -= 0.032 if txt else 0.014
        T.button(b, "CLOSE (T / Esc)", self.on_close, pos=(-1.25, 0, -0.86), size=(0.25, 0.04), text_scale=0.03,
                 accent=T.GOLD)
        if self.mcts:
            self._draw_mcts(b, self.stepper)
        else:
            self._draw_tree(b, self.trace)

    def destroy(self) -> None:
        self.root.destroy()
