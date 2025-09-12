import warnings
from typing import Sequence

import numpy as np
from manimlib import *

from .quera_colors import *


class SingleQubitGate(VGroup):
    """
    Universal 1-qubit boxed gate (square/round/D) with optional control dot.

    type:  "H","X","Y","Z","S","T","U","SQX","SQY","SQZ","R","RX","RY","RZ"
           or any custom TeX string (interpreted as the label), or None.
    label: explicit TeX override. If provided, it wins.
    dag:   bool → adds ^{\dagger} when applicable.
    axis:  for R/RX/RY/RZ (subscript).
    angle: for R/RX/RY/RZ (if None → no parentheses).
    """

    def __init__(
        self,
        *,
        type: str | None = None,
        slot: int,
        start: np.ndarray,
        pitch: float,
        size: float = 0.5,
        dot_size: float | None = None,
        color: Color = WHITE,
        bg_color: Color = BLACK,
        shift: str | None = None,
        shape: str = "square",
        stroke_width: float | None = None,
        control_y: float | None = None,
        dag: bool = False,
        axis: str | None = None,
        angle: str | None = None,
        label: str | None = None,
        **kwargs,
    ):
        self._color = color
        self._dot_size = dot_size
        self._dag = bool(dag)
        self._gate_type = type.upper() if isinstance(type, str) else None
        sw = stroke_width if stroke_width is not None else 4 * size
        self._orig_sw = sw

        super().__init__(**kwargs)

        # 1) horizontal nudge
        off = 0.0
        if isinstance(shift, str):
            s = shift.lower()
            if s in ("right", "r"):
                off = pitch / 3
            elif s in ("left", "l"):
                off = -pitch / 3

        # 2) geometry
        pt = start + RIGHT * (pitch * slot + off)
        half = size / 2.0
        key = (shape or "square").lower()

        # 3) body
        if key == "d":
            rect_fill = (
                Rectangle(width=half, height=size)
                .move_to(pt + np.array([-half / 2, 0, 0]))
                .set_fill(bg_color, 1)
                .set_stroke(width=0)
            )
            circ_fill = (
                Circle(radius=half)
                .move_to(pt)
                .set_fill(bg_color, 1)
                .set_stroke(width=0)
            )
            fill_group = VGroup(rect_fill, circ_fill).set_z_index(0)

            left_edge = Line(
                pt + np.array([-half, -half, 0]), pt + np.array([-half, half, 0])
            )
            top_edge = Line(
                pt + np.array([-half, half, 0]), pt + np.array([0, half, 0])
            )
            bot_edge = Line(
                pt + np.array([-half, -half, 0]), pt + np.array([0, -half, 0])
            )
            semi = Arc(start_angle=PI / 2, angle=-PI, radius=half, arc_center=pt)
            outline = VGroup(left_edge, top_edge, bot_edge, semi).set_z_index(1)
            outline.set_stroke(color, sw)
            self.add(fill_group, outline)

        elif key in ("r", "round"):
            fill_circle = (
                Circle(radius=half)
                .move_to(pt)
                .set_fill(bg_color, 1)
                .set_stroke(width=0)
                .set_z_index(0)
            )
            outline = (
                Circle(radius=half)
                .move_to(pt)
                .set_stroke(color, sw)
                .set_fill(opacity=0)
                .set_z_index(1)
            )
            self.add(fill_circle, outline)

        else:  # square
            sq = (
                Square(side_length=size)
                .move_to(pt)
                .set_fill(bg_color, 1)
                .set_stroke(color, sw)
            )
            outline = sq
            self.add(sq)

        # 4) label (resolve order: explicit label → known type → R-logic → unknown type as TeX → blank)
        label_tex = self._resolve_label_tex(
            type_upper=self._gate_type,
            dag=self._dag,
            axis=axis,
            angle=angle,
            label_override=label,
        )
        if label_tex:
            # Heuristic: treat \sqrt, daggers, or explicit parentheses as “long” labels
            prefer_fill = any(
                tag in label_tex for tag in (r"\sqrt", r"^{", r"\left(", r"(")
            )
            lbl = self._place_label(label_tex, pt, size, prefer_fill=prefer_fill)
            self.add(lbl)
            self._label = lbl
        else:
            self._label = None

        # 5) optional control dot + vertical line
        if control_y is not None:
            ctrl_pt = np.array([pt[0], control_y, pt[2]])
            if self._dot_size is not None:
                r = float(self._dot_size) / 2.0
                dot = Dot(radius=r)
            else:
                dot = Dot()
                r = dot.get_width() / 2.0
            dot.set_color(color).set_fill(color, 1).move_to(ctrl_pt).set_z_index(1)

            if control_y > pt[1]:
                start_line = ctrl_pt - UP * r
                end_line = pt + UP * half
            else:
                start_line = ctrl_pt + UP * r
                end_line = pt + DOWN * half

            vline = Line(start_line, end_line).set_stroke(color, sw).set_z_index(1)
            self.add(dot, vline)
            self._control_dot = dot
            self._control_line = vline

        # 6) finalize
        self.set_z_index(2)
        self.slot = slot
        self._box = outline

    def _place_label(
        self,
        tex: str,
        center: np.ndarray,
        box_size: float,
        *,
        prefer_fill: bool = False,
    ):
        """
        Fit a TeX label inside a single-qubit box/circle/D body.
        prefer_fill=True lets long expressions (\sqrt, daggers, parentheses) use a slightly larger fill.
        """
        lbl = Tex(tex, font_size=box_size * 60)

        # available box size (single-qubit body is square-ish with side = `box_size`)
        avail_w = avail_h = float(box_size)

        # choose the target coverage:
        # - "normal" ≈ 72% of the box
        # - "fill"   ≈ 80% (for \sqrt{…}, daggers, or labels with parentheses)
        target_ratio = 0.80 if prefer_fill else 0.72

        w, h = max(lbl.get_width(), 1e-9), max(lbl.get_height(), 1e-9)
        scale = min((target_ratio * avail_w) / w, (target_ratio * avail_h) / h, 1.0)
        if scale < 1.0:
            lbl.scale(scale)

        lbl.move_to(center).set_color(self._color).set_z_index(2)
        return lbl

    # ---------- label resolver ----------
    @staticmethod
    def _resolve_label_tex(
        *,
        type_upper: str | None,
        dag: bool,
        axis: str | None,
        angle: str | None,
        label_override: str | None,
    ) -> str | None:
        # 1) explicit label wins; put dagger OUTSIDE the whole label
        if label_override is not None:
            return label_override + (r"^{\dagger}" if dag else "")

        # 2) blank if nothing specified
        if type_upper is None:
            return None

        # 3) R family: dagger belongs on R_axis; angle (if any) follows
        if type_upper in ("R", "RX", "RY", "RZ"):
            ax = axis
            if type_upper != "R":
                ax = type_upper[1]
            if ax is None:
                ax = "X"
            base = rf"R_{{{ax}}}" + (r"^{\dagger}" if dag else "")
            return base if angle is None else base + rf"\left({angle}\right)"

        # 4) sqrt family: put dagger OUTSIDE the radical
        if type_upper in ("SQX", "SQY", "SQZ"):
            core = type_upper[2]  # X/Y/Z
            tex = rf"\sqrt{{{core}}}"
            return tex + (r"^{\dagger}" if dag else "")

        # 5) simple named gates (H, X, Y, Z, S, T, U): dagger at the end
        if type_upper in ("H", "X", "Y", "Z", "S", "T", "U"):
            tex = rf"\mathit{{{type_upper}}}"
            return tex + (r"^{\dagger}" if dag else "")

        # 6) unknown token → treat as raw TeX; dagger appended OUTSIDE
        return type_upper + (r"^{\dagger}" if dag else "")

    # ---------- styling ----------
    def set_color(self, new_color: Color, bg_color: Color | None = None):
        self._color = new_color
        if isinstance(self._box, VGroup):
            for piece in self._box:
                piece.set_stroke(new_color, width=self._orig_sw)
        else:
            self._box.set_stroke(new_color, width=self._orig_sw)
        if bg_color is not None:
            for mob in self.submobjects:
                try:
                    mob.set_fill(bg_color, 1)
                except Exception:
                    pass
        if getattr(self, "_label", None) is not None:
            self._label.set_color(new_color)
        if hasattr(self, "_control_dot"):
            self._control_dot.set_color(new_color)
            self._control_line.set_stroke(new_color, self._orig_sw)
        return self

    def get_color(self) -> Color:
        return self._color

    def get_backgroundless(self, pulse_color, stroke_width=1):
        parts = []
        if isinstance(self._box, VGroup):
            for piece in self._box:
                parts.append(
                    piece.copy()
                    .set_fill(opacity=0)
                    .set_stroke(pulse_color, stroke_width)
                )
        else:
            parts.append(
                self._box.copy()
                .set_fill(opacity=0)
                .set_stroke(pulse_color, stroke_width)
            )
        if getattr(self, "_label", None) is not None:
            parts.append(
                self._label.copy()
                .set_fill(opacity=0)
                .set_stroke(pulse_color, stroke_width)
            )
        if hasattr(self, "_control_dot"):
            parts.extend(
                [
                    self._control_dot.copy()
                    .set_fill(opacity=0)
                    .set_stroke(pulse_color, stroke_width),
                    self._control_line.copy().set_stroke(pulse_color, stroke_width),
                ]
            )
        return VGroup(*parts)


class CustomGate(VGroup):
    """
    A labeled gate that can be:
      • Single-slot (square/round/D), or
      • Multi-slot rectangle spanning [start..end] with extra downward coverage,
        and/or vertically spanning multiple wires via `gate_height`.

    Parameters
    ----------
    slot : int | tuple[int,int]
        Slot index, or inclusive (start,end) slot range.
    start : np.ndarray
        Left endpoint of the target wire (contains the wire's y).
    pitch : float
        Slot spacing.
    size : float
        Base side/height for a single-qubit body (and the baseline height for the rectangle).
    color : Color
        Stroke/text color.
    bg_color : Color
        Fill color.
    shift : str | None
        "left"/"l" or "right"/"r" (± pitch/3) like other gates.
    shape : str
        For single-slot with gate_height==1: "square" (default), "round"/"R", or "D".
        For multi-slot: if (slot_span == gate_height) you may also use "round"/"R" or "D"
        to draw a square round/D body; else a rectangle is used (with a warning).
    stroke_width : float | None
        Outline stroke width; if None → 4*size.
    control_y : float | None
        Optional control dot y with a vertical connector to the body.
    dot_size : float | None
        Optional DIAMETER for the control dot. If None, use the default Dot() radius.
    name : str | Tex
        The label to draw inside (string or a prebuilt Tex).
    wire_spacing : float | None
        Spacing between wires for vertical extension; defaults to 1.2*pitch if None.
    gate_height : int
        Number of wires the gate spans vertically. The bottom edge extends
        downward by `wire_spacing * (gate_height - 1)`. Must be ≥ 1.
    label_size : "normal" | "fill"
        "normal" → Hadamard-like sizing (fit to ~70% if needed).
        "fill"   → scale up to the maximum that fits within 80% of both width & height.
    """

    def __init__(
        self,
        *,
        slot: int | tuple[int, int],
        start: np.ndarray,
        pitch: float,
        size: float = 0.5,
        color: Color = WHITE,
        bg_color: Color = BLACK,
        shift: str | None = None,
        shape: str = "square",
        stroke_width: float | None = None,
        control_y: float | None = None,
        dot_size: float | None = None,
        name: "str | Tex" = r"\mathit{H}",
        wire_spacing: float | None = None,
        gate_height: int = 1,
        label_size: str = "normal",
        **kwargs,
    ):

        self._color = color
        self._dot_size = dot_size  # <-- new (diameter)
        sw = stroke_width if stroke_width is not None else 4 * size
        self._orig_sw = sw
        super().__init__(**kwargs)

        # ---- helpers --------------------------------------------------------
        def build_label_obj(name_obj):
            return (
                Tex(name_obj, font_size=size * 60)
                if isinstance(name_obj, str)
                else name_obj.copy()
            )

        def place_label(label, center_xy, avail_w, avail_h, mode: str):
            label.set_color(self._color)
            w, h = label.get_width(), label.get_height()
            if mode.lower() == "fill":
                if max(w, h) > 0:
                    scale = min(
                        (0.8 * avail_w) / max(w, 1e-9), (0.8 * avail_h) / max(h, 1e-9)
                    )
                    label.scale(scale)
            else:
                target = 0.70 * min(avail_w, avail_h)
                if max(w, h) > target > 0:
                    label.scale(target / max(w, h))
            label.move_to(center_xy).set_z_index(2)
            return label

        def make_control_dot_and_line(
            ctrl_xy: np.ndarray, top_y: float, bot_y: float, center_y: float
        ):
            """Create a control dot (using self._dot_size if provided) and a vertical line to the body."""
            # dot with explicit radius if diameter given
            if self._dot_size is not None:
                r = float(self._dot_size) / 2.0
                dot = Dot(radius=r)
            else:
                dot = Dot()
                r = dot.get_width() / 2.0
            dot.set_color(color).set_fill(color, 1).move_to(ctrl_xy).set_z_index(3)

            if ctrl_xy[1] > center_y:
                start_line = ctrl_xy - UP * r
                end_line = np.array([ctrl_xy[0], top_y, ctrl_xy[2]])
            else:
                start_line = ctrl_xy + UP * r
                end_line = np.array([ctrl_xy[0], bot_y, ctrl_xy[2]])

            vline = Line(start_line, end_line).set_stroke(color, sw).set_z_index(2)
            return dot, vline

        # ---------------------------------------------------------------------
        gate_height = max(1, int(gate_height))
        spacing = (1.2 * pitch) if wire_spacing is None else float(wire_spacing)
        extra_dn = spacing * (gate_height - 1)

        # nudge
        off = 0.0
        if shift and shift.lower() in ("right", "r"):
            off = pitch / 3
        elif shift and shift.lower() in ("left", "l"):
            off = -pitch / 3

        self.slot = slot
        self._size = size

        def x_at(s: int) -> float:
            return (start + RIGHT * (pitch * s + off))[0]

        wire_y = float(start[1])
        key = (shape or "square").lower()
        is_range = isinstance(slot, (tuple, list)) and len(slot) == 2

        # ---------- MULTI-SLOT ----------
        if is_range:
            s0, s1 = int(slot[0]), int(slot[1])
            if s1 < s0:
                s0, s1 = s1, s0

            buf = size / 2
            xL = x_at(s0) - buf
            xR = x_at(s1) + buf
            width = max(1e-6, xR - xL)

            top_y = wire_y + size / 2
            bot_y = wire_y - size / 2 - extra_dn
            height = max(1e-6, top_y - bot_y)

            center = np.array([(xL + xR) / 2, (top_y + bot_y) / 2, 0.0])

            slot_span = s1 - s0 + 1
            is_square_span = slot_span == gate_height

            drew_nonrect = False
            if key in ("r", "round", "d"):
                if is_square_span:
                    # square body → circle or D with side = min(width, height)
                    side = min(width, height)
                    half = side / 2.0
                    if key in ("r", "round"):
                        fill_circle = (
                            Circle(radius=half)
                            .move_to(center)
                            .set_fill(bg_color, 1)
                            .set_stroke(width=0)
                            .set_z_index(0)
                        )
                        outline = (
                            Circle(radius=half)
                            .move_to(center)
                            .set_fill(opacity=0)
                            .set_stroke(color, sw)
                            .set_z_index(1)
                        )
                        self.add(fill_circle, outline)
                        self._box = outline
                    else:
                        rect_fill = (
                            Rectangle(width=half, height=side)
                            .move_to(center + np.array([-half / 2, 0, 0]))
                            .set_fill(bg_color, 1)
                            .set_stroke(width=0)
                        )
                        circ_fill = (
                            Circle(radius=half)
                            .move_to(center)
                            .set_fill(bg_color, 1)
                            .set_stroke(width=0)
                        )
                        fill_group = VGroup(rect_fill, circ_fill).set_z_index(0)

                        left_edge = Line(
                            center + np.array([-half, -half, 0]),
                            center + np.array([-half, half, 0]),
                        )
                        top_edge = Line(
                            center + np.array([-half, half, 0]),
                            center + np.array([0, half, 0]),
                        )
                        bot_edge = Line(
                            center + np.array([-half, -half, 0]),
                            center + np.array([0, -half, 0]),
                        )
                        semi = Arc(
                            start_angle=PI / 2,
                            angle=-PI,
                            radius=half,
                            arc_center=center,
                        )
                        outline = VGroup(
                            left_edge, top_edge, bot_edge, semi
                        ).set_z_index(1)
                        outline.set_stroke(color, sw)

                        self.add(fill_group, outline)
                        self._box = outline

                    # label
                    lbl = build_label_obj(name)
                    lbl = place_label(lbl, center, side, side, label_size)
                    self.add(lbl)
                    self._label = lbl

                    # control
                    if control_y is not None:
                        ctrl_pt = np.array([center[0], control_y, 0.0])
                        dot, vline = make_control_dot_and_line(
                            ctrl_pt,
                            top_y=center[1] + half,
                            bot_y=center[1] - half,
                            center_y=center[1],
                        )
                        self.add(dot, vline)
                        self._control_dot = dot
                        self._control_line = vline

                    self.set_z_index(2)
                    drew_nonrect = True
                else:
                    warnings.warn(
                        "CustomGate: 'round'/'d' shapes for multi-slot are only supported "
                        "when (slot_span == gate_height). Falling back to rectangular body."
                    )

            if not drew_nonrect:
                # rectangle
                fill_rect = (
                    Rectangle(width=width, height=height)
                    .move_to(center)
                    .set_fill(bg_color, 1)
                    .set_stroke(width=0)
                    .set_z_index(0)
                )
                outline = (
                    Rectangle(width=width, height=height)
                    .move_to(center)
                    .set_fill(opacity=0)
                    .set_stroke(color, sw)
                    .set_z_index(1)
                )
                self.add(fill_rect, outline)
                self._box = outline

                lbl = build_label_obj(name)
                lbl = place_label(lbl, center, width, height, label_size)
                self.add(lbl)
                self._label = lbl

                if control_y is not None:
                    ctrl_pt = np.array([center[0], control_y, 0.0])
                    dot, vline = make_control_dot_and_line(
                        ctrl_pt, top_y=top_y, bot_y=bot_y, center_y=center[1]
                    )
                    self.add(dot, vline)
                    self._control_dot = dot
                    self._control_line = vline

                self.set_z_index(2)
            return

        # ---------- SINGLE-SLOT ----------
        s = int(slot)
        pt = start + RIGHT * (pitch * s + off)
        half = size / 2

        if gate_height > 1:
            # tall rectangle
            top_y = wire_y + half
            bot_y = wire_y - half - extra_dn
            height = max(1e-6, top_y - bot_y)
            width = size
            center = np.array([pt[0], (top_y + bot_y) / 2, pt[2]])

            fill_rect = (
                Rectangle(width=width, height=height)
                .move_to(center)
                .set_fill(bg_color, 1)
                .set_stroke(width=0)
                .set_z_index(0)
            )
            outline = (
                Rectangle(width=width, height=height)
                .move_to(center)
                .set_fill(opacity=0)
                .set_stroke(color, sw)
                .set_z_index(1)
            )
            self.add(fill_rect, outline)
            self._box = outline

            lbl = build_label_obj(name)
            lbl = place_label(lbl, center, width, height, label_size)
            self.add(lbl)
            self._label = lbl

            if control_y is not None:
                ctrl_pt = np.array([pt[0], control_y, pt[2]])
                dot, vline = make_control_dot_and_line(
                    ctrl_pt, top_y=top_y, bot_y=bot_y, center_y=center[1]
                )
                self.add(dot, vline)
                self._control_dot = dot
                self._control_line = vline

            self.set_z_index(2)
            return

        # classic single-qubit shapes
        key = key.lower()
        if key == "d":
            rect_fill = (
                Rectangle(width=half, height=size)
                .move_to(pt + np.array([-half / 2, 0, 0]))
                .set_fill(bg_color, 1)
                .set_stroke(width=0)
            )
            circ_fill = (
                Circle(radius=half)
                .move_to(pt)
                .set_fill(bg_color, 1)
                .set_stroke(width=0)
            )
            fill_group = VGroup(rect_fill, circ_fill).set_z_index(0)

            left_edge = Line(
                pt + np.array([-half, -half, 0]), pt + np.array([-half, half, 0])
            )
            top_edge = Line(
                pt + np.array([-half, half, 0]), pt + np.array([0, half, 0])
            )
            bot_edge = Line(
                pt + np.array([-half, -half, 0]), pt + np.array([0, -half, 0])
            )
            semi = Arc(start_angle=PI / 2, angle=-PI, radius=half, arc_center=pt)
            outline = (
                VGroup(left_edge, top_edge, bot_edge, semi)
                .set_z_index(1)
                .set_stroke(color, sw)
            )
            self.add(fill_group, outline)
            self._box = outline
            avail_w = avail_h = size

        elif key in ("r", "round"):
            fill_circle = (
                Circle(radius=half)
                .move_to(pt)
                .set_fill(bg_color, 1)
                .set_stroke(width=0)
                .set_z_index(0)
            )
            outline = (
                Circle(radius=half)
                .move_to(pt)
                .set_fill(opacity=0)
                .set_stroke(color, sw)
                .set_z_index(1)
            )
            self.add(fill_circle, outline)
            self._box = outline
            avail_w = avail_h = size

        else:  # square
            sq = (
                Square(side_length=size)
                .move_to(pt)
                .set_fill(bg_color, 1)
                .set_stroke(color, sw)
            )
            self.add(sq)
            self._box = sq
            avail_w = avail_h = size

        lbl = build_label_obj(name)
        lbl = place_label(lbl, pt, avail_w, avail_h, label_size)
        self.add(lbl)
        self._label = lbl

        if control_y is not None:
            ctrl_pt = np.array([pt[0], control_y, pt[2]])
            # top/bot edges of a single-qubit box:
            top_y = pt[1] + half
            bot_y = pt[1] - half
            dot, vline = make_control_dot_and_line(
                ctrl_pt, top_y=top_y, bot_y=bot_y, center_y=pt[1]
            )
            self.add(dot, vline)
            self._control_dot = dot
            self._control_line = vline

        self.set_z_index(2)

    # ----- Coloring / pulse helpers -----
    def set_color(self, new_color: Color, bg_color: Color | None = None):
        self._color = new_color
        if isinstance(self._box, VGroup):
            for piece in self._box:
                piece.set_stroke(new_color, width=self._orig_sw)
        else:
            self._box.set_stroke(new_color, width=self._orig_sw)
        if bg_color is not None:
            for mob in self.submobjects:
                try:
                    mob.set_fill(bg_color, 1)
                except Exception:
                    pass
        if hasattr(self, "_label"):
            self._label.set_color(new_color)
        if hasattr(self, "_control_dot"):
            self._control_dot.set_color(new_color).set_fill(new_color, 1)
            self._control_line.set_stroke(new_color, self._orig_sw)
        return self

    def get_color(self) -> Color:
        return self._color

    def get_backgroundless(self, pulse_color, stroke_width=1):
        parts = []
        if isinstance(self._box, VGroup):
            for piece in self._box:
                parts.append(
                    piece.copy()
                    .set_fill(opacity=0)
                    .set_stroke(pulse_color, stroke_width)
                )
        else:
            parts.append(
                self._box.copy()
                .set_fill(opacity=0)
                .set_stroke(pulse_color, stroke_width)
            )
        if hasattr(self, "_label"):
            parts.append(
                self._label.copy()
                .set_fill(opacity=0)
                .set_stroke(pulse_color, stroke_width)
            )
        if hasattr(self, "_control_dot"):
            parts.append(
                self._control_dot.copy()
                .set_fill(opacity=0)
                .set_stroke(pulse_color, stroke_width)
            )
            parts.append(
                self._control_line.copy().set_stroke(pulse_color, stroke_width)
            )
        return VGroup(*parts)


class CNOTGate(VGroup):
    """
    Two-wire CNOT at discrete `slot`.

    Control wire: filled dot.
    Target wire : unfilled circle + plus (two lines), with a filled background
                  circle (bg_color) to mask the underlying wire.

    Notes
    -----
    - The target-side circle/plus uses a fixed radius of pitch/3 (independent of dot_size).
    - The control-side dot uses `dot_size` (DIAMETER) if provided.
    """

    def __init__(
        self,
        *,
        slot: int,
        start_control: np.ndarray,
        start_target: np.ndarray,
        pitch: float,
        color: Color = WHITE,
        bg_color: Color = BLACK,
        stroke_width: float | None = None,
        dot_size: float | None = None,  # DIAMETER for the control dot only
        shift: str | None = None,
        **kwargs,
    ):
        # style
        self._color = color
        self._bg_color = bg_color
        self._dot_size = dot_size  # DIAMETER (may be None)
        sw = 2 if stroke_width is None else stroke_width
        self._sw = sw

        super().__init__(**kwargs)

        # horizontal nudge
        off = 0
        if shift and shift.lower() in ("right", "r"):
            off = pitch / 3
        elif shift and shift.lower() in ("left", "l"):
            off = -pitch / 3

        # positions (dot on control, circle+plus on target)
        pos_ctrl = start_control + RIGHT * (pitch * slot + off)
        pos_tgt = start_target + RIGHT * (pitch * slot + off)

        # control dot (filled), using explicit radius if dot_size provided
        if self._dot_size is not None:
            r_dot = float(self._dot_size) / 2.0
            self._dot = (
                Dot(radius=r_dot)
                .set_color(color)
                .set_fill(color, 1)
                .move_to(pos_ctrl)
                .set_z_index(1)
            )
        else:
            self._dot = Dot(color=color).move_to(pos_ctrl).set_z_index(1)
            r_dot = self._dot.get_width() / 2

        # target circle/plus radius is FIXED from pitch (not dot_size)
        r_plus = pitch / 4.0

        # background circle (filled, no stroke) — masks the wire behind the plus
        self._bgcirc = (
            Circle(radius=r_plus)
            .move_to(pos_tgt)
            .set_fill(bg_color, 1)
            .set_stroke(width=0)
            .set_z_index(0)
        )

        # outline circle (unfilled)
        self._circle = (
            Circle(radius=r_plus)
            .move_to(pos_tgt)
            .set_fill(opacity=0)
            .set_stroke(color, sw)
            .set_z_index(1)
        )

        # plus = two lines
        self._vplus = (
            Line(pos_tgt + UP * r_plus, pos_tgt + DOWN * r_plus)
            .set_stroke(color, sw)
            .set_z_index(1)
        )
        self._hplus = (
            Line(pos_tgt + LEFT * r_plus, pos_tgt + RIGHT * r_plus)
            .set_stroke(color, sw)
            .set_z_index(1)
        )

        # connector: from dot edge to circle edge
        vec = pos_tgt - pos_ctrl
        nrm = np.linalg.norm(vec)
        unit = vec / nrm if nrm else UP
        start_pt = pos_ctrl + unit * r_dot
        end_pt = pos_tgt - unit * r_plus
        self._line = Line(
            start_pt, end_pt, stroke_color=color, stroke_width=sw
        ).set_z_index(1)

        # assemble
        self.add(
            self._bgcirc, self._circle, self._vplus, self._hplus, self._line, self._dot
        )
        self.set_z_index(1)
        self.slot = slot

    def set_color(self, new_color: Color, bg_color: Color | None = None):
        """Recolor dot, circle/plus, connector; optionally recolor background fill."""
        self._color = new_color
        if bg_color is not None:
            self._bg_color = bg_color
            self._bgcirc.set_fill(bg_color, 1)

        self._dot.set_fill(new_color, 1).set_color(new_color)
        self._circle.set_stroke(new_color, self._sw)
        self._vplus.set_stroke(new_color, self._sw)
        self._hplus.set_stroke(new_color, self._sw)
        self._line.set_stroke(new_color, self._sw)
        return self

    def get_color(self) -> Color:
        return self._color

    def get_backgroundless(self, pulse_color, stroke_width=1):
        """
        Return copies of strokes only (no background circle), tinted for pulse:
        dot outline, connector, circle outline, and plus lines.
        """
        dot_outline = (
            self._dot.copy().set_fill(opacity=0).set_stroke(pulse_color, stroke_width)
        )
        circ = (
            self._circle.copy()
            .set_fill(opacity=0)
            .set_stroke(pulse_color, stroke_width)
        )
        v = self._vplus.copy().set_stroke(pulse_color, stroke_width)
        h = self._hplus.copy().set_stroke(pulse_color, stroke_width)
        line = self._line.copy().set_stroke(pulse_color, stroke_width)
        return VGroup(dot_outline, line, circ, v, h)


class CZGate(VGroup):
    """
    A two-wire CZ at discrete `slot` using `start_control`, `start_target`, and `pitch`.

    Control: filled dot
    Target : filled dot
    Connector line runs between the **edges** of the dots.

    Parameters
    ----------
    slot : int
        Which slot to place the gate at.
    start_control : np.ndarray
        Left endpoint of the control wire.
    start_target : np.ndarray
        Left endpoint of the target wire.
    pitch : float
        Distance between adjacent slots.
    color : Color
        Stroke/text color.
    stroke_width : float
        Width of the connector line (defaults to 2).
    dot_size : float | None
        Optional DIAMETER for both dots. If None, use Manim Dot() default size.
    shift : str | None
        Optional horizontal shift: "left"/"l" or "right"/"r" to move gate by ±(pitch/3).
    """

    def __init__(
        self,
        *,
        slot: int,
        start_control: np.ndarray,
        start_target: np.ndarray,
        pitch: float,
        color: Color = WHITE,
        stroke_width: float = 2,
        dot_size: float | None = None,
        shift: str | None = None,
        **kwargs,
    ):
        # 1) stash style before super()
        self._color = color
        self._stroke_width = stroke_width
        self._dot_size = dot_size  # DIAMETER (may be None)

        super().__init__(**kwargs)

        # 2) horizontal offset
        off = 0
        if shift and shift.lower() in ("right", "r"):
            off = pitch / 3
        elif shift and shift.lower() in ("left", "l"):
            off = -pitch / 3

        # 3) positions
        pos_c = start_control + RIGHT * (pitch * slot + off)
        pos_t = start_target + RIGHT * (pitch * slot + off)

        # 4) dots (explicit radius if dot_size provided)
        if self._dot_size is not None:
            r = float(self._dot_size) / 2.0
            self._dot_c = (
                Dot(radius=r)
                .set_color(color)
                .set_fill(color, 1)
                .move_to(pos_c)
                .set_z_index(1)
            )
            self._dot_t = (
                Dot(radius=r)
                .set_color(color)
                .set_fill(color, 1)
                .move_to(pos_t)
                .set_z_index(1)
            )
            r_c = r_t = r
        else:
            self._dot_c = Dot(color=color).move_to(pos_c).set_z_index(1)
            self._dot_t = Dot(color=color).move_to(pos_t).set_z_index(1)
            r_c = self._dot_c.get_width() / 2
            r_t = self._dot_t.get_width() / 2

        # 5) connector between dot *edges*
        vec = pos_t - pos_c
        nrm = np.linalg.norm(vec)
        unit = vec / nrm if nrm else RIGHT
        start_pt = pos_c + unit * r_c
        end_pt = pos_t - unit * r_t
        self._line = Line(
            start_pt, end_pt, stroke_color=color, stroke_width=stroke_width
        ).set_z_index(1)

        # 6) assemble
        self.add(self._dot_c, self._line, self._dot_t)
        self.set_z_index(1)
        self.slot = slot

    def set_color(self, new_color: Color):
        """Recolor both dots and the connector."""
        self._color = new_color
        self._dot_c.set_fill(new_color, 1).set_color(new_color)
        self._dot_t.set_fill(new_color, 1).set_color(new_color)
        self._line.set_stroke(new_color, self._stroke_width)
        return self

    def get_color(self) -> Color:
        return self._color

    def get_backgroundless(self, pulse_color, stroke_width=1):
        """Return just dot outlines + connector, tinted."""
        dot_c = (
            self._dot_c.copy().set_fill(opacity=0).set_stroke(pulse_color, stroke_width)
        )
        dot_t = (
            self._dot_t.copy().set_fill(opacity=0).set_stroke(pulse_color, stroke_width)
        )
        connector = self._line.copy().set_stroke(pulse_color, stroke_width)
        return VGroup(dot_c, connector, dot_t)


class SwapGate(VGroup):
    """
    A two-wire SWAP at discrete `slot` using `start_control`, `start_target`, and `pitch`.
    Draws the two little '×'s as crossed lines (no filled Tex), so ghost-glow never
    shows any interior shading.

    Parameters
    ----------
    slot : int
        Which slot to place the gate at.
    start_control : np.ndarray
        Left endpoint of the control wire.
    start_target : np.ndarray
        Left endpoint of the target wire.
    pitch : float
        Distance between adjacent slots.
    size : float
        Overall visual scale of the little '×'s.
    color : Color
        Stroke/text color.
    stroke_width : float
        Width of all strokes.
    shift : str | None
        Optional horizontal shift: "left"/"l" or "right"/"r".
    """

    def __init__(
        self,
        *,
        slot: int,
        start_control: np.ndarray,
        start_target: np.ndarray,
        pitch: float,
        size: float = 0.5,
        color: Color = WHITE,
        stroke_width: float | None = None,
        shift: str | None = None,
        **kwargs,
    ):
        # 0) stash color early so init_colors() can find it
        self._color = color
        sw = stroke_width if stroke_width is not None else 4 * size
        self._sw = sw

        super().__init__(**kwargs)

        # 1) optional nudge
        off = 0
        if shift and shift.lower() in ("right", "r"):
            off = pitch / 3
        elif shift and shift.lower() in ("left", "l"):
            off = -pitch / 3

        # 2) compute positions
        pos_c = start_control + RIGHT * (pitch * slot + off)
        pos_t = start_target + RIGHT * (pitch * slot + off)

        # 3) build the two '×' shapes as pure Lines
        cross_size = 0.3 * size
        d = cross_size / 2
        # control-X
        c1 = Line(pos_c + UP * d + LEFT * d, pos_c + DOWN * d + RIGHT * d)
        c2 = Line(pos_c + DOWN * d + LEFT * d, pos_c + UP * d + RIGHT * d)
        x_ctrl = VGroup(c1, c2).set_stroke(color, sw).set_z_index(1)
        # target-X
        t1 = Line(pos_t + UP * d + LEFT * d, pos_t + DOWN * d + RIGHT * d)
        t2 = Line(pos_t + DOWN * d + LEFT * d, pos_t + UP * d + RIGHT * d)
        x_tgt = VGroup(t1, t2).set_stroke(color, sw).set_z_index(1)

        # 4) connector line
        connector = Line(pos_c, pos_t, stroke_color=color, stroke_width=sw).set_z_index(
            1
        )

        # 5) assemble
        self.add(x_ctrl, connector, x_tgt)
        self.slot = slot

        # stash for recoloring
        self._crosses = (x_ctrl, x_tgt)
        self._connector = connector

    def set_color(self, new_color: Color):
        """
        Change the gate’s color (both ×’s and connector) to `new_color`.
        """
        self._color = new_color
        for cross in self._crosses:
            cross.set_stroke(new_color, self._sw)
        self._connector.set_stroke(new_color, self._sw)
        return self

    def get_color(self) -> Color:
        """Return the current gate color."""
        return self._color


class MeasurementGate(VGroup):
    """
    A 1-qubit measurement gate at discrete `slot` using `start` and `pitch`,
    optionally controlled by a dot on another wire.

    Parameters
    ----------
    slot : int
        Which slot to place the gate at.
    start : np.ndarray
        Left endpoint of the wire.
    pitch : float
        Distance between adjacent slots.
    size : float
        Side length of the square gate.
    color : Color
        Stroke/text color.
    bg_color : Color
        Fill color of the square.
    stroke_width : float | None
        Width of strokes. If None, defaults to 4*size.
    label : bool | str
        False for no label, True for wire-index label, or custom TeX string.
    wire_index : int
        Index of the wire (for default label).
    shift : str | None
        Optional horizontal shift: "left"/"l" or "right"/"r".
    axis : str | None
        Optional axis indicator: one of "X","Y","Z" to draw that letter inside
        the bottom-right of the measurement box.
    control_y : float | None
        If given, draw a control dot at this y and a vertical line (or two lines)
        connecting it to the measurement box.
    control_type : str
        "Quantum" (default) → one line; "Classical" or "C" → two parallel lines.
    dot_size : float | None
        **Diameter** of the control dot (and used for geometry); if None, uses Manim default.
    """

    def __init__(
        self,
        *,
        slot: int,
        start: np.ndarray,
        pitch: float,
        size: float = 0.5,
        color: Color = WHITE,
        bg_color: Color = BLACK,
        stroke_width: float | None = None,
        label: bool | str = False,
        wire_index: int = None,
        shift: str | None = None,
        axis: str | None = None,
        control_y: float | None = None,
        control_type: str = "Quantum",
        dot_size: float | None = None,
        **kwargs,
    ):
        # 0) determine stroke width and stash color before super()
        if stroke_width is None:
            stroke_width = 4 * size
        self._sw = stroke_width
        self._color = color
        self._control_y = control_y
        self._dot_size = dot_size  # DIAMETER (may be None)
        super().__init__(**kwargs)

        # 1) horizontal nudge
        off = 0
        if shift and shift.lower() in ("right", "r"):
            off = pitch / 3
        elif shift and shift.lower() in ("left", "l"):
            off = -pitch / 3

        # 2) compute center point
        pt = start + RIGHT * (pitch * slot + off)
        self.slot = slot
        self._size = size
        self.wire_index = wire_index

        # small lever offset if we're drawing the switch arc
        lever_offset = UP * (0.1 * size) if axis is not None else ORIGIN

        # 3) draw box, arc, and lever‐lines
        self.box = (
            Square(side_length=size)
            .set_stroke(color, stroke_width)
            .set_fill(bg_color, 1)
            .move_to(pt)
        )
        self.arc = (
            Arc(start_angle=PI / 4, angle=PI / 2)
            .set_width(0.7 * size)
            .set_stroke(color, stroke_width)
            .move_to(pt + lever_offset)
        )
        self.lines = (
            VGroup(Line(ORIGIN, 0.2 * UL), Line(ORIGIN, 0.2 * UR))
            .set_stroke(color, stroke_width)
            .move_to(pt + lever_offset)
        )
        # hide the second tick until we switch
        self.lines[1].set_opacity(0)

        # 4) optional wire‐index or custom label
        self.label_mob = None
        if label:
            txt = label if isinstance(label, str) else rf"q_{{{wire_index+1}}}"
            lab = Tex(txt, font_size=size * 40, color=color)
            lab.next_to(self.box, RIGHT, buff=0.05).shift(DOWN * (0.15 * size))
            lab.set_z_index(2)
            self.add(lab)
            self.label_mob = lab

        # assemble core shapes
        self.add(self.box, self.arc, self.lines)

        # 5) optional axis letter inside bottom‐right
        self.letter = None
        if isinstance(axis, str) and axis.upper() in ("X", "Y", "Z"):
            letter = Tex(axis.upper(), font_size=size * 40)
            letter.set_color(color).set_opacity(1).set_z_index(3)
            margin = 0.1 * size
            corner = self.box.get_corner(DR)
            w_, h_ = letter.get_width(), letter.get_height()
            target = (
                corner + UP * margin + LEFT * margin + LEFT * (w_ / 2) + UP * (h_ / 2)
            )
            letter.move_to(target)
            self.add(letter)
            self.letter = letter

        # 6) optional control dot + line(s)
        self._control_lines = VGroup()
        self._control_dot = None
        if control_y is not None:
            ctrl_pt = np.array([pt[0], control_y, pt[2]])
            # build dot with explicit radius when dot_size provided
            if self._dot_size is not None:
                r = float(self._dot_size) / 2.0
                dot = (
                    Dot(radius=r)
                    .set_color(color)
                    .set_fill(color, 1)
                    .move_to(ctrl_pt)
                    .set_z_index(4)
                )
            else:
                dot = Dot().set_color(color).move_to(ctrl_pt).set_z_index(4)
                r = dot.get_width() / 2

            half = size / 2
            kind = control_type.lower()

            if kind in ("classical", "c"):
                # two parallel lines tangent to the dot at x ± dx
                dx = 0.05 * size
                r_vert = float(np.sqrt(max(r * r - dx * dx, 0.0)))  # sqrt(r^2 - dx^2)
                if control_y > pt[1]:
                    base = ctrl_pt - UP * r_vert
                    end_box = pt + UP * half
                else:
                    base = ctrl_pt + UP * r_vert
                    end_box = pt + DOWN * half

                for sign in (-1, 1):
                    ln = Line(
                        base + RIGHT * (sign * dx),
                        end_box + RIGHT * (sign * dx),
                        stroke_color=color,
                        stroke_width=stroke_width,
                    ).set_z_index(3)
                    self._control_lines.add(ln)
            else:
                # single quantum-style line (no horizontal offset)
                if control_y > pt[1]:
                    start_box = ctrl_pt - UP * r
                    end_box = pt + UP * half
                else:
                    start_box = ctrl_pt + UP * r
                    end_box = pt + DOWN * half

                ln = Line(
                    start_box, end_box, stroke_color=color, stroke_width=stroke_width
                ).set_z_index(3)
                self._control_lines.add(ln)

            self.add(dot, self._control_lines)
            self._control_dot = dot

        # 7) finalize z‐index (keep above wires/labels)
        self.set_z_index(2)

    def set_color(self, new_color: Color):
        """Recolor box, arc, ticks, optional label/axis, and control if present."""
        self._color = new_color
        self.box.set_stroke(new_color, self._sw)
        self.arc.set_stroke(new_color, self._sw)
        self.lines.set_stroke(new_color, self._sw)
        if self.label_mob is not None:
            self.label_mob.set_color(new_color)
        if self.letter is not None:
            self.letter.set_color(new_color)
        if self._control_y is not None:
            # dot fill + outline, and the connector lines
            self._control_dot.set_fill(new_color, 1).set_color(new_color)
            for ln in self._control_lines:
                ln.set_stroke(new_color, self._sw)
        return self

    def get_color(self) -> Color:
        return self._color

    def get_backgroundless(self, pulse_color, stroke_width=1):
        """Return only stroke‐outlines (box, lever, ticks, label, axis, control)."""
        parts: list = []
        for sub in (self.box, self.arc, *self.lines):
            parts.append(
                sub.copy().set_fill(opacity=0).set_stroke(pulse_color, stroke_width)
            )
        if self.label_mob:
            parts.append(
                self.label_mob.copy()
                .set_fill(opacity=0)
                .set_stroke(pulse_color, stroke_width)
            )
        if self.letter:
            parts.append(
                self.letter.copy()
                .set_fill(opacity=0)
                .set_stroke(pulse_color, stroke_width)
            )
        if self._control_y is not None:
            parts.append(
                self._control_dot.copy()
                .set_fill(opacity=0)
                .set_stroke(pulse_color, stroke_width)
            )
            for ln in self._control_lines:
                parts.append(
                    ln.copy().set_fill(opacity=0).set_stroke(pulse_color, stroke_width)
                )
        return VGroup(*parts)


class PhaseGate(VGroup):
    """
    A one–wire phase marker: a filled dot at (wire, slot) with a TeX label above-right.
    """

    def __init__(
        self,
        *,
        slot: int,
        start: np.ndarray,
        pitch: float,
        label: str | Tex = r"\alpha",
        dot_size: float | None = None,  # DIAMETER
        color: Color = WHITE,
        **kwargs,
    ):
        # --- STASH FIRST (Manim calls get_color() inside super().__init__) ---
        self.slot = slot
        self._color = color
        self._dot_size = float(dot_size) if dot_size is not None else 0.15

        super().__init__(**kwargs)

        # Center point on the wire
        pt = start + RIGHT * (pitch * slot)

        # Dot using DIAMETER→radius
        r = self._dot_size / 2.0
        dot = (
            Dot(radius=r)
            .set_color(self._color)
            .set_fill(self._color, 1)
            .move_to(pt)
            .set_z_index(3)
        )

        # Label (accept str or Tex)
        if isinstance(label, Tex):
            lbl = label.copy().set_color(self._color)
        else:
            # readable heuristic
            fs = int(max(self._dot_size * 150.0, pitch * 20.0))
            lbl = Tex(label, font_size=fs).set_color(self._color)

        # Place label up-and-right from the dot
        margin = max(pitch * 0.07, self._dot_size * 0.18)
        lbl.next_to(dot, direction=UP + RIGHT, buff=margin).set_z_index(3)

        self.dot = dot
        self.label = lbl
        self.add(dot, lbl)

    def set_color(self, new_color: Color):
        self._color = new_color
        self.dot.set_fill(new_color, 1).set_color(new_color)
        self.label.set_color(new_color)
        return self

    def get_color(self) -> Color:
        return self._color

    def get_backgroundless(self, pulse_color, stroke_width=1):
        d = self.dot.copy().set_fill(opacity=0).set_stroke(pulse_color, stroke_width)
        l = self.label.copy().set_fill(opacity=0).set_stroke(pulse_color, stroke_width)
        d.move_to(self.dot.get_center())
        l.move_to(self.label.get_center())
        return VGroup(d, l)


class WireBundle(VGroup):
    """
    A “bundle” marker on a single wire/slot, showing a diagonal slash
    with a count label above it.

    Parameters
    ----------
    slot : int
        Which slot to place the bundle at.
    start : np.ndarray
        Left endpoint of the wire.
    pitch : float
        Distance between adjacent slots.
    count : int
        The number to display above the slash.
    size : float
        Overall length of the slash (height in scene units).
    color : Color
        Stroke/text color.
    thickness : float
        Stroke width of the slash.
    """

    def __init__(
        self,
        *,
        slot: int,
        start: np.ndarray,
        pitch: float,
        count: int,
        size: float = 0.3,
        color: Color = WHITE,
        thickness: float = 2,
        **kwargs,
    ):

        self.slot = slot
        self.count = count
        self._color = color
        self._thickness = thickness
        self._size = size / 2

        super().__init__(**kwargs)

        # center point on the wire
        pt = start + RIGHT * (pitch * slot)

        # diagonal slash of total length `size`
        half_len = self._size / 2
        # unit vector at 45°
        diag = (RIGHT + UP) / np.sqrt(2)
        v = diag * half_len
        slash = Line(pt - v, pt + v).set_stroke(color, thickness).set_z_index(1)

        # label above slash
        lbl = Tex(str(count), font_size=size * 40, color=color).set_color(color)
        # shift up so bottom of label sits a bit above slash
        margin = 0.1 * size
        lbl.move_to(pt + (UP + RIGHT) * (half_len + margin)).set_z_index(1)

        self.slash = slash
        self.label = lbl
        self.add(slash, lbl)

    def set_color(self, new_color: Color):
        """
        Recolor the slash and the count label.
        """
        self._color = new_color
        self.slash.set_stroke(new_color, self._thickness)
        self.label.set_color(new_color)
        return self

    def get_color(self) -> Color:
        return self._color


class Wire(VGroup):
    """
    A horizontal wire that can be “snipped” into slot‐aligned segments.

    Parameters
    ----------
    start : np.ndarray
        Leftmost base coordinate.
    end : np.ndarray
        Rightmost base coordinate.
    num_slots : int
        Total number of slots; there are (num_slots + 1) positions.
    pitch : float
        Distance between adjacent slot positions.
    color : Color
        Stroke color of the wire.
    thickness : float
        Stroke width.
    start_slot : int
        First slot to include (default 0).
    end_slot : int | None
        Last slot to include (default num_slots-1).
    """

    def __init__(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        num_slots: int,
        pitch: float,
        color=WHITE,
        thickness: float = 2,
        start_slot: int = 0,
        end_slot: int | None = None,
        **kwargs,
    ):
        # never forward color into super()
        super().__init__(**kwargs)
        self.start = start
        self.end = end
        self.num_slots = num_slots
        self.pitch = pitch
        self.color = color
        self.thickness = thickness
        last = end_slot if end_slot is not None else num_slots - 1
        self.intervals: list[tuple[int, int]] = [(start_slot, last)]

        self._segments = VGroup()
        self._rebuild_segments()

    def _slot_point(self, slot: int) -> np.ndarray:
        vec = self.end - self.start
        unit = vec / np.linalg.norm(vec) if np.linalg.norm(vec) else RIGHT
        return self.start + unit * (self.pitch * slot)

    def point_from_slot(self, slot: int) -> np.ndarray:
        """
        Public API: get the (x,y) of a discrete slot along this wire.
        """
        if slot < 0 or slot > self.num_slots:
            raise ValueError(f"slot must be in [0..{self.num_slots}]")
        return self._slot_point(slot)

    def _rebuild_segments(self):
        self.remove(self._segments)
        self._segments = VGroup()
        for a, b in self.intervals:
            p0 = self._slot_point(a)
            p1 = self._slot_point(b)
            seg = Line(p0, p1)
            seg.set_stroke(self.color, width=self.thickness, behind=True)
            seg.set_z_index(-2)
            self._segments.add(seg)
        self.add(self._segments)

    def remove_slots(self, start: int, end: int):
        new_intervals = []
        for a, b in self.intervals:
            if b < start or a > end:
                new_intervals.append((a, b))
            else:
                if a < start:
                    new_intervals.append((a, start - 1))
                if b > end:
                    new_intervals.append((end + 1, b))
        self.intervals = new_intervals
        self._rebuild_segments()
        return self

    def add_slots(self, start: int, end: int):
        all_ints = self.intervals + [(start, end)]
        all_ints.sort(key=lambda x: x[0])
        merged = []
        for a, b in all_ints:
            if not merged or a > merged[-1][1] + 1:
                merged.append((a, b))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        self.intervals = merged
        self._rebuild_segments()
        return self

    def point_from_proportion(self, alpha: float) -> np.ndarray:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0,1]")
        slot_f = alpha * self.num_slots
        s = int(np.floor(slot_f))
        t = float(np.clip(slot_f - s, 0, 1))
        if s >= self.num_slots:
            return self._slot_point(self.num_slots)
        p0 = self._slot_point(s)
        p1 = self._slot_point(s + 1)
        return interpolate(p0, p1, t)

    def get_start(self) -> np.ndarray:
        return self._slot_point(0)

    def get_end(self) -> np.ndarray:
        return self._slot_point(self.num_slots)

    def shift(self, vec: np.ndarray):
        super().shift(vec)
        self.start += vec
        self.end += vec
        return self

    def set_slots(self, slots: str | tuple[int, int] | list[tuple[int, int]]):
        if slots == "all":
            self.intervals = [(0, self.num_slots - 1)]
        elif isinstance(slots, tuple) and len(slots) == 2:
            self.intervals = [slots]
        elif isinstance(slots, list) and all(
            isinstance(x, tuple) and len(x) == 2 for x in slots
        ):
            self.intervals = slots
        else:
            raise ValueError(
                "Expected 'all', (start,end), or list of (start,end) tuples"
            )
        self._rebuild_segments()
        return self

    def set_color(self, new_color: Color):
        self.color = new_color
        for seg in self._segments:
            seg.set_stroke(new_color, width=self.thickness)
        return self

    def get_color(self) -> Color:
        return self.color


class ClassicalWire(VGroup):
    """
    A horizontal “classical” wire drawn as two parallel lines, which can be
    “snipped” into slot-aligned segments.

    Parameters
    ----------
    start : np.ndarray
        Leftmost coordinate.
    end : np.ndarray
        Rightmost coordinate.
    num_slots : int
        Total number of slots; there are (num_slots + 1) positions.
    pitch : float
        Distance between adjacent slot positions.
    color : Color
        Stroke color for both lines.
    thickness : float
        Stroke width of each line.
    start_slot : int
        First slot to include (default 0).
    end_slot : int | None
        Last slot to include (default num_slots-1).
    """

    def __init__(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        num_slots: int,
        pitch: float,
        color=WHITE,
        thickness: float = 2,
        start_slot: int = 0,
        end_slot: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.start = start
        self.end = end
        self.num_slots = num_slots
        self.pitch = pitch
        self.color = color
        self.thickness = thickness
        last = end_slot if end_slot is not None else num_slots - 1
        self.intervals: list[tuple[int, int]] = [(start_slot, last)]
        self._segments = VGroup()
        self._rebuild_segments()

    def _slot_point(self, slot: int) -> np.ndarray:
        vec = self.end - self.start
        unit = vec / np.linalg.norm(vec) if np.linalg.norm(vec) else RIGHT
        return self.start + unit * (self.pitch * slot)

    def point_from_slot(self, slot: int) -> np.ndarray:
        if not (0 <= slot <= self.num_slots):
            raise ValueError(f"slot must be in [0..{self.num_slots}]")
        return self._slot_point(slot)

    def _rebuild_segments(self):
        self.remove(self._segments)
        self._segments = VGroup()
        sep = 0.02  # <<<<< very narrow separation
        for a, b in self.intervals:
            p0 = self._slot_point(a)
            p1 = self._slot_point(b)
            top = Line(p0 + UP * sep, p1 + UP * sep)
            bot = Line(p0 + DOWN * sep, p1 + DOWN * sep)
            top.set_stroke(self.color, width=self.thickness, behind=True)
            bot.set_stroke(self.color, width=self.thickness, behind=True)
            top.set_z_index(-2)
            bot.set_z_index(-2)
            self._segments.add(top, bot)
        self.add(self._segments)

    def remove_slots(self, start: int, end: int):
        new = []
        for a, b in self.intervals:
            if b < start or a > end:
                new.append((a, b))
            else:
                if a < start:
                    new.append((a, start - 1))
                if b > end:
                    new.append((end + 1, b))
        self.intervals = new
        self._rebuild_segments()
        return self

    def add_slots(self, start: int, end: int):
        ints = self.intervals + [(start, end)]
        ints.sort(key=lambda x: x[0])
        merged = []
        for a, b in ints:
            if not merged or a > merged[-1][1] + 1:
                merged.append((a, b))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        self.intervals = merged
        self._rebuild_segments()
        return self

    def set_slots(self, slots: str | tuple[int, int] | list[tuple[int, int]]):
        if slots == "all":
            self.intervals = [(0, self.num_slots - 1)]
        elif isinstance(slots, tuple) and len(slots) == 2:
            self.intervals = [slots]
        elif isinstance(slots, list) and all(
            isinstance(x, tuple) and len(x) == 2 for x in slots
        ):
            self.intervals = slots
        else:
            raise ValueError("Expected 'all', (start,end), or list of such tuples")
        self._rebuild_segments()
        return self

    def point_from_proportion(self, alpha: float) -> np.ndarray:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0,1]")
        slot_f = alpha * self.num_slots
        s = int(np.floor(slot_f))
        t = float(np.clip(slot_f - s, 0, 1))
        if s >= self.num_slots:
            return self._slot_point(self.num_slots)
        return interpolate(self._slot_point(s), self._slot_point(s + 1), t)

    def get_start(self) -> np.ndarray:
        return self._slot_point(0)

    def get_end(self) -> np.ndarray:
        return self._slot_point(self.num_slots)

    def shift(self, vec: np.ndarray):
        super().shift(vec)
        self.start += vec
        self.end += vec
        return self

    def set_color(self, new_color: Color):
        self.color = new_color
        for seg in self._segments:
            seg.set_stroke(new_color, width=self.thickness)
        return self

    def get_color(self) -> Color:
        return self.color


def SurroundingRectangleDir(
    mobject,
    buff_left: float = 0.0,
    buff_right: float = 0.0,
    buff_up: float = 0.0,
    buff_down: float = 0.0,
    **kwargs,
):
    """
    Like SurroundingRectangle, but with independent left/right/up/down buffers.
    """
    # 1) Grab the raw bounding box corners
    x0, y0, _ = mobject.get_bounding_box()[0]  # bottom-left
    x1, y1, _ = mobject.get_bounding_box()[2]  # top-right

    # 2) Compute the new width/height including the extra buffers
    new_width = (x1 - x0) + buff_left + buff_right
    new_height = (y1 - y0) + buff_up + buff_down

    # 3) Create a zero-buff rectangle of that size
    rect = Rectangle(width=new_width, height=new_height, **kwargs)

    # 4) Center it on the mobject...
    rect.move_to(mobject.get_center())
    # 5) ...then shift it so that the extra horizontal margin
    #    splits buff_left on the left and buff_right on the right
    dx = (buff_right - buff_left) / 2
    dy = (buff_up - buff_down) / 2
    rect.shift(dx * RIGHT + dy * UP)
    return rect


class QuantumCircuit(VGroup):
    """
    A quantum circuit with equally‐spaced discrete gate slots along each wire.
    """

    def __init__(
        self,
        num_wires: int = 3,
        wire_length: float = 6,
        wire_thickness: float = 2,
        wire_spacing: float = 1.0,
        left_labels: bool | Sequence[str] = False,
        right_labels: bool | Sequence[str] = False,
        boxed: bool = False,
        box_color: Color = WHITE,
        box_buffer: float = 0.2,
        gate_size: float = 0.5,
        gate_spacing: float = 0.1,
        gate_shape: str | None = "square",
        *,
        bg_color: Color = BLACK,  # <<< NEW
        gate_palette: str | None = None,  # <<< NEW
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_wires = num_wires
        self.wire_length = wire_length
        self.wire_spacing = wire_spacing
        self.boxed = boxed
        self.box_color = box_color
        self.box_buffer = box_buffer
        self.gate_size = gate_size
        self._wire_label_font = max(1, int(round(36 * (self.gate_size / 0.5))))
        self.dot_size = gate_size * 0.3
        self.gate_spacing = gate_spacing
        self.wire_thickness = wire_thickness
        self._gate_shape = gate_shape
        self.bg_color = bg_color
        self._scheme_name = (gate_palette or "").strip().lower() or None
        self._palette, self._palette_i = self._make_palette(self._scheme_name), 0
        self._kind_to_color: dict[str, Color] = {}

        # --- slot geometry ---
        self.pitch = gate_size + gate_spacing
        self.num_slots = int(wire_length // self.pitch) + 1
        self._slot_alphas = [
            (i * self.pitch) / wire_length for i in range(self.num_slots)
        ]
        self._slot_xs = [
            -wire_length / 2 + i * self.pitch for i in range(self.num_slots)
        ]

        # --- storage created early so placers can safely read them ---
        self.gates: list[Mobject] = []
        self._surbox = None
        self._perm_groups: list[Mobject] = []
        self._perm_windows: list[dict] = []
        self._post_perm_maps: list[dict] = []
        self._pulse_routes: dict[int, list] = {i: [] for i in range(num_wires)}

        # --- build wires and labels ---
        self.wires = VGroup()
        self.labels = VGroup()

        def resolve_labels(val, fallback_fn):
            if val is False:
                return [None] * self.num_wires
            if val is True:
                return [fallback_fn(i) for i in range(self.num_wires)]
            if isinstance(val, (list, tuple)):
                labels = list(val)
                while len(labels) < self.num_wires:
                    labels.append(None)
                return [lbl if lbl != "" else None for lbl in labels]
            raise TypeError("Labels must be False, True, or a list of strings.")

        lefts = resolve_labels(left_labels, lambda i: f"q_{{{i}}}")
        rights = resolve_labels(right_labels, lambda i: f"q_{{{i}}}")

        for i in range(num_wires):
            w = (
                Wire(
                    LEFT * wire_length / 2,
                    RIGHT * wire_length / 2,
                    num_slots=self.num_slots,
                    pitch=self.pitch,
                    color=WHITE,
                    thickness=self.wire_thickness,
                )
                .set_stroke(WHITE, self.wire_thickness)
                .set_z_index(-1)
            )
            w.shift(DOWN * (i - (num_wires - 1) / 2) * wire_spacing)
            self.wires.add(w)

            if lefts[i]:
                lbl = Tex(lefts[i], font_size=self._wire_label_font).set_z_index(1)
                lbl._qc_label_side = "left"
                self.labels.add(lbl)
                self._place_left_label_on_wire(i, lbl)
            if rights[i]:
                lbl = Tex(rights[i], font_size=self._wire_label_font).set_z_index(1)
                lbl._qc_label_side = "right"
                self.labels.add(lbl)
                self._place_right_label_on_wire(i, lbl)

        # segment→labels bookkeeping (unchanged)
        self.qubit_labels = []
        for i, wire in enumerate(self.wires):
            segments = []
            left = lefts[i]
            right = rights[i]
            for a, b in wire.intervals:
                l = left if a == 0 else None
                r = right if b == self.num_slots - 1 else None
                segments.append((l, r))
            self.qubit_labels.append(segments)

        self.add(self.wires, self.labels)
        if self.boxed:
            self._make_surrounding_box()

    # -------------------------------------------------------------------------
    # Gate placement
    # -------------------------------------------------------------------------
    def add_gate(
        self,
        kind: str,
        wire_idx: int | tuple[int, int],
        slot: int | tuple[int, int],
        *,
        shift: str | None = None,
        axis: str | None = None,
        gate_shape: str | None = None,
        control_idx: int | None = None,
        **params,
    ):
        """
        Place a gate (single, controlled, two-qubit, phase, custom, bundle) on the circuit.

        Updates:
          • Single-qubit gates are created with SingleQubitGate (H, X, Y, Z, S, T, U, SQX, SQY, SQZ, R, RX, RY, RZ,
            or any custom TeX label).
          • A leading 'C' makes a controlled *single-qubit* gate (e.g. CX, CH, CRZ, CSQY, C\\alpha, ...),
            distinct from the *two-qubit* special gates like CNOT/CZ.
          • Two-qubit special gates remain: CNOT, CZ, SWAP.
        """

        # ---------- helpers for permutation-aware remapping ----------
        def _ensure_perm_maps():
            if not hasattr(self, "_post_perm_maps"):
                self._post_perm_maps = []
            if not hasattr(self, "_pulse_routes"):
                self._pulse_routes = {i: [] for i in range(len(self.wires))}
            if not hasattr(self, "_perm_groups"):
                self._perm_groups = []
            if not hasattr(self, "_perm_windows"):
                self._perm_windows = []

        def _remap_index_after_slot(idx: int, slot_at: int) -> int:
            _ensure_perm_maps()
            new_idx = int(idx)
            for win in sorted(self._post_perm_maps, key=lambda d: d["cut_slot"]):
                if slot_at >= win["cut_slot"]:
                    new_idx = win["map"].get(new_idx, new_idx)
            return new_idx

        # ---------- normalize inputs ----------
        kind_raw = str(kind)
        kind_up = kind_raw.upper()
        dot_size = params.pop("dot_size", self.dot_size)

        # --- choose color (if a scheme is active) and default bg fill ---
        peek_label = params.get("label_tex", params.get("label", None))
        peek_name = params.get("name", None)  # for CUSTOM
        pal_key = self._palette_key(
            kind_up,
            axis=axis,
            label_override=peek_label,
            custom_name=str(peek_name) if peek_name is not None else None,
        )
        auto_color = self._color_for_kind(pal_key)
        if auto_color is not None and "color" not in params:
            params["color"] = auto_color

        # wire index (allow vertical span)
        if isinstance(wire_idx, (tuple, list)) and len(wire_idx) == 2:
            top_idx, bot_idx = int(wire_idx[0]), int(wire_idx[1])
            if bot_idx < top_idx:
                top_idx, bot_idx = bot_idx, top_idx
            if not (0 <= top_idx < len(self.wires)) or not (
                0 <= bot_idx < len(self.wires)
            ):
                raise IndexError("wire_idx tuple out of range")
            tgt_wire = self.wires[top_idx]
            gate_height = bot_idx - top_idx + 1
        else:
            if not (0 <= int(wire_idx) < len(self.wires)):
                raise IndexError(
                    f"wire_idx {wire_idx} out of range (0..{self.wires.__len__()-1})"
                )
            top_idx = int(wire_idx)
            tgt_wire = self.wires[top_idx]
            gate_height = 1

        # slot (allow horizontal span)
        def _check_slot_bound(s: int):
            if not (0 <= s <= self.num_slots):
                raise ValueError(f"slot must be in [0..{self.num_slots}], got {s}")

        if isinstance(slot, (tuple, list)) and len(slot) == 2:
            s0, s1 = int(slot[0]), int(slot[1])
            if s1 < s0:
                s0, s1 = s1, s0
            _check_slot_bound(s0)
            _check_slot_bound(s1)
            slot_is_range, slot_lo, slot_hi, slot_for_checks = True, s0, s1, s0
        else:
            s = int(slot)
            _check_slot_bound(s)
            slot_is_range, slot_lo, slot_hi, slot_for_checks = False, s, s, s

        # visible on the top wire?
        visible = [(a, b) for a, b in tgt_wire.intervals if a <= slot_for_checks <= b]
        if not visible:
            raise ValueError(
                f"Cannot place {kind_up} on wire {top_idx}@{slot_for_checks}: "
                f"visible only in {tgt_wire.intervals}"
            )
        seg_a, seg_b = visible[0]
        start = tgt_wire.start
        pitch = self.pitch

        # ---------- permutation-aware remap (single-slot only) ----------
        orig_target_idx = top_idx
        orig_control_idx = None if control_idx is None else int(control_idx)

        _ensure_perm_maps()
        remapped_top_idx = top_idx
        remapped_ctrl_idx = orig_control_idx

        if not slot_is_range:
            if gate_height == 1:
                remapped_top_idx = _remap_index_after_slot(top_idx, slot_for_checks)
                tgt_wire = self.wires[remapped_top_idx]
                start = tgt_wire.start
            if orig_control_idx is not None:
                remapped_ctrl_idx = _remap_index_after_slot(
                    orig_control_idx, slot_for_checks
                )

        # ---------- classify the request ----------
        # Known single-qubit tokens (handled by SingleQubitGate)
        SINGLE_QUBIT_TYPES = {
            "H",
            "X",
            "Y",
            "Z",
            "S",
            "T",
            "U",
            "SQX",
            "SQY",
            "SQZ",
            "R",
            "RX",
            "RY",
            "RZ",
        }

        is_controlled_single = False
        single_token_raw = None
        if kind_up.startswith("C") and kind_up not in {"CNOT", "CZ", "SWAP", "CUSTOM"}:
            is_controlled_single = True
            single_token_raw = kind_raw[1:]
            # single_token_up = single_token_raw.upper()

        # Helper to build SingleQubitGate
        def _build_single_qubit_gate(
            token_raw: str | None, *, control_y_val: float | None
        ):
            # accept both spellings from callers
            label_override = params.pop("label_tex", params.pop("label", None))

            if token_raw is None:
                gate_type = params.pop("type", None)
                gate_label = label_override
            else:
                if token_raw.upper() in SINGLE_QUBIT_TYPES:
                    gate_type = token_raw.upper()
                    gate_label = (
                        label_override  # <-- honor explicit label for known types too
                    )
                else:
                    gate_type = None
                    gate_label = (
                        label_override if label_override is not None else token_raw
                    )

            axis_local = axis
            angle_local = params.pop("angle", params.pop("angle_tex", None))
            if gate_type in {"RX", "RY", "RZ"}:
                axis_local = gate_type[1]

            shape = gate_shape or self._gate_shape
            return SingleQubitGate(
                type=gate_type,
                label=gate_label,
                slot=slot_for_checks,
                start=start,
                pitch=pitch,
                shift=shift,
                size=self.gate_size,
                shape=shape,
                dot_size=dot_size,
                control_y=control_y_val,
                dag=bool(params.pop("dag", params.pop("dagger", False))),
                axis=axis_local,
                angle=angle_local,
                color=params.get("color", WHITE),
                bg_color=params.get("bg_color", self.bg_color),
                stroke_width=params.get("stroke_width", None),
            )

        # ---------- construct the gate ----------
        if slot_is_range and gate_height == 1:
            # single-qubit gates cannot span slots (keep behavior consistent)
            # allow CUSTOM spanning as before
            pass

        if is_controlled_single:
            if remapped_ctrl_idx is None:
                raise ValueError(f"{kind_raw} requires a control_idx")
            ctrl_wire = self.wires[remapped_ctrl_idx]
            gate = _build_single_qubit_gate(
                single_token_raw, control_y_val=ctrl_wire.get_y()
            )

        elif (kind_up in SINGLE_QUBIT_TYPES) or (
            kind_up
            not in {"M", "CNOT", "CZ", "SWAP", "WB", "BUNDLE", "P", "PHASE", "CUSTOM"}
        ):
            # Uncontrolled single-qubit: known token OR unknown token treated as TeX label
            if gate_height != 1 or slot_is_range:
                raise ValueError(
                    "Single-qubit gates must be placed on a single wire and single slot."
                )
            # If it's a known non-single keyword, it will be handled below; otherwise, treat as single-qubit/label.
            token_for_gate = kind_raw if (kind_up in SINGLE_QUBIT_TYPES) else kind_raw
            gate = _build_single_qubit_gate(token_for_gate, control_y_val=None)

        elif kind_up == "M":
            lbl = params.pop("label", False)
            rd = params.pop("readout_idx", None)
            if rd is not None:
                rd_idx = _remap_index_after_slot(int(rd), slot_for_checks)
                rd_wire = self.wires[rd_idx]
                control_y_val = rd_wire.get_y()
                control_type = (
                    "Classical" if isinstance(rd_wire, ClassicalWire) else "Quantum"
                )
            else:
                control_y_val = None
                control_type = "Quantum"
            gate = MeasurementGate(
                slot=slot_for_checks,
                start=start,
                pitch=pitch,
                shift=shift,
                axis=axis,
                size=self.gate_size,
                wire_index=remapped_top_idx,
                label=lbl,
                control_y=control_y_val,
                control_type=control_type,
                dot_size=dot_size,
                bg_color=params.get("bg_color", self.bg_color),
                **params,
            )

        elif kind_up in {"CNOT", "CZ", "SWAP"}:
            if remapped_ctrl_idx is None:
                raise ValueError(f"{kind_up} requires a control_idx")
            ctrl_wire = self.wires[remapped_ctrl_idx]

            params_nobg = dict(params)
            params_nobg.pop("bg_color", None)  # swallow unsupported kw

            if kind_up == "CNOT":
                gate = CNOTGate(
                    slot=slot_for_checks,
                    start_control=ctrl_wire.start,
                    start_target=start,
                    pitch=pitch,
                    shift=shift,
                    dot_size=dot_size,
                    bg_color=params.get("bg_color", self.bg_color),  # CNOT uses bg fill
                    **params_nobg,
                )
            elif kind_up == "CZ":
                gate = CZGate(
                    slot=slot_for_checks,
                    start_control=ctrl_wire.start,
                    start_target=start,
                    pitch=pitch,
                    shift=shift,
                    dot_size=dot_size,
                    **params_nobg,  # no bg_color for CZ
                )
            else:  # SWAP
                gate = SwapGate(
                    slot=slot_for_checks,
                    start_control=ctrl_wire.start,
                    start_target=start,
                    pitch=pitch,
                    shift=shift,
                    **params_nobg,  # no bg_color for SWAP
                )

        elif kind_up in {"WB", "BUNDLE"}:
            count = params.pop("count", None)
            if count is None:
                raise ValueError("WireBundle requires a `count` parameter")
            gate = WireBundle(
                slot=slot_for_checks,
                start=start,
                pitch=pitch,
                count=count,
                size=self.gate_size,
                color=params.get("color", WHITE),
                thickness=params.get("thickness", 2),
            )

        elif kind_up in {"P", "PHASE"}:
            label_tex = params.pop("label", r"\alpha")
            gate = PhaseGate(
                slot=slot_for_checks,
                start=start,
                pitch=pitch,
                label=label_tex,
                dot_size=dot_size,
                color=params.get("color", WHITE),
            )

        elif kind_up == "CUSTOM":
            name = params.pop("name", params.pop("label", "U"))
            control_y_val = None
            if remapped_ctrl_idx is not None:
                control_y_val = self.wires[remapped_ctrl_idx].get_y()
            shape_for_custom = gate_shape if gate_shape is not None else "square"
            gate = CustomGate(
                slot=slot if slot_is_range else slot_for_checks,
                start=start,
                pitch=pitch,
                size=self.gate_size,
                color=params.get("color", WHITE),
                bg_color=params.get("bg_color", self.bg_color),
                shape=shape_for_custom,
                stroke_width=params.get("stroke_width", None),
                control_y=control_y_val,
                name=name,
                wire_spacing=self.wire_spacing,
                gate_height=gate_height,
                shift=shift,
                label_size=params.get("label_size", "normal"),
                dot_size=dot_size,
            )
            # Overlap warning (unchanged)
            if slot_is_range or gate_height > 1:
                import warnings

                def _rect(mob):
                    c = mob.get_center()
                    w = max(1e-9, mob.get_width())
                    h = max(1e-9, mob.get_height())
                    return (c[0] - w / 2, c[0] + w / 2, c[1] - h / 2, c[1] + h / 2)

                xL, xR, yB, yT = _rect(gate)
                overlaps = []
                for g in self.gates:
                    try:
                        aL, aR, aB, aT = _rect(g)
                    except Exception:
                        continue
                    if xL < aR and xR > aL and yB < aT and yT > aB:
                        overlaps.append(g)
                if overlaps:
                    names = ", ".join(type(g).__name__ for g in overlaps[:5])
                    more = "" if len(overlaps) <= 5 else f" (+{len(overlaps)-5} more)"
                    warnings.warn(
                        f"CustomGate at wire(s) {wire_idx}, slot(s) {slot} overlaps "
                        f"{len(overlaps)} existing gate(s): {names}{more}. Adding anyway."
                    )
        else:
            raise ValueError(
                f"Unknown gate kind {kind!r}; expected one of single-qubit tokens "
                f"(H,X,Y,Z,S,T,U,SQX,SQY,SQZ,R,RX,RY,RZ or any custom TeX), "
                f"controlled single-qubit prefixed with 'C' (e.g. CX, CH, CRZ, CSQY, C\\phi), "
                f"or special two-qubit: 'CNOT','CZ','SWAP', or 'M','WB','BUNDLE','P','PHASE','CUSTOM'."
            )

        # ---------- attach metadata (before label nudging) ----------
        try:
            color_now = (
                gate.get_color()
                if hasattr(gate, "get_color")
                else params.get("color", WHITE)
            )
            gate._qc_meta = {
                "kind": kind_up,
                "slot": None if slot_is_range else slot_for_checks,
                "slot_range": (slot_lo, slot_hi) if slot_is_range else None,
                "orig_wire_idx": (
                    orig_target_idx
                    if gate_height == 1
                    else (orig_target_idx, orig_target_idx + gate_height - 1)
                ),
                "orig_control_idx": orig_control_idx,
                "placed_wire_idx": (
                    remapped_top_idx
                    if gate_height == 1
                    else (remapped_top_idx, remapped_top_idx + gate_height - 1)
                ),
                "placed_control_idx": remapped_ctrl_idx,
                "shift": shift,
                "axis": axis,
                "gate_shape": gate_shape or self._gate_shape,
                "color": color_now,
            }
        except Exception:
            pass

        # ---------- add to scene & nudge labels ----------
        self.gates.append(gate)
        self.add(gate)

        if kind_up == "CUSTOM" and slot_is_range:
            bot_idx_eff = remapped_top_idx + gate_height - 1
            slot_lo_eff, slot_hi_eff = slot_lo, slot_hi
            for wi in range(remapped_top_idx, bot_idx_eff + 1):
                w = self.wires[wi]
                left_hits = [(a, b) for (a, b) in w.intervals if a <= slot_lo_eff <= b]
                right_hits = [(a, b) for (a, b) in w.intervals if a <= slot_hi_eff <= b]
                if left_hits and slot_lo_eff == left_hits[0][0]:
                    self._shift_left_label(wi, shift)
                if right_hits and slot_hi_eff == right_hits[0][1]:
                    self._shift_right_label(wi, shift)
        else:
            if slot_for_checks == seg_b:
                self._shift_right_label(remapped_top_idx, shift)
            elif slot_for_checks == seg_a:
                self._shift_left_label(remapped_top_idx, shift)

        # refresh surround if touching boundary
        if self.boxed and (
            remapped_top_idx in (0, self.num_wires - 1)
            or slot_for_checks in (0, self.num_slots - 1)
        ):
            self._make_surrounding_box()

        return gate

    # -------------------------------------------------------------------------
    # Boxes & utilities
    # -------------------------------------------------------------------------
    def box_gate(
        self,
        *,
        wire_idx: int | tuple[int, int],
        slot_idx: int | tuple[int, int],
        shift: str | None = None,
        color: Color = WHITE,
        stroke_width: float | None = None,
        dash_length: float = 0.08,
        buffer: float | None = None,
        z_index: int = 3,
        stroke: str = "dashed",
        dot_spacing: float | None = None,
        dot_radius: float | None = None,
    ) -> VGroup:
        def _wire_y(i: int) -> float:
            return self.wires[i].get_y()

        def _x_at_slot(s: int, off: float = 0.0) -> float:
            start = self.wires[0].start
            return (start + RIGHT * (self.pitch * s + off))[0]

        def _solid(a, b):
            return Line(a, b).set_stroke(color, sw).set_z_index(z_index)

        def _dashed(a, b):
            return (
                DashedLine(a, b, dash_length=dash_length)
                .set_stroke(color, sw)
                .set_z_index(z_index)
            )

        def _dotted(a, b):
            p0, p1 = np.array(a), np.array(b)
            v = p1 - p0
            L = float(np.linalg.norm(v))
            if L == 0:
                return VGroup()
            u = v / L
            spacing = dot_spacing if dot_spacing is not None else dash_length
            n = max(2, int(np.floor(L / spacing)) + 1)
            r = dot_radius if dot_radius is not None else min(0.06, 0.25 * spacing)
            dots = VGroup()
            for i in range(n):
                pos = p0 + u * (i * L / (n - 1))
                dots.add(
                    Dot(radius=r)
                    .set_fill(color, 1)
                    .set_stroke(width=0)
                    .move_to(pos)
                    .set_z_index(z_index)
                )
            return dots

        maker = {"solid": _solid, "dashed": _dashed, "dotted": _dotted}.get(
            stroke.lower(), _dashed
        )

        if isinstance(wire_idx, tuple) and len(wire_idx) == 2:
            w0, w1 = sorted(wire_idx)
            y0, y1 = _wire_y(w0), _wire_y(w1)
            span_wires = True
        elif isinstance(wire_idx, int):
            y0 = y1 = _wire_y(wire_idx)
            span_wires = False
        else:
            raise TypeError("wire_idx must be int or (start,end)")

        if isinstance(slot_idx, tuple) and len(slot_idx) == 2:
            s0, s1 = sorted(slot_idx)
            off = 0.0
            x0, x1 = _x_at_slot(s0, off), _x_at_slot(s1, off)
            span_slots = True
        elif isinstance(slot_idx, int):
            off = 0.0
            if shift:
                s = shift.lower()
                off = (
                    self.pitch / 3
                    if s in ("r", "right")
                    else (-self.pitch / 3 if s in ("l", "left") else 0.0)
                )
            x0 = x1 = _x_at_slot(slot_idx, off)
            span_slots = False
        else:
            raise TypeError("slot_idx must be int or (start,end)")

        pitch = self.pitch
        sw = stroke_width if stroke_width is not None else 4 * self.gate_size
        buf = buffer if buffer is not None else pitch / 2
        xL = min(x0, x1) - buf if span_slots else x0 - buf
        xR = max(x0, x1) + buf if span_slots else x0 + buf
        yB = min(y0, y1) - buf if span_wires else y0 - buf
        yT = max(y0, y1) + buf if span_wires else y0 + buf
        top = maker([xL, yT, 0], [xR, yT, 0])
        bottom = maker([xL, yB, 0], [xR, yB, 0])
        left = maker([xL, yB, 0], [xL, yT, 0])
        right = maker([xR, yB, 0], [xR, yT, 0])
        box = VGroup(left, right, top, bottom)
        self.add(box)
        return box

    def _make_surrounding_box(self):
        if self._surbox is not None:
            self.remove(self._surbox)
        content = VGroup(self.wires, self.labels, *self.gates)
        rect = SurroundingRectangle(
            content, buff=self.box_buffer, color=self.box_color, stroke_width=1
        ).set_fill(opacity=0)
        rect.set_z_index(-2)
        self._surbox = rect
        self.add(rect)

    # -------------------------------------------------------------------------
    # Pulse sweep (unchanged except for style)
    # -------------------------------------------------------------------------
    def _ensure_perm_maps(self):
        if not hasattr(self, "_post_perm_maps"):
            self._post_perm_maps = []
        if not hasattr(self, "_pulse_routes"):
            self._pulse_routes = {i: [] for i in range(len(self.wires))}
        if not hasattr(self, "_perm_groups"):
            self._perm_groups = []
        if not hasattr(self, "_perm_windows"):
            self._perm_windows = []

    def _remap_index_after_slot(self, idx: int, slot: int) -> int:
        self._ensure_perm_maps()
        new_idx = int(idx)
        for win in sorted(self._post_perm_maps, key=lambda d: d["cut_slot"]):
            if slot >= win["cut_slot"]:
                new_idx = win["map"].get(new_idx, new_idx)
        return new_idx

    def pulse_all(
        self,
        scene,
        run_time: float = 2.0,
        pulse_color=TEAL,
        animate: bool = True,
        repetitions: int = 1,
    ):
        """
        Sweep a soft pulse left→right across this circuit.

        Parameters
        ----------
        scene : Scene
        run_time : float
            Duration PER pulse.
        pulse_color : Color
        animate : bool
            If True, plays now. If False, returns an Animation.
        repetitions : int
            Number of times to repeat the sweep (back-to-start each time).
        """
        # --- guard / normalize reps ---------------------------------------------
        reps = max(1, int(repetitions))

        # --- sweep extents in world coords (based on current center) -------------
        pitch = self.gate_size + self.gate_spacing
        half_pitch = pitch / 2.0
        cx = float(self.get_center()[0])
        ext_start = cx + (-self.wire_length / 2.0 - 2.0 * half_pitch)
        ext_end = cx + (+self.wire_length / 2.0 + 2.0 * half_pitch)

        tracker = ValueTracker(ext_start)

        # --- precompute straight spans (world coords) ----------------------------
        line_segments: list[list[tuple[float, float, float]]] = []
        for w in self.wires:
            segs = []
            for a, b in w.intervals:
                x_a = w.point_from_slot(a)[0]
                x_b = w.point_from_slot(b)[0]
                if x_b < x_a:
                    x_a, x_b = x_b, x_a
                segs.append((x_a, x_b, w.get_y()))
            line_segments.append(segs)

        # optional curved routes (absolute)
        curve_routes = getattr(
            self, "_pulse_routes", {i: [] for i in range(len(self.wires))}
        )

        def _y_for_x_on_wire(wi: int, x: float):
            # straight spans first
            for x_a, x_b, y in line_segments[wi]:
                if x_a <= x <= x_b:
                    return float(y), 1.0
            # curved route beziers
            for route in curve_routes.get(wi, []):
                xL, xR = float(route["xL"]), float(route["xR"])
                if xL <= x <= xR and xR > xL:
                    t = (x - xL) / (xR - xL)
                    p0, p1, p2, p3 = route["p0"], route["p1"], route["p2"], route["p3"]
                    a = interpolate(p0, p1, t)
                    b = interpolate(p1, p2, t)
                    c = interpolate(p2, p3, t)
                    d = interpolate(a, b, t)
                    e = interpolate(b, c, t)
                    pt = interpolate(d, e, t)
                    return float(pt[1]), 1.0
            # not visible at this x
            return float(self.wires[wi].get_y()), 0.0

        # --- overlays ------------------------------------------------------------
        wire_glows = [
            GlowDot(color=pulse_color, radius=0.25).set_z_index(-1) for _ in self.wires
        ]
        for wi, glow in enumerate(wire_glows):

            def updater(mob, idx=wi):
                x = tracker.get_value()
                y, vis = _y_for_x_on_wire(idx, x)
                mob.move_to([x, y, 0]).set_opacity(vis)

            glow.add_updater(updater)

        ghosts = []
        skip_groups = set(getattr(self, "_perm_groups", []))
        for gate in self.gates:
            if gate in skip_groups:
                continue
            if hasattr(gate, "get_backgroundless"):
                ghost = gate.get_backgroundless(pulse_color, stroke_width=1)
            else:
                ghost = gate.copy()
                for sub in ghost.submobjects:
                    try:
                        sub.set_fill(opacity=0)
                    except Exception:
                        pass
                ghost.set_stroke(pulse_color, width=1)

            def update_gate_glow(mob):
                sweep_x = tracker.get_value()
                gate_x = mob.get_center()[0]
                sigma = 1.5 * half_pitch
                width = 4.0 * np.exp(-0.5 * ((gate_x - sweep_x) / sigma) ** 2)
                mob.set_stroke(width=width)

            ghost.add_updater(update_gate_glow)
            ghosts.append(ghost)

        # --- driver --------------------------------------------------------------
        driver = Dot(radius=0.01, color=BLACK).set_opacity(0.0)
        added = {"ok": False}
        span = ext_end - ext_start

        def drive(_mob, a):
            if not added["ok"]:
                scene.add(*wire_glows, *ghosts)
                added["ok"] = True
            # map [0,1] → repeated [0,1] cycles
            local_a = (a * reps) % 1.0 if reps > 1 else a
            tracker.set_value(ext_start + local_a * span)
            if a >= 1.0:
                # cleanup at the very end only
                try:
                    scene.remove(*wire_glows, *ghosts)
                except Exception:
                    pass
                for o in wire_glows:
                    o.clear_updaters()
                for o in ghosts:
                    o.clear_updaters()

        anim = UpdateFromAlphaFunc(driver, drive)

        if animate:
            # keep per-pass speed the same
            scene.play(anim, run_time=run_time * reps, rate_func=linear)
            return
        else:
            # When returning the Animation, remember to multiply run_time at play time:
            # scene.play(qc.pulse_all(scene, animate=False, repetitions=3), run_time=1*3)
            return anim

    # -------------------------------------------------------------------------
    # Label anchoring & nudgers
    # -------------------------------------------------------------------------
    def _gate_right_extent_at(self, wi: int, slot: int) -> float:
        """
        Return the rightmost x-coordinate among the slot center and any gate drawn
        on wire `wi` at `slot`. Works even if a gate was just added and its
        `_qc_meta` isn't set yet by falling back to a geometry check.
        """
        base_x = self.wires[wi].point_from_slot(slot)[0]
        y = self.wires[wi].get_y()
        rightmost = base_x

        tol_x = self.pitch * 0.55  # half-slot tolerance
        tol_y = max(1e-3, 0.6 * self.wire_spacing)

        for g in getattr(self, "gates", []):
            try:
                meta = getattr(g, "_qc_meta", None)
                if isinstance(meta, dict):
                    # Prefer exact metadata match
                    placed = meta.get("placed_wire_idx")
                    on_wire = (
                        isinstance(placed, tuple) and placed[0] <= wi <= placed[1]
                    ) or (placed == wi)
                    if on_wire and meta.get("slot") == slot:
                        rightmost = max(rightmost, g.get_right()[0])
                        continue

                # Geometry fallback: same slot x, and wire y lies within gate's vertical span
                cx, cy, _ = g.get_center()
                if abs(cx - base_x) <= tol_x:
                    top_y = g.get_top()[1]
                    bot_y = g.get_bottom()[1]
                    if (bot_y - tol_y) <= y <= (top_y + tol_y):
                        rightmost = max(rightmost, g.get_right()[0])
            except Exception:
                # best-effort; ignore malformed gates
                continue

        return rightmost

    def _place_left_label_on_wire(
        self, wi: int, lbl: "Mobject", shift: str | None = None
    ):
        w = self.wires[wi]
        first_slot = min(a for (a, b) in w.intervals)
        anchor = np.array([w.point_from_slot(first_slot)[0], w.get_y(), 0.0])
        buff = self.pitch / 3
        if shift and shift.lower() in ("left", "l"):
            buff += self.pitch / 3
        elif shift and shift.lower() in ("right", "r"):
            buff = max(0.0, buff - self.pitch / 3)
        lbl.next_to(anchor, LEFT, buff=buff, aligned_edge=RIGHT)

    def _place_right_label_on_wire(
        self, wi: int, lbl: "Mobject", shift: str | None = None
    ):
        w = self.wires[wi]
        last_slot = max(b for (a, b) in w.intervals)
        anchor_x = self._gate_right_extent_at(wi, last_slot)
        anchor = np.array([anchor_x, w.get_y(), 0.0])
        buff = self.pitch / 3
        if shift and shift.lower() in ("right", "r"):
            buff += self.pitch / 3
        elif shift and shift.lower() in ("left", "l"):
            buff = max(0.0, buff - self.pitch / 3)
        lbl.next_to(anchor, RIGHT, buff=buff, aligned_edge=LEFT)

    def _shift_left_label(self, wire_idx: int, shift: str | None = None):
        wy = self.wires[wire_idx].get_y()
        tol = 1e-3
        cands = [
            m
            for m in self.labels
            if getattr(m, "_qc_label_side", None) == "left"
            and abs(m.get_center()[1] - wy) < tol
        ]
        if not cands:
            return  # no left label on this wire → nothing to do
        lbl = min(cands, key=lambda m: m.get_center()[0])  # leftmost among left labels
        self._place_left_label_on_wire(wire_idx, lbl, shift)

    def _shift_right_label(self, wire_idx: int, shift: str | None = None):
        wy = self.wires[wire_idx].get_y()
        tol = 1e-3
        cands = [
            m
            for m in self.labels
            if getattr(m, "_qc_label_side", None) == "right"
            and abs(m.get_center()[1] - wy) < tol
        ]
        if not cands:
            return  # no right label on this wire → leave left label alone
        lbl = max(
            cands, key=lambda m: m.get_center()[0]
        )  # rightmost among right labels
        self._place_right_label_on_wire(wire_idx, lbl, shift)

    # -------------------------------------------------------------------------
    # Higher-level helpers
    # -------------------------------------------------------------------------
    def end_measurements(
        self,
        wires: list[int] | None = None,
        shift: str | None = None,
        axis: str | None = None,
        **gate_kwargs,
    ) -> VGroup:
        """
        Place a measurement gate in the *last visible slot* on selected wires.
        Note: add_gate(...) already nudges end labels; we do not re-nudge here.
        """
        out = VGroup()
        wire_indices = wires if wires is not None else range(len(self.wires))
        for wi in wire_indices:
            wire = self.wires[wi]
            for a, b in reversed(wire.intervals):
                if a <= b:
                    last_slot = b
                    break
            else:
                continue
            m = self.add_gate(
                "M", wi, slot=last_slot, shift=shift, axis=axis, **gate_kwargs
            )
            out.add(m)
        if self.boxed:
            self._make_surrounding_box()
        return out

    def hadamard_transform(
        self,
        slot: int,
        shift: str | None = None,
        gate_shape: str | None = None,
        **gate_kwargs,
    ) -> VGroup:
        out = VGroup()
        for wi, wire in enumerate(self.wires):
            h = self.add_gate(
                "H", wi, slot=slot, shift=shift, gate_shape=gate_shape, **gate_kwargs
            )
            out.add(h)
            for a, b in wire.intervals:
                if a <= slot <= b:
                    if slot == b:
                        self._shift_right_label(wi, shift=shift)
                    elif slot == a:
                        self._shift_left_label(wi, shift=shift)
                    break
        if self.boxed:
            self._make_surrounding_box()
        return out

    def gate_transform(
        self,
        kind_or_label: str | None,
        *,
        slot: int,
        shift: str | None = None,
        gate_shape: str | None = None,
        dag: bool = False,
        axis: str | None = None,  # for R / RX / RY / RZ
        angle_tex: str | None = None,  # for R / RX / RY / RZ
        label_tex: str | None = None,  # for U/custom
        **gate_kwargs,
    ) -> VGroup:
        """
        Place the same single-qubit gate on every visible wire at `slot`.

        - If `kind_or_label` is one of: H,X,Y,Z,S,T,SQX,SQY,R,RX,RY,RZ,U → we use that.
        - Otherwise we treat `kind_or_label` as a custom LaTeX label (same as kind 'U').
        - For rotations: you can use 'R' with `axis=` and optional `angle_tex=`, or 'RX'/'RY'/'RZ' with `angle_tex=`.
        - `dag=True` applies a dagger to types that support it (S, T, SQX, SQY, R...).
        """
        known = {"H", "X", "Y", "Z", "S", "T", "SQX", "SQY", "R", "RX", "RY", "RZ", "U"}
        if kind_or_label is None:
            gate_kind = "U"
        else:
            k = str(kind_or_label).upper()
            gate_kind = k if k in known else "U"
            if gate_kind == "U" and label_tex is None:
                label_tex = str(kind_or_label)  # treat input as the custom label

        out = VGroup()
        for wi, wire in enumerate(self.wires):
            # only place if slot is visible on this wire segment
            vis = [(a, b) for (a, b) in wire.intervals if a <= slot <= b]
            if not vis:
                continue

            g = self.add_gate(
                gate_kind,
                wi,
                slot=slot,
                shift=shift,
                gate_shape=gate_shape,
                dag=dag,
                axis=axis,
                angle_tex=angle_tex,
                label_tex=label_tex,
                **gate_kwargs,
            )
            out.add(g)

            a, b = vis[0]
            if slot == b:
                self._shift_right_label(wi, shift)
            elif slot == a:
                self._shift_left_label(wi, shift)

        if self.boxed:
            self._make_surrounding_box()
        return out

    # -------------------------------------------------------------------------
    # Wires & recolor
    # -------------------------------------------------------------------------
    def add_wire(
        self,
        position: float | str = "below",
        slot_range: tuple[int, int] | list[tuple[int, int]] | None = None,
        label: (
            bool | str | tuple[str, str] | list[tuple[str | None, str | None]]
        ) = False,
        wire_type: str = "quantum",
    ):
        if slot_range is None:
            intervals = [(0, self.num_slots - 1)]
        elif isinstance(slot_range, (list, tuple)) and all(
            isinstance(seg, tuple)
            and len(seg) == 2
            and isinstance(seg[0], int)
            and isinstance(seg[1], int)
            for seg in slot_range
        ):
            intervals = list(slot_range)
        elif (
            isinstance(slot_range, tuple)
            and len(slot_range) == 2
            and isinstance(slot_range[0], int)
            and isinstance(slot_range[1], int)
        ):
            intervals = [slot_range]
        else:
            raise TypeError(
                "slot_range must be None, a (start,end) tuple, or a list/tuple of such tuples"
            )

        for a, b in intervals:
            if not (0 <= a <= b <= self.num_slots - 1):
                raise ValueError(f"Invalid slot range ({a},{b})")

        if isinstance(position, (float, int)):
            y = position
        else:
            wire_ys = [w.get_y() for w in self.wires]
            if not wire_ys:
                y = 0
            elif position == "above":
                y = max(wire_ys) + self.wire_spacing
            elif position == "below":
                y = min(wire_ys) - self.wire_spacing
            else:
                raise ValueError(f"Invalid position {position!r}")

        WireClass = ClassicalWire if wire_type == "classical" else Wire
        w = WireClass(
            LEFT * self.wire_length / 2,
            RIGHT * self.wire_length / 2,
            num_slots=self.num_slots,
            pitch=self.pitch,
            color=WHITE,
            thickness=self.wire_thickness,
            start_slot=intervals[0][0],
            end_slot=intervals[0][1],
        )
        w.intervals = intervals
        w._rebuild_segments()
        w.shift(UP * y)
        self.wires.add(w)

        n = len(intervals)
        if label is True:
            q = rf"q_{{{len(self.wires) - 1}}}"
            segments = [(q, q)] + [(None, None)] * (n - 1)
        elif isinstance(label, str):
            segments = [(label, None)] + [(None, None)] * (n - 1)
        elif isinstance(label, tuple) and len(label) == 2 and n == 1:
            segments = [label]
        elif isinstance(label, list):
            segments = list(label)
            segments += [(None, None)] * max(0, n - len(segments))
        else:
            segments = [(None, None)] * n

        for (a, b), (l_lbl, r_lbl) in zip(intervals, segments):
            if l_lbl:
                m = Tex(l_lbl, font_size=self._wire_label_font)
                m._qc_label_side = "left"  # << tag
                pt = w.point_from_slot(a)
                m.next_to(pt, LEFT, buff=self.pitch / 3).set_z_index(1)
                self.labels.add(m)

            if r_lbl:
                m = Tex(r_lbl, font_size=self._wire_label_font)
                m._qc_label_side = "right"  # << tag
                pt = w.point_from_slot(b)
                m.next_to(pt, RIGHT, buff=self.pitch / 3).set_z_index(1)
                self.labels.add(m)

        self.add(w)
        self.num_wires += 1
        if self.boxed:
            self._make_surrounding_box()
        return w

    def edit_wire(
        self,
        index: int,
        slot_range: tuple[int, int] | list[tuple[int, int]] | None = None,
        label: (
            bool | str | tuple[str, str] | list[tuple[str | None, str | None]]
        ) = False,
    ):
        """
        Edit one wire's visible slot segments and (optionally) its per-segment labels.

        Parameters
        ----------
        index : int
            Wire index to edit.
        slot_range : (start, end) | [(start, end), ...] | None
            New visible segments for this wire. None → a single full-span segment.
        label : bool | str | (left,right) | [(left,right), ...]
            Per-segment labels. True → q_{index}. str → left-only for first segment.
            Provide tuples for explicit per-segment (left, right) labels.

        Returns
        -------
        Wire
            The edited wire (for chaining).
        """
        w = self.wires[index]
        y = w.get_y()

        # Remove ALL existing labels on this wire (both sides)
        for m in list(self.labels):
            if abs(m.get_center()[1] - y) < 1e-3:
                self.labels.remove(m)

        # Normalize slot_range -> intervals
        if slot_range is None:
            intervals = [(0, self.num_slots - 1)]
        elif isinstance(slot_range, (list, tuple)) and all(
            isinstance(seg, tuple)
            and len(seg) == 2
            and isinstance(seg[0], int)
            and isinstance(seg[1], int)
            for seg in slot_range
        ):
            intervals = list(slot_range)
        else:
            raise TypeError(
                "slot_range must be None, a (start,end) tuple, or a list/tuple of such tuples"
            )

        for a, b in intervals:
            if not (0 <= a <= b <= self.num_slots - 1):
                raise ValueError(f"Invalid slot range ({a},{b})")

        # Apply new segments to the wire
        w.intervals = intervals
        w._rebuild_segments()

        # Build per-segment label spec
        n = len(intervals)
        if label is True:
            q = rf"q_{{{index}}}"
            segments = [(q, q)] + [(None, None)] * (n - 1)
        elif isinstance(label, str):
            segments = [(label, None)] + [(None, None)] * (n - 1)
        elif isinstance(label, tuple) and len(label) == 2 and n == 1:
            segments = [label]
        elif isinstance(label, list):
            segments = list(label) + [(None, None)] * max(0, n - len(label))
        else:
            segments = [(None, None)] * n

        # Recreate labels (now with the critical _qc_label_side tag!)
        for (a, b), (l_lbl, r_lbl) in zip(intervals, segments):
            if l_lbl:
                m = Tex(l_lbl, font_size=self._wire_label_font).set_z_index(1)
                m._qc_label_side = "left"  #  <<<<<<  tag it
                pt = w.point_from_slot(a)
                m.next_to(pt, LEFT, buff=self.pitch / 3)
                self.labels.add(m)

            if r_lbl:
                m = Tex(r_lbl, font_size=self._wire_label_font).set_z_index(1)
                m._qc_label_side = "right"  #  <<<<<<  tag it
                pt = w.point_from_slot(b)
                m.next_to(pt, RIGHT, buff=self.pitch / 3)
                self.labels.add(m)

        if self.boxed:
            self._make_surrounding_box()
        return w

    # -------------------- color scheme plumbing --------------------
    def _make_palette(self, name: str | None) -> list[Color]:
        """Return a long, vibe-consistent list of colors for a scheme name."""
        if not name:
            return []

        def H(*hexes: str) -> list[str]:
            return [h if h.startswith("#") else ("#" + h) for h in hexes]

        # From your screenshots (first 5 exactly), extended with close matches
        muted = H(
            "A53F2B",
            "F1EDEE",
            "2660A4",
            "F6AE2D",
            "17BEBB",
            "6B705C",
            "CB997E",
            "B7B7A4",
            "386FA4",
            "C7702D",
            "5E548E",
            "9A8C98",
        )
        cmyk = H(
            "FCE762",
            "FC60A8",
            "1BE7FF",
            "D4F5F5",
            "00B3FF",
            "FF4DA6",
            "FFD166",
            "111111",
            "A0F0FF",
            "FF8BD2",
        )
        beach = H(
            "166CBC",
            "17BEBB",
            "FFFFFF",
            "FFEE7F",
            "FF9985",
            "FCE4A8",
            "7FC8A9",
            "F4A259",
            "2B9EB3",
            "E3F2FD",
            "FFE0B2",
        )
        # Broad, well-known vibes
        warm = H(
            "C14953",
            "E36414",
            "F28F3B",
            "F6C77D",
            "FFD6A5",
            "8C1C13",
            "A44A3F",
            "E76F51",
            "E9C46A",
            "DDB892",
        )
        cool = H(
            "1F3A5F",
            "166CBC",
            "17BEBB",
            "A8E6CF",
            "E0FBFC",
            "3D5A80",
            "98C1D9",
            "5AA9E6",
            "BEE3DB",
            "89C2D9",
        )
        vibrant = H(
            "1BE7FF",
            "FF3F8E",
            "FF9F1C",
            "2EC4B6",
            "E71D36",
            "8338EC",
            "3A86FF",
            "FFBE0B",
            "06D6A0",
            "FB5607",
        )

        table = {
            "muted": muted,
            "cmyk": cmyk,
            "beach": beach,
            "warm": warm,
            "cool": cool,
            "vibrant": vibrant,
        }
        return table.get(name, [])

    def _color_for_kind(self, key: str) -> Color | None:
        """
        Deterministically assign the next palette color to a *kind key*
        the first time we see it; reuse thereafter.
        """
        if not self._palette:
            return None
        if key not in self._kind_to_color:
            col = self._palette[self._palette_i % len(self._palette)]
            self._kind_to_color[key] = col
            self._palette_i += 1
        return self._kind_to_color[key]

    @staticmethod
    def _palette_key(
        kind_up: str,
        *,
        axis: str | None = None,
        label_override: str | None = None,
        custom_name: str | None = None,
    ) -> str:
        """
        Normalize a color-key across all gate kinds.
        Examples: 'H', 'M', 'CNOT', 'CZ', 'SWAP', 'R_X', 'CH', 'U:alpha', 'CUSTOM:MyGate'.
        """
        k = kind_up.upper()
        if k == "R" and axis:  # e.g., R with axis arg
            return f"R_{axis.upper()}"
        if k in {"RX", "RY", "RZ"}:
            return k
        if k in {"M", "CNOT", "CZ", "SWAP", "WB", "BUNDLE", "PHASE", "P"}:
            return {"P": "PHASE"}.get(k, k)
        if k == "CUSTOM":
            return f"CUSTOM:{custom_name or 'U'}"
        # controlled single (e.g., CH, CY, CRZ, C√X …) – keep the whole token
        if k.startswith("C") and k not in {"CNOT"}:
            return k
        # unknown single-qubit label → key by the visible label
        if k not in {"H", "X", "Y", "Z", "S", "T", "U", "SQX", "SQY", "SQZ", "R"}:
            return f"U:{label_override or k}"
        return k

    def recolor_gates(self, kind: str | None, color: Color):
        """
        Recolor gates by logical kind. Uses _qc_meta['kind'] when available.
        Falls back to class-based matching only for distinct gate classes
        (MeasurementGate, CNOTGate, CZGate, SwapGate, PhaseGate, CustomGate, WireBundle).
        This avoids referencing removed single-qubit classes after the SingleQubitGate consolidation.
        """
        # 1) "all" fast path
        if kind is None or str(kind).strip() == "" or str(kind).lower() == "all":
            for g in self.gates:
                if hasattr(g, "set_color"):
                    g.set_color(color)
            return self

        # 2) normalize requested kind
        k = str(kind).upper().strip()
        synonyms = {
            "CX": "CNOT",
            "P": "PHASE",
            "WB": "BUNDLE",
        }
        k_norm = synonyms.get(k, k)

        # 3) build a safe fallback class map (only classes that are distinct and present)
        def _maybe(cls_name: str):
            return globals().get(cls_name, None)

        class_map = {}
        for key, cls_name in [
            ("M", "MeasurementGate"),
            ("MEAS", "MeasurementGate"),
            ("MEASURE", "MeasurementGate"),
            ("PHASE", "PhaseGate"),
            ("CNOT", "CNOTGate"),
            ("CZ", "CZGate"),
            ("SWAP", "SwapGate"),
            ("CUSTOM", "CustomGate"),
            ("BUNDLE", "WireBundle"),
        ]:
            cls = _maybe(cls_name)
            if cls is not None:
                class_map[key] = cls

        GateClass = class_map.get(k_norm, None)

        # 4) recolor pass
        for g in self.gates:
            # Prefer metadata (set in add_gate early)
            meta = getattr(g, "_qc_meta", None)
            if isinstance(meta, dict):
                mk_raw = str(meta.get("kind", "")).upper()
                mk = synonyms.get(mk_raw, mk_raw)
                if mk == k_norm:
                    if hasattr(g, "set_color"):
                        g.set_color(color)
                    continue  # go to next gate

            # Fallback: only for distinct gate classes (when no/old metadata)
            if GateClass is not None and isinstance(g, GateClass):
                # Avoid accidental H/Y misses for CH/CY when metadata is missing:
                # We never fallback by class for CH/CY/etc.; those are single-qubit boxes.
                if k_norm in {"CH", "CY", "CSQY", "CRX", "CRY", "CRZ"}:
                    continue
                if hasattr(g, "set_color"):
                    g.set_color(color)

        return self

    def recolor_wires(
        self, indices: Sequence[int] | str | None = None, color: Color = WHITE
    ):
        if indices is None or indices == "all":
            target_idxs = range(len(self.wires))
        elif isinstance(indices, int):
            target_idxs = [indices]
        else:
            target_idxs = indices
        for i in target_idxs:
            if not (0 <= i < len(self.wires)):
                raise IndexError(f"wire index {i} out of range")
            self.wires[i].set_color(color)
        return self

    def recolor_labels(
        self, indices: Sequence[int] | str | None = None, color: Color = WHITE
    ):
        if indices is None or indices == "all":
            target = range(len(self.wires))
        elif isinstance(indices, int):
            target = [indices]
        else:
            target = indices
        ys = {i: self.wires[i].get_y() for i in target}
        tol = 1e-3
        for lbl in self.labels:
            _, y, _ = lbl.get_center()
            if any(abs(y - wy) < tol for wy in ys.values()):
                lbl.set_color(color)
        return self

    # -------------------------------------------------------------------------
    # Wire permutation (unchanged except style)
    # -------------------------------------------------------------------------
    def permute_wires(
        self,
        wire_order: tuple[int, ...] | list[int],
        slot_range: tuple[int, int],
        permute_labels: bool = True,
        permute_wiredata: bool = True,
    ):
        if not hasattr(self, "_perm_windows"):
            self._perm_windows = []
        if not hasattr(self, "_pulse_routes"):
            self._pulse_routes = {i: [] for i in range(len(self.wires))}
        if not hasattr(self, "_perm_groups"):
            self._perm_groups = []
        if not hasattr(self, "_post_perm_maps"):
            self._post_perm_maps = []

        order = [int(i) for i in wire_order]
        if len(order) < 2:
            raise ValueError("wire_order must include at least two wire indices")
        nw = len(self.wires)
        if any(i < 0 or i >= nw for i in order):
            raise IndexError(f"wire indices must be in 0..{nw-1}, got {order}")
        if len(set(order)) != len(order):
            raise ValueError("wire_order must not contain duplicate indices")

        s0, s1 = int(slot_range[0]), int(slot_range[1])
        if s0 > s1:
            s0, s1 = s1, s0
        if not (0 <= s0 < self.num_slots) or not (0 <= s1 < self.num_slots):
            raise ValueError(
                f"slot_range {slot_range} out of bounds 0..{self.num_slots-1}"
            )
        if (s1 - s0) < 1:
            raise ValueError(
                "slot_range must span at least adjacent slots (s1 - s0 >= 1)"
            )

        wire_ys = [w.get_y() for w in self.wires]

        def _x_at_slot(s: int) -> float:
            start = self.wires[0].start
            return (start + RIGHT * (self.pitch * s))[0]

        def _y_of(wi: int) -> float:
            return wire_ys[wi]

        def _find_right_label_mob(wi: int):
            y = _y_of(wi)
            cands = [m for m in self.labels if abs(m.get_center()[1] - y) < 1e-3]
            return max(cands, key=lambda m: m.get_center()[0]) if cands else None

        def _place_right_label_on_wire(mob, wire):
            if mob is None:
                return
            last_slot = max(
                (b for (a, b) in wire.intervals), default=self.num_slots - 1
            )
            pt = wire.point_from_slot(last_slot)
            mob.next_to(pt, RIGHT, buff=self.pitch / 3).set_z_index(1)

        xL, xR = _x_at_slot(s0), _x_at_slot(s1)
        W = xR - xL
        cp_dx = 0.5 * W

        src_color = {i: self.wires[i].get_color() for i in range(nw)}
        src_thick = {
            i: getattr(self.wires[i], "thickness", self.wire_thickness)
            for i in range(nw)
        }
        try:
            bg_color = config.get("background_color", BLACK)
        except Exception:
            bg_color = BLACK

        def _trim_keep_boundaries(w):
            new_ints = []
            for a, b in w.intervals:
                if b < s0 or a > s1:
                    new_ints.append((a, b))
                else:
                    if a <= s0:
                        new_ints.append((a, min(b, s0)))
                    if b >= s1:
                        new_ints.append((max(a, s1), b))
            w.intervals = [(a, b) for (a, b) in new_ints if a <= b]
            w._rebuild_segments()

        for wi in order:
            _trim_keep_boundaries(self.wires[wi])

        selected_tb = sorted(order, key=lambda i: _y_of(i), reverse=True)
        dest_y_by_rank = [_y_of(i) for i in selected_tb]
        dest_idx_by_rank = selected_tb[:]
        dest_idx_for = {src: dest_idx_by_rank[rank] for rank, src in enumerate(order)}
        dest_y_for = {src: dest_y_by_rank[rank] for rank, src in enumerate(order)}

        underlays = VGroup()
        permuted = VGroup()
        window_record = {"xL": xL, "xR": xR, "s0": s0, "s1": s1, "routes": {}}

        for src in order:
            y0 = _y_of(src)
            y1 = dest_y_for[src]
            wid = float(src_thick[src])
            p0 = np.array([xL, y0, 0.0])
            p1 = np.array([xL + cp_dx, y0, 0.0])
            p2 = np.array([xR - cp_dx, y1, 0.0])
            p3 = np.array([xR, y1, 0.0])

            ul = (
                CubicBezier(p0, p1, p2, p3)
                .set_stroke(bg_color, width=wid + 2)
                .set_z_index(-3)
            )
            cv = (
                CubicBezier(p0, p1, p2, p3)
                .set_stroke(src_color[src], width=wid)
                .set_z_index(1)
            )
            underlays.add(ul)
            permuted.add(cv)

            self._pulse_routes.setdefault(src, []).append(
                {"xL": xL, "xR": xR, "p0": p0, "p1": p1, "p2": p2, "p3": p3}
            )
            window_record["routes"][src] = {"p0": p0, "p1": p1, "p2": p2, "p3": p3}

            dw_idx = dest_idx_for[src]
            dw = self.wires[dw_idx]
            needs_split = None
            for a, b in dw.intervals:
                if a < s1 < b:
                    needs_split = (a, b)
                    break
            if needs_split:
                a, b = needs_split
                new_ints = []
                for aa, bb in dw.intervals:
                    if (aa, bb) == (a, b):
                        new_ints.extend([(aa, s1), (s1, bb)])
                    else:
                        new_ints.append((aa, bb))
                dw.intervals = new_ints
                dw._rebuild_segments()
            for (a, b), seg in zip(dw.intervals, dw._segments):
                if a >= s1:
                    seg.set_stroke(src_color[src], width=wid)

        group = VGroup(underlays, permuted)
        self.add(group)
        self.gates.append(group)
        self._perm_groups.append(group)
        self._perm_windows.append(window_record)

        if permute_labels:
            right_lbl_for_src = {src: _find_right_label_mob(src) for src in order}
            for src in order:
                lbl = right_lbl_for_src.get(src)
                if lbl is None:
                    continue
                dest_idx = dest_idx_for[src]
                _place_right_label_on_wire(lbl, self.wires[dest_idx])

        if permute_wiredata:
            right_map = {src: dest_idx_for[src] for src in order}
            self._post_perm_maps.append({"cut_slot": s1, "map": right_map})

            to_rebuild = []
            for g in list(self.gates):
                if g in self._perm_groups:
                    continue
                meta = getattr(g, "_qc_meta", None)
                if not meta:
                    continue
                slot = meta.get("slot", None)
                if slot is None or slot < s1:
                    continue
                to_rebuild.append((g, meta))

            for g, meta in to_rebuild:
                kind = meta.get("kind")
                slot = meta.get("slot")
                orig_wire_idx = meta.get("orig_wire_idx", meta.get("wire_idx", None))
                orig_ctrl_idx = meta.get(
                    "orig_control_idx", meta.get("control_idx", None)
                )
                shift = meta.get("shift", None)
                axis = meta.get("axis", None)
                gshape = meta.get("gate_shape", None)
                color_now = meta.get("color", WHITE)

                if orig_wire_idx is None:
                    continue
                try:
                    self.remove(g)
                    self.gates.remove(g)
                except Exception:
                    pass

                if isinstance(
                    orig_wire_idx, tuple
                ):  # advanced custom; skip retro-remap
                    continue

                tgt = int(orig_wire_idx)
                ctrl = None if orig_ctrl_idx is None else int(orig_ctrl_idx)

                if kind in ("H", "X", "Y", "Z"):
                    self.add_gate(
                        kind,
                        tgt,
                        slot=slot,
                        shift=shift,
                        gate_shape=gshape,
                        color=color_now,
                    )
                elif kind == "M":
                    self.add_gate(
                        kind,
                        tgt,
                        slot=slot,
                        shift=shift,
                        axis=axis,
                        color=color_now,
                        readout_idx=None,
                    )
                elif kind in ("P", "PHASE"):
                    self.add_gate(
                        kind, tgt, slot=slot, shift=shift, axis=axis, color=color_now
                    )
                elif kind in ("CH", "CY", "CNOT", "CX", "CZ", "SWAP"):
                    if ctrl is None:
                        continue
                    if kind in ("CH", "CY"):
                        self.add_gate(
                            kind,
                            tgt,
                            slot=slot,
                            control_idx=ctrl,
                            gate_shape=gshape,
                            color=color_now,
                        )
                    elif kind in ("CNOT", "CX"):
                        self.add_gate(
                            "CNOT", tgt, slot=slot, control_idx=ctrl, shift=shift
                        )
                    elif kind == "CZ":
                        self.add_gate(
                            "CZ", tgt, slot=slot, control_idx=ctrl, shift=shift
                        )
                    elif kind == "SWAP":
                        self.add_gate(
                            "SWAP", tgt, slot=slot, control_idx=ctrl, shift=shift
                        )

        if self.boxed:
            self._make_surrounding_box()
        return group

    def add_bracket(
        self,
        *,
        side: str,  # "left" | "right" | "top" | "bottom"
        idx: tuple[int, int],  # wires (for left/right) or slots (for top/bottom)
        label: Tex | str | None = None,  # LaTeX string or a ready Tex
        color: Color = WHITE,
        stroke_width: float = 2.0,
        buffer: float | None = None,  # distance from the circuit/labels
        label_buff: float = 0.25,  # gap between brace and label
        font_size: int = 36,
        z_index: int = 3,
        fill_opacity: float = 1.0,  # <-- solid fill by default
    ) -> VGroup:
        """
        Curly bracket hugging the circuit on a chosen side. For left/right, `idx`
        are wire indices (inclusive). For top/bottom, `idx` are slot indices
        (inclusive). The brace is positioned beyond the furthest gates *and*
        wire labels, similar to _make_surrounding_box.
        """
        side_l = str(side).lower()
        if side_l not in ("left", "right", "top", "bottom"):
            raise ValueError("side must be one of 'left','right','top','bottom'")

        # --- small helpers ------------------------------------------------------
        def _wire_y(i: int) -> float:
            return self.wires[i].get_y()

        def _x_at_slot(s: int) -> float:
            start = self.wires[0].start
            return (start + RIGHT * (self.pitch * s))[0]

        def _first_slot_on(w) -> int:
            return min(a for (a, b) in w.intervals)

        def _last_slot_on(w) -> int:
            return max(b for (a, b) in w.intervals)

        # furthest right gate edge at the end of a wire (accounts for custom sizes)
        def _right_extent_on_wire(wi: int) -> float:
            last = _last_slot_on(self.wires[wi])
            return self._gate_right_extent_at(wi, last)

        # leftmost visible slot x (start of first segment)
        def _left_extent_on_wire(wi: int) -> float:
            w = self.wires[wi]
            first = _first_slot_on(w)
            return w.point_from_slot(first)[0]

        # include wire labels in extents
        def _right_label_x_on_wire(wi: int) -> float | None:
            wy = _wire_y(wi)
            tol = 1e-3
            cand = [m for m in self.labels if abs(m.get_center()[1] - wy) < tol]
            return max((m.get_right()[0] for m in cand), default=None) if cand else None

        def _left_label_x_on_wire(wi: int) -> float | None:
            wy = _wire_y(wi)
            tol = 1e-3
            cand = [m for m in self.labels if abs(m.get_center()[1] - wy) < tol]
            return min((m.get_left()[0] for m in cand), default=None) if cand else None

        buf = self.pitch / 2 if buffer is None else float(buffer)
        grp = VGroup()

        # -----------------------------------------------------------------------
        # LEFT / RIGHT: vertical brace spanning wires, x chosen beyond gates+labels
        # -----------------------------------------------------------------------
        if side_l in ("left", "right"):
            w0, w1 = int(min(idx)), int(max(idx))
            if not (0 <= w0 <= w1 < len(self.wires)):
                raise IndexError("wire indices out of range")

            y_top = _wire_y(w0)  # higher y
            y_bot = _wire_y(w1)  # lower y
            if y_top < y_bot:
                y_top, y_bot = y_bot, y_top

            # compute outward x
            if side_l == "right":
                xs = []
                for wi in range(w0, w1 + 1):
                    xs.append(_right_extent_on_wire(wi))
                    xlab = _right_label_x_on_wire(wi)
                    if xlab is not None:
                        xs.append(xlab)
                x_for_line = max(xs) + buf
                direction = RIGHT
                label_side = RIGHT
            else:  # left
                xs = []
                for wi in range(w0, w1 + 1):
                    xs.append(_left_extent_on_wire(wi))
                    xlab = _left_label_x_on_wire(wi)
                    if xlab is not None:
                        xs.append(xlab)
                x_for_line = min(xs) - buf
                direction = LEFT
                label_side = LEFT

            # base segment the Brace uses to create the curly shape
            brace_line = Line([x_for_line, y_bot, 0], [x_for_line, y_top, 0])
            brace = Brace(brace_line, direction=direction, buff=0.0)
            brace.set_stroke(color, stroke_width, opacity=1.0)
            brace.set_fill(color, opacity=float(fill_opacity))
            brace.set_z_index(z_index)
            grp.add(brace)

            # label
            if label is not None:
                label_mob = (
                    Tex(label, font_size=font_size)
                    if isinstance(label, str)
                    else label.copy()
                )
                label_mob.set_color(color)
                label_mob.next_to(brace, label_side, buff=label_buff).set_z_index(
                    z_index
                )
                grp.add(label_mob)

            self.add(grp)
            return grp

        # -----------------------------------------------------------------------
        # TOP / BOTTOM: horizontal brace spanning slots, y chosen above/below wires
        # -----------------------------------------------------------------------
        s0, s1 = int(min(idx)), int(max(idx))
        if not (0 <= s0 <= s1 <= self.num_slots):
            raise ValueError("slot indices out of range")

        xL, xR = _x_at_slot(s0), _x_at_slot(s1)
        y_all = [w.get_y() for w in self.wires]
        y_top_all, y_bot_all = max(y_all), min(y_all)

        if side_l == "top":
            y_for_line = y_top_all + self.wire_spacing + buf
            direction = UP
            label_side = UP
        else:
            y_for_line = y_bot_all - self.wire_spacing - buf
            direction = DOWN
            label_side = DOWN

        brace_line = Line([xL, y_for_line, 0], [xR, y_for_line, 0])
        brace = Brace(brace_line, direction=direction, buff=0.0)
        brace.set_stroke(color, stroke_width, opacity=1.0)
        brace.set_fill(color, opacity=float(fill_opacity))
        brace.set_z_index(z_index)
        grp.add(brace)

        if label is not None:
            label_mob = (
                Tex(label, font_size=font_size)
                if isinstance(label, str)
                else label.copy()
            )
            label_mob.set_color(color)
            label_mob.next_to(brace, label_side, buff=label_buff).set_z_index(z_index)
            grp.add(label_mob)

        self.add(grp)
        return grp

    def add_divider(
        self,
        idx: int | float | tuple[int | float, ...] | list[int | float] | None = None,
        *,
        slot: int | float | tuple[int | float, ...] | list[int | float] | None = None,
        wire: int | tuple[int, ...] | list[int] | None = None,
        orientation: str = "vertical",  # "vertical" or "horizontal"
        color: Color = WHITE,
        stroke_width: float | None = None,
        dash_length: float = 0.08,
        buffer: (
            float | None
        ) = None,  # vertical extra span (vertical) or horizontal (horizontal)
        z_index: int = 3,
        stroke: str = "dashed",  # "dashed" | "solid" | "dotted"
        dot_spacing: float | None = None,
        dot_radius: float | None = None,
        shift: str | None = None,  # vertical: "left"/"right"; horizontal: "up"/"down"
    ) -> VGroup:
        """
        Draw one or more dividers and register them on the circuit so they
        can be morphed (cross-faded) by `morph_to`.
        """

        # --- ensure divider registry ---
        if not hasattr(self, "dividers"):
            self.dividers = []
        if not hasattr(self, "_divider_group"):
            self._divider_group = VGroup().set_z_index(3)
            self.add(self._divider_group)

        # ---------- helpers ----------
        def _norm(v):
            if v is None:
                return []
            if isinstance(v, (list, tuple)):
                return list(v)
            return [v]

        def _x_at_slot(s: float, off: float = 0.0) -> float:
            start = self.wires[0].start
            return (start + RIGHT * (self.pitch * s + off))[0]

        def _solid(a, b):
            return Line(a, b).set_stroke(color, sw).set_z_index(z_index)

        def _dashed(a, b):
            return (
                DashedLine(a, b, dash_length=dash_length)
                .set_stroke(color, sw)
                .set_z_index(z_index)
            )

        def _dotted(a, b):
            p0, p1 = np.array(a), np.array(b)
            v = p1 - p0
            L = float(np.linalg.norm(v))
            if L <= 0:
                return VGroup()
            u = v / L
            spacing = dot_spacing if dot_spacing is not None else dash_length
            n = max(2, int(np.floor(L / spacing)) + 1)
            r = dot_radius if dot_radius is not None else min(0.06, 0.25 * spacing)
            dots = VGroup()
            for i in range(n):
                pos = p0 + u * (i * L / (n - 1))
                dots.add(
                    Dot(radius=r)
                    .set_fill(color, 1)
                    .set_stroke(width=0)
                    .move_to(pos)
                    .set_z_index(z_index)
                )
            return dots

        maker = {"solid": _solid, "dashed": _dashed, "dotted": _dotted}.get(
            str(stroke).lower(), _dashed
        )
        sw = stroke_width if stroke_width is not None else 4 * self.gate_size
        grp = VGroup()

        ori = str(orientation).lower()
        if ori not in ("vertical", "horizontal"):
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

        if ori == "vertical":
            idx_list = _norm(idx)
            slot_list = _norm(slot)
            items = idx_list or slot_list
            if not items:
                raise ValueError("Provide `idx` or `slot` for a vertical divider.")

            # record which parameter was used ("idx" or "slot")
            index_field = "idx" if idx_list else "slot"

            wire_ys = [w.get_y() for w in self.wires]
            if not wire_ys:
                raise RuntimeError("No wires to span for vertical divider.")
            ext_y = self.wire_spacing if buffer is None else float(buffer)
            y_bot = min(wire_ys) - ext_y
            y_top = max(wire_ys) + ext_y

            for s in items:
                base_x = _x_at_slot(float(s)) + 0.5 * self.pitch
                if shift:
                    sh = shift.lower()
                    base_x += (
                        self.pitch / 3
                        if sh in ("r", "right")
                        else (-self.pitch / 3 if sh in ("l", "left") else 0.0)
                    )
                a = [base_x, y_bot, 0.0]
                b = [base_x, y_top, 0.0]
                grp.add(maker(a, b))

        else:  # horizontal
            items = _norm(idx) or _norm(wire)
            if not items:
                raise ValueError("Provide `idx` or `wire` for a horizontal divider.")
            index_field = "wire"

            n_w = len(self.wires)
            if n_w < 2:
                raise RuntimeError("Need at least two wires for a horizontal divider.")

            # left/right extents include label-aware rightmost gate reach
            def _first_slot_on(w):
                return min(a for (a, b) in w.intervals)

            def _last_slot_on(w):
                return max(b for (a, b) in w.intervals)

            x_lefts = []
            x_rights = []
            for wi in range(n_w):
                w = self.wires[wi]
                x_lefts.append(w.point_from_slot(_first_slot_on(w))[0])
                x_rights.append(self._gate_right_extent_at(wi, _last_slot_on(w)))
            x_min = min(x_lefts)
            x_max = max(x_rights)

            ext_x = (self.pitch / 2) if buffer is None else float(buffer)
            xL = x_min - ext_x
            xR = x_max + ext_x

            for w_idx in items:
                wi = int(w_idx)
                if not (0 <= wi < n_w - 1):
                    raise IndexError(f"wire index {wi} invalid (needs 0..{n_w-2}).")
                y0 = self.wires[wi].get_y()
                y1 = self.wires[wi + 1].get_y()
                base_y = 0.5 * (y0 + y1)
                if shift:
                    sh = shift.lower()
                    base_y += (
                        self.wire_spacing / 3
                        if sh in ("up", "u")
                        else (-self.wire_spacing / 3 if sh in ("down", "d") else 0.0)
                    )
                a = [xL, base_y, 0.0]
                b = [xR, base_y, 0.0]
                grp.add(maker(a, b))

        # ---- attach metadata for morphing ----
        grp._qc_divider_meta = {
            "orientation": ori,  # "vertical" | "horizontal"
            "index_field": index_field,  # "idx" | "slot" | "wire"
            "indices": [float(x) for x in items],  # numbers only
            "color": color,
            "stroke_width": sw,
            "dash_length": float(dash_length),
            "buffer": (None if buffer is None else float(buffer)),
            "z_index": int(z_index),
            "stroke": str(stroke).lower(),  # "dashed" | "solid" | "dotted"
            "dot_spacing": (None if dot_spacing is None else float(dot_spacing)),
            "dot_radius": (None if dot_radius is None else float(dot_radius)),
            "shift": (None if shift is None else str(shift)),
        }

        # ---- register on circuit (NOT in self.gates) ----
        self.dividers.append(grp)
        self._divider_group.add(grp)
        self.add(grp)  # keep behavior identical to before

        return grp

    def morph_to(
        self,
        scene,
        target_qc: "QuantumCircuit",
        *,
        run_time: float = 1.0,
        rate_func=smooth,
        keep_palette: bool = True,
    ):
        """
        Morph this circuit into `target_qc` via a cross-fade.

        Parameters
        ----------
        scene : Scene
            The Manim scene playing the animation.
        target_qc : QuantumCircuit
            The circuit whose gates/dividers to morph into.
        run_time : float, optional
            Duration of the cross-fade animation (seconds).
        rate_func : Callable, optional
            Manim rate function used for the animation.
        keep_palette : bool, optional
            If True (default), keep this circuit’s bg/palette and only change geometry.
            If False, also adopt target’s bg_color, palette, and per-gate colors.

        Returns
        -------
        QuantumCircuit
            self
        """
        # Adopt target palette/colors only when requested.
        if not keep_palette:
            self.bg_color = getattr(target_qc, "bg_color", self.bg_color)
            self._scheme_name = getattr(target_qc, "_scheme_name", None)
            self._palette = list(getattr(target_qc, "_palette", []))
            self._palette_i = int(getattr(target_qc, "_palette_i", 0))
            self._kind_to_color = dict(getattr(target_qc, "_kind_to_color", {}))

        # Fill background to use for gates we know draw a filled body.
        bg_for_new = (
            self.bg_color
            if keep_palette
            else getattr(target_qc, "bg_color", self.bg_color)
        )

        # Ensure divider registry exists.
        if not hasattr(self, "dividers"):
            self.dividers = []
        if not hasattr(self, "_divider_group"):
            self._divider_group = VGroup().set_z_index(3)
            self.add(self._divider_group)

        # ----------------- helpers -----------------
        def _is_single(g):
            return g is not None and hasattr(g, "_box") and hasattr(g, "_label")

        def _spec_from_gate(g):
            meta = getattr(g, "_qc_meta", {}) or {}
            kind = (meta.get("kind") or "").upper()
            slot = meta.get("slot", None)
            placed = meta.get("placed_wire_idx", None)
            ctrl = meta.get("placed_control_idx", None)
            shape = meta.get("gate_shape", None)
            shift = meta.get("shift", None)
            axis = meta.get("axis", None)
            angle = meta.get("angle_tex", None)
            color = g.get_color() if hasattr(g, "get_color") else None

            # Reconstruct dagger if needed from label.
            dag = bool(meta.get("dag", False))
            if not dag and _is_single(g):
                ts = getattr(getattr(g, "_label", None), "tex_string", "") or ""
                if r"^{\dagger}" in ts:
                    dag = True

            label_tex = (
                getattr(getattr(g, "_label", None), "tex_string", None)
                if _is_single(g)
                else None
            )

            return {
                "kind": kind,
                "slot": slot,
                "wire_idx": placed if not isinstance(placed, tuple) else None,
                "control_idx": ctrl,
                "gate_shape": shape,
                "shift": shift,
                "axis": axis,
                "angle_tex": angle,
                "dag": dag,
                "color": color,
                "label_tex": label_tex,
            }

        def _set_alpha_vm(mob, a: float):
            """Recursively set both fill and stroke opacity (ManimGL)."""
            try:
                mob.set_opacity(a)
            except Exception:
                pass
            try:
                mob.set_stroke(opacity=a)
            except Exception:
                pass
            for sm in getattr(mob, "submobjects", []):
                _set_alpha_vm(sm, a)

        def _fade_anim(group, start_alpha, end_alpha):
            def upd(_m, t):
                a = start_alpha + (end_alpha - start_alpha) * t
                for sm in group.submobjects:
                    _set_alpha_vm(sm, a)

            driver = Dot(radius=0.001, color=BLACK).set_opacity(0)
            return UpdateFromAlphaFunc(driver, upd)

        def _copy_divider_specs(qc: "QuantumCircuit"):
            specs = []
            for d in getattr(qc, "dividers", []):
                meta = getattr(d, "_qc_divider_meta", None)
                if meta:
                    specs.append(dict(meta))  # shallow copy
            return specs

        def _build_dividers_from_specs(specs: list[dict]):
            new_divs = []
            for sp in specs:
                ori = sp["orientation"]
                field = sp["index_field"]
                vals = sp["indices"]
                kwargs = dict(
                    orientation=ori,
                    color=sp["color"],
                    stroke_width=sp["stroke_width"],
                    dash_length=sp["dash_length"],
                    buffer=sp["buffer"],
                    z_index=sp["z_index"],
                    stroke=sp["stroke"],
                    dot_spacing=sp["dot_spacing"],
                    dot_radius=sp["dot_radius"],
                    shift=sp["shift"],
                )
                if ori == "vertical":
                    div = (
                        self.add_divider(idx=vals, **kwargs)
                        if field == "idx"
                        else self.add_divider(slot=vals, **kwargs)
                    )
                else:
                    div = self.add_divider(wire=vals, **kwargs)
                _set_alpha_vm(div, 0.0)  # start invisible for cross-fade
                new_divs.append(div)
            return new_divs

        # ----------------- capture current + build new (hidden) -----------------
        old_gates = list(self.gates)
        old_divs = list(self.dividers)
        old_group = VGroup(*old_gates, *old_divs)

        gate_specs = [_spec_from_gate(g) for g in target_qc.gates]
        new_gates: list = []

        def _wants_bg(k: str) -> bool:
            ku = (k or "").upper()
            # Filled bodies
            if ku in {
                "M",
                "CNOT",
                "CUSTOM",
                "H",
                "X",
                "Y",
                "Z",
                "S",
                "T",
                "U",
                "SQX",
                "SQY",
                "SQZ",
                "R",
                "RX",
                "RY",
                "RZ",
            }:
                return True
            if ku.startswith("C") and ku != "CNOT":  # CH, CY, CRZ, ...
                return True
            # Outline-only gates
            return False  # CZ, SWAP, PHASE, BUNDLE, etc.

        for sp in gate_specs:
            kind = sp["kind"] or "U"
            slot = sp["slot"]
            widx = sp["wire_idx"]
            if slot is None or widx is None:
                continue

            params = {
                "shift": sp["shift"],
                "gate_shape": sp["gate_shape"],
            }

            # Rotations
            if kind in {"R", "RX", "RY", "RZ"}:
                if kind in {"RX", "RY", "RZ"}:
                    params["axis"] = {"RX": "X", "RY": "Y", "RZ": "Z"}[kind]
                    kind = "R"
                else:
                    params["axis"] = sp["axis"]
                params["angle"] = sp["angle_tex"]

            # Custom/labelled single
            if kind == "U" and sp["label_tex"]:
                params["label"] = sp["label_tex"]

            # Dagger for single-qubit families
            if kind in {"H", "X", "Y", "Z", "S", "T", "SQX", "SQY", "SQZ", "R", "U"}:
                params["dag"] = bool(sp["dag"])

            # Two-qubit / controlled-single: keep control_idx
            if kind in {"CNOT", "CZ", "SWAP", "CH", "CY", "CSQX", "CSQY", "CSQZ"}:
                params["control_idx"] = sp["control_idx"]

            # Create new gate (only pass bg_color to gate types that use a fill).
            if _wants_bg(kind):
                newg = self.add_gate(
                    kind, widx, slot=int(slot), bg_color=bg_for_new, **params
                )
            else:
                newg = self.add_gate(kind, widx, slot=int(slot), **params)

            # Only copy target per-gate color when adopting its palette.
            if (
                not keep_palette
                and hasattr(newg, "set_color")
                and sp["color"] is not None
            ):
                newg.set_color(sp["color"])

            _set_alpha_vm(newg, 0.0)  # start invisible
            new_gates.append(newg)

        # Dividers after gates (so horizontal extents see new geometry)
        divider_specs = _copy_divider_specs(target_qc)
        new_divs = _build_dividers_from_specs(divider_specs)

        new_group = VGroup(*new_gates, *new_divs)

        # ----------------- animate cross-fade -----------------
        fade_out_old = _fade_anim(old_group, start_alpha=1.0, end_alpha=0.0)
        fade_in_new = _fade_anim(new_group, start_alpha=0.0, end_alpha=1.0)
        scene.play(
            AnimationGroup(fade_out_old, fade_in_new, lag_ratio=0.0),
            run_time=run_time,
            rate_func=rate_func,
        )

        # ----------------- finalize -----------------
        try:
            self.remove(*old_gates, *old_divs)
        except Exception:
            pass
        for og in old_gates + old_divs:
            og.clear_updaters()

        self.dividers = new_divs
        try:
            self.remove(self._divider_group)
        except Exception:
            pass
        self._divider_group = VGroup(*new_divs).set_z_index(3)
        self.add(self._divider_group)

        if getattr(self, "boxed", False):
            self._make_surrounding_box()

        return self
