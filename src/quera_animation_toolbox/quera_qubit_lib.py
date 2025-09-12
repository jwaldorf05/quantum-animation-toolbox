import time
from functools import cached_property
from typing import Sequence  # Kept Sequence, removed Generic & TypeVar

import numpy as np
from colour import Color
from manimlib import *
from scipy.spatial import KDTree

from .quera_colors import *

# # GlowDot Object Presets
#
# DEFAULT_DOT_RADIUS = 0.05
# DEFAULT_GLOW_DOT_RADIUS = 0.2
# DEFAULT_GRID_HEIGHT = 6
# DEFAULT_BUFF_RATIO = 0.5


class Qubit(Group):
    """
    A qubit: visible Dot + faint GlowDot. Layering is deterministic:
      glow  z = -2
      dot   z =  0
      label z = +1
    """

    def __init__(
        self,
        position=ORIGIN,
        radius=0.15,
        glow_radius=0.4,
        color=QUERA_PURPLE,
        **kwargs,
    ):
        dot = Dot(position, radius=radius).set_z_index(0)
        glow = GlowDot(position, radius=glow_radius, color=color)
        glow.add_updater(lambda m: m.move_to(dot.get_center()))
        glow.set_opacity(0).set_z_index(-2)

        # store parts before super
        self.dot = dot
        self.glow = glow
        self._orig_color = color  # single source of truth for baseline
        self.label = None

        super().__init__(dot, glow, **kwargs)

        # initialize visual color from baseline (non-animated)
        self.set_color(color, animate=False, update_internal=False)

    # ---- helpers ----
    def baseline_color(self):
        """Return the stored, intended baseline color."""
        return getattr(self, "_orig_color", self.dot.get_color())

    # return the *visual* color, not the bookkeeping field
    def get_color(self):
        return self.dot.get_color()

    def set_color(self, new_color, animate=False, update_internal=False):
        """Set visual color; optionally also update the baseline color."""
        if update_internal:
            self._orig_color = (
                new_color  # mutate baseline only when explicitly requested
            )

        if animate:
            return [
                self.dot.animate.set_color(new_color),
                self.glow.animate.set_color(new_color),
            ]
        else:
            self.dot.set_color(new_color)
            self.glow.set_color(new_color)

    def reset_color(self, animate=False):
        return self.set_color(self.baseline_color(), animate, update_internal=False)

    def turn_glow(self, on=True, animate=False):
        target = 1.0 if on else 0.0
        return (
            [self.glow.animate.set_opacity(target)]
            if animate
            else self.glow.set_opacity(target)
        )

    def pulse_once(self, run_time=3.0, scale_factor=2.5):
        base_r = self.glow.get_radius()
        self.glow.set_opacity(0)
        self.turn_glow(True)

        fade_in = ApplyMethod(self.glow.set_opacity, 1, run_time=0)
        grow = ApplyMethod(
            self.glow.set_radius,
            base_r * scale_factor,
            rate_func=there_and_back,
            run_time=run_time,
        )
        fade_out = ApplyMethod(self.glow.set_opacity, 0, run_time=0)

        return Succession(fade_in, grow, fade_out, lag_ratio=0.0)

    def start_pulsing(self, pulse_period=1.0, scale_factor=1.5):
        self.turn_glow(True)
        base_r = self.glow.get_radius()
        t0 = time.time()

        def _pulse(m, dt):
            elapsed = time.time() - t0
            phase = 2 * np.pi * elapsed / pulse_period
            opacity = 0.5 * (1 - np.cos(phase))
            m.set_opacity(opacity).set_radius(base_r * scale_factor)

        self.glow.add_updater(_pulse)

    def stop_pulsing(self):
        self.glow.clear_updaters()
        self.turn_glow(False)

    def get_z_index(self):
        return self.dot.z_index

    def add_label(self, tex, color=WHITE, scale: float | None = None):
        """
        Add a LaTeX label. If scale is None, scale with radius
        (baseline: radius=0.15 → scale≈0.5).
        """
        if scale is None:
            base_r = 0.15
            r_now = 0.5 * self.dot.get_width()
            scale = 0.5 * (r_now / base_r)

        label = Tex(str(tex), color=color).scale(float(scale))
        label.move_to(self.dot.get_center())
        label.set_z_index(self.get_z_index() + 1)
        self.label = label
        self.add(label)

    def remove_label(self):
        if hasattr(self, "label") and self.label is not None:
            self.remove(self.label)
            self.label = None

    def set_label_color(self, new_color, animate: bool = False):
        """Set the label color if a label exists."""
        if getattr(self, "label", None) is None:
            return self  # no-op if there’s no label
        if animate:
            return self.label.animate.set_color(new_color)
        else:
            self.label.set_color(new_color)
            return self


class DashedObj(VMobject):
    """
    Fixed the ManimGL DashedObject class
    """

    def __init__(
        self,
        vmobject: VMobject,
        num_dashes: int = 15,
        positive_space_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if num_dashes > 0:
            # End points of the unit interval for division
            alphas = np.linspace(0, 1, num_dashes + 2)

            # This determines the length of each "dash"
            full_d_alpha = 1.0 / (num_dashes + 1)
            partial_d_alpha = full_d_alpha * positive_space_ratio

            # Rescale so that the last point of vmobject will
            # be the end of the last dash
            alphas /= 1 - full_d_alpha

            self.add(
                *[
                    vmobject.get_subcurve(alpha, alpha + partial_d_alpha)
                    for alpha in alphas[:-1]
                ]
            )
        # Family is already taken care of by get_subcurve
        # implementation
        self.match_style(vmobject, recurse=False)


class Vacancy(VGroup):
    """
    Dashed circular placeholder with an optional subtle center label.

    Layering (no behind=True anywhere):
      arcs  z = -4
      label z = -5
    """

    _Z_ARCS = -4
    _Z_LABEL = -5

    def __init__(
        self,
        position=ORIGIN,
        radius=0.15,
        color=QUERA_MGRAY,
        stroke_width: float | None = None,
        num_dashes=8,
        dash_ratio=0.55,
        start_angle=0.0,
        **kwargs,
    ):
        self.label = None
        super().__init__(**kwargs)

        self._radius = float(radius)
        self._color = color
        # proportional thickness; matches dividers below
        self._stroke_width = (
            float(stroke_width) if stroke_width is not None else 20.0 * self._radius
        )
        self._n = max(1, int(num_dashes))
        self._dash_ratio = float(np.clip(dash_ratio, 0.0, 1.0))
        self._theta0 = float(start_angle)

        self._build_arcs()
        self.move_to(
            np.array(
                position if len(position) == 3 else [position[0], position[1], 0.0]
            )
        )

    # ---------- internals ----------

    def _build_arcs(self):
        existing_label = self.label
        self.submobjects = []

        dθ = (2 * np.pi) / self._n
        dashθ = max(1e-4, dθ * self._dash_ratio)
        for k in range(self._n):
            a0 = self._theta0 + k * dθ
            arc = Arc(start_angle=a0, angle=dashθ, radius=self._radius)
            arc.set_fill(opacity=0.0)
            arc.set_stroke(self._color, width=self._stroke_width, opacity=1.0)
            arc.set_z_index(self._Z_ARCS)
            self.add(arc)

        if existing_label is not None:
            self.label = existing_label
            if self.label not in self.submobjects:
                self.add(self.label)
            self._center_label()
            self.label.set_z_index(self._Z_LABEL)

    def _center_label(self):
        if self.label is not None:
            self.label.move_to(self.get_center())

    def _label_tone(self, base):
        try:
            return interpolate_color(base, BLACK, 0.55)
        except Exception:
            return QUERA_MGRAY

    # ---------- public API ----------

    def set_color(self, new_color, new_opacity=None):
        self._color = new_color
        for m in self.submobjects:
            if m is self.label:
                continue
            m.set_stroke(
                new_color,
                width=self._stroke_width,
                opacity=(
                    m.get_stroke_opacity() if new_opacity is None else new_opacity
                ),
            )
            m.set_z_index(self._Z_ARCS)
        if self.label is not None:
            self.label.set_color(self._label_tone(new_color)).set_z_index(self._Z_LABEL)
        return self

    def set_stroke(self, color=None, width=None, opacity=None, **kwargs):
        if color is not None:
            self._color = color
        if width is not None:
            self._stroke_width = float(width)
        for m in self.submobjects:
            if m is self.label:
                continue
            m.set_stroke(
                self._color if color is None else color,
                width=self._stroke_width,
                opacity=m.get_stroke_opacity() if opacity is None else opacity,
                **kwargs,
            ).set_z_index(self._Z_ARCS)
        if color is not None and self.label is not None:
            self.label.set_color(self._label_tone(color)).set_z_index(self._Z_LABEL)
        return self

    def add_label(
        self,
        tex,
        *,
        color=None,
        scale: float | None = None,  # auto-scale with radius if None
        opacity: float | None = None,
        z_bump: int = 0,
        **tex_kwargs,
    ):
        if scale is None:
            base_r = 0.15
            scale = 0.5 * (self._radius / base_r)

        lbl = Tex(
            str(tex),
            color=(color if color is not None else self._label_tone(self._color)),
            **tex_kwargs,
        ).scale(scale)
        if opacity is not None:
            lbl.set_opacity(float(opacity))

        self.label = lbl
        self._center_label()
        self.label.set_z_index(min(self._Z_LABEL + int(z_bump), self._Z_ARCS - 1))

        if self.label not in self.submobjects:
            self.add(self.label)
        return self

    def remove_label(self):
        if self.label is not None:
            try:
                self.remove(self.label)
            except Exception:
                pass
            self.label = None

    def set_radius(self, r):
        self._radius = float(r)
        # keep thickness proportional on radius changes
        self._stroke_width = (
            20.0 * self._radius if self._stroke_width is None else self._stroke_width
        )
        self._build_arcs()
        self._center_label()
        return self

    def set_num_dashes(self, n, dash_ratio=None):
        self._n = max(1, int(n))
        if dash_ratio is not None:
            self._dash_ratio = float(np.clip(dash_ratio, 0.0, 1.0))
        self._build_arcs()
        self._center_label()
        return self


class QubitArray(Group):
    """
    A container that can lay out *qubits* and/or *vacancies* in three styles:
    ``linear``, ``grid`` or ``bilinear`` (two rows).

    Parameters
    ----------
    layout          : {"linear", "grid", "bilinear"}
    num_qubits      : int                – linear / bilinear
    rows, cols      : int                – grid
    group_spacing   : float              – gap between two lines or grids
    qubit_spacing   : float              – centre-to-centre spacing (x direction)
    use_vacancies   : bool               – create Vacancy markers as well?
    fill_pattern    : {"all","none","alternate", "checkerboard", "checkerboard2"} or set[int]
                      When vacancies **and** qubits are both requested, this
                      decides which vacancy indices get an initial qubit.

    Notes
    -----
    ``fill_pattern`` is ignored when *use_vacancies=False* (only qubits
    are created) **or** when *use_vacancies=True* but you pass
    ``fill_pattern="none"``.c
    """

    def __init__(self, **params):
        group_kwargs = params.pop("group_kwargs", {})
        super().__init__(**group_kwargs)

        defaults = {
            "layout": "linear",
            "num_qubits": 6,
            "rows": None,
            "cols": None,
            "size1": None,
            "size2": None,  # for bigrid
            "qubit_spacing": 1.0,
            "row_spacing": None,  # per-axis step in Y; if None → use qubit_spacing
            "col_spacing": None,  # per-axis step in X; if None → use qubit_spacing
            "group_spacing": 2.0,
            "orientation": "horizontal",
            "divider": False,
            "use_vacancies": True,
            "fill_pattern": "all",
            "qubit_labels": False,
            "vacancy_labels": False,
            # styling
            "qubit_radius": 0.15,
            "glow_radius": 0.3,
            "color": QUERA_PURPLE,
            "vac_radius": None,  # if None → tie to qubit_radius
            "vac_color": QUERA_MGRAY,
        }
        defaults.update(params)
        uv_raw = defaults.get("use_vacancies", False)
        vac_invisible = False
        if isinstance(uv_raw, str) and uv_raw.lower() in (
            "invisible",
            "invis",
            "ghost",
            "hidden",
            "hide",
        ):
            # Treat as enabled vacancies, but invisible & unlabeled.
            defaults["use_vacancies"] = True
            defaults["vacancy_labels"] = False
            vac_invisible = True

        defaults["_vac_invisible"] = vac_invisible

        # normalize orientation
        ori = defaults.get("orientation")
        if isinstance(ori, str) and ori.lower() in ("v", "vertical"):
            defaults["orientation"] = "vertical"
        elif isinstance(ori, str) and ori.lower() in ("h", "horizontal"):
            defaults["orientation"] = "horizontal"

        # allow 'radius' alias for qubit
        if "radius" in defaults and "qubit_radius" not in defaults:
            defaults["qubit_radius"] = defaults["radius"]

        # tie vacancy radius to qubit radius if not provided
        if defaults.get("vac_radius") is None:
            defaults["vac_radius"] = 1.08 * float(defaults["qubit_radius"])

        # stash attrs
        for k, v in defaults.items():
            setattr(self, k, v)

        # stroke width used by Vacancy and dividers (kept identical)
        self._vac_stroke_width = 20.0 * float(self.vac_radius)

        # styles passed into Qubit / Vacancy
        self._q_style = dict(
            radius=float(self.qubit_radius),
            glow_radius=float(self.glow_radius),
            color=self.color,
        )
        self._v_style = dict(
            radius=float(self.vac_radius),
            color=self.vac_color,
            stroke_width=self._vac_stroke_width,  # <— match divider thickness
        )

        self.qubits = []
        self.vacancies = []

        # dispatch
        match self.layout:
            case "linear":
                self._build_linear()
            case "grid":
                self._create_grid(self.rows, self.cols)
            case "bilinear":
                self._build_bilinear()
            case "bigrid":
                if self.size1 is None or self.size2 is None:
                    raise ValueError("bigrid requires size1 and size2 tuples")
                self._build_bigrid()
            case other:
                raise ValueError(f"Unknown layout {other!r}")

    # ——— private builders —————————————————————————————————————————————
    def _build_linear(self):
        """Lay out self.num_qubits in one row at y=0, with vacancies if requested."""
        self._create_row(
            n=self.num_qubits, y=0.0, build_qubit=True, build_vac=self.use_vacancies
        )

    def _build_grid(self):
        """Lay out a self.rows×self.cols grid with per-axis spacing."""
        rs = float(
            self.row_spacing if self.row_spacing is not None else self.qubit_spacing
        )  # Y step
        cs = float(
            self.col_spacing if self.col_spacing is not None else self.qubit_spacing
        )  # X step

        for r in range(self.rows):
            for c in range(self.cols):
                x = (c - (self.cols - 1) / 2.0) * cs
                y = -(r - (self.rows - 1) / 2.0) * rs
                pos = np.array([x, y, 0.0])

                if self.use_vacancies:
                    self._spawn_vac(pos)
                    if self._want_qubit_here(idx=r * self.cols + c, row=r, col=c):
                        self._spawn_qubit(pos)
                else:
                    self._spawn_qubit(pos)

    def _build_bilinear(self):
        n = self.num_qubits
        s = self.qubit_spacing
        g = self.group_spacing

        if self.orientation == "horizontal":
            offsets = [(-g / 2, 0), (+g / 2, 0)]
            half_h = (n - 1) * s / 2
            divider_p1 = np.array([0.0, half_h + s / 2, 0.0])
            divider_p2 = np.array([0.0, -half_h - s / 2, 0.0])
        else:
            offsets = [(0, +g / 2), (0, -g / 2)]
            half_w = (n - 1) * s / 2
            divider_p1 = np.array([half_w + s / 2, 0.0, 0.0])
            divider_p2 = np.array([-half_w - s / 2, 0.0, 0.0])

        # first half
        self._in_first_grid = True
        xo, yo = offsets[0]
        if self.orientation == "horizontal":
            self._create_column(
                n=n, x=xo, build_qubit=True, build_vac=self.use_vacancies, base_index=0
            )
        else:
            self._create_row(
                n=n, y=yo, build_qubit=True, build_vac=self.use_vacancies, base_index=0
            )

        # second half
        self._in_first_grid = False
        xo, yo = offsets[1]
        if self.orientation == "horizontal":
            self._create_column(
                n=n, x=xo, build_qubit=True, build_vac=self.use_vacancies, base_index=n
            )
        else:
            self._create_row(
                n=n, y=yo, build_qubit=True, build_vac=self.use_vacancies, base_index=n
            )
        del self._in_first_grid

        # divider (thickness & dash length proportional to radius; color = vacancy)
        if self.divider:
            dash_len = (2.0 / 3.0) * float(self.vac_radius)  # ≈0.1 when radius=0.15
            divider = DashedLine(divider_p1, divider_p2, dash_length=dash_len)
            divider.set_stroke(self._v_style["color"], width=self._vac_stroke_width)
            divider.set_z_index(-3)  # always below qubit dots (z=0), above bg
            self.add(divider)

    def _build_bigrid(self):
        rows1, cols1 = self.size1
        rows2, cols2 = self.size2

        # Per-axis steps (fallback to qubit_spacing if unspecified)
        rs = float(
            self.row_spacing if self.row_spacing is not None else self.qubit_spacing
        )  # row (Y) step
        cs = float(
            self.col_spacing if self.col_spacing is not None else self.qubit_spacing
        )  # col (X) step
        g = float(self.group_spacing)

        if self.orientation == "horizontal":
            # two blocks left/right → separate along X by half-widths + group gap
            half_w1 = (cols1 - 1) * cs / 2.0
            half_w2 = (cols2 - 1) * cs / 2.0
            center_dist = half_w1 + half_w2 + g
            offsets = [(-center_dist / 2.0, 0.0), (+center_dist / 2.0, 0.0)]
        else:
            # two blocks top/bottom → separate along Y by half-heights + group gap
            half_h1 = (rows1 - 1) * rs / 2.0
            half_h2 = (rows2 - 1) * rs / 2.0
            center_dist = half_h1 + half_h2 + g
            offsets = [(0.0, +center_dist / 2.0), (0.0, -center_dist / 2.0)]

        # ---- stable row-major indices per subgrid ----
        def make_subgrid(rows, cols, xo, yo, base_idx):
            for r in range(rows):
                for c in range(cols):
                    x = xo + (c - (cols - 1) / 2.0) * cs
                    y = yo - (r - (rows - 1) / 2.0) * rs
                    pos = np.array([x, y, 0.0])

                    if self.use_vacancies:
                        self._spawn_vac(pos)
                        idx = base_idx + (r * cols + c)  # stable vacancy index
                        if self._want_qubit_here(idx=idx, row=r, col=c):
                            self._spawn_qubit(pos)
                    else:
                        self._spawn_qubit(pos)

        # First block (top/left) occupies [0 .. rows1*cols1-1]
        base0 = 0
        self._in_first_grid = True
        make_subgrid(rows1, cols1, *offsets[0], base0)

        # Second block continues after the first
        base1 = base0 + rows1 * cols1
        self._in_first_grid = False
        make_subgrid(rows2, cols2, *offsets[1], base1)
        del self._in_first_grid

        # Optional divider: draw between blocks using the larger opposing span
        if getattr(self, "divider", False):
            dash_len = (2.0 / 3.0) * float(self.vac_radius)

            if self.orientation == "horizontal":
                # vertical divider at x = x_mid
                half_h1 = (rows1 - 1) * rs / 2.0
                half_h2 = (rows2 - 1) * rs / 2.0
                y_ext = max(half_h1, half_h2) + rs / 2.0

                half_w1 = (cols1 - 1) * cs / 2.0
                half_w2 = (cols2 - 1) * cs / 2.0
                x_mid = (half_w1 - half_w2) / 2.0  # == (cols1 - cols2) * cs / 4

                p1 = np.array([x_mid, +y_ext, 0.0])
                p2 = np.array([x_mid, -y_ext, 0.0])

            else:  # vertical orientation
                # horizontal divider at y = y_mid
                half_w1 = (cols1 - 1) * cs / 2.0
                half_w2 = (cols2 - 1) * cs / 2.0
                x_ext = max(half_w1, half_w2) + cs / 2.0

                half_h1 = (rows1 - 1) * rs / 2.0
                half_h2 = (rows2 - 1) * rs / 2.0
                y_mid = (half_h2 - half_h1) / 2.0  # == (rows2 - rows1) * rs / 4

                p1 = np.array([-x_ext, y_mid, 0.0])
                p2 = np.array([+x_ext, y_mid, 0.0])

            divider = DashedLine(p1, p2, dash_length=dash_len)
            divider.set_stroke(self._v_style["color"], width=self._vac_stroke_width)
            divider.set_z_index(-3)
            self.add(divider)

    def _create_subgrid(self, rows, cols, x0=0.0, y0=0.0):
        rs = float(
            self.row_spacing if self.row_spacing is not None else self.qubit_spacing
        )
        cs = float(
            self.col_spacing if self.col_spacing is not None else self.qubit_spacing
        )

        for r in range(rows):
            for c in range(cols):
                x = x0 + (c - (cols - 1) / 2.0) * cs
                y = y0 - (r - (rows - 1) / 2.0) * rs
                pos = np.array([x, y, 0.0])

                if self.use_vacancies:
                    self._spawn_vac(pos)
                    idx = len(self.qubits) + len(self.vacancies) - 1
                    if self._want_qubit_here(idx=idx, row=r, col=c):
                        self._spawn_qubit(pos)
                else:
                    self._spawn_qubit(pos)

    @classmethod
    def linear(
        cls,
        num_qubits: int = 6,
        qubit_spacing: float = 1.0,
        use_vacancies: bool = True,
        **kwargs,
    ):
        return cls(
            layout="linear",
            num_qubits=num_qubits,
            qubit_spacing=qubit_spacing,
            use_vacancies=use_vacancies,  # <-- fix
            **kwargs,
        )

    @classmethod
    def grid(
        cls,
        rows: int,
        cols: int,
        qubit_spacing: float = 1.0,
        *,
        row_spacing: float | None = None,  # NEW
        col_spacing: float | None = None,  # NEW
        **kwargs,
    ):
        return cls(
            layout="grid",
            rows=rows,
            cols=cols,
            qubit_spacing=qubit_spacing,
            row_spacing=row_spacing,  # pass through
            col_spacing=col_spacing,  # pass through
            **kwargs,
        )

    @classmethod
    def bilinear(
        cls,
        num_qubits: int = 6,
        qubit_spacing: float = 1.0,
        group_spacing: float = 2.0,
        orientation: str = "vertical",  # vertical → rows; horizontal → cols
        divider: bool = False,
        **kwargs,
    ):
        return cls(
            layout="bilinear",
            num_qubits=num_qubits,
            qubit_spacing=qubit_spacing,
            group_spacing=group_spacing,
            orientation=orientation,
            divider=divider,
            **kwargs,
        )

    @classmethod
    def bigrid(
        cls,
        size1: tuple[int, int],
        size2: tuple[int, int],
        *,
        orientation: str = "horizontal",
        qubit_spacing: float = 1.0,
        row_spacing: float | None = None,  # NEW
        col_spacing: float | None = None,  # NEW
        group_spacing: float = 2.0,
        use_vacancies: bool = True,
        fill_pattern="all",
        qubit_labels: bool = False,
        # styling ↓
        radius: float = 0.15,
        glow_radius: float = 0.3,
        color=QUERA_PURPLE,
        vac_radius: float = 0.15,
        vac_color=QUERA_MGRAY,
        **kwargs,
    ) -> "QubitArray":
        rows1, cols1 = size1
        rows2, cols2 = size2
        total_qubits = rows1 * cols1 + rows2 * cols2

        return cls(
            layout="bigrid",
            num_qubits=total_qubits,
            size1=size1,
            size2=size2,
            orientation=orientation,
            group_spacing=group_spacing,
            qubit_spacing=qubit_spacing,
            row_spacing=row_spacing,  # pass through
            col_spacing=col_spacing,  # pass through
            use_vacancies=use_vacancies,
            fill_pattern=fill_pattern,
            qubit_labels=qubit_labels,
            radius=radius,
            glow_radius=glow_radius,
            color=color,
            vac_radius=vac_radius,
            vac_color=vac_color,
            **kwargs,
        )

    @cached_property
    def kdtree(self) -> KDTree:
        """
        A spatial index for qubit center locations.
        """
        positions = [pos for _, pos in self.qubits]
        return KDTree(positions)

    # ------------------------------------------------------------------
    # internal creators -------------------------------------------------
    def _spawn_qubit(self, pos):
        q = Qubit(position=pos, **self._q_style)
        if self.qubit_labels:
            q.add_label(f"{len(self.qubits)}")
        # cache baseline color so any operation can restore it
        try:
            q._orig_color = getattr(q, "_orig_color", q.dot.get_color())
        except Exception:
            pass
        self.qubits.append([q, pos.copy()])
        self.add(q)
        if hasattr(self, "__dict__") and "kdtree" in self.__dict__:
            del self.__dict__["kdtree"]
        return q

    def _spawn_vac(self, pos):
        v = Vacancy(position=pos, **self._v_style)
        idx = len(self.vacancies)  # index BEFORE appending

        if getattr(self, "_vac_invisible", False):
            # Force all vacancy strokes fully transparent and ensure no label.
            v.set_stroke(opacity=0.0)
            v.remove_label()
        else:
            # Optional subtle label (kept exactly as before)
            if getattr(self, "vacancy_labels", False):
                v.add_label(
                    str(idx),
                    color=self._v_style.get("color", QUERA_MGRAY),
                    opacity=0.65,
                    scale=0.45,
                    z_bump=1,
                )

        self.vacancies.append([v, pos.copy()])
        self.add(v)

        # Invalidate KDTree if present
        if hasattr(self, "__dict__") and "kdtree" in self.__dict__:
            del self.__dict__["kdtree"]

        return v

    def _want_qubit_here(self, *, idx=None, row=None, col=None):
        fp = self.fill_pattern

        # 1) trivial
        if fp == "all":
            return True
        if fp == "none":
            return False

        # 2) explicit index- or (row,col)-lists/sets
        if isinstance(fp, (set, list, tuple)):
            if all(isinstance(x, int) for x in fp):
                return idx in fp
            if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in fp):
                return (row, col) in fp

        # 3) string shortcuts
        if isinstance(fp, str):
            key = fp.lower()

            # -------- BIGRID: handle blocks first, and EXIT (no fall-through) --------
            if getattr(self, "layout", "") == "bigrid":
                in_first = getattr(self, "_in_first_grid", True)
                rows1, cols1 = self.size1
                rows2, cols2 = self.size2

                if self.orientation == "horizontal":
                    # left/right blocks
                    if key == "left":
                        return in_first
                    if key == "right":
                        return not in_first
                    # optional within-block half-rows
                    if row is not None:
                        max_r = rows1 if in_first else rows2
                        if key == "top":
                            return row < (max_r + 1) // 2
                        if key == "bottom":
                            return row >= max_r // 2
                    # default: don't filter further
                    return True

                else:  # vertical bigrid = top/bottom blocks
                    if key == "top":
                        return in_first  # only the top block
                    if key == "bottom":
                        return not in_first  # only the bottom block
                    # optional within-block half-cols
                    if col is not None:
                        max_c = cols1 if in_first else cols2
                        if key == "left":
                            return col < (max_c + 1) // 2
                        if key == "right":
                            return col >= max_c // 2
                    return True

            # -------- SINGLE GRID (not bigrid) --------
            if (
                getattr(self, "layout", "") == "grid"
                and (row is not None)
                and (getattr(self, "rows", None) is not None)
                and (getattr(self, "cols", None) is not None)
            ):
                import math

                top_rows = math.ceil(self.rows / 2)
                bottom_rows = math.ceil(self.rows / 2)
                left_cols = math.ceil(self.cols / 2)
                right_cols = math.ceil(self.cols / 2)

                if key == "top":
                    return row < top_rows
                if key == "bottom":
                    return row >= (self.rows - bottom_rows)
                if key == "left":
                    return col < left_cols
                if key == "right":
                    return col >= (self.cols - right_cols)

            # -------- LINEAR / BILINEAR halves --------
            if (
                getattr(self, "layout", "") in ("linear", "bilinear")
                and idx is not None
            ):
                import math

                total = self.num_qubits * (2 if self.layout == "bilinear" else 1)
                half = math.ceil(total / 2)
                if key in ("top", "left"):
                    return idx < half
                if key in ("bottom", "right"):
                    return idx >= (total - half)

        # 4) fallback patterns
        if idx is not None and fp == "alternate":
            return (idx % 2) == 0
        if (row is not None) and (col is not None):
            if fp == "alternate":
                return (col % 2) == 0
            if fp == "checkerboard":
                return ((row + col) % 2) == 0
            if fp == "checkerboard2":
                return ((row + col) % 2) == 1

        # 5) default: fill
        return True

    def _create_row(self, n, y, *, build_qubit=True, build_vac=False, base_index=0):
        for i in range(n):
            x = (i - (n - 1) / 2) * self.qubit_spacing
            pos = np.array([x, y, 0.0])

            if build_vac:
                self._spawn_vac(pos)
                if self._want_qubit_here(idx=base_index + i):
                    self._spawn_qubit(pos)
            else:
                if build_qubit:
                    self._spawn_qubit(pos)

    def _create_column(
        self,
        *,
        n: int,
        x: float,
        build_qubit: bool = True,
        build_vac: bool = False,
        base_index: int = 0,
    ):
        half = (n - 1) / 2
        for i in range(n):
            y = (half - i) * self.qubit_spacing
            pos = np.array([x, y, 0.0])
            if build_vac:
                self._spawn_vac(pos)
                if self._want_qubit_here(idx=base_index + i):
                    self._spawn_qubit(pos)
            elif build_qubit:
                self._spawn_qubit(pos)

    def _create_grid(self, rows, cols):
        rs = float(
            self.row_spacing if self.row_spacing is not None else self.qubit_spacing
        )  # Y step
        cs = float(
            self.col_spacing if self.col_spacing is not None else self.qubit_spacing
        )  # X step

        for r in range(rows):
            for c in range(cols):
                x = (c - (cols - 1) / 2.0) * cs
                y = -(r - (rows - 1) / 2.0) * rs
                pos = np.array([x, y, 0.0])

                if self.use_vacancies:
                    self._spawn_vac(pos)
                    if self._want_qubit_here(idx=r * cols + c, row=r, col=c):
                        self._spawn_qubit(pos)
                else:
                    self._spawn_qubit(pos)

    # ------------------------------------------------------------------
    def _apply_moves(self, scene, items, moves, run_time, animate):
        anims = []
        for idx, dx, dy in moves:
            obj, pos = items[idx]
            vec = np.array([dx, dy, 0])
            if animate:
                anims.append(obj.animate.shift(vec))
            else:
                obj.shift(vec)
            items[idx][1] = pos + vec
        if animate:
            if hasattr(self, "__dict__") and "kdtree" in self.__dict__:
                del self.__dict__["kdtree"]
            scene.play(*anims, run_time=run_time)
        else:
            if hasattr(self, "__dict__") and "kdtree" in self.__dict__:
                del self.__dict__["kdtree"]
            return anims

    # --- color safety helpers -------------------------------------------------
    def _capture_colors(self, indices: list[int]):
        """Return [(qubit, original_color)] for qubits that will move."""
        out = []
        for qi in indices:
            q, _ = self.qubits[int(qi)]
            try:
                c = q.get_color()
            except Exception:
                c = q.dot.get_color()
            out.append((q, c))
        return out

    def _restore_colors(self, snapshot):
        """Restore colors previously captured by _capture_colors()."""
        for q, c in snapshot:
            try:
                q.set_color(c)
            except Exception:
                if hasattr(q, "dot"):
                    q.dot.set_color(c)

    def move_qubits(
        self,
        scene,
        shift_list,
        run_time: float = 1.0,
        animate: bool = True,
        use_lasers: bool = False,
        extra_lasers: list[tuple[tuple, tuple]] | None = None,
    ):
        def _to_vec(pt):
            a = np.array(pt, dtype=float)
            return a if a.size == 3 else np.array([a[0], a[1], 0.0])

        # clear any cached kdtree (will be rebuilt lazily later)
        self.__dict__.pop("kdtree", None)

        if use_lasers:
            real_tweezers = []
            real_anims = []
            orig_colors: dict[int, any] = {}

            # 1) real tweezers for qubits being moved
            for idx, dx, dy in shift_list:
                q, pos = self.qubits[idx]
                target = pos + np.array([dx, dy, 0.0])

                # snapshot baseline color so the tweezer doesn't leave qubits white
                try:
                    c0 = getattr(q, "_orig_color", q.dot.get_color())
                    q._orig_color = c0
                    orig_colors[idx] = c0
                except Exception:
                    orig_colors[idx] = None

                tweezer = DotLaserTweezer(
                    position=pos,
                    qubit_radius=float(self._q_style.get("radius", self.qubit_radius)),
                )
                tweezer.pick_up(q)
                scene.add(tweezer)
                real_tweezers.append((tweezer, idx, target))

                if animate:
                    real_anims.append(tweezer.animate.move_to(target))
                else:
                    # instant move (no play)
                    tweezer.move_to(target)
                    tweezer.release()
                    scene.remove(tweezer)
                    self.qubits[idx][1] = target
                    c0 = orig_colors.get(idx, None)
                    if c0 is not None:
                        try:
                            q.set_color(c0, animate=False)
                        except Exception:
                            pass

            # 2) ghost tweezers (optional)
            ghosts, ghost_anims = [], []
            for start_pt, end_pt in extra_lasers or []:
                s = _to_vec(start_pt)
                e = _to_vec(end_pt)
                ghost = DotLaserTweezer(
                    position=s,
                    qubit_radius=float(self._q_style.get("radius", self.qubit_radius)),
                ).set_opacity(0.0)
                scene.add(ghost)
                ghosts.append(ghost)

                if animate:
                    ghost_anims.append(ghost.animate.move_to(e))
                else:
                    # in non-animated mode, don't leave ghosts behind
                    ghost.move_to(e)
                    scene.remove(ghost)

            # 3) play animations together; then release & cleanup
            if animate:
                scene.play(*(real_anims + ghost_anims), run_time=run_time)

                for tweezer, idx, target in real_tweezers:
                    q, _ = self.qubits[idx]
                    tweezer.release()
                    scene.remove(tweezer)
                    self.qubits[idx][1] = target
                    c0 = orig_colors.get(idx, None)
                    if c0 is not None:
                        try:
                            q.set_color(c0, animate=False)
                        except Exception:
                            pass

                for (
                    g
                ) in ghosts:  # <- was ghost_tweezers (NameError); use the right list
                    scene.remove(g)

            # refresh spatial cache after moves
            self.__dict__.pop("kdtree", None)
            return

        # ---- non-laser path (unchanged) ----
        return self._apply_moves(scene, self.qubits, shift_list, run_time, animate)

    def move_qubits_to_vacancies(
        self,
        scene,
        mapping: list[tuple[int, int]],
        *,
        run_time: float = 1.0,
        animate: bool = True,
        use_lasers: bool = True,
        extra_lasers: list[tuple[tuple, tuple]] | None = None,
    ):
        """
        Move qubits to specific vacancy *indices*.

        mapping: list of (qubit_index, vacancy_index)
                 e.g. [(0,4),(1,5)] moves q0→v4, q1→v5

        Uses the same animation engine as move_qubits, including tweezers.
        """
        if not getattr(self, "use_vacancies", False) or not self.vacancies:
            raise RuntimeError("Vacancies are not enabled in this QubitArray.")

        shifts = []
        for qi, vi in mapping:
            q, qpos = self.qubits[int(qi)]
            v, vpos = self.vacancies[int(vi)]
            d = np.array(vpos) - np.array(qpos)
            shifts.append((int(qi), float(d[0]), float(d[1])))
        return self.move_qubits(
            scene,
            shifts,
            run_time=run_time,
            animate=animate,
            use_lasers=use_lasers,
            extra_lasers=extra_lasers,
        )

    def place_qubits_at_vacancies(self, scene, mapping: list[tuple[int, int]]):
        """
        Instantly place qubits at vacancy centers without animation.

        mapping: list[(qubit_index, vacancy_index)]
        """
        for qi, vi in mapping:
            q, _ = self.qubits[int(qi)]
            _, vpos = self.vacancies[int(vi)]
            delta = np.array(vpos) - q.get_center()
            q.shift(delta)
            self.qubits[int(qi)][1] = np.array(vpos, dtype=float)

        # invalidate KDTree
        self.__dict__.pop("kdtree", None)

    def shift_row(
        self,
        scene,
        row: int,
        shift: tuple[float, float],
        run_time: float = 1.0,
        animate: bool = True,
        use_lasers: bool = False,
        extra_lasers: (
            list[tuple[tuple[float, float], tuple[float, float]]] | None
        ) = None,
    ):
        """
        Shift all qubits whose y-coordinate lies in the row‐band indexed by `row`.
        Uses a KDTree to first grab only those near the row, then filters exactly.
        """
        # 1) choose the target y‐level exactly as before
        ys = sorted({round(pos[1], 3) for _, pos in self.qubits}, reverse=True)
        if not (0 <= row < len(ys)):
            raise IndexError(f"Row {row} out of range (0..{len(ys)-1})")
        target_y = ys[row]
        dx, dy = shift

        # 2) compute a quick horizontal‐span to bound our ball query
        xs = [pos[0] for _, pos in self.qubits]
        half_span = (max(xs) - min(xs)) / 2
        # 3) query KDTree for a small band around (mid_x, target_y)
        mid_x = (max(xs) + min(xs)) / 2
        # a tiny tolerance for floating‐point rounding
        tol = 1e-6
        # Note: self.kdtree builds on the same list of positions
        cand_idxs = self.kdtree.query_ball_point(
            [mid_x, target_y, 0.0], half_span + tol
        )

        # 4) exact pick: those whose |y − target_y|<tol
        shifts = [
            (i, dx, dy) for i in cand_idxs if abs(self.qubits[i][1][1] - target_y) < tol
        ]

        # 5) delegate to move_qubits
        return self.move_qubits(
            scene,
            shifts,
            run_time=run_time,
            animate=animate,
            use_lasers=use_lasers,
            extra_lasers=extra_lasers,
        )

    def shift_column(
        self,
        scene,
        col: int | None = None,
        column: int | None = None,
        shift: tuple[float, float] = (0.0, 0.0),
        run_time: float = 1.0,
        animate: bool = True,
        use_lasers: bool = False,
        extra_lasers: (
            list[tuple[tuple[float, float], tuple[float, float]]] | None
        ) = None,
    ):
        """
        Shift all qubits whose x-coordinate lies in the column‐band indexed by `col`
        (or `column`).  Uses a KDTree to first grab only those near the column, then filters exactly.
        """
        # allow either name
        if column is not None:
            col = column
        if col is None:
            raise TypeError(
                "shift_column() missing required argument: 'col' or 'column'"
            )

        # 1) pick the target x‐level
        xs = sorted({round(pos[0], 3) for _, pos in self.qubits})
        if not (0 <= col < len(xs)):
            raise IndexError(f"Column {col} out of range (0..{len(xs)-1})")
        target_x = xs[col]
        dx, dy = shift

        # 2) bound vertical span for KDTree query
        ys = [pos[1] for _, pos in self.qubits]
        half_span = (max(ys) - min(ys)) / 2
        mid_y = (max(ys) + min(ys)) / 2
        tol = 1e-6

        cand_idxs = self.kdtree.query_ball_point(
            [target_x, mid_y, 0.0], half_span + tol
        )

        # 3) exact pick: those whose |x − target_x|<tol
        shifts = [
            (i, dx, dy) for i in cand_idxs if abs(self.qubits[i][1][0] - target_x) < tol
        ]

        # 4) delegate
        return self.move_qubits(
            scene,
            shifts,
            run_time=run_time,
            animate=animate,
            use_lasers=use_lasers,
            extra_lasers=extra_lasers,
        )

    def move_to_atomstate(self, scene, atom_state, run_time=1.0, animate=True):
        """
        Move the Qubits in this QubitArray to match the positions in the given AtomState.

        Parameters
        ----------
        scene : Scene
            Manim scene context.
        atom_state : AtomState
            Target positions to animate toward.
        run_time : float
            Animation duration.
        animate : bool
            If True, animates the move. If False, applies instantly.
        """
        if len(atom_state.positions) != len(self.qubits):
            raise ValueError("AtomState must contain exactly one position per qubit")

        anims = []
        for (q, _), new_pos in zip(self.qubits, atom_state.positions):
            new_pos_3d = np.array([new_pos[0], new_pos[1], 0])
            delta = new_pos_3d - q.get_center()
            if animate:
                anims.append(q.animate.shift(delta))
            else:
                q.shift(delta)

        # Update internal qubit position tracking
        for i, new_pos in enumerate(atom_state.positions):
            self.qubits[i][1] = np.array([new_pos[0], new_pos[1], 0])

        if animate and anims:
            scene.play(*anims, run_time=run_time)

        # Clear stale KDTree if you're using it
        if hasattr(self, "__dict__") and "kdtree" in self.__dict__:
            del self.__dict__["kdtree"]

    def move_vacancies(self, scene, shift_list, run_time=1, animate=True):
        if not self.use_vacancies:
            raise RuntimeError("Vacancies not enabled.")
        self._apply_moves(scene, self.vacancies, shift_list, run_time, animate)

    def place_qubits(self, scene, pos_list, run_time=1, animate=True):
        self._place_items(scene, self.qubits, pos_list, run_time, animate)

    def place_vacancies(self, scene, pos_list, run_time=1, animate=True):
        if not self.use_vacancies:
            raise RuntimeError("Vacancies not enabled.")
        self._place_items(scene, self.vacancies, pos_list, run_time, animate)

    def _place_items(self, scene, items, pos_list, run_time, animate):
        anims = []
        for idx, x, y in pos_list:
            obj, _ = items[idx]
            target = np.array([x, y, 0])
            delta = target - obj.get_center()
            if animate:
                anims.append(obj.animate.shift(delta))
            else:
                obj.shift(delta)
            items[idx][1] = target
        if animate and anims:
            scene.play(*anims, run_time=run_time)

    def add_qubit_labels(self, color=BLACK):
        for i, (q, _) in enumerate(self.qubits):
            q.add_label(str(i), color=color)

    def remove_qubit_labels(self):
        for q, _ in self.qubits:
            q.remove_label()

    # Getter/Setter Functions ------------------------------------------

    def get_qubit_count(self):
        return len(self.qubits)

    def get_vacancies(self):
        return len(self.vacancies)

    def to_atomstate(self):
        """Return an AtomState representing the current Qubit positions."""
        positions = tuple(tuple(pos[:2]) for _, pos in self.qubits)
        return AtomState(positions)

    def get_pairs(self, index_pairs):
        """
        Return a list of (Qubit, Qubit) pairs from index pairs.

        Parameters
        ----------
        index_pairs : list of (int, int)
            Each tuple specifies two indices in the internal `qubits` list.

        Returns
        -------
        list of (Qubit, Qubit)
        """
        return [(self.qubits[i][0], self.qubits[j][0]) for i, j in index_pairs]

    def get_qubit(self, index):
        """
        Return a Qubit object by index.

        Parameters: index (int)
        Returns: Qubit
        """
        return self.qubits[index][0]

    def get_vacancy(self, index):
        """
        Return a Qubit object by index.

        Parameters: index (int)
        Returns: Qubit
        """
        return self.vacancies[index][0]

    def find_nearest_qubit(self, target_pos, radius=1e-3):
        from scipy.spatial import KDTree

        positions = [pos[:2] for _, pos in self.qubits]
        kdtree = KDTree(positions)
        idxs = kdtree.query_ball_point(target_pos, r=radius)
        return [self.qubits[i][0] for i in idxs]

    def find_vacancy_by_pos(self, target_pos, tol=1e-3):
        """
        Find the Vacancy object located at a given position (within a tolerance).

        Parameters
        ----------
        target_pos : array-like (list, tuple, or np.ndarray)
            The position to search for, e.g. [1.0, 0.0, 0.0].

        tol : float
            Euclidean distance tolerance for matching (default 1e-3).

        Returns
        -------
        Vacancy or None
            Returns the Vacancy at the given position, or None if not found.
        """
        target_pos = np.array(target_pos)
        for v, pos in self.vacancies:
            if np.linalg.norm(pos - target_pos) <= tol:
                return v
        return None

    def pair_off_row(
        self,
        scene,
        row: int,
        *,
        target: str = "vacancies",  # "vacancies" | "qubits" | "both"
        squeeze: float = 0.20,  # fraction of qubit_spacing to move inward
        run_time: float = 0.25,
        animate: bool = True,
        use_lasers: bool = False,  # only for qubits (DotLaserTweezer)
    ):
        """
        Pair adjacent items in the given row (0 = topmost by Y) and pull them
        inward along X by ±(squeeze * qubit_spacing):
          (0,1) → (+step, -step), (2,3) → (+step, -step), ...

        Works for any layout that produces horizontal rows (grid / bilinear / etc.).
        By default, pairs **vacancies**; set target="qubits" or "both" to affect qubits.
        """
        if target not in {"vacancies", "qubits", "both"}:
            raise ValueError("target must be 'vacancies', 'qubits', or 'both'")

        tol = 1e-3
        step = float(squeeze) * float(self.qubit_spacing)  # scale with spacing

        def _pair_moves_from(source_list):
            """
            Build [(idx, dx, dy), ...] for items in the requested row of `source_list`
            (either self.vacancies or self.qubits). Items are paired left→right.
            """
            if not source_list:
                return []

            # Distinct Y levels for this source, top (max Y) first
            y_levels = sorted(
                {round(obj.get_center()[1], 3) for (obj, _) in source_list},
                reverse=True,
            )
            if not (0 <= row < len(y_levels)):
                raise IndexError(
                    f"row {row} out of range for this source (0..{len(y_levels)-1})"
                )

            y_target = y_levels[row]

            # indices in that row, sorted left→right by X
            idxs = [
                i
                for i, (obj, _) in enumerate(source_list)
                if abs(obj.get_center()[1] - y_target) < tol
            ]
            idxs.sort(key=lambda i: source_list[i][0].get_center()[0])

            # pair up: (0,1), (2,3), ...
            moves = []
            for j in range(0, len(idxs) - 1, 2):
                iL, iR = idxs[j], idxs[j + 1]
                moves.append((iL, +step, 0.0))
                moves.append((iR, -step, 0.0))
            return moves

        did_anything = False

        # vacancies
        if target in {"vacancies", "both"} and getattr(self, "use_vacancies", False):
            v_moves = _pair_moves_from(self.vacancies)
            if v_moves:
                self.move_vacancies(scene, v_moves, run_time=run_time, animate=animate)
                did_anything = True

        # qubits
        if target in {"qubits", "both"} and self.qubits:
            q_moves = _pair_moves_from(self.qubits)
            if q_moves:
                self.move_qubits(
                    scene,
                    q_moves,
                    run_time=run_time,
                    animate=animate,
                    use_lasers=use_lasers,
                )
                did_anything = True

        if not did_anything:
            print("[pair_off_row] nothing to move on requested row/target.")

    # ------------------------------------------------------------------
    def remove_vacancies(self):
        for v, _ in self.vacancies:
            self.remove(v)
        self.vacancies.clear()
        self.use_vacancies = False
        if hasattr(self, "__dict__") and "kdtree" in self.__dict__:
            del self.__dict__["kdtree"]

    def unitary_operations(
        self,
        scene,
        indices,
        color=QUERA_YELLOW,
        animate=True,
        run_time=0.6,
    ):
        """
        Flash selected qubits to `color` and back WITHOUT changing baseline.
        """
        # gather targets & snapshot baseline
        per_qubit = []
        for idx in indices:
            q, _ = self.qubits[int(idx)]
            orig = q.baseline_color()
            per_qubit.append((q, orig))

        if animate:
            drivers = []

            def make_driver(q, orig, color):
                def _driver(mob, a, orig=orig, color=color):
                    s = float(np.sin(np.pi * a))  # 0→1→0
                    mob.set_color(
                        interpolate_color(orig, color, s), update_internal=False
                    )

                return UpdateFromAlphaFunc(q, _driver)

            for q, orig in per_qubit:
                drivers.append(make_driver(q, orig, color))

            if drivers:
                scene.play(*drivers, run_time=run_time)
                # snap back exactly to baseline (non-mutating)
                for q, orig in per_qubit:
                    q.set_color(orig, animate=False, update_internal=False)
            return

        else:
            anims_in, anims_out = [], []
            for q, orig in per_qubit:

                def drive_in(m, a, o=orig, c=color):
                    m.set_color(interpolate_color(o, c, a), update_internal=False)

                def drive_out(m, a, o=orig, c=color):
                    m.set_color(interpolate_color(c, o, a), update_internal=False)

                anims_in.append(UpdateFromAlphaFunc(q, drive_in))
                anims_out.append(UpdateFromAlphaFunc(q, drive_out))
            return anims_in, anims_out

    def blast_points(
        self,
        scene,
        indices: list[int],
        *,
        radius: float | None = None,
        color: Color = QUERA_YELLOW,
        run_time: float = 0.6,
        animate: bool | str = True,  # ← now accepts "static"
        peak_opacity: float = 0.65,
        layers: int = 12,
        spread: float = 1.5,
        sigma: float = 0.50,
        brightness_scale: float = 1.0,
    ):
        """
        Gentle radial halos behind selected qubits.

        - animate=True    : animated halos + synced non-mutating qubit tint
        - animate=False   : return drivers (halos + tint) without playing
        - animate='static': paint halos to peak immediately (SoftPoint look).
                            No qubit color tweening in static mode.
        """
        # ---- validate & normalize indices ----
        nq = len(self.qubits)
        idxs = sorted({int(i) for i in indices})
        bad = [i for i in idxs if i < 0 or i >= nq]
        if bad:
            raise IndexError(f"blast_points: indices out of range: {bad} (0..{nq-1})")

        # ---- default radius proportional to qubit size ----
        qr = float(self._q_style.get("radius", 0.15))
        if radius is None:
            radius = 0.25 * (qr / 0.15)

        # ---- build blasts (transparent initially) ----
        blasts: list[OperationPointBlast] = []
        for i in idxs:
            q, _ = self.qubits[i]
            b = OperationPointBlast(
                q.get_center(),
                radius,
                color=color,
                peak_opacity=peak_opacity * brightness_scale,
                layers=layers,
                spread=spread,
                sigma=sigma,
                z_index=-2,
            )
            blasts.append(b)

        # ---------- STATIC path ----------
        if isinstance(animate, str) and animate.lower() == "static":
            for b in blasts:
                b.pulse(scene, animate="static")  # paints to peak and stays
            return {"blasts": blasts, "indices": idxs}

        # ---------- Non-static paths (same behavior as before) ----------
        # add blasts and init transparent
        for b in blasts:
            for disk in b._disks:
                disk.set_fill(color, 0.0)
            scene.add(b)

        # snapshot baselines (non-mutating)
        targets = []
        for i in idxs:
            q, _ = self.qubits[i]
            orig = q.baseline_color()
            targets.append((q, orig))

        # === non-animated path: return drivers ===
        if not animate:
            blast_anims = [
                b.pulse(scene, run_time=run_time, animate=False) for b in blasts
            ]

            color_anims = []
            for q, orig in targets:

                def _drv(m, a, o=orig, c=color):
                    s = 0.5 * (1.0 - np.cos(2.0 * np.pi * a))
                    m.set_color(interpolate_color(o, c, s), update_internal=False)

                color_anims.append(UpdateFromAlphaFunc(q, _drv))

            return {"blasts": blast_anims, "color": color_anims, "indices": idxs}

        # === animated path: updaters with cleanup ===
        t0 = time.time()

        def make_blast_updater(b: OperationPointBlast):
            def _blast_updater(m, dt, t0=t0):
                t = time.time() - t0
                if t > run_time:
                    for d in m._disks:
                        d.set_fill(m._color, 0.0)
                    m.remove_updater(_blast_updater)
                    scene.remove(m)
                else:
                    s = 0.5 * (1.0 - np.cos(2.0 * np.pi * (t / run_time)))
                    for d, a in zip(m._disks, m._base_alphas):
                        d.set_fill(m._color, float(a) * s)

            return _blast_updater

        for b in blasts:
            b.add_updater(make_blast_updater(b))

        def make_q_updater(q: "Qubit", orig_color):
            def _q_updater(m, dt, t0=t0, o=orig_color, c=color):
                t = time.time() - t0
                if t > run_time:
                    m.set_color(o, update_internal=False)
                    m.remove_updater(_q_updater)
                else:
                    s = 0.5 * (1.0 - np.cos(2.0 * np.pi * (t / run_time)))
                    m.set_color(interpolate_color(o, c, s), update_internal=False)

            return _q_updater

        for q, orig in targets:
            q.add_updater(make_q_updater(q, orig))

        scene.wait(run_time)

    def blast_lines(
        self,
        scene,
        line_specs: list[tuple[Sequence[float], Sequence[float]]],
        *,
        thickness: float = 0.8,
        color: Color = QUERA_RED,
        opacity: float = 0.6,
        run_time: float = 0.6,
        animate: bool | str = True,
        unitary_operations: bool = True,
        extra_lasers: list[tuple[tuple, tuple]] | None = None,
        curved: bool = False,
        bulge_factor: float = 0.5,
        infinite: bool = True,
        z_index: int | None = None,
    ):
        """
        Soft, layered beams for each (start, end).
          - animate=True  : cosine pulse with optional unitary flashes (scene.play)
          - animate=False : return UpdateFromAlphaFunc drivers (no play)
          - animate="static": paint beams to peak immediately (SoftBeam look),
                              no unitary color changes, and leave them on scene.
        """

        def _to_vec(pt):
            a = np.array(pt, dtype=float)
            return a if a.size == 3 else np.array([a[0], a[1], 0.0])

        # Build beams
        beams: list[OperationLineBlast] = []
        for start, end in line_specs:
            beams.append(
                OperationLineBlast(
                    start,
                    end,
                    thickness=thickness,
                    color=color,
                    opacity=float(opacity),
                    infinite=infinite,
                    curved=curved,
                    bulge_factor=bulge_factor,
                    z_index=z_index,
                )
            )

        # Static hit collection (for return value)
        all_qubits = [q for q, _ in self.qubits]
        hit_qubits = set()
        for b in beams:
            hit_qubits.update(b.find_qubit_hits(all_qubits))

        # ---------- STATIC path ----------
        if isinstance(animate, str) and animate.lower() == "static":
            # Paint each beam to its composite-peak frame; leave on scene
            for b in beams:
                b.pulse(scene, animate="static")
            # No ghosts / unitary in static mode
            return {"lines": beams, "hits": list(hit_qubits)}

        # Optional ghost tweezers (only meaningful for animated / driver modes)
        ghosts, ghost_anims = [], []
        for start_pt, end_pt in extra_lasers or []:
            s = _to_vec(start_pt)
            e = _to_vec(end_pt)
            ghost = DotLaserTweezer(
                position=s,
                qubit_radius=float(self._q_style.get("radius", self.qubit_radius)),
            ).set_opacity(0.0)
            scene.add(ghost)
            ghosts.append(ghost)
            if animate:
                ghost_anims.append(ghost.animate.move_to(e))
            else:
                ghost.move_to(e)

        # Drivers (sync-able UpdateFromAlphaFunc) for non-static modes
        drivers = [
            b.pulse(
                scene,
                run_time=run_time,
                animate=False,
                unitary_operations=unitary_operations,
                qubits=self.qubits,
                unitary_color=color,
            )
            for b in beams
        ]

        # Return drivers if caller wants to compose manually
        if not animate:
            return {
                "lines": drivers,
                "ghosts": ghost_anims,
                "hits": list(hit_qubits),
            }

        # Otherwise play them
        scene.play(*(drivers + ghost_anims), run_time=run_time)

        # Cleanup ghosts; beams self-manage via their drivers
        for g in ghosts:
            scene.remove(g)

        return hit_qubits

    def blast_plane(
        self,
        scene,
        span: "tuple[float, float] | str",
        *,
        # plane build knobs
        orientation: str = "horizontal",
        color: Color = QUERA_RED,
        opacity: float = 0.6,
        infinite: bool = True,
        layers: int | None = None,
        spread: float | None = None,
        sigma: float | None = None,
        curved: bool = False,
        bulge_factor: float = 1.4,
        z_index: int | None = None,
        # animation / selection
        run_time: float = 1.0,
        animate: "bool | str" = True,  # True | False | "static" | "static_remove"/"static_off"
        unitary_operations: "bool | list[int] | str" = True,  # True | list[...] | "pairs"
        entangle_distance: "float | str" = 0.5,  # only used when unitary_operations=="pairs"
        unitary_width: float | None = None,
        unitary_margin: float = 0.0,
        **plane_kwargs,
    ):
        """
        Build an OperationPlaneBlast over `span=(start, end)` or "all".

        animate:
          • True / False  : pulsed driver (play or return driver)
          • "static"      : paint band and statically tint targets (no baseline change)
          • "static_remove"/"static_off": remove ALL static bands and restore colors.

        unitary_operations:
          • True  → tint qubits hit by the band.
          • list  → tint exactly those qubit indices.
          • "pairs" → tint qubits that have another qubit within `entangle_distance`
                      **and** both are inside the band (or all, if span="all").
        """
        # storage for statics
        if not hasattr(self, "_static_blasts"):
            self._static_blasts = (
                []
            )  # list[{"plane": plane, "tinted": [(q, orig_color), ...]}]

        if isinstance(animate, str) and animate.lower() in {
            "static_remove",
            "static_off",
        }:
            return self.remove_static_blasts(scene)

        # Build the plane
        blast = OperationPlaneBlast(
            span,
            orientation=orientation,
            color=color,
            opacity=opacity,
            infinite=infinite,
            layers=layers,
            spread=spread,
            sigma=sigma,
            curved=curved,
            bulge_factor=bulge_factor,
            z_index=z_index,
            **plane_kwargs,
        )

        # ---- choose target qubits for unitary color ----
        targets: list["Qubit"] = []

        if isinstance(unitary_operations, (list, tuple, set)):
            # explicit indices
            for qi in unitary_operations:
                q, _ = self.qubits[int(qi)]
                targets.append(q)

        elif (
            isinstance(unitary_operations, str)
            and unitary_operations.lower() == "pairs"
        ):
            # candidates are band hits (or all)
            candidates = (
                [q for q, _ in self.qubits]
                if (isinstance(span, str) and str(span).lower() == "all")
                else list(
                    blast.find_qubit_hits(
                        self.qubits, hit_width=unitary_width, margin=unitary_margin
                    )
                )
            )
            pts = [np.array(q.get_center(), dtype=float) for q in candidates]
            dist = (
                float(entangle_distance)
                if not isinstance(entangle_distance, (tuple, list))
                else float(entangle_distance[0])
            )

            in_pair = set()
            n = len(candidates)
            for i in range(n):
                for j in range(i + 1, n):
                    if np.linalg.norm(pts[i] - pts[j]) <= dist:
                        in_pair.add(candidates[i])
                        in_pair.add(candidates[j])
            targets = list(in_pair)

        elif unitary_operations is True:
            # band hits (or all)
            if isinstance(span, str) and str(span).lower() == "all":
                targets = [q for q, _ in self.qubits]
            else:
                targets = list(
                    blast.find_qubit_hits(
                        self.qubits, hit_width=unitary_width, margin=unitary_margin
                    )
                )

        else:
            targets = []

        # ---- "static" painter mode ----
        if isinstance(animate, str) and animate.lower() == "static":
            scene.add(blast)
            blast.paint_peak()

            tinted = []
            for q in targets:
                try:
                    prev = q.get_color()  # restore-to color
                    tinted.append((q, prev))
                    q.set_color(color, animate=False, update_internal=False)
                except Exception:
                    pass

            self._static_blasts.append({"plane": blast, "tinted": tinted})
            return {"plane": blast, "targets": [q for q, _ in tinted]}

        # ---- pulsed path (animated or return driver) ----
        pulse_qubits = (
            [(q, None) for q in targets]
            if isinstance(unitary_operations, (list, tuple, set))
            or (
                isinstance(unitary_operations, str)
                and unitary_operations.lower() == "pairs"
            )
            else self.qubits
        )

        driver = blast.pulse(
            scene,
            run_time=run_time,
            animate=False,
            unitary_operations=(
                bool(targets)
                if (
                    isinstance(unitary_operations, (list, tuple, set))
                    or (
                        isinstance(unitary_operations, str)
                        and unitary_operations.lower() == "pairs"
                    )
                )
                else unitary_operations
            ),
            qubits=pulse_qubits,
            unitary_color=color,
            unitary_width=unitary_width,
            unitary_margin=unitary_margin,
        )

        if not animate:
            return {"plane": driver, "targets": targets}

        scene.play(driver, run_time=run_time)
        return set(targets)

    def register_static_blast(self, mob):
        """Optional: allow manual registration of a static blast VMobject."""
        if not hasattr(self, "_static_blasts"):
            self._static_blasts = []
        self._static_blasts.append(mob)

    def remove_static_blasts(self, scene):
        """
        Remove all static blasts:
          1) anything registered via register_static_blast()
          2) anything in the scene tagged with `_is_static_blast == True`
        """
        # 1) Remove anything we explicitly tracked
        if hasattr(self, "_static_blasts"):
            for m in list(self._static_blasts):
                try:
                    scene.remove(m)
                except Exception:
                    pass
            self._static_blasts.clear()

        # 2) Sweep the scene for ad-hoc statics (like your manual line_blast)
        def _collect_static(m):
            found = []
            try:
                if getattr(m, "_is_static_blast", False):
                    found.append(m)
            except Exception:
                pass
            for sm in getattr(m, "submobjects", []):
                found.extend(_collect_static(sm))
            return found

        to_remove = set()
        for top in list(scene.mobjects):
            to_remove.update(_collect_static(top))

        for m in to_remove:
            try:
                scene.remove(m)
            except Exception:
                pass


class DotLaserTweezer(GlowDot):
    """
    A glowing laser tweezer dot used to pick up and move qubits freely
    around the scene. Appears as a bright BLUE_C GlowDot.

    Parameters
    ----------
    position : Sequence[3] or np.ndarray
        Initial location of the tweezer (default: ORIGIN).
    radius   : float
        Radius of the glow effect (default: 0.25).
    color    : Color
        Color of the laser glow (default: BLUE_C).
    opacity  : float
        Opacity of the laser (default: 1.0).

    Methods
    -------
    move_to(target)
        Moves the tweezer to a new position.
    pick_up(qubit)
        Moves the qubit to the tweezer and locks to its position.
    release()
        Releases the qubit (no longer follows).
    update()
        Continuously moves the qubit with the tweezer if attached.
    """

    BASE_QR = 0.15
    BASE_RADIUS = 0.40
    SCALE = BASE_RADIUS / BASE_QR  # 8/3

    def __init__(
        self,
        position=ORIGIN,
        *,
        qubit_radius: float | None = None,  # ← required driver of size
        color=BLUE_C,
        opacity=1.0,
        glow_factor=3.0,
        **kwargs,
    ):
        qr = float(self.BASE_QR if qubit_radius is None else qubit_radius)
        eff_radius = qr * self.SCALE  # e.g., 0.30 → 0.80

        super().__init__(
            center=position,
            radius=eff_radius,
            color=color,
            glow_factor=glow_factor,
            opacity=opacity,
            **kwargs,
        )

        self.position = np.array(position)
        self.attached_qubit = None
        self.add_updater(lambda m, dt: self._update_qubit_follow())

        super().__init__(
            center=position,
            radius=eff_radius,  # ← use scaled radius
            color=color,
            glow_factor=3.0,
            opacity=opacity,
            **kwargs,
        )

        self.position = np.array(position)
        self.attached_qubit = None

        # Follow the qubit while carrying (no size syncing needed)
        self.add_updater(lambda m, dt: self._update_qubit_follow())

    def move_to(self, point_or_tweezer, **kwargs):
        """Move the tweezer (and attached qubit) to a new location."""
        target = (
            point_or_tweezer.position
            if isinstance(point_or_tweezer, DotLaserTweezer)
            else point_or_tweezer
        )
        super().move_to(target, **kwargs)
        self.position = np.array(target)
        return self

    def pick_up(self, qubit, *, show: bool = False):
        """
        Attach *qubit* and – if `show` is True – return a Fade-IN animation
        so the caller can decide how long the fade should last:

            self.play(*tweezer.pick_up(q, show=True), run_time=0.3)
        """
        anims = []
        if show and self.get_opacity() == 0:
            anims.append(self.animate.set_opacity(1))  #  ← no .set_run_time()

        self.attached_qubit = qubit
        qubit.move_to(self.get_center())
        return anims

    def release(self, *, hide: bool = False):
        """
        Detach the qubit and – if `hide` – return a Fade-OUT animation.

            self.play(*tweezer.release(hide=True), run_time=0.3)
        """
        self.attached_qubit = None
        return [self.animate.set_opacity(0)] if hide and self.get_opacity() else []

    def _update_qubit_follow(self):
        if self.attached_qubit:
            self.attached_qubit.move_to(self.get_center())


class OperationPointBlast(VMobject):
    """
    A composited, layered radial halo centered at a point.
    Designed to sit behind qubits and animate with a single temporal envelope.
    """

    _LAYERS_DEFAULT = 12
    _SPREAD_DEFAULT = 1.5  # outer radius multiplier (gentler than lines)
    _SIGMA_DEFAULT = (
        0.50  # controls how fast the edge falls off (smaller = softer edge)
    )
    _PEAK_DEFAULT = 0.65  # composite peak opacity at center

    def __init__(
        self,
        center,
        radius: float,
        *,
        color=QUERA_YELLOW,
        peak_opacity: float | None = None,
        layers: int | None = None,
        spread: float | None = None,
        sigma: float | None = None,
        z_index: int = -2,
    ):
        super().__init__()

        # normalize center to 3D
        c = np.array(center, dtype=float)
        if c.size == 2:
            c = np.array([c[0], c[1], 0.0])

        self._center = c
        self._radius = float(radius)
        self._color = color
        self._layers = int(self._LAYERS_DEFAULT if layers is None else layers)
        self._spread = float(self._SPREAD_DEFAULT if spread is None else spread)
        self._sigma = float(self._SIGMA_DEFAULT if sigma is None else sigma)

        peak = self._PEAK_DEFAULT if peak_opacity is None else float(peak_opacity)
        self._peak = float(np.clip(peak, 0.0, 1.0))

        self._disks = []
        self._base_alphas = []

        # ensure all children render behind qubits
        self._z = int(z_index)
        self.set_z_index(self._z)

        # compositing-correct per-layer alphas (gaussian weights across layers)
        xs = np.linspace(0.0, 1.0, self._layers)
        raw = np.exp(-0.5 * (xs / max(1e-6, self._sigma)) ** 2)
        wts = raw / np.sum(raw)
        alphas = [1.0 - (1.0 - self._peak) ** wk for wk in wts]

        # build largest→smallest disks so larger ones sit farther back
        for k, a_k in enumerate(alphas):
            t_scale = 1.0 + (self._spread - 1.0) * (k / max(1, self._layers - 1))
            rk = self._radius * t_scale
            dk = (
                Dot(self._center, radius=rk)
                .set_fill(self._color, 0.0)
                .set_stroke(width=0)
            )
            dk.set_z_index(self._z)  # <<< keep halos behind qubit dots
            self._disks.append(dk)
            self._base_alphas.append(float(a_k))
            self.add(dk)

    def pulse(self, scene, run_time: float = 0.6, animate: bool | str = True):
        """
        animate=True     : cosine in/out, auto-cleanup
        animate=False    : return UpdateFromAlphaFunc driver
        animate='static' : paint to composite-peak and keep on scene (tagged)
        """
        # --- STATIC: paint to peak and keep; tag for later removal ---
        if isinstance(animate, str) and animate.lower() == "static":
            scene.add(self)  # harmless if already present
            for d, a in zip(self._disks, self._base_alphas):
                d.set_fill(self._color, float(a))
            # tag so remove_static_blasts() can find it
            self._is_static_blast = True
            self._static_blast_kind = "point"
            self._static_restore = []  # points don’t tint qubits; kept for API symmetry
            return self

        # --- Animated & driver modes (unchanged) ---
        for d in self._disks:
            d.set_fill(self._color, 0.0)
        scene.add(self)

        if animate:
            t0 = time.time()

            def _update(m, dt):
                t = time.time() - t0
                if t > run_time:
                    for d in self._disks:
                        d.set_fill(self._color, 0.0)
                    m.clear_updaters()
                    scene.remove(m)
                else:
                    s = 0.5 * (1.0 - np.cos(2.0 * np.pi * (t / run_time)))
                    for d, a in zip(self._disks, self._base_alphas):
                        d.set_fill(self._color, float(a) * s)

            self.add_updater(_update)
            scene.wait(run_time)
            return

        def _fn(m, a):
            s = 0.5 * (1.0 - np.cos(2.0 * np.pi * a))
            for d, b in zip(self._disks, self._base_alphas):
                d.set_fill(self._color, float(b) * s)

        return UpdateFromAlphaFunc(self, _fn)


class OperationLineBlast(VMobject):
    """
    A glowing line “blast” between two points.

    - curved=False: layered flat rectangles (softlaser look, no bulge)
    - curved=True : layered hyperbolic flare (softlaser look with bulge)

    The constructor is unchanged; `opacity` is interpreted as the desired
    *composite peak opacity* of the beam (centerline) in both modes.
    """

    # internal soft-beam knobs (hidden from public API)
    _LAYERS_DEFAULT = 14
    _SPREAD_DEFAULT = 2.0  # widest layer thickness multiplier
    _SIGMA_DEFAULT = 0.60  # Gaussian falloff across layers

    def __init__(
        self,
        start,
        end,
        *,
        thickness=0.4,
        color=QUERA_RED,
        opacity=1.0,  # composite peak
        glow_factor=11.0,
        infinite=True,
        curved=False,
        bulge_factor=1.0,
        z_index=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._thickness = float(thickness)
        self._color = color
        self._opacity = float(opacity)  # composite peak for layered stack
        self._glow_factor = float(glow_factor)
        self._infinite = bool(infinite)
        self._curved = bool(curved)
        self._bulge_factor = float(bulge_factor)

        # soft-beam internals
        self._layers_list: list[VMobject] = []
        self._base_alphas: list[float] = []
        self._spread = self._SPREAD_DEFAULT

        self.set_z_index(-2 if z_index is None else int(z_index))

        # --- endpoints to 3D
        p0 = np.array(start, dtype=float)
        p0 = p0 if p0.size == 3 else np.array([p0[0], p0[1], 0.0])
        p1 = np.array(end, dtype=float)
        p1 = p1 if p1.size == 3 else np.array([p1[0], p1[1], 0.0])

        # direction & normal
        vec = p1 - p0
        length = float(np.linalg.norm(vec)) if np.linalg.norm(vec) > 1e-9 else 1.0
        u = vec / length
        v = np.array([-u[1], u[0], 0.0])
        self._u, self._v = u, v

        # optionally extend for “infinite” look & hit-tests
        if self._infinite:
            p0 = p0 - u * 20.0
            p1 = p1 + u * 20.0

        self._start = p0
        self._end = p1
        self._length = float(np.linalg.norm(p1 - p0))

        # ---- layered geometry (softlaser in both modes) ------------------
        LAYERS = self._LAYERS_DEFAULT
        SPREAD = self._SPREAD_DEFAULT
        SIGMA = self._SIGMA_DEFAULT
        self._spread = SPREAD

        # Gaussian weights → compositing-correct per-layer alphas so that
        # 1 - Π(1-α_k) = peak at the centerline
        xs = np.linspace(0.0, 1.0, LAYERS)
        raw = np.exp(-0.5 * (xs / max(1e-6, SIGMA)) ** 2)
        w = raw / raw.sum()
        peak = float(self._opacity)
        alphas = [1.0 - (1.0 - peak) ** wk for wk in w]

        def _build_rect_layer(eff_thickness: float) -> VMobject:
            half = eff_thickness * 0.5
            A = p0 + v * half
            B = p1 + v * half
            C = p1 - v * half
            D = p0 - v * half
            poly = Polygon(A, B, C, D).set_stroke(width=0).set_fill(self._color, 0.0)
            return poly

        def _build_flare_layer(eff_thickness: float) -> VMobject:
            n_pts = 60
            xs = np.linspace(-self._length / 2, self._length / 2, n_pts)

            def half_width(x):
                full = (
                    eff_thickness
                    * self._bulge_factor
                    * np.sqrt(1.0 + (x / (0.3 * self._length)) ** 2)
                )
                return 0.5 * full

            upp, low = [], []
            for x in xs:
                center = self._start + u * (x + self._length * 0.5)
                h = half_width(x)
                upp.append(center + v * h)
                low.append(center - v * h)

            poly = (
                Polygon(*upp, *reversed(low))
                .set_stroke(width=0)
                .set_fill(self._color, 0.0)
            )
            return poly

        for k, a_k in enumerate(alphas):
            t_scale = 1.0 + (SPREAD - 1.0) * (k / max(1, LAYERS - 1))
            eff_t = self._thickness * t_scale
            layer = (
                _build_flare_layer(eff_t) if self._curved else _build_rect_layer(eff_t)
            )
            self._layers_list.append(layer)
            self._base_alphas.append(float(a_k))
            self.add(layer)

    # ---------------------------------------------------------------------
    def find_qubit_hits(
        self,
        qubits: list[VMobject],
        *,
        hit_width: float | None = None,
        margin: float = 0.0,
    ):
        """
        Distance-to-centerline test using the *widest* layer’s width.
        """
        p0, u, L = self._start, self._u, self._length
        # widest thickness accounts for spread (flat) or outer skirt (curved)
        core = (
            float(hit_width)
            if hit_width is not None
            else (self._thickness * self._spread)
        )
        half = 0.5 * core

        items = (
            qubits
            if (qubits and not isinstance(qubits[0], (tuple, list)))
            else [m for m, _ in qubits]
        )
        hits = []
        for q in items:
            c = np.array(q.get_center(), dtype=float)
            t = float(np.dot(c - p0, u))
            if not self._infinite and not (0.0 <= t <= L):
                continue
            t_clamped = (
                0.0
                if (not self._infinite and t < 0.0)
                else (L if (not self._infinite and t > L) else t)
            )
            closest = p0 + u * t_clamped
            d = float(np.linalg.norm(c - closest))
            if d <= (half + margin):
                hits.append(q)
        return hits

    # ---------------------------------------------------------------------
    def pulse(
        self,
        scene,
        run_time: float = 1.0,
        animate: bool | str = True,
        *,
        unitary_operations: bool = False,
        qubits: "list[VMobject] | list[tuple[VMobject, Any]] | None" = None,
        unitary_color=QUERA_RED,
        unitary_width: float | None = None,
        unitary_margin: float = 0.0,
    ):
        scene.add(self)

        # --- STATIC: paint to peak and keep; tag for later removal ---
        if isinstance(animate, str) and animate.lower() == "static":
            for i, lyr in enumerate(self._layers_list):
                lyr.set_fill(self._color, float(self._base_alphas[i]))
            self._is_static_blast = True
            self._static_blast_kind = "line"
            self._static_restore = []  # lines in static mode don’t tint qubits
            return self

        # --- animated pulse (unchanged) ---
        items = []
        if qubits:
            items = (
                [m for m, _ in qubits]
                if isinstance(qubits[0], (tuple, list))
                else list(qubits)
            )

        hit_qubits = []
        if unitary_operations and items:
            hit_qubits = list(
                self.find_qubit_hits(
                    items, hit_width=unitary_width, margin=unitary_margin
                )
            )

        orig_colors = {}
        for q in hit_qubits:
            try:
                orig_colors[q] = q.baseline_color()
            except Exception:
                orig_colors[q] = unitary_color

        t0 = time.time()
        if animate is True:

            def _update(m, dt):
                elapsed = time.time() - t0
                if elapsed > run_time:
                    for lyr in self._layers_list:
                        lyr.set_fill(self._color, 0.0)
                    if unitary_operations:
                        for q in hit_qubits:
                            try:
                                q.set_color(
                                    orig_colors[q], animate=False, update_internal=False
                                )
                            except Exception:
                                pass
                    m.clear_updaters()
                    scene.remove(m)
                else:
                    amp = 0.5 * (1.0 - np.cos(2.0 * np.pi * (elapsed / run_time)))
                    for i, lyr in enumerate(self._layers_list):
                        lyr.set_fill(self._color, float(self._base_alphas[i]) * amp)
                    if unitary_operations:
                        for q in hit_qubits:
                            try:
                                base = orig_colors[q]
                                q.set_color(
                                    interpolate_color(base, unitary_color, amp),
                                    update_internal=False,
                                )
                            except Exception:
                                pass

            self.add_updater(_update)
            scene.wait(run_time)
            return

        # driver
        for lyr in self._layers_list:
            lyr.set_fill(self._color, 0.0)

        def _fn(mob, a):
            amp = 0.5 * (1.0 - np.cos(2.0 * np.pi * a))
            for i, lyr in enumerate(self._layers_list):
                lyr.set_fill(self._color, float(self._base_alphas[i]) * amp)
            if unitary_operations:
                for q in hit_qubits:
                    base = orig_colors[q]
                    q.set_color(
                        interpolate_color(base, unitary_color, amp),
                        update_internal=False,
                    )

        return UpdateFromAlphaFunc(self, _fn)


class OperationPlaneBlast(VMobject):
    """
    Soft 'beam/stripe' like OperationLineBlast, spanning a band.

    span:
      • (start, end) along the NORMAL axis
          - orientation="horizontal": y in [start, end]
          - orientation="vertical"  : x in [start, end]
      • "all": full-frame, uniform (no spatial gradient)

    With curved=True, the band has *minimum* thickness exactly equal to |end-start|
    at the center, and bulges outward away from center. The center never shrinks.
    """

    def __init__(
        self,
        span: "tuple[float, float] | str",
        *,
        orientation: str = "horizontal",
        color=QUERA_RED,
        opacity: float = 1.0,
        infinite: bool = True,
        layers: int | None = None,
        spread: float | None = None,
        curved: bool = False,
        bulge_factor: float = 1.0,
        sigma: float | None = None,
        pad: float = 0.0,
        z_index: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._is_all = isinstance(span, str) and str(span).lower() == "all"
        self._orientation = (
            "horizontal" if str(orientation).lower().startswith("h") else "vertical"
        )
        self._color = color
        self._opacity = float(opacity)
        self._infinite = bool(infinite)
        self._curved = bool(curved)
        self._bulge = float(bulge_factor)
        self._pad = float(pad)  # <-- store it

        LAYERS = 32 if layers is None else int(layers)
        SPREAD = 2.0 if spread is None else float(spread)
        SIGMA = 0.60 if sigma is None else float(sigma)
        self._spread = SPREAD

        self.set_z_index(-5 if z_index is None else int(z_index))
        self._layers_list: list[VMobject] = []
        self._base_alphas: list[float] = []

        # "all" -> uniform full-frame plate
        if self._is_all:
            rect = (
                Polygon(
                    [-FRAME_WIDTH, -FRAME_HEIGHT, 0],
                    [FRAME_WIDTH, -FRAME_HEIGHT, 0],
                    [FRAME_WIDTH, FRAME_HEIGHT, 0],
                    [-FRAME_WIDTH, FRAME_HEIGHT, 0],
                )
                .set_stroke(width=0)
                .set_fill(self._color, 0.0)
            )
            self._center = 0.0
            self._base_thickness = max(FRAME_WIDTH, FRAME_HEIGHT) * 2.0
            self._layers_list.append(rect)
            self._base_alphas.append(float(self._opacity))
            self.add(rect)
            return

        # regular band
        s, e = float(span[0]), float(span[1])
        self._center = 0.5 * (s + e)
        # --- key change: minimum/waist thickness has padding ---
        core = abs(e - s) + 2.0 * self._pad
        self._base_thickness = core

        # compositing-correct per-layer alphas (peak is `opacity`)
        xs = np.linspace(0.0, 1.0, LAYERS)
        raw = np.exp(-0.5 * (xs / max(1e-6, SIGMA)) ** 2)
        w = raw / raw.sum()
        alphas = [1.0 - (1.0 - float(self._opacity)) ** wk for wk in w]

        if self._orientation == "horizontal":
            # tangent = X, normal = Y
            xL = -FRAME_WIDTH if self._infinite else -FRAME_WIDTH / 2.0
            xR = +FRAME_WIDTH if self._infinite else +FRAME_WIDTH / 2.0
            span = xR - xL
            xc = 0.5 * (xL + xR)

            def _rect_for_thickness(t):
                half = 0.5 * t
                y0, y1 = self._center - half, self._center + half
                return (
                    Polygon([xL, y0, 0], [xR, y0, 0], [xR, y1, 0], [xL, y1, 0])
                    .set_stroke(0)
                    .set_fill(self._color, 0.0)
                )

            def _flare_for_thickness(t):
                """
                Keep center thickness == t, let sides bulge outward.
                """
                B = max(self._bulge, 1.0)
                k = 0.30 * span  # bulge length scale along X
                n_pts = 80
                xs = np.linspace(xL, xR, n_pts)

                upp, low = [], []
                for x in xs:
                    xi = x - xc
                    # scale(0) = 1 → half-width at center = 0.5 * t
                    scale = 1.0 + (B - 1.0) * (np.sqrt(1.0 + (xi / k) ** 2) - 1.0)
                    h = 0.5 * t * scale
                    upp.append([x, self._center + h, 0.0])
                    low.append([x, self._center - h, 0.0])

                return (
                    Polygon(
                        *[np.array(p) for p in upp],
                        *[np.array(p) for p in reversed(low)],
                    )
                    .set_stroke(width=0)
                    .set_fill(self._color, 0.0)
                )

            _layer = _flare_for_thickness if self._curved else _rect_for_thickness

        else:
            yB = -FRAME_HEIGHT if self._infinite else -FRAME_HEIGHT / 2.0
            yT = +FRAME_HEIGHT if self._infinite else +FRAME_HEIGHT / 2.0
            span = yT - yB
            yc = 0.5 * (yB + yT)

            def _rect_for_thickness(t):
                half = 0.5 * t
                x0, x1 = self._center - half, self._center + half
                return (
                    Polygon([x0, yB, 0], [x1, yB, 0], [x1, yT, 0], [x0, yT, 0])
                    .set_stroke(0)
                    .set_fill(self._color, 0.0)
                )

            def _flare_for_thickness(t):
                B = max(self._bulge, 1.0)
                k = 0.30 * span  # bulge length scale along Y
                n_pts = 80
                ys = np.linspace(yB, yT, n_pts)

                upp, low = [], []
                for y in ys:
                    yi = y - yc
                    scale = 1.0 + (B - 1.0) * (np.sqrt(1.0 + (yi / k) ** 2) - 1.0)
                    h = 0.5 * t * scale
                    upp.append([self._center + h, y, 0.0])
                    low.append([self._center - h, y, 0.0])

                return (
                    Polygon(
                        *[np.array(p) for p in upp],
                        *[np.array(p) for p in reversed(low)],
                    )
                    .set_stroke(width=0)
                    .set_fill(self._color, 0.0)
                )

            _layer = _flare_for_thickness if self._curved else _rect_for_thickness

        # inner→outer: from core/SPREAD up to core
        t_min = 1.0 / float(SPREAD)
        for k_idx, a_k in enumerate(alphas):
            a = k_idx / max(1, LAYERS - 1)
            t_scale = t_min + (1.0 - t_min) * a
            eff_t = core * t_scale
            layer = _layer(eff_t)
            self._layers_list.append(layer)
            self._base_alphas.append(float(a_k))
            self.add(layer)

    # hit-test: default uses the *waist* width (includes pad)
    def find_qubit_hits(
        self, qubits, *, hit_width: float | None = None, margin: float = 0.0
    ):
        if self._is_all:
            return (
                qubits
                if (qubits and not isinstance(qubits[0], (tuple, list)))
                else [m for m, _ in qubits]
            )
        half = 0.5 * (
            float(hit_width) if hit_width is not None else self._base_thickness
        )
        items = (
            qubits
            if (qubits and not isinstance(qubits[0], (tuple, list)))
            else [m for m, _ in qubits]
        )
        hits = []
        if self._orientation == "horizontal":
            for q in items:
                if abs(q.get_center()[1] - self._center) <= (half + margin):
                    hits.append(q)
        else:
            for q in items:
                if abs(q.get_center()[0] - self._center) <= (half + margin):
                    hits.append(q)
        return hits

    # --- static painters ---
    def paint_peak(self):
        for i, lyr in enumerate(self._layers_list):
            lyr.set_fill(self._color, float(self._base_alphas[i]))
        return self

    def clear_fill(self):
        for lyr in self._layers_list:
            lyr.set_fill(self._color, 0.0)
        return self

    # --- cosine pulse driver (unchanged semantics) ---
    def pulse(
        self,
        scene,
        run_time: float = 1.0,
        animate: bool | str = True,
        *,
        unitary_operations: bool = False,
        qubits: "list[VMobject] | list[tuple[VMobject, any]] | None" = None,
        unitary_color=QUERA_RED,
        unitary_width: float | None = None,
        unitary_margin: float = 0.0,
    ):
        scene.add(self)

        # Normalize qubits list
        items = []
        if qubits:
            items = (
                [m for m, _ in qubits]
                if isinstance(qubits[0], (tuple, list))
                else list(qubits)
            )

        # --- STATIC: paint to peak and keep; tag + (optional) static tint restore ---
        if isinstance(animate, str) and animate.lower() == "static":
            # paint to peak
            for i, lyr in enumerate(self._layers_list):
                lyr.set_fill(self._color, float(self._base_alphas[i]))

            # If caller asked for unitary tint in static mode, apply now and remember originals
            restore = []
            if unitary_operations and items:
                hit_qubits = list(
                    self.find_qubit_hits(
                        items, hit_width=unitary_width, margin=unitary_margin
                    )
                )
                for q in hit_qubits:
                    try:
                        orig = q.baseline_color()
                    except Exception:
                        orig = unitary_color
                    restore.append((q, orig))
                    try:
                        q.set_color(unitary_color, animate=False, update_internal=False)
                    except Exception:
                        pass

            # tag so remove_static_blasts() can find & optionally restore
            self._is_static_blast = True
            self._static_blast_kind = "plane"
            self._static_restore = restore  # list[(qubit, orig_color)]
            return self

        # --- animated pulse (unchanged) ---
        hit_qubits = []
        if unitary_operations and items:
            hit_qubits = list(
                self.find_qubit_hits(
                    items, hit_width=unitary_width, margin=unitary_margin
                )
            )

        orig_colors = {}
        for q in hit_qubits:
            try:
                orig_colors[q] = q.baseline_color()
            except Exception:
                orig_colors[q] = unitary_color

        if animate is True:
            t0 = time.time()

            def _update(m, dt):
                elapsed = time.time() - t0
                if elapsed > run_time:
                    for lyr in self._layers_list:
                        lyr.set_fill(self._color, 0.0)
                    if unitary_operations:
                        for q in hit_qubits:
                            try:
                                q.set_color(
                                    orig_colors[q], animate=False, update_internal=False
                                )
                            except Exception:
                                pass
                    m.clear_updaters()
                    scene.remove(m)
                else:
                    amp = 0.5 * (1.0 - np.cos(2.0 * np.pi * (elapsed / run_time)))
                    for i, lyr in enumerate(self._layers_list):
                        lyr.set_fill(self._color, float(self._base_alphas[i]) * amp)
                    if unitary_operations:
                        for q in hit_qubits:
                            try:
                                base = orig_colors[q]
                                q.set_color(
                                    interpolate_color(base, unitary_color, amp),
                                    update_internal=False,
                                )
                            except Exception:
                                pass

            self.add_updater(_update)
            scene.wait(run_time)
            return

        # driver
        for lyr in self._layers_list:
            lyr.set_fill(self._color, 0.0)

        def _fn(mob, a):
            amp = 0.5 * (1.0 - np.cos(2.0 * np.pi * a))
            for i, lyr in enumerate(self._layers_list):
                lyr.set_fill(self._color, float(self._base_alphas[i]) * amp)
            if unitary_operations:
                for q in hit_qubits:
                    base = orig_colors[q]
                    q.set_color(
                        interpolate_color(base, unitary_color, amp),
                        update_internal=False,
                    )

        return UpdateFromAlphaFunc(self, _fn)


# -----------------------------------------------------------
#              Glow Objects Based on GlowDot
# -----------------------------------------------------------


class GlowEllipse(VMobject):
    """
    Soft elliptical glow spanning two points.

    - Major axis auto-fits |p2 - p1| + width (margin).
    - Minor axis = height.
    - Layer stack uses compositing-correct alphas (like beams).
    - `glow_factor` tightens the core; `peak_opacity` keeps a center/edge contrast
      even at the top of the pulse.
    """

    _LAYERS_DEFAULT = 32
    _SPREAD_DEFAULT = 1.6
    _SIGMA_DEFAULT = 0.8

    def __init__(
        self,
        point1,
        point2,
        *,
        color=QUERA_RED,
        width=0.6,
        height=0.15,
        layers: int | None = None,
        spread: float | None = None,
        sigma: float | None = None,
        glow_factor: float = 2.0,  # ↑ for tighter core / sharper edge
        peak_opacity: float = 0.92,  # composite centerline @ pulse peak (<1 keeps contrast)
        z_index: int = -2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # endpoints → 3D
        p1 = np.array(point1, dtype=float)
        p1 = p1 if p1.size == 3 else np.array([p1[0], p1[1], 0.0])
        p2 = np.array(point2, dtype=float)
        p2 = p2 if p2.size == 3 else np.array([p2[0], p2[1], 0.0])

        self._c = 0.5 * (p1 + p2)
        self._color = color
        self.set_z_index(z_index)

        LAYERS = int(self._LAYERS_DEFAULT if layers is None else layers)
        # Sharper falloff: reduce sigma as glow_factor grows (stronger than before)
        if sigma is None:
            SIGMA = float(self._SIGMA_DEFAULT / (max(1.0, glow_factor) ** 0.9))
        else:
            SIGMA = float(sigma)
        # A touch wider outer skirt to preserve the “old” feel while sharpening edges
        if spread is None:
            SPREAD = float(self._SPREAD_DEFAULT + 0.28 * max(0.0, glow_factor - 1.0))
        else:
            SPREAD = float(spread)

        self._layers = LAYERS
        self._spread = SPREAD
        self._sigma = SIGMA

        # geometry (same eccentricity as before)
        d = float(np.linalg.norm(p2 - p1))
        theta = angle_of_vector(p2 - p1)
        a_base = d + float(width)
        b_base = float(height)

        # ---- per-layer “base” alphas (kept even at pulse peak) ----
        # Gaussian weights across layer index → compositing-correct opacities.
        xs = np.linspace(0.0, 1.0, LAYERS)
        raw = np.exp(-0.5 * (xs / max(1e-6, SIGMA)) ** 2)
        wts = raw / raw.sum()

        # IMPORTANT: peak_opacity < 1 keeps center/edge contrast even at max pulse
        peak = float(np.clip(peak_opacity, 0.0, 1.0))
        base_alphas = [1.0 - (1.0 - peak) ** wk for wk in wts]

        self._ellipses: list[VMobject] = []
        self._base_alphas: list[float] = []

        for k, a_k in enumerate(base_alphas):
            t = 1.0 + (SPREAD - 1.0) * (k / max(1, LAYERS - 1))
            a = a_base * t
            b = b_base * t
            e = (
                Ellipse(width=a, height=b)
                .rotate(theta)
                .move_to(self._c)
                .set_stroke(width=0)
                .set_fill(self._color, 0.0)
            )
            self._ellipses.append(e)
            self._base_alphas.append(float(a_k))
            self.add(e)

    def set_opacity(self, alpha: float):
        """
        Temporal envelope driver (0..1). We scale the *base* per-layer alphas,
        which preserves the inner/outer contrast at all times.
        """
        a = float(np.clip(alpha, 0.0, 1.0))
        for e, b in zip(self._ellipses, self._base_alphas):
            e.set_fill(self._color, b * a)
        return self

    def set_color(self, color):
        self._color = color
        for e in self._ellipses:
            e.set_fill(color, e.get_fill_opacity())
        return self


class GlowLine(DotCloud):
    """
    A glowing line rendered using DotCloud, optionally infinite, and designed to
    always render behind other elements like Qubits.

    Parameters
    ----------
    point1, point2 : np.ndarray
        Endpoints of the line segment.
    color : Color
        Glow color.
    radius : float
        Radius of each glow dot (default 0.3).
    glow_factor : float
        Intensity of the glow (default 4.0).
    opacity : float
        Opacity of the glow dots (default 1.0).
    infinite : bool
        If True, extends the line in both directions.
    extension : float
        Total length of the line when infinite=True.
    density : float
        Number of dots per unit length (default 50).
    """

    def __init__(
        self,
        point1=LEFT,
        point2=RIGHT,
        color=BLUE_C,
        radius=0.3,
        glow_factor=4.0,
        opacity=1.0,
        infinite=False,
        extension=FRAME_WIDTH * 2,
        density=50,
        **kwargs,
    ):
        p1 = np.array(point1)
        p2 = np.array(point2)

        if infinite:
            dir_vec = normalize(p2 - p1)
            mid = (p1 + p2) / 2
            p1 = mid - 0.5 * extension * dir_vec
            p2 = mid + 0.5 * extension * dir_vec

        length = np.linalg.norm(p2 - p1)
        num_points = max(int(length * density), 2)

        points = np.array(
            [p1 + alpha * (p2 - p1) for alpha in np.linspace(0, 1, num_points)]
        )

        super().__init__(
            points=points,
            color=color,
            radius=radius,
            glow_factor=glow_factor,
            opacity=opacity,
            **kwargs,
        )

        self.set_z_index(-1)  # always render below Qubits


class LineLaserTweezer(GlowLine):
    """
    A glowing laser tweezer rendered as a line that can interact with all Qubits
    along its path.

    Use this to drag multiple qubits at once. Inherits from GlowLine.

    Parameters
    ----------
    point1, point2 : np.ndarray
        Line endpoints.
    qubit_array : QubitArray
        Reference to a QubitArray object from which to collect qubits.
    catch_radius : float
        Distance threshold to include qubits (default 0.15).
    """

    def __init__(
        self,
        point1,
        point2,
        qubit_array,
        catch_radius=0.15,
        color=BLUE,
        radius=0.3,
        glow_factor=4.0,
        opacity=1.0,
        infinite=False,
        **kwargs,
    ):
        super().__init__(
            point1=point1,
            point2=point2,
            color=color,
            radius=radius,
            glow_factor=glow_factor,
            opacity=opacity,
            infinite=infinite,
            **kwargs,
        )
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)
        self.qubit_array = qubit_array  # reference to QubitArray
        self.catch_radius = catch_radius
        self._collected = []

        self.update_collected()

    # -------------------------------------------------------------
    def update_collected(self):
        """Find and store qubits near the laser line."""
        self._collected.clear()
        direction = normalize(self.point2 - self.point1)

        for idx, (q, pos) in enumerate(self.qubit_array.qubits):
            proj = self.point1 + np.dot(pos - self.point1, direction) * direction
            if np.linalg.norm(pos - proj) <= self.catch_radius:
                self._collected.append((idx, q))

    # -------------------------------------------------------------
    def print_collected_indices(self):
        """Print the indices of currently collected qubits."""
        indices = [idx for idx, _ in self._collected]
        print("Collected Qubit Indices:", indices)

    # -------------------------------------------------------------
    def move_by(self, scene, delta, run_time=1.0, animate=True):
        """Move laser and collected qubits together by delta vector and update QubitArray."""
        self.update_collected()  # Make sure _collected is fresh

        # Prepare animations
        if animate:
            anims = [q.animate.shift(delta) for _, q in self._collected]
            anims.append(self.animate.shift(delta))
            scene.play(*anims, run_time=run_time)
        else:
            for _, q in self._collected:
                q.shift(delta)
            self.shift(delta)
            scene.add(self)

        # Update internal state
        self.point1 += delta
        self.point2 += delta

        # Sync qubit positions inside QubitArray
        for idx, (_, q) in zip([i for i, _ in self._collected], self._collected):
            self.qubit_array.qubits[idx][1] += delta

    # -------------------------------------------------------------
    def turn_on(self, *, scene=None, run_time=0.5, animate=False):
        """Fade the laser in.  If animate=True you must supply `scene`."""
        if animate:
            if scene is None:
                raise ValueError("scene must be provided when animate=True")
            scene.play(self.animate.set_opacity(1.0), run_time=run_time)
        else:
            self.set_opacity(1.0)
        return self

    def turn_off(self, *, scene=None, run_time=0.5, animate=False):
        """Fade the laser out."""
        if animate:
            if scene is None:
                raise ValueError("scene must be provided when animate=True")
            scene.play(self.animate.set_opacity(0.0), run_time=run_time)
        else:
            self.set_opacity(0.0)
        return self

    def pulse(self, scene, run_time=1.0, scale_factor=1.3):
        orig_radius = self.get_radius()
        scene.play(
            self.animate.set_opacity(1.0).scale(scale_factor),
            rate_func=there_and_back,
            run_time=run_time,
        )
        self.set_radius(orig_radius)


def _cache_orig_color(q):
    if not hasattr(q, "_orig_color"):
        q._orig_color = q.get_color()


def entanglements(
    scene,
    pairs,
    *,
    color=QUERA_RED,
    width: float | None = None,
    height: float | None = None,
    glow_factor: float = 2.0,
    run_time: float = 0.4,
    animate: bool | str = True,  # True | False | "static"
    z_index: int = -2,
):
    """
    Draw entanglement glows between qubit pairs.

    Parameters
    ----------
    scene : Scene
        Manim scene to add animations/objects to.
    pairs : list[tuple[q1, q2]]
        Qubit pairs to connect (each q has .get_center() and .get_color()).
    color : Color
        Glow + tint color.
    width, height : float | None
        Ellipse dimensions. If None, sensible defaults are chosen per mode:
        - animate=True/False → width=0.6, height=0.15
        - animate="static"   → width=0.4, height=0.4
    glow_factor : float
        Extra blur/glow for GlowEllipse.
    run_time : float
        Duration for the pulsing animation (animate=True).
    animate : bool | str
        True  → play the pulsing animation now.
        False → return animation drivers (qubits, ellipses, objs) without playing.
        "static" → add static peak-frame glow and tint qubits.
    z_index : int
        Z index for the ellipses.

    Returns
    -------
    - animate=True  → None
    - animate=False → (qubit_anims, ellipse_anims, ellipses)
    - animate="static" → [ellipses]
    """
    if not pairs:
        if animate is True:
            return None
        if animate is False:
            return ([], [], [])
        return []  # "static"

    # Mode & defaults
    is_static = isinstance(animate, str) and animate.lower() == "static"
    if width is None or height is None:
        if is_static:
            width = 0.4 if width is None else width
            height = 0.4 if height is None else height
        else:
            width = 0.6 if width is None else width
            height = 0.15 if height is None else height

    # Build ellipses
    ellipses = []
    for q1, q2 in pairs:
        e = GlowEllipse(
            q1.get_center(),
            q2.get_center(),
            color=color,
            width=width,
            height=height,
            glow_factor=glow_factor,
        ).set_z_index(z_index)
        scene.add(e)
        ellipses.append(e)

    if is_static:
        # Tag for remove_static_blasts() sweeper
        for (q1, q2), e in zip(pairs, ellipses):
            try:
                e._is_static_blast = True
                e._blast_pair = (id(q1), id(q2))  # optional breadcrumb
            except Exception:
                pass
            e.set_opacity(1.0)
        # Full tint (no auto-restore in static mode)
        qubits = {q for pair in pairs for q in pair}
        for q in qubits:
            q.set_color(color, animate=False, update_internal=False)
        return ellipses

    # ---- Pulsing animation (sine envelope) ----
    for e in ellipses:
        e.set_opacity(0.0)

    qubits = {q for pair in pairs for q in pair}
    bases = {q: q.get_color() for q in qubits}

    def env(a: float) -> float:
        # 0 → 1 → 0
        return float(np.sin(np.pi * float(a)))

    ellipse_drivers = []
    for e in ellipses:

        def _ellipse_drv(_mob, a, e=e):
            e.set_opacity(env(a))

        ellipse_drivers.append(UpdateFromAlphaFunc(e, _ellipse_drv))

    qubit_drivers = []
    for q in qubits:
        orig = bases[q]

        def _q_drv(_mob, a, q=q, o=orig):
            s = env(a)
            q.set_color(interpolate_color(o, color, s), update_internal=False)

        qubit_drivers.append(UpdateFromAlphaFunc(q, _q_drv))

    if animate is False:
        # return drivers for the caller to scene.play(...)
        return qubit_drivers, ellipse_drivers, ellipses

    # animate is True → play now & restore baselines; remove ellipses
    scene.play(*(qubit_drivers + ellipse_drivers), run_time=run_time)
    for q in qubits:
        q.set_color(bases[q], animate=False, update_internal=False)
    for e in ellipses:
        scene.remove(e)


def entangle(scene, q1, q2, **kwargs):
    """Single-pair convenience wrapper."""
    return entanglements(scene, [(q1, q2)], **kwargs)


def parallel_operations(
    scene,
    qa: "QubitArray",
    entangle_pairs: list[tuple[int, int]] | None = None,
    unitary_indices: list[int] | None = None,
    entangle_color=QUERA_RED,
    unitary_color=QUERA_YELLOW,
    run_time: float = 1.0,
    animate: bool = True,
):
    """
    Run entanglements and single-qubit flashes on one unified timeline.
    Uses a single sine pulse: s(a) = sin(pi * a), a∈[0,1].
    No mid-animation waits/splits → no end-frame blinks.

    If animate=False, returns (qubit_anims, ellipse_anims, ellipses).
    """
    entangle_pairs = entangle_pairs or []
    unitary_indices = unitary_indices or []

    # ---- Build one target per qubit (avoid double writers) ----
    # If a qubit is in both lists, unitary_color takes precedence.
    targets: dict[Qubit, tuple[Color, Color]] = {}

    def _cache_target(q: Qubit, tgt: Color):
        orig = getattr(q, "_orig_color", q.dot.get_color())
        q._orig_color = orig
        targets[q] = (orig, tgt)

    for i, j in entangle_pairs:
        _cache_target(qa.get_qubit(i), entangle_color)
        _cache_target(qa.get_qubit(j), entangle_color)

    for idx in unitary_indices:
        _cache_target(qa.get_qubit(idx), unitary_color)  # overwrites if present

    # ---- Ellipses for entanglement (opacity driven by same pulse) ----
    ellipses: list[VMobject] = []
    ellipse_anims: list[Animation] = []

    def _ellipse_anim(e):
        def drv(_, a, e=e):
            s = np.sin(np.pi * a)  # 0→1→0
            e.set_opacity(s)

        return UpdateFromAlphaFunc(e, drv)

    for i, j in entangle_pairs:
        q1, q2 = qa.get_qubit(i), qa.get_qubit(j)
        e = (
            GlowEllipse(
                q1.get_center(),
                q2.get_center(),
                color=entangle_color,
                width=0.6,
                height=0.15,
                glow_factor=2.0,
            )
            .set_opacity(0.0)
            .set_z_index(-2)
        )
        scene.add(e)
        ellipses.append(e)
        ellipse_anims.append(_ellipse_anim(e))

    # ---- One updater per qubit, single sine pulse ----
    qubit_anims: list[Animation] = []

    def _qubit_anim(q: Qubit, orig: Color, target: Color):
        def drv(mob, a, orig=orig, target=target):
            s = np.sin(np.pi * a)  # smooth in/out, total time = run_time
            mob.set_color(interpolate_color(orig, target, s))

        return UpdateFromAlphaFunc(q, drv)

    for q, (orig, tgt) in targets.items():
        qubit_anims.append(_qubit_anim(q, orig, tgt))

    if not animate:
        return qubit_anims, ellipse_anims, ellipses

    # ---- Single play → exact timing, no flicker ----
    scene.play(*(qubit_anims + ellipse_anims), run_time=run_time)

    # Cleanup ellipses; qubits already back at orig because s(1)=0
    for e in ellipses:
        scene.remove(e)


def unitary_all(
    scene,
    qa,
    color=QUERA_YELLOW,
    run_time=1.0,
    animate=True,
):
    """
    Perform a simultaneous unitary operation (color flash) on all qubits.

    Parameters
    ----------
    scene : Scene
        Scene context for animations.
    qa : QubitArray
        The qubit array whose qubits to flash.
    color : Color
        Flash color (default: QUERA_YELLOW).
    run_time : float
        Total duration of the animation.
    animate : bool
        If False, returns (in_anims, out_anims); caller controls timing.

    Returns
    -------
    If animate=False:
        tuple[list[Animation], list[Animation]]
    Else:
        None
    """
    indices = list(range(qa.get_qubit_count()))
    result = qa.unitary_operations(
        scene=scene, indices=indices, color=color, run_time=run_time, animate=False
    )

    if result is not None:
        in_anims, out_anims = result
    else:
        in_anims, out_anims = [], []

    if animate:
        scene.play(*in_anims, run_time=run_time * 0.4)
        scene.wait(run_time * 0.2)
        scene.play(*out_anims, run_time=run_time * 0.4)
    else:
        return in_anims, out_anims
