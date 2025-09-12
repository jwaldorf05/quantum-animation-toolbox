from manimlib import *

QUERA_PURPLE = Color("#6437FF")
QUERA_RED = Color("#FF505D")
QUERA_YELLOW = Color("#F0FF48")
QUERA_LGRAY = Color("#E6E6E6")
QUERA_MGRAY = Color("#979797")
QUERA_DGRAY = Color("#333333")

# Defaults
QUERA_LOGO_HEIGHT = 1.5


# Quera Logo Objects


class QuEraLogo(SVGMobject):
    """
    Unified QuEra logo.
    Pass *any* ManimColor or hex-string for `color`.

    Parameters
    ----------
    color : str | ManimColor
        Fill/stroke color for the logo.
    height : float, optional
        Desired height of the logo in scene units.
    include_text : bool, optional
        If True → “QuEra Computing Inc” version,
        else     → icon + “QuEra”.
    """

    def __init__(
        self, color=QUERA_PURPLE, height=QUERA_LOGO_HEIGHT, include_text=False, **kwargs
    ):
        # pick correct SVG file
        filename = "quera_logo_text" if include_text else "quera_logo"
        super().__init__(filename, **kwargs)

        # accept hex-string or ManimColor instance
        if isinstance(color, str):
            color = ManimColor(*hex_to_rgb(color))
        self.set_color(color)
        self.set_height(height)


class BackgroundColor(VGroup):
    """
    Simple full-screen background-color panel.

    Parameters
    ----------
    color : Color
        The fill colour you want behind everything else.
    z_index : int, optional
        Layers the rectangle *under* other mobjects by default.
    """

    def __init__(self, color=BLACK, z_index=-100, **kwargs):
        super().__init__(**kwargs)

        # A FullScreenRectangle automatically matches the camera frame
        rect = FullScreenRectangle().set_fill(color, opacity=1.0).set_stroke(width=0)

        rect.set_z_index(z_index)
        self.add(rect).set_z_index(z_index)


class QuEraSlideBG(ImageMobject):
    """
    Load a QuEra slide template background stored under
    raster_images/quera_bgs/

    Parameters
    ----------
    key : int | str
        * int  -> slide_bg_00.png … slide_bg_99.png
        * str  -> slide_bg_<key>.png

    slide_bg_00: Purple Title and Textbox
    slide_bg_01: White Logo Blank
    slide_bg_02: Dark Purple Title and space
    slide_bg_03: Light Purple Blank
    slide_bg_04: White Title and space
    slide_bg_05: Dark Purple Blank
    """

    def __init__(self, key: int | str = 0, **kwargs):
        # 1) filename --------------------------------------------------
        if isinstance(key, int):
            filename = f"slide_bg_{key:02d}.png"
        else:
            filename = f"slide_bg_{key}.png"

        # 2) relative to raster_images dir ----------------------------
        super().__init__(f"quera_bgs/{filename}", **kwargs)

        # 3) scale / position -----------------------------------------
        # Manim’s default 16×9 “camera frame” is 14.22 × 8 (≈1.777 ratio).
        # Using set_height alone preserves aspect-ratio and is usually enough.
        self.set_height(FRAME_HEIGHT)
        self.set_z_index(-100)
        self.move_to(ORIGIN)  # centre on screen


def _style_to_weight_slant(style: str):
    s = (style or "regular").lower()
    # ManimGL Text accepts 'weight' and 'slant'
    try:
        from manimlib.mobject.svg.text_mobject import (
            BOLD,
            ITALIC,
            LIGHT,
            NORMAL,
            OBLIQUE,
        )
    except Exception:
        # fallback names used by ManimGL
        NORMAL = "NORMAL"
        ITALIC = "ITALIC"
        OBLIQUE = "OBLIQUE"
        BOLD = "BOLD"
        LIGHT = "LIGHT"

    table = {
        "regular": (NORMAL, NORMAL),
        "bold": (BOLD, NORMAL),
        "italic": (NORMAL, ITALIC),
        "bold-italic": (BOLD, ITALIC),
        "light": (LIGHT, NORMAL),
        "oblique": (NORMAL, OBLIQUE),
    }
    return table.get(s, (NORMAL, NORMAL))


def _auto_wrap_text(raw: str, target_width: float, font: str, font_size: int):
    """
    Greedy word-wrap: build lines so each line's rendered width <= target_width.
    Uses temporary Text objects to measure.
    """
    from manimlib.mobject.svg.text_mobject import Text as _T

    words = raw.split()
    lines, cur = [], []

    def width_of(s):
        if not s:
            return 0.0
        tmp = _T(s, font=font, font_size=font_size)
        w = tmp.get_width()
        try:
            tmp.remove()  # avoid leaving temp mobjects around
        except Exception:
            pass
        return w

    for w in words:
        candidate = (" ".join(cur + [w])).strip()
        if width_of(candidate) <= target_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


class QuEraSlideTitle(Text):
    """
    One-line Radion-B slide title. Defaults mirror PPT-ish spacing:
      • line height ≈ 1.05 (extra line spacing = 0.05 * font_size)
      • centered
    """

    def __init__(
        self,
        text: str,
        color=BLACK,
        y_offset: float = -0.6,
        font_size: int = 48,
        font_style: str = "bold",  # default bold
        line_spacing_scale: float = 1.0,  # 1.0 (single); titles are single-line anyway
        **kwargs,
    ):
        weight, slant = _style_to_weight_slant(font_style)
        super().__init__(
            text,
            font="Radion B",
            color=color,
            font_size=font_size,
            weight=weight,
            slant=slant,
            **kwargs,
        )
        # apply line spacing AFTER init (even though title is usually single-line)
        try:
            extra = (line_spacing_scale - 1.0) * font_size
            self.set_line_spacing(extra)
        except Exception:
            pass

        # place near the top
        self.set_stroke(color, width=0)
        self.to_edge(UP)
        self.shift(y_offset * UP)


class QuEraSlideText(Text):
    """
    Radion-B paragraph block sized & positioned for QuEra slides.

    Defaults (chosen to feel like PowerPoint body text):
      • wrap_width = 8.2 scene units (narrower column than before)
      • line height ≈ 1.15  → extra = 0.15 * font_size
      • left-aligned
    """

    def __init__(
        self,
        text: str,
        color=WHITE,
        wrap_width: float = 8,  # a bit narrower by default
        y_offset: float = -0.2,
        font_size: int = 30,  # ≈ 18pt visual on 1080p
        font_style: str = "regular",
        line_spacing_scale: float = 1.15,  # PPT-ish default
        **kwargs,
    ):
        weight, slant = _style_to_weight_slant(font_style)

        # auto-insert newlines to fit target width
        wrapped = _auto_wrap_text(
            text, target_width=wrap_width, font="Radion B", font_size=font_size
        )

        # build the Text (do NOT pass line_spacing here)
        super().__init__(
            wrapped,
            font="Radion B",
            color=color,
            font_size=font_size,
            weight=weight,
            slant=slant,
            **kwargs,
        )

        # line spacing AFTER construction (scale by font size)
        try:
            extra = (line_spacing_scale - 1.0) * font_size
            self.set_line_spacing(extra)
        except Exception:
            pass

        # position in the slide’s text region
        self.set_stroke(color, width=0)
        self.move_to(ORIGIN)
        self.shift(y_offset * UP)

        # safety: if someone passes a too-small wrap_width, keep it from over-wrapping visually
        if self.get_width() > wrap_width + 0.01:
            # tiny scale-down to respect the box; should almost never trigger because of wrapping
            self.set(width=wrap_width)


class QuEraBanner(VGroup):
    def __init__(self, color=QUERA_PURPLE, direction="center", show=True):
        super().__init__()
        self.color = color
        self.direction = direction

        # Load parts
        self.ket_left = SVGMobject("quera_ket_left.svg", height=2.0).set_color(color)
        self.ket_right = SVGMobject("quera_ket_right.svg", height=2.0).set_color(color)
        self.q_ket = SVGMobject("quera_q_ket.svg", height=2.0).set_color(color)
        self.anim = SVGMobject("split_logo.svg", height=2.0).set_color(color)
        self.logo_ref = self.anim.copy().set_opacity(0)

        print("q_ket width:", self.q_ket.get_width())
        print("anim width:", self.anim.get_width())

        # Normalize heights
        self.q_ket.scale(142 / 143)
        self.ket_left.scale(142 / 143)
        self.ket_right.scale(130 / 143)

        # q_ket is invisible but must be scaled and added
        self.q_ket.set_opacity(0)
        self.anim.set_opacity(0)

        # Add all elements so .scale(...) applies properly
        self.add(self.ket_left, self.ket_right, self.q_ket, self.anim, self.logo_ref)
        if not show:
            self.set_opacity(0)
        self.initialize_layout()

    def initialize_layout(self):
        # Set baseline for anim
        y_pos = self.q_ket.get_center()[1]

        # Align left and right kets to q_ket
        dx_left = self.q_ket.get_left()[0] - self.ket_left.get_left()[0]
        dx_right = self.q_ket.get_right()[0] - self.ket_right.get_right()[0]
        self.ket_left.move_to(self.ket_left.get_center() + RIGHT * dx_left)
        self.ket_right.move_to(self.ket_right.get_center() + RIGHT * dx_right)

        if self.direction == "center":
            lx = self.ket_left.get_left()[0]
            rx = self.ket_right.get_right()[0]
            mid_x = (lx + rx) / 2
            anim_start_x = (
                self.ket_left.get_center()[0]
                + (self.anim.get_width() - self.ket_left.get_width()) / 2
            )
            self.anim.move_to([mid_x + anim_start_x, y_pos, 0])
            self.logo_ref.move_to([mid_x, y_pos, 0])
        elif self.direction == "right":
            lx = self.ket_left.get_left()[0]
            self.anim.move_to([-(self.anim.get_left()[0] - lx), y_pos, 0])
            self.logo_ref.move_to([(self.anim.get_width() / 2) + lx, y_pos, 0])
        elif self.direction == "left":
            rx = self.ket_right.get_right()[0]
            # anim_start_x = self.ket_left.get_center()[0] + (self.anim.get_width()-self.ket_left.get_width())/2
            # Then shift right so right edge aligns with ket_right's right edge
            self.anim.move_to(
                [self.ket_left.get_left()[0] + (self.anim.get_width() / 2), y_pos, 0]
            )
            # self.anim.move_to([rx - self.anim.get_right()[0] + anim_start_x,y_pos,0])
            self.logo_ref.move_to([rx - (self.anim.get_width() / 2), y_pos, 0])

        self._refresh_motions()

    def _refresh_motions(self):
        """Re-measure vectors using the stationary reference copy."""
        self.left_motion = self.logo_ref.get_left() - self.ket_left.get_left()
        self.right_motion = self.logo_ref.get_right() - self.ket_right.get_right()

    def _refresh_reverse_motions(self):
        """Re-measure vectors using the stationary reference copy."""
        self.left_motion = -self.left_motion
        self.right_motion = -self.right_motion

    def center_expansion(self, side: str):
        """
        Shifts unexpanded logo so that when expanded to the initialized size the
        full banner will be centered.
        """
        if side not in {"left", "right"}:
            raise ValueError("center_expansion(side) must be 'left' or 'right'")

        offset = 0.5 * (self.anim.get_width() - self.q_ket.get_width())

        if side == "left":
            self.shift(RIGHT * offset)
        elif side == "right":
            self.shift(LEFT * offset)

    def appear(self):
        """Instantly set both kets fully visible."""
        return AnimationGroup(
            ApplyMethod(self.ket_left.set_opacity, 1, run_time=0),
            ApplyMethod(self.ket_right.set_opacity, 1, run_time=0),
            lag_ratio=0.0,
        )

    def disappear(self):
        """Instantly hide both kets."""
        return AnimationGroup(
            ApplyMethod(self.ket_right.set_opacity, 0, run_time=0),
            ApplyMethod(self.ket_left.set_opacity, 0, run_time=0),
            lag_ratio=0.0,
        )

    def create(self, run_time=1):
        return AnimationGroup(
            Write(self.ket_left, run_time=run_time),
            Write(self.ket_right, run_time=run_time),
            lag_ratio=0.1,
        )

    def uncreate(self, run_time=1):
        """Reverse of create(): erase right-ket first, then left-ket."""
        return AnimationGroup(
            Write(
                self.ket_right, run_time=run_time, rate_func=lambda t: 1 - t
            ),  # reverse order
            Write(self.ket_left, run_time=run_time, rate_func=lambda t: 1 - t),
            lag_ratio=0.1,
        )

    def expand(self, run_time=0.75):
        self._refresh_motions()
        self.ket_left.save_state()
        self.ket_right.save_state()
        self.anim.save_state()
        print("anim saved x: ", self.anim.saved_state.get_center()[0])
        letters = self.anim.family_members_with_points()
        for l in letters:
            l.set_opacity(0)

        def get_motions(alpha):
            if self.direction == "center":
                return (
                    alpha * self.left_motion,
                    alpha * self.right_motion,
                    alpha * self.left_motion,
                )
            elif self.direction == "right":
                return ORIGIN, alpha * self.right_motion, ORIGIN
            elif self.direction == "left":
                return alpha * self.left_motion, ORIGIN, alpha * self.left_motion

        def shift(alpha):
            lm, rm, cm = get_motions(alpha)

            # Move kets
            self.ket_left.move_to(self.ket_left.saved_state.get_center() + lm)
            self.ket_right.move_to(self.ket_right.saved_state.get_center() + rm)
            self.anim.move_to(self.anim.saved_state.get_center() + cm)

        def slide_and_reveal(mob, alpha):
            shift(alpha)
            for letter in letters:
                x = letter.get_center()[0]
                if self.ket_left.get_right()[0] < x < self.ket_right.get_left()[0]:
                    letter.set_opacity(1)
                    self.add_to_back(letter)
            if alpha == 1:
                self.ket_left.set_opacity(0)
                self.ket_right.set_opacity(0)
                self.add_to_back(self.anim)
                self.anim.set_opacity(1)

        return Succession(
            UpdateFromAlphaFunc(
                self, slide_and_reveal, run_time=run_time, rate_func=smooth
            )
        )

    def contract(self, run_time=0.75):
        # make sure vectors reflect the banner’s current size / position
        self._refresh_reverse_motions()

        # bring kets back to full opacity, hide the big logo
        self.ket_left.set_opacity(1)
        self.ket_right.set_opacity(1)
        # self.anim.set_opacity(0)

        #  save "expanded" positions so we can interpolate back
        self.ket_left.save_state()
        self.ket_right.save_state()
        self.anim.save_state()

        letters = self.anim.family_members_with_points()
        # start with all letters visible
        for l in letters:
            l.set_opacity(1)

        def get_motions(alpha):
            """reverse of expand(): alpha=0 => expanded, alpha=1 => closed"""
            if self.direction == "center":
                return (
                    alpha * self.left_motion,
                    alpha * self.right_motion,
                    alpha * self.left_motion,
                )
            elif self.direction == "right":
                return ORIGIN, alpha * self.right_motion, ORIGIN
            elif self.direction == "left":
                return alpha * self.left_motion, ORIGIN, alpha * self.left_motion

        def shift(alpha):
            lm, rm, cm = get_motions(alpha)
            self.ket_left.move_to(self.ket_left.saved_state.get_center() + lm)
            self.ket_right.move_to(self.ket_right.saved_state.get_center() + rm)
            self.anim.move_to(self.anim.saved_state.get_center() + cm)

        def cover_and_hide(mob, alpha):
            shift(alpha)
            # gradually hide letters as they get re-covered
            for l in letters:
                x = l.get_center()[0]
                if (
                    self.ket_left.get_right()[0] >= x
                    or self.ket_right.get_left()[0] <= x
                ):
                    l.set_opacity(0)
            # at the end, hide the logo completely
            if alpha == 1:
                self.anim.set_opacity(0)

        return Succession(
            UpdateFromAlphaFunc(
                self, cover_and_hide, run_time=run_time, rate_func=smooth
            )
        )

    def scale(self, factor, **kwargs):
        super().scale(
            factor, about_point=self.q_ket.get_center(), **kwargs
        )  # do the actual scaling
        self.initialize_layout()  # refresh dx_left / dx_right
        return self

    def shift(self, *vectors, **kwargs):
        super().shift(*vectors, **kwargs)  # do the actual shift
        self.initialize_layout()  # recompute dx_left/right, etc.
        return self


class QuEraScatter(VGroup):
    def __init__(self, color=QUERA_PURPLE, height=2.0):
        super().__init__()
        true_r = 0.06098851551322548  # Radius of the QuEra dots for a logo of height 1

        self.color = color
        self.num_dots = 88
        self.aspect_ratio = 1112 / 252
        self.height = height
        self.dot_radius = height * true_r
        self.width = self.aspect_ratio * self.height

        # Generate non-overlapping dot positions
        self.start_points = self._generate_non_overlapping_points()
        self.dots = VGroup(
            *[
                Dot(point=pt, radius=self.dot_radius, color=color).set_fill(color, 1)
                for pt in self.start_points
            ]
        )
        self.add(self.dots)

    def _generate_non_overlapping_points(self, max_attempts=10000):
        points = []
        r = self.dot_radius
        for _ in range(max_attempts):
            if len(points) >= self.num_dots:
                break
            x = np.random.uniform(
                -self.width / 2 - self.height / 2 + r,
                self.width / 2 + self.height / 2 - r,
            )
            y = np.random.uniform(-self.height + r, self.height - r)
            pt = np.array([x, y, 0])
            if all(np.linalg.norm(pt - p) >= 2 * r for p in points):
                points.append(pt)
        if len(points) < self.num_dots:
            raise ValueError("Failed to place all dots non-overlapping.")
        return np.array(points)

    def form_logo(self, duration=2.0):
        """
        Transform the dots from the random scatter into the logo layout
        """
        # Load target layout
        target_logo = SVGMobject("split_logo_88.svg", height=self.height).set_color(
            self.color
        )
        target_dots = target_logo.family_members_with_points()

        if len(target_dots) != self.num_dots:
            raise ValueError(
                f"Expected {self.num_dots} target dots, got {len(target_dots)}."
            )

        tgt = np.array([d.get_center() for d in target_dots])
        src = np.array([d.get_center() for d in self.dots])
        cost = np.linalg.norm(src[:, None, :2] - tgt[None, :, :2], axis=2)
        rows, cols = linear_sum_assignment(cost)

        moves = [self.dots[r].animate.move_to(tgt[c]) for r, c in zip(rows, cols)]
        return AnimationGroup(*moves, run_time=duration, lag_ratio=0)

        return AnimationGroup(*transforms, run_time=duration, lag_ratio=0)

    def unform_logo(self, duration=2.0):
        """
        Transform the dots from the logo layout back to the
        original random scatter stored in self.start_points.
        """
        target_positions = np.array(self.start_points)
        source_positions = np.array([d.get_center() for d in self.dots])

        # optimal assignment back to the scatter points
        cost = np.linalg.norm(
            source_positions[:, None, :2] - target_positions[None, :, :2], axis=2
        )
        rows, cols = linear_sum_assignment(cost)

        moves = [
            self.dots[r].animate.move_to(target_positions[c])
            for r, c in zip(rows, cols)
        ]
        return AnimationGroup(*moves, run_time=duration, lag_ratio=0)


class BloqadeScatter(VGroup):
    """
    A scatter of (n) dots that will transform into bloqade_logo.svg
    and back again.
    """

    def __init__(self, color=QUERA_PURPLE, height=2.0):
        super().__init__()
        true_r = (
            0.047901424723754416  # Radius of the Bloqade dots for a logo of height 1
        )

        self.color = color
        self.num_dots = 100
        self.aspect_ratio = 760.91 / 175.9
        self.height = height
        self.dot_radius = height * true_r
        self.width = self.aspect_ratio * self.height

        # Generate non-overlapping dot positions
        self.start_points = self._generate_non_overlapping_points()
        self.dots = VGroup(
            *[
                Dot(point=pt, radius=self.dot_radius, color=color).set_fill(color, 1)
                for pt in self.start_points
            ]
        )
        self.add(self.dots)

    def _generate_non_overlapping_points(self, max_attempts=10000):
        points = []
        r = self.dot_radius
        for _ in range(max_attempts):
            if len(points) >= self.num_dots:
                break
            x = np.random.uniform(
                -self.width / 2 - self.height / 2 + r,
                self.width / 2 + self.height / 2 - r,
            )
            y = np.random.uniform(-self.height + r, self.height - r)
            pt = np.array([x, y, 0])
            if all(np.linalg.norm(pt - p) >= 2 * r for p in points):
                points.append(pt)
        if len(points) < self.num_dots:
            raise ValueError("Failed to place all dots non-overlapping.")
        return np.array(points)

    def form_logo(self, duration=2.0):
        """
        Transform the dots from the random scatter into the logo layout
        """
        # Load target layout
        target_logo = SVGMobject("bloqade_logo_100.svg", height=self.height).set_color(
            self.color
        )
        target_dots = target_logo.family_members_with_points()

        if len(target_dots) != self.num_dots:
            raise ValueError(
                f"Expected {self.num_dots} target dots, got {len(target_dots)}."
            )

        tgt = np.array([d.get_center() for d in target_dots])
        src = np.array([d.get_center() for d in self.dots])
        cost = np.linalg.norm(src[:, None, :2] - tgt[None, :, :2], axis=2)
        rows, cols = linear_sum_assignment(cost)

        moves = [self.dots[r].animate.move_to(tgt[c]) for r, c in zip(rows, cols)]
        return AnimationGroup(*moves, run_time=duration, lag_ratio=0)

        return AnimationGroup(*transforms, run_time=duration, lag_ratio=0)

    def unform_logo(self, duration=2.0):
        """
        Transform the dots from the logo layout back to the
        original random scatter stored in self.start_points.
        """
        target_positions = np.array(self.start_points)
        source_positions = np.array([d.get_center() for d in self.dots])

        # optimal assignment back to the scatter points
        cost = np.linalg.norm(
            source_positions[:, None, :2] - target_positions[None, :, :2], axis=2
        )
        rows, cols = linear_sum_assignment(cost)

        moves = [
            self.dots[r].animate.move_to(target_positions[c])
            for r, c in zip(rows, cols)
        ]
        return AnimationGroup(*moves, run_time=duration, lag_ratio=0)
