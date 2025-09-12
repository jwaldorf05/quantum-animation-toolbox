![AnimationToolboxLogo](example_images/LandscapeLogo.png)
# Quantum Animation Toolbox

**Disclaimer** this library was initially developed as an internship project at QuEra. It is **not** an official QuEra producted and is **not maintained by QuEra**.

This library was made to create quantum circuits and qubit visualization of QuEra processors in [ManimGL](https://github.com/3b1b/manim), as well as providing QuEra styling for animations. 

QuEra logos, images and stylings can be found on [our website](https://www.quera.com/media-kit).

Thank you to [Grant Sanderson](https://github.com/3b1b) for creating the Manim framework on which this package is based. Note that this package runs on [ManimGL](https://github.com/3b1b/manim) (also known as 3b1b Manim) as opposed to ManimCE. Ensure you are using the correct version on Manim to run this package, the ManimGL installation guide can be found [here](https://3b1b.github.io/manim/getting_started/installation.html). Documentation for ManimGL can be found [here](https://3b1b.github.io/manim/).

To see this package documentation, testing, and installation guides, check out our [wiki page](../../wiki).


## Installation

To use, make sure that your files have this structure:
```text
manim_stuff/
├── venv/                            # virtual environment
├── project_folder/                  # contains your Manim script
│   └── your_script.py
├── quera-animation-toolbox/
│   └── src/
│       └── quantum_animation_toolbox/
│           ├── quera_colors.py
│           ├── quera_circuit_lib.py
│           └── quera_qubit_lib.py
```

Then once you're in your script file add to the header:
```python
from manimlib import *
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.normpath(
	os.path.join(current_dir, "..", "quera-animation-toolbox", "src", "quantum_animation_toolbox")
)
sys.path.append(lib_path)

from quera_colors import *
from quera_qubit_lib import *
```

## Testing

This library contains tests to make sure that the classes work properly. In order to run these tests after installaction, enter a terminal window in your `quera-animation-toolbox` folder and run:

```zsh
python generate_control_frames.py 
```
followed by:
```zsh
pytest -q
```

If you get a message saying "6 passed", you're all good to go!

## Qubit Visualization Toolkit: `quera_qubit_lib.py`

A compact Manim GL helper-library that lets you **draw, animate, and interact with
arrays of qubits**—plus the laser tweezers and glow effects you need to represent qubit motions within a quantum processor.

### Classes

**Qubit(self, position = ORIGIN, radius = 0.15, glow_radius = 0.4, color)**  
A drawable qubit: a colored **Dot** with a faint **GlowDot** centred behind it.  
You can recolor it, make the glow breathe continuously, or trigger individual glow pulses.

**Methods**  
`set_color(new_color, animate=False)` – change colour (returns a list of animations when `animate=True`)  
`reset_color(animate=False)` – restore the original colour  
`turn_glow(on=True, animate=False)` – show / hide the glow ring  
`pulse_once(run_time=1.0, scale_factor=1.5)` – *returns* a single in-and-out glow pulse you can pass to `self.play`  
`start_pulsing(pulse_period=1.0, scale_factor=1.5)` – begin a continuous “breathing” glow  
`stop_pulsing()` – halt the breathing effect and hide the glow

![QubitPulse](example_videos/QubitDemo.gif)
```python
class QubitDemo(Scene):
    def construct(self):
        q = Qubit(ORIGIN, color=QUERA_PURPLE)
        self.add(q)

        # breathe for three seconds
        q.start_pulsing(scale_factor=1.5)
        self.wait(3)
        q.stop_pulsing()
        self.wait(0.3)

        # change color to red
        self.play(*q.set_color(QUERA_RED, animate=True))
        self.wait(0.2)

        # one stronger pulse
        q.start_pulsing(scale_factor=2)
        self.wait(1)
        q.stop_pulsing()
        self.wait(0.5)

        # return to original purple
        self.play(*q.reset_color(animate=True))
        self.wait()
```

**Vacancy(self, position, radius, color, stroke_width, num_dashes)**  
Dashed-circle placeholder indicating where a **Qubit** *could* sit.

Methods  
`set_color(new_color, new_opacity=None)` – change stroke (and optional opacity)

![VacancyWithQubit](example_videos/VacancyQubitScene.gif)

```python
class VacancyQubitScene(Scene):
	def construct(self):
		# two side-by-side vacancies
		left  = Vacancy(position=LEFT  * 0.5, color=QUERA_MGRAY)
		right = Vacancy(position=RIGHT * 0.5, color=QUERA_MGRAY)

		# qubit starts in the left vacancy
		q = Qubit(position=left.get_center(), color=QUERA_PURPLE)

		self.add(left, right, q)
		self.wait(0.5)

		# move qubit to the right vacancy …
		self.play(q.animate.move_to(right.get_center()), run_time=1)

		# … then give it a single pulse
		q.pulse_once(self, run_time=1, scale_factor=1.5)
		self.wait()
```

**QubitArray(self, layout, …)**  
Versatile container that arranges **Qubit** and **Vacancy** objects in a *linear* row, rectangular *grid*, or two-row *bilinear* pattern.  
It also provides convenience methods to move / place qubits or vacancies after creation.

Constructor highlights  

| Arg | Meaning | Default |
|-----|---------|---------|
| `layout` | `"linear"`, `"grid"`, or `"bilinear"` | `"linear"` |
| `num_qubits` | how many qubits (linear / bilinear) | `6` |
| `rows`, `cols` | grid dimensions | `2`, `3` |
| `qubit_spacing` | horizontal spacing | `1.0` |
| `line_spacing` | vertical gap (bilinear) | `2` |
| `use_vacancies` | draw dashed placeholders too? | `True` |
| `fill_pattern` | `"all"`, `"none"`, `"alternate"` or *set[int]* | `"all"` |

Selected methods  

* `move_qubits(scene, [(idx, dx, dy), …])` – shift chosen qubits  
* `move_vacancies(scene, …)` – same for vacancies  
* `place_qubits(scene, [(idx, x, y), …])` – absolute positioning  
* `get_pairs([(i,j), …])` – return list of *(Qubit, Qubit)* pairs  
* `remove_vacancies()` – delete all vacancy markers

![QubitArray](example_videos/QubitArrayScene.gif)

```python
class QubitArrayScene(Scene):
    def construct(self):
        # 4×4 grid with a “column-alternate” fill
        array = QubitArray(
            layout="grid",
            rows=4, cols=4,
            qubit_spacing=1.0,
            use_vacancies=True,
            fill_pattern="checkerboard"      # whole columns 0 & 2 filled
        )
        glow_group = Group(*(q.glow for q, _ in array.qubits))
        self.add(glow_group, array)          # glows first → higher z-index
        self.wait(0.5)

        # --- build shifts: odd ROW → right, even ROW → left -------------
        shifts = []
        qubits_per_row = 2
        for idx, (q, pos) in enumerate(array.qubits):
            row = idx // qubits_per_row       # row number 0-based
            dx  = -1 if row % 2 else +1   # odd row → +1, even row → –1
            shifts.append((idx, dx, 0))

        # animate the slide
        array.move_qubits(self, shifts, run_time=1.2, animate=True)

        # pulse each qubit once after arriving
        self.play(*[q.pulse_once(scale_factor=2.5)
                    for q, _ in array.qubits])
        array.move_qubits(self, [(indiv[0], -indiv[1], indiv[2]) for indiv in shifts], run_time=1.2, animate=True)
        self.wait()
```

**DotLaserTweezer(self, position, radius, color, opacity)**  
Glowing “tweezer-dot” that can grab a **Qubit**, drag it around, and release it anywhere in the scene.  
Useful for interactive demos where atoms are rearranged by optical tweezers.

Key methods  
* `move_to(target)` – jump/animate to a new location (another tweezer or point)  
* `pick_up(qubit)` – attach a qubit; it will follow the tweezer  
* `release()` – drop the qubit (stops following)  
* *(auto-called)* tweezer keeps the qubit centred via an updater while attached

![DotLaserTweezerDemo](example_videos/DotLaserTweezerScene.gif)

```python
class DotLaserTweezerScene(Scene):
	def construct(self):
		# --- build a 2×2 array: vacancies everywhere,
		#     qubits only in the two left-hand sites ---------
		array = QubitArray(layout="grid",
		                   rows=2, cols=2,
		                   use_vacancies=True,
		                   fill_pattern={0, 2})   # indices: 0 upper-left, 2 lower-left
		self.add(array)
		self.wait(0.5)

		# Tweezer starts over the upper-left qubit
		tweezer = DotLaserTweezer()
		tweezer.move_to(array.get_qubit(0))
		tweezer.set_opacity(0)
		self.add(tweezer)

		# --- move first qubit from left to right -----------------------
		self.play(*tweezer.pick_up(array.get_qubit(0), show=True), run_time=0.3)
		self.play(tweezer.animate.move_to(array.get_vacancy(1).get_center()),
		          run_time=1.0)
		self.play(*tweezer.release(hide=True), run_time=0.3)
		self.wait(0.5)

		# --- move second qubit (lower-left) to lower-right -------------
		tweezer.move_to(array.get_qubit(1))  # jump down
		self.play(*tweezer.pick_up(array.get_qubit(1), show=True), run_time=0.3)
		self.play(tweezer.animate.move_to(array.get_vacancy(3).get_center()),
		          run_time=1.0)
		self.play(*tweezer.release(hide=True), run_time=0.3)

		# Pulse both relocated qubits
		self.play(*[array.get_qubit(idx).pulse_once() for idx in (0,1)])
		self.wait()
```

**LineLaserTweezer(self, point1, point2, qubits, catch_radius, …)**  
Glowing *line* tweezer that “grabs” **every Qubit** lying within `catch_radius` of its path and drags them together when it moves. Perfect for demonstrating bulk transport of atoms with an optical conveyor belt.

Key methods  
* `move_by(scene, delta, run_time=1.0)` – shift laser **and** all collected qubits by a vector  
* `turn_on(animate=False)` / `turn_off()` – fade the beam in / out  
* `pulse(scene, run_time=1.0)` – quick brightening flash  
*(collection list auto-updates after every move)*

![LineLaserTweezerDemo](example_videos/LineLaserTweezerScene.gif)

```python
class LineLaserTweezerScene(Scene):
    def construct(self):
        array = QubitArray(
            layout="grid",
            rows=2, cols=2,
            use_vacancies=True,
            fill_pattern={0, 2}          # Left side qubits start filled
        )
        self.add(array)

        left_vac = array.get_vacancy(0)
        p1 = left_vac.get_center() + UP * 0.2
        p2 = left_vac.get_center() + DOWN * 1.2

        laser = LineLaserTweezer(
            p1, p2,
            qubit_array=array,
            catch_radius=0.2,
            opacity=0,
            infinite=True
        )
        self.wait(0.5)
        self.add(laser)
        laser.turn_on(scene=self, animate=True, run_time=0.5)
        self.wait(0.3)

        shift_vec = RIGHT * array.qubit_spacing
        laser.move_by(
            scene=self,
            delta=shift_vec,
            run_time=1.2,
            animate=True                 # show the motion
        )

        laser.turn_off(scene=self, animate=True, run_time=0.5)
        self.wait(0.5)
        laser.turn_on(scene=self, animate=True, run_time=0.5)

        laser.move_by(
            scene=self,
            delta=-shift_vec,
            run_time=1.2,
            animate=True                 # show the motion
        )

        laser.turn_off(scene=self, animate=True, run_time=0.5)

        self.wait(0.5)
```

**entanglements(scene, pairs, color, width, height, run_time)**  
Animates *all* supplied qubit–qubit pairs **in parallel**:  
1. A glowing `GlowEllipse` fades-in to connect each pair.  
2. Both qubits smoothly blend from their own colour to `color`.  
3. After a short pause everything plays backward, restoring the original state and hiding the ellipses.  

Parameters  
| name | purpose | default |
|------|---------|---------|
| `scene` | current `Scene` instance | – |
| `pairs` | iterable of `(q1, q2)` `Qubit` tuples | – |
| `color` | entanglement highlight colour | `QUERA_RED` |
| `width` / `height` | ellipse size | `0.6` / `0.15` |
| `run_time` | fade-in (and fade-out) duration | `0.4` |



![EntanglementDemo](example_videos/EntanglementsScene.gif)

```python
class EntanglementsScene(Scene):
    def construct(self):
        array = QubitArray(
            layout="grid",
            rows=2, cols=4,
            use_vacancies=True,
            fill_pattern="all"
        )
        self.add(array)

        self.wait(0.5)

        dx = 0.5 * array.qubit_spacing

        moves = [
            (0,  dx, 0),   # qubit-0  → right
            (4,  dx, 0),   # qubit-4  → right
            (3, -dx, 0),   # qubit-3  ← left
            (7, -dx, 0)    # qubit-7  ← left
        ]

        # use DotLaserTweezer internally
        array.move_qubits(
            scene=self,
            shift_list=moves,
            run_time=0.5,
            animate=True,
            use_lasers=True
        )

        self.wait(0.3)

        pairs = array.get_pairs([(0, 1), (2, 3), (4, 5), (6, 7)])

        entanglements(
            self, pairs,
            color=QUERA_RED,
            width=0.8, height=0.2,
            run_time=0.5,
            animate=True
        )

        self.wait(0.3)

        # Qubits moving in reverse
        array.move_qubits(
            scene=self,
            shift_list=[(indiv[0], -indiv[1], -indiv[2]) for indiv in moves],
            run_time=0.5,
            animate=True,
            use_lasers=True
        )

        self.wait(0.5)
```


## Quantum Circuit Maker: `quera_circuit_lib.py`

**HadamardGate(slot, start, pitch, size=0.5, color=WHITE, bg_color=BLACK, shift=None)**  
A drawable 1-qubit **H** gate that snaps to a specified `slot` on a horizontal quantum wire.  
Gate placement is determined by the wire’s `start` point, the slot number, and the uniform `pitch` between slots.  
Use `shift="left"` or `"right"` to nudge the gate sideways by ±(pitch / 3) for better layout in multi-gate circuits.

No animations or interactive methods are defined (use Manim’s standard transforms to move or fade gates).

![HadamardGateExample](example_videos/HadamardGateScene.png)

```python
class HadamardGateScene(Scene):
	def construct(self):
		bg = BackgroundColor(BLACK)
		h = HadamardGate(start=ORIGIN, slot=0, size=2, pitch =2.4, color=WHITE)
		self.add(bg, h)
```


### Classes

## QuEra Styling Toolkit: `quera_colors.py`
A drop-in helper-package that skins **Manim GL** with QuEra’s look-and-feel and a handful of higher-level animations.

### Classes:
**QuEraLogo(color, height, include_text)**  
A lightweight SVG wrapper for the QuEra company logo. Accepts any color and can render either the simple “QuEra” mark or the full “QuEra Computing Inc” version.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `color` | `str` or `ManimColor` | `QUERA_PURPLE` | Hex string (`"#FF9900"`) or a Manim `Color` object. |
| `height` | `float` | `QUERA_LOGO_HEIGHT` | Desired logo height in scene units. |
| `include_text` | `bool` | `False` | `True` → full-text SVG • `False` → icon + “QuEra”. |

Methods (inherited from **`SVGMobject`**)
* `.set_color(new_color)` &nbsp;– change fill/stroke colour.  
* `.set_height(h)` &nbsp;– resize while preserving aspect ratio.  
* `.scale(factor)` &nbsp;– uniformly scale about the logo’s centre.  
* `.shift(vector)` / `.move_to(point)` &nbsp;– translate.  
* Any standard SVGMobject transform (rotate, flip, fade, etc.).

> **File expectation** The class looks for  
> `quera_logo.svg` (icon + wordmark) and `quera_logo_text.svg`  
> (long “QuEra Computing Inc”) in your `vector_images/` folder.  

**BackgroundColor(color, z_index)**  
A full-screen rectangle that tints the entire scene background.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `color`   | `ManimColor` | `BLACK` | Solid fill colour behind every other mobject. |
| `z_index` | `int`        | `-100` | Negative index places the panel *under* normal content. |  

**QuEraSlideBG(self, key)**  
This is how you set the background of your Manim scene to match the QuEra Slide templates, key argument goes from 1 to 6 for the following backgrounds:
- slide_bg_00: Purple Title and Textbox
![Background 0](raster_images/quera_bgs/slide_bg_00.png)
- slide_bg_01: White Logo Blank
![Background 1](raster_images/quera_bgs/slide_bg_01.png)
- slide_bg_02: Dark Purple Title and space
![Background 2](raster_images/quera_bgs/slide_bg_02.png)
- slide_bg_03: Light Purple Blank
![Background 3](raster_images/quera_bgs/slide_bg_03.png)
- slide_bg_04: White Title and space
![Background 4](raster_images/quera_bgs/slide_bg_04.png)
- slide_bg_05: Dark Purple Blank
![Background 5](raster_images/quera_bgs/slide_bg_05.png)

**QueraBanner(self, color, direction)**
Creates the QuEra logo packed into $\ket{Q}$ form, which can be expanded and contracted using methods

Parameters:
color - logo color, defaults to QUERA_PURPLE
direction - direction to which the banner opens

Methods:
`create(run_time=1)` - draws folded $\ket{Q}$ logo dot by dot
`expand(run_time=0.75)` - expands logo to $\ket{QuEra}$ in the direction specified by initialization
`contract(run_time=0.75)` - undoes `expand()` and returns to $\ket{Q}$ form

![QuEraLogoExpanding](example_videos/QuEraRevealScene.gif)
```python
class QuEraRevealScene(Scene):
	def construct(self):
		banner = QuEraBanner(color=QUERA_PURPLE, direction="center")
		self.add(banner)
		self.play(banner.create())
		self.wait(0.3)
		self.play(banner.expand())
		self.wait()
```

![LeftRightCenter](example_videos/ExpandDirections.gif)
```python
class ExpandDirections(Scene):
	def construct(self):
		banner1 = QuEraBanner(color=QUERA_PURPLE, direction="left")
		banner2 = QuEraBanner(color=QUERA_PURPLE, direction="center")
		banner3 = QuEraBanner(color=QUERA_PURPLE, direction="right")
		banner1.scale(0.5).shift(2*UP)
		banner2.scale(0.5)
		banner3.scale(0.5).shift(2*DOWN)

		self.add(banner1, banner2, banner3)
		self.play(banner1.appear(),banner2.appear(),banner3.appear())
		self.wait(0.5)
		self.play(banner1.expand(),banner2.expand(),banner3.expand())
		self.wait(1)
		self.play(banner1.contract(),banner2.contract(),banner3.contract())
		self.wait(0.5)
```

**QueraScatter(self, color, height)**
Creates a random arrangement of dots that when `form_logo()` is called transform into the QuEra logo.

Methods:
`form_logo(duration=2.0)` - Makes dots transform themselves into QuEra logo
`form_logo(duration=2.0)` - Makes dots transform themselves into QuEra logo

![QuEraLogoForm](example_videos/QuEraScatterScene.gif)
```python
class QuEraScatterScene(Scene):
	def construct(self):
		scatter = QuEraScatter(color=QUERA_PURPLE)
		self.add(scatter)
		self.wait(0.5)
		self.play(scatter.form_logo())
		self.wait(0.5)
		self.play(scatter.unform_logo())
		self.wait()
```

**BloqadeScatter(self, color, height)**
Creates a random arrangement of dots that when `form_logo()` is called transform into the QuEra logo.

Methods:
`form_logo(duration=2.0)` - Makes dots transform themselves into QuEra logo
`form_logo(duration=2.0)` - Makes dots transform themselves into QuEra logo

![BloqadeLogoForm](example_videos/BloqadeScatterScene.gif)
```python
class BloqadeScatterScene(Scene):
	def construct(self):
		scatter = BloqadeScatter(color=QUERA_PURPLE)
		self.add(scatter)
		self.wait(0.5)
		self.play(scatter.form_logo())
		self.wait(0.5)
		self.play(scatter.unform_logo())
		self.wait()
```







