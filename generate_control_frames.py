#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
from pathlib import Path

# 1)  toolbox src  (already in your snippet) ---------------------------
HERE = Path(__file__).parent.resolve()
SRC     = HERE / "src" / "quantum_animation_toolbox"
sys.path.insert(0, SRC.as_posix())

from quera_colors    import *
from quera_qubit_lib import *

# 2)  tests / demos ----------------------------------------------------
TESTS = HERE / "demos"          # “.. / .. / tests”
sys.path.insert(0, TESTS.as_posix())             # add *in front* of PYTHONPATH

from qubit_demo import QubitDemo
from vacancy_qubit_scene import VacancyQubitScene
from qubit_array_scene import QubitArrayScene
from dot_laser_tweezer_scene import DotLaserTweezerScene
from line_laser_tweezer_scene import LineLaserTweezerScene
from entanglements_scene import EntanglementsScene

from conftest import *

# 4) where to dump them
CTRL_DIR = HERE / "control_data"
CTRL_DIR.mkdir(parents=True, exist_ok=True)

def render_last_frame(scene_cls, *, pixel=300):
    """
    Run a Manim GL scene_cls and return its final RGB frame as a numpy array.
    """
    with tempconfig({
        "pixel_height": pixel,
        "pixel_width":  pixel,
        "frame_rate":   1,      # one step per frame
        "dry_run":      True,   # don’t write a video
    }):
        scene = scene_cls()
        scene.run()
        # GL stores its buffer in scene.camera, not scene.renderer:
        return scene.camera.get_image()

if __name__ == "__main__":
    # list your scenes here
    scenes = [
        QubitDemo,
        VacancyQubitScene,
        QubitArrayScene,
        DotLaserTweezerScene,
        LineLaserTweezerScene,
        EntanglementsScene,
    ]
    for cls in scenes:
        print(f"Rendering → {cls.__name__}…", end="", flush=True)
        frame = render_last_frame(cls, pixel=300)
        out = CTRL_DIR / f"{cls.__name__}.npz"
        np.savez_compressed(out, frame=frame)
        print(" saved.")