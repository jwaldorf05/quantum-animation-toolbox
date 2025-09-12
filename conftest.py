# from __future__ import annotations

# from contextlib import contextmanager
# from pathlib import Path
# from typing import Callable, Dict

# import numpy as np
# import pytest
# from manimlib import Scene
# from manimlib.config import manim_config as mconfig

# # ------------------------------------------------------------------ #
# #  tempconfig – tiny replacement for the CE helper (ManimGL only)    #
# # ------------------------------------------------------------------ #
# import sys

# @contextmanager
# def tempconfig(overrides: Dict):
#     """
#     Temporarily override manimlib.manim_config inside a with-block,
#     suppressing CLI argument parsing side effects from ManimGL.
#     """
#     # Step 1: Suppress CLI args to prevent ManimGL from reading pytest flags
#     original_argv = sys.argv[:]
#     sys.argv = [sys.argv[0]]  # Prevent argparse from seeing --set_test etc.

#     # Step 2: Import Manim only inside this context to avoid early CLI parsing
#     from manimlib.config import manim_config as mconfig
#     backup = mconfig.copy()
#     mconfig.update(overrides)

#     try:
#         yield
#     finally:
#         # Step 3: Restore everything to avoid side effects
#         mconfig.clear()
#         mconfig.update(backup)
#         sys.argv = original_argv
# # ------------------------------------------------------------------ #
# #  Location for control frames                                       #
# # ------------------------------------------------------------------ #
# CTRL_ROOT = Path(__file__).parent / "control_data"
# CTRL_ROOT.mkdir(exist_ok=True)

# # ------------------------------------------------------------------ #
# #  Helper: render last frame of a scene at low resolution            #
# # ------------------------------------------------------------------ #
# def _render_last_frame(scene_cls: type[Scene], *, pixel: int = 480) -> np.ndarray:
#     """
#     Render `scene_cls` once at low-res and return the final RGB frame.

#     • Works in **Manim GL** (scene.run → scene.camera.get_image())  
#     • Still works in **Manim CE** (scene.render → scene.renderer.get_frame())
#     """
#     with tempconfig(
#         {
#             "pixel_height": pixel,
#             "pixel_width": pixel,
#             "frame_rate": 1,      # 1 fps ⇒ one frame per animation step
#             "dry_run": True,      # no video file, but frame-buffer is filled
#         }
#     ):
#         scene = scene_cls()

#         # --- run the scene --------------------------------------------------
#         if hasattr(scene, "render"):      # Community-Edition
#             scene.render()
#             frame = scene.renderer.get_frame()
#         else:                             # Manim GL
#             scene.run()
#             frame = scene.camera.get_image()   # <─ this is the missing bit!

#         return frame

# # ------------------------------------------------------------------ #
# #  Register custom CLI options                                       #
# # ------------------------------------------------------------------ #
# def pytest_addoption(parser: pytest.Parser) -> None:
#     parser.addoption(
#         "--set_test",
#         action="store_true",
#         help="Generate control frames instead of comparing them.",
#     )
#     parser.addoption(
#         "--show_diff",
#         action="store_true",
#         help="Save & open pixel-diff images when a graphical test fails.",
#     )

# # ------------------------------------------------------------------ #
# #  Public decorator                                                  #
# # ------------------------------------------------------------------ #
# def frames_comparison(*, pixel: int = 480, atol: int = 1) -> Callable[[type], Callable]:
#     """
#     Decorator for **Scene classes**:

#         @frames_comparison(pixel=300)
#         class MyScene(Scene): ...
#     """

#     def decorator(scene_cls: type[Scene]) -> Callable[..., None]:
#         test_name  = scene_cls.__name__
#         ctrl_file  = CTRL_ROOT / f"{test_name}.npz"

#         @pytest.mark.usefixtures("request")
#         def _test(request: pytest.FixtureRequest) -> None:
#             generate = request.config.getoption("--set_test")
#             showdiff = request.config.getoption("--show_diff")
#             frame    = _render_last_frame(scene_cls, pixel=pixel)

#             if generate:
#                 np.savez_compressed(ctrl_file, frame=frame)
#                 pytest.skip(f"Control data written for {test_name}")
#                 return

#             if not ctrl_file.exists():
#                 pytest.skip(f"No control data for {test_name}. Run once with --set_test.")
#                 return

#             ref = np.load(ctrl_file)["frame"]
#             try:
#                 np.testing.assert_allclose(frame, ref, atol=atol)
#             except AssertionError:
#                 if showdiff:
#                     _save_and_open_diff(frame, ref, test_name)
#                 raise

#         _test.__name__ = f"test_{test_name}"
#         return _test

#     return decorator

# # ------------------------------------------------------------------ #
# #  Optional helper to visualise diff (PNG)                           #
# # ------------------------------------------------------------------ #
# def _save_and_open_diff(frame: np.ndarray, ref: np.ndarray, name: str) -> None:
#     import imageio.v3 as iio
#     diff   = np.abs(frame.astype(int) - ref.astype(int)).clip(0, 255).astype(np.uint8)
#     outdir = CTRL_ROOT / "_diff"
#     outdir.mkdir(exist_ok=True)
#     path   = outdir / f"{name}_diff.png"
#     iio.imwrite(path, diff)
#     try:  # macOS preview, adapt for Linux/Windows if you like
#         import subprocess, sys
#         if sys.platform == "darwin":
#             subprocess.run(["open", path])
#     except Exception:
#         pass

from __future__ import annotations
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pytest

# Directory to store control frames
CTRL_ROOT = Path(__file__).parent / "control_data"
CTRL_ROOT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# CLI Flags
# ---------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--set_test",
        action="store_true",
        help="Generate control frames instead of comparing them."
    )
    parser.addoption(
        "--show_diff",
        action="store_true",
        help="Save and open pixel-diff images when a graphical test fails."
    )


# ---------------------------------------------------------------------
# Temporary config override for ManimGL
# ---------------------------------------------------------------------
@contextmanager
def tempconfig(overrides: Dict):
    """
    Temporarily override manimlib.manim_config inside a with-block.
    This safely disables CLI parsing from manimlib.config.
    """
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]  # Strip flags like --set_test

    from manimlib.config import manim_config as mconfig
    backup = mconfig.copy()
    mconfig.update(overrides)

    try:
        yield
    finally:
        mconfig.clear()
        mconfig.update(backup)
        sys.argv = original_argv


# ---------------------------------------------------------------------
# Helper: render last frame of a scene
# ---------------------------------------------------------------------
def _render_last_frame(scene_cls: type, *, pixel: int = 480) -> np.ndarray:
    with tempconfig({
        "pixel_height": pixel,
        "pixel_width": pixel,
        "frame_rate": 1,
        "dry_run": True,
    }):
        scene = scene_cls()
        if hasattr(scene, "render"):  # ManimCE
            scene.render()
            frame = scene.renderer.get_frame()
        else:  # ManimGL
            scene.run()
            frame = scene.camera.get_image()
        return frame


# ---------------------------------------------------------------------
# Main public decorator
# ---------------------------------------------------------------------
def frames_comparison(*, pixel: int = 480, atol: int = 1) -> Callable[[type], Callable]:
    """
    Decorator for Scene classes:
        @frames_comparison(pixel=300)
        class MyScene(Scene): ...
    """
    def decorator(scene_cls: type) -> Callable[..., None]:
        test_name = scene_cls.__name__
        ctrl_file = CTRL_ROOT / f"{test_name}.npz"

        @pytest.mark.usefixtures("request")
        def _test(request: pytest.FixtureRequest) -> None:
            generate = request.config.getoption("--set_test")
            showdiff = request.config.getoption("--show_diff")
            frame = _render_last_frame(scene_cls, pixel=pixel)

            if generate:
                np.savez_compressed(ctrl_file, frame=frame)
                pytest.skip(f"Control data written for {test_name}")
                return

            if not ctrl_file.exists():
                pytest.skip(f"No control data for {test_name}. Run once with --set_test.")
                return

            ref = np.load(ctrl_file)["frame"]
            try:
                np.testing.assert_allclose(frame, ref, atol=atol)
            except AssertionError:
                if showdiff:
                    _save_and_open_diff(frame, ref, test_name)
                raise

        _test.__name__ = f"test_{test_name}"
        return _test

    return decorator


# ---------------------------------------------------------------------
# Helper: show visual diff if needed
# ---------------------------------------------------------------------
def _save_and_open_diff(frame: np.ndarray, ref: np.ndarray, name: str) -> None:
    import imageio.v3 as iio
    import subprocess
    import sys

    diff = np.abs(frame.astype(int) - ref.astype(int)).clip(0, 255).astype(np.uint8)
    outdir = CTRL_ROOT / "_diff"
    outdir.mkdir(exist_ok=True)
    path = outdir / f"{name}_diff.png"

    iio.imwrite(path, diff)

    if sys.platform == "darwin":
        subprocess.run(["open", path])
    elif sys.platform == "linux":
        subprocess.run(["xdg-open", path])
    elif sys.platform == "win32":
        os.startfile(path)