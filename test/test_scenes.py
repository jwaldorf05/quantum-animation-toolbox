"""
Graphical regression tests for demo scenes in `quera-animation-toolbox`.

Run from the *repo root* with:
    pytest test -q                # Compare scenes to control frames
    pytest test --set_test        # Generate new control frames
    pytest test --show_diff       # Show pixel diffs on mismatch
"""
from __future__ import annotations
import sys
from pathlib import Path

# ── Set up paths to toolbox and demo sources ───────────────────────
CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parent.parent
SRC = ROOT / "src" / "quantum_animation_toolbox"
TESTS = ROOT / "tests"

sys.path.insert(0, SRC.as_posix())
sys.path.insert(0, TESTS.as_posix())

_PIXEL = 300  # test resolution

# ── Lazy scene imports to delay manimlib initialization ────────────
def get_scene_classes():
    from demos.qubit_demo import QubitDemo
    from demos.vacancy_qubit_scene import VacancyQubitScene
    from demos.qubit_array_scene import QubitArrayScene
    from demos.dot_laser_tweezer_scene import DotLaserTweezerScene
    from demos.line_laser_tweezer_scene import LineLaserTweezerScene
    from demos.entanglements_scene import EntanglementsScene
    from demos.hadamard_scene import HadamardGateScene
    from demos.measurement_scene import MeasurementGateScene
    from demos.wire_scene import WireScene
    from demos.circuit_scene import CircuitScene
    from demos.pulse_scene import PulseScene

    return [
        QubitDemo,
        VacancyQubitScene,
        QubitArrayScene,
        DotLaserTweezerScene,
        LineLaserTweezerScene,
        EntanglementsScene,
        HadamardGateScene,
        MeasurementGateScene,
        WireScene,
        CircuitScene,
        PulseScene
    ]

# ── Lazy decorator import to delay conftest/manimlib loading ───────
def register_tests():
    from conftest import frames_comparison  # do this only after CLI parsing
    for scene_cls in get_scene_classes():
        test_func = frames_comparison(pixel=_PIXEL)(scene_cls)
        globals()[f"test_{scene_cls.__name__}"] = test_func

# ── Trigger test registration AFTER pytest CLI is ready ────────────
register_tests()