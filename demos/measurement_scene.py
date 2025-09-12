from manimlib import *

from quantum_animation_toolbox.quera_colors import *
from quantum_animation_toolbox.quera_circuit_lib import *


class MeasurementScene(Scene):
    def construct(self):
        bg = BackgroundColor(BLACK)
        m1 = MeasurementGate(start=2.4 * LEFT, slot=0, size=2, pitch =2.4, color=RED_E, axis="X")
        m2 = MeasurementGate(start=2.4 * LEFT, slot=1, size=2, pitch =2.4, color=GREEN_E, axis="Y")
        m3 = MeasurementGate(start=2.4 * LEFT, slot=2, size=2, pitch =2.4, color=BLUE_E, axis="Z")
        self.add(bg, m1, m2, m3)