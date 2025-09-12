from manimlib import *

from quantum_animation_toolbox.quera_colors import *
from quantum_animation_toolbox.quera_circuit_lib import *


class HadamardScene(Scene):
    def construct(self):
        bg = BackgroundColor(BLACK)
        h1 = HadamardGate(start=2.4 * LEFT, slot=0, size=2, pitch =2.4, color=RED_E, bg_color=RED_C)
        h2 = HadamardGate(start=2.4 * LEFT, slot=1, size=2, pitch =2.4, color=GREEN_E, bg_color=GREEN_C)
        h3 = HadamardGate(start=2.4 * LEFT, slot=2, size=2, pitch =2.4, color=BLUE_E, bg_color=BLUE_C)
        self.add(bg, h1, h2, h3)