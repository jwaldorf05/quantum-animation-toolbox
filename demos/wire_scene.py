from manimlib import *

from quantum_animation_toolbox.quera_colors import *
from quantum_animation_toolbox.quera_circuit_lib import *



class WireScene(Scene):
    def construct(self):
        bg = BackgroundColor(BLACK)
        w1 = Wire(start=2*LEFT, end=2*RIGHT, num_slots = 5, pitch = 1.0)
        w2 = ClassicalWire(start=2*LEFT + DOWN, end=2*RIGHT + DOWN, num_slots = 5, pitch=1.0)
        g1 = HadamardGate(slot=0, start=2*LEFT, pitch=1, size=0.8, color=RED)
        g2 = MeasurementGate(slot=0, start=2*LEFT+DOWN, pitch=1, size=0.8, color=GREEN)
        g3 = CNOTGate(slot=1, start_control=2*LEFT, start_target=2*LEFT+DOWN, pitch=1, color=BLUE)
        g4 = SwapGate(slot=2, start_control=2*LEFT, start_target=2*LEFT+DOWN, pitch=1, color=ORANGE)
        g5 = CZGate(slot=3, start_control=2*LEFT, start_target=2*LEFT+DOWN, pitch=1, color=PURPLE)
        # g6 = WireBundle()


        self.add(bg, w1, w2, g1, g2, g3, g4, g5)