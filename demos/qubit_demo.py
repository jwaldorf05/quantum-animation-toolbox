from manimlib import *

from quantum_animation_toolbox.quera_colors import *
from quantum_animation_toolbox.quera_qubit_lib import *


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
        self.wait(0.5)

        # clear the pulsing updater, but don’t hide the glow
        q.glow.clear_updaters()
