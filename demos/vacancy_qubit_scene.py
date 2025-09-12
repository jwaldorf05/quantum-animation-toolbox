from manimlib import *

from quantum_animation_toolbox.quera_colors import *
from quantum_animation_toolbox.quera_qubit_lib import *


class VacancyQubitScene(Scene):
    def construct(self):
        # two side-by-side vacancies
        left = Vacancy(position=LEFT * 0.5, color=QUERA_MGRAY)
        right = Vacancy(position=RIGHT * 0.5, color=QUERA_MGRAY)

        # qubit starts in the left vacancy
        q = Qubit(position=left.get_center(), color=QUERA_PURPLE)

        self.add(left, right, q)
        self.wait(0.5)

        # move qubit to the right vacancy
        self.play(q.animate.move_to(right.get_center()), run_time=1)

        # then give it a single pulse
        q.start_pulsing(scale_factor=1.5)
        self.wait(0.5)
        # clear the pulsing updater, but don’t hide the glow
        q.glow.clear_updaters()
