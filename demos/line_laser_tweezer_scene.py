from manimlib import *

from quantum_animation_toolbox.quera_colors import *
from quantum_animation_toolbox.quera_qubit_lib import *


class LineLaserTweezerScene(Scene):
    def construct(self):
        array = QubitArray(
            layout="grid",
            rows=2,
            cols=2,
            use_vacancies=True,
            fill_pattern={0, 2},  # Left side qubits start filled
        )
        self.add(array)

        left_vac = array.get_vacancy(0)
        p1 = left_vac.get_center() + UP * 0.2
        p2 = left_vac.get_center() + DOWN * 1.2

        laser = LineLaserTweezer(
            p1, p2, qubit_array=array, catch_radius=0.2, opacity=0, infinite=True
        )
        self.wait(0.5)
        self.add(laser)
        laser.turn_on(scene=self, animate=True, run_time=0.5)
        self.wait(0.3)

        shift_vec = RIGHT * array.qubit_spacing
        laser.move_by(
            scene=self, delta=shift_vec, run_time=1.2, animate=True  # show the motion
        )
