from manimlib import *

from quantum_animation_toolbox.quera_colors import *
from quantum_animation_toolbox.quera_qubit_lib import *


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