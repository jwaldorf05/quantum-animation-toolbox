from manimlib import *

from quantum_animation_toolbox.quera_colors import *
from quantum_animation_toolbox.quera_qubit_lib import *


class EntanglementsScene(Scene):
    def construct(self):
        array = QubitArray(
            layout="grid", rows=2, cols=4, use_vacancies=True, fill_pattern="all"
        )
        self.add(array)

        self.wait(0.5)

        dx = 0.5 * array.qubit_spacing

        moves = [
            (0, dx, 0),  # qubit-0  → right
            (4, dx, 0),  # qubit-4  → right
            (3, -dx, 0),  # qubit-3  ← left
            (7, -dx, 0),  # qubit-7  ← left
        ]

        # use DotLaserTweezer internally
        array.move_qubits(
            scene=self, shift_list=moves, run_time=0.5, animate=True, use_lasers=True
        )

        self.wait(0.3)

        pairs = array.get_pairs([(0, 1), (2, 3), (4, 5), (6, 7)])

        entanglements(
            self,
            pairs,
            color=QUERA_RED,
            width=0.4,
            height=0.4,
            run_time=0.5,
            animate="static",
        )
