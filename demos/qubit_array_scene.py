from manimlib import *

from quantum_animation_toolbox.quera_colors import *
from quantum_animation_toolbox.quera_qubit_lib import *


class QubitArrayScene(Scene):
    def construct(self):
        # 4×4 grid with a “column-alternate” fill
        array = QubitArray(
            layout="grid",
            rows=4,
            cols=4,
            qubit_spacing=1.0,
            use_vacancies=True,
            fill_pattern="checkerboard",  # whole columns 0 & 2 filled
        )
        glow_group = Group(*(q.glow for q, _ in array.qubits))
        self.add(glow_group, array)  # glows first → higher z-index
        self.wait(0.5)

        # --- build shifts: odd ROW → right, even ROW → left -------------
        shifts = []
        qubits_per_row = 2
        for idx, (q, pos) in enumerate(array.qubits):
            row = idx // qubits_per_row  # row number 0-based
            dx = -1 if row % 2 else +1  # odd row → +1, even row → –1
            shifts.append((idx, dx, 0))

        # animate the slide
        array.move_qubits(self, shifts, run_time=1.2, animate=True)
