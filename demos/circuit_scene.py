from manimlib import *

from quantum_animation_toolbox.quera_circuit_lib import *
from quantum_animation_toolbox.quera_colors import *


class CircuitScene(Scene):
    def construct(self):
        bg = BackgroundColor(BLACK)
        qc = QuantumCircuit(
            num_wires=5,
            wire_length=6,
            left_labels=[r"\ket{0}", r"\ket{1}", r"\ket{+}", r"\ket{-}", r"\ket{\psi}"],
            right_labels=True,
            wire_spacing=0.6,
            boxed=True,
            gate_size=0.5,
            gate_spacing=0.1,
            # gate_shape   = "round",
        )
        # Put a column of Hs in slot 1 (x ≃ -1.8)
        qc.hadamard_transform(slot=1)
        # hs = qc.hadamard_transform(slot=2)

        # Place other gates by slot index instead of alpha
        qc.edit_wire(
            index=4,
            slot_range=((0, 4), (7, 10)),
            label=[(r"q_4", r"q_4"), (r"\ket{\psi}", r"\ket{\psi}")],
        )
        qc.add_wire(
            position="above",
            slot_range=(3, 5),
            label=r"\ket{\varphi}",
            wire_type="classical",
        )
        qc.add_wire(
            position="below",
            slot_range=((0, 4), (7, 10)),
            label=[(r"q_5", r"q_5"), (r"\ket{\psi}", r"\ket{\psi}")],
        )
        # qc.add_wire(position="below")
        qc.add_gate("CNOT", wire_idx=1, control_idx=0, slot=3)
        qc.add_gate("CNOT", wire_idx=2, control_idx=1, slot=4)
        qc.add_gate("CNOT", wire_idx=0, control_idx=5, slot=4)
        # qc.add_gate("CZ",   wire_idx=0, target_idx=1, slot=2)
        # qc.add_gate("CZ",   wire_idx=3, target_idx=4, slot=2)
        qc.add_gate("SWAP", wire_idx=0, control_idx=1, slot=5)
        qc.add_gate("CZ", wire_idx=3, control_idx=2, slot=5)
        qc.add_gate("CNOT", wire_idx=0, control_idx=1, slot=7, shift="l")
        qc.add_gate("SWAP", wire_idx=2, control_idx=1, slot=7)
        qc.add_gate("CNOT", wire_idx=2, control_idx=0, slot=8)
        qc.add_gate("CZ", wire_idx=0, control_idx=1, slot=6)
        qc.add_gate("CZ", wire_idx=1, control_idx=2, slot=6, shift="right")
        qc.add_gate("CZ", wire_idx=3, control_idx=2, slot=6)
        qc.add_gate("SWAP", wire_idx=0, control_idx=1, slot=9)
        qc.add_gate("SWAP", wire_idx=3, control_idx=4, slot=9)
        # qc.add_gate("WB", wire_idx=6, slot=2, count=7, color=YELLOW, thickness=3)

        # measure at final slot
        qc.end_measurements(axis="Z")
        qc.add_gate("M", wire_idx=6, slot=1, axis="X")
        qc.recolor_gates("all", color=BLUE_C)
        qc.recolor_gates("H", color=GREEN_C)
        qc.recolor_gates("CZ", color=QUERA_PURPLE)
        qc.recolor_gates("CNOT", color=ORANGE)
        qc.recolor_gates("M", color=WHITE)
        qc.recolor_wires(color=QUERA_RED)
        qc.recolor_labels(color=QUERA_YELLOW)

        self.add(qc, bg)
