from manimlib import *

from quantum_animation_toolbox.quera_colors import *
from quantum_animation_toolbox.quera_circuit_lib import *


class PulseScene(Scene):
	def construct(self):
		bg = BackgroundColor(BLACK)
		qc = QuantumCircuit(
			num_wires    = 5,
			wire_length  = 6,
			left_labels  = [r"\ket{0}", r"\ket{1}", r"\ket{+}", r"\ket{-}", r"\ket{\psi}"],
			right_labels = True,
			wire_spacing = 0.6,
			boxed        = True,
			gate_size    = 0.5,
			gate_spacing = 0.1,
		)

		qc.hadamard_transform(slot=1)
		qc.hadamard_transform(slot=2)

		qc.edit_wire(index=4, slot_range=((0,4),(7,10)), label=[(r"q_4",r"q_4"),(r"\ket{\psi}",r"\ket{\psi}")])
		qc.add_wire(position="above", slot_range=(3,5), label=r"\ket{\varphi}")
		qc.add_wire(position="below", slot_range=((0,4),(7,10)), label=[(r"q_5",r"q_5"),(r"\ket{\psi}",r"\ket{\psi}")])
		qc.add_gate("CNOT", wire_idx=1, target_idx=0, slot=3)
		qc.add_gate("CNOT", wire_idx=2, target_idx=1, slot=4)
		qc.add_gate("CNOT", wire_idx=0, target_idx=5, slot=4)
		qc.add_gate("SWAP", wire_idx=0, target_idx=1, slot=5)
		qc.add_gate("CZ",   wire_idx=3, target_idx=2, slot=5)
		qc.add_gate("CNOT", wire_idx=0, target_idx=1, slot=7, shift="l")
		qc.add_gate("SWAP", wire_idx=2, target_idx=1, slot=7)
		qc.add_gate("CNOT", wire_idx=2, target_idx=0, slot=8)
		qc.add_gate("CZ",   wire_idx=0, target_idx=1, slot=6)
		qc.add_gate("CZ",   wire_idx=1, target_idx=2, slot=6, shift="right")
		qc.add_gate("CZ",   wire_idx=3, target_idx=2, slot=6)
		qc.add_gate("SWAP", wire_idx=0, target_idx=1, slot=9)
		qc.add_gate("SWAP", wire_idx=3, target_idx=4, slot=9)

		qc.end_measurements(axis="Z")
		qc.add_gate("M", wire_idx=6, slot=1, axis="X")

		self.add(bg, qc)

		# ----- Create glows for halfway pulse frame -----
		pitch = qc.gate_size + qc.gate_spacing
		half_pitch = pitch / 2
		ext_start = -qc.wire_length / 2 - 2 * half_pitch
		ext_end = qc.wire_length / 2 + 2 * half_pitch
		mid_x = (ext_start + ext_end) / 2

		# Glow dots positioned halfway
		wire_glows = Group(
			*[
				GlowDot(color=QUERA_PURPLE, radius=0.25).move_to([mid_x, wire.get_y(), 0])
				for wire in qc.wires
			]
		).set_z_index(-1)

		# Opacity masking (clip to visible intervals)
		for glow, wire in zip(wire_glows, qc.wires):
			def clip(mob, w=wire):
				x = mob.get_x()
				for a, b in w.intervals:
					x_a = w.point_from_slot(a)[0]
					x_b = w.point_from_slot(b)[0]
					if x_a <= x <= x_b:
						mob.set_opacity(1)
						return
				mob.set_opacity(0)
			glow.add_updater(clip)

		# Gate glows (Gaussian stroke profile based on proximity to mid_x)
		gate_glows = VGroup(
			*[
				g.copy().set_stroke(QUERA_PURPLE, width=1).set_fill(opacity=0)
				for g in qc.gates
			]
		)
		for gg in gate_glows:
			def update_gate_glow(mob):
				gate_x = mob.get_center()[0]
				sigma = 1.5 * half_pitch
				width = 4 * np.exp(-0.5 * ((gate_x - mid_x) / sigma) ** 2)
				mob.set_stroke(width=width)
			gg.add_updater(update_gate_glow)

		self.add(wire_glows, gate_glows)
