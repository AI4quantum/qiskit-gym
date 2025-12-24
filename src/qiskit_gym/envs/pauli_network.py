# -*- coding: utf-8 -*-

# (C) Copyright 2025 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Pauli


def make_pauli_string(num_qubits: int, qubit: int, pauli_type: str) -> str:
    pattern = ["I"] * num_qubits
    pattern[qubit] = pauli_type
    return "".join(pattern[::-1])


def pauli_payload_from_circuit(circuit: QuantumCircuit) -> tuple[np.ndarray, list[str]]:
    """Return (tableau, rotation labels) describing the Pauli network target."""

    if circuit.num_qubits == 0:
        raise ValueError("Pauli synthesis requires at least one qubit")

    circuitb = QuantumCircuit(circuit.num_qubits).compose(circuit)
    rotations: list[Pauli] = []
    clifford = Clifford(np.eye(2 * circuitb.num_qubits, dtype=bool))

    for gate in circuitb:
        g_name = gate.operation.name
        g_qubits = [qubit._index for qubit in gate.qubits]

        if g_name in {"rx", "ry", "rz"}:
            pauli = Pauli(
                make_pauli_string(circuitb.num_qubits, g_qubits[0], g_name[1].upper())
            )
            pauli = pauli.evolve(clifford)
            rotations.append(pauli.adjoint())
            continue

        try:
            clifford = clifford.compose(gate.operation, g_qubits)
        except Exception as exc:  # pragma: no cover - defensive path
            raise TypeError(
                f"Gate {g_name} on qubits {g_qubits} not supported for Pauli synthesis"
            ) from exc

    tableau = clifford.adjoint().tableau[:, :-1].T.astype(int)
    rotation_labels = [pauli.to_label() for pauli in rotations]
    return tableau, rotation_labels
