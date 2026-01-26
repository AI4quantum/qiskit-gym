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

from qiskit_gym import qiskit_gym_rs

from .adapters import gym_adapter
from qiskit.transpiler import CouplingMap
from typing import List, Tuple, Iterable, ClassVar

from abc import ABC, abstractmethod

import numpy as np
from qiskit import QuantumCircuit


ONE_Q_GATES = ["H", "S", "Sdg", "SX", "SXdg"]
TWO_Q_GATES = ["CX", "CZ", "SWAP"]


# ------------- Base Synth Env class -------------


class BaseSynthesisEnv(ABC):
    cls_name: ClassVar[str]
    allowed_gates: ClassVar[List[str]]

    @classmethod
    def from_coupling_map(
        cls,
        coupling_map: CouplingMap | List[Tuple[int, int]],
        basis_gates: Tuple[str] = None,
        difficulty: int = 1,
        depth_slope: int = 2,
        max_depth: int = 128,
        metrics_weights: dict[str, float] | None = None,
        add_inverts: bool = True,
        add_perms: bool = True,
    ):
        if basis_gates is None:
            basis_gates = tuple(cls.allowed_gates)
        assert all(g in cls.allowed_gates for g in basis_gates), (
            f"Some provided gates are not allowed (allowed: {cls.allowed_gates})."
        )

        if isinstance(coupling_map, CouplingMap):
            coupling_map = list(coupling_map.get_edges())
        coupling_map = sorted(coupling_map)

        num_qubits = max(max(qubits) for qubits in coupling_map) + 1

        gateset = []
        for gate_name in basis_gates:
            if gate_name in ONE_Q_GATES:
                for q in range(num_qubits):
                    gateset.append((gate_name, (q,)))
            else:
                assert gate_name in TWO_Q_GATES, f"Gate {gate_name} not supported!"
                for q1, q2 in coupling_map:
                    gateset.append((gate_name, (q1, q2)))

        config = {
            "num_qubits": num_qubits,
            "difficulty": difficulty,
            "gateset": gateset,
            "depth_slope": depth_slope,
            "max_depth": max_depth,
            "metrics_weights": metrics_weights,
            "add_inverts": add_inverts,
            "add_perms": add_perms,
        }
        # Filter config to only include parameters accepted by the class __init__
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        filtered_config = {k: v for k, v in config.items() if k in valid_params}
        return cls(**filtered_config)

    @classmethod
    def from_json(cls, env_config):
        # Filter config to only include parameters accepted by the class __init__
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        filtered_config = {k: v for k, v in env_config.items() if k in valid_params}
        return cls(**filtered_config)

    @classmethod
    @abstractmethod
    def get_state(cls, input):
        pass

    def post_process_synthesis(self, synth_circuit: QuantumCircuit, input_state):
        return synth_circuit

# ---------------------------------------
# ------------- Env classes -------------
# ---------------------------------------

# ------------- Clifford -------------
from qiskit.quantum_info import Clifford

CliffordEnv = gym_adapter(qiskit_gym_rs.CliffordEnv)

def _solve_phases(clifford_cpy):
    num_qubits = clifford_cpy.num_qubits
    out = QuantumCircuit(num_qubits)

    # Add the phases (Pauli gates) to the Clifford circuit
    for qubit in range(num_qubits):
        stab = clifford_cpy.stab_phase[qubit]
        destab = clifford_cpy.destab_phase[qubit]
        if destab and stab:
            out.y(qubit)
        elif not destab and stab:
            out.x(qubit)
        elif destab and not stab:
            out.z(qubit)

    return out

class CliffordGym(CliffordEnv, BaseSynthesisEnv):
    cls_name = "CliffordEnv"
    allowed_gates = ONE_Q_GATES + TWO_Q_GATES

    def __init__(
        self,
        num_qubits: int,
        gateset: List[Tuple[str, List[int]]],
        difficulty: int = 1,
        depth_slope: int = 2,
        max_depth: int = 128,
        metrics_weights: dict[str, float] | None = None,
        add_inverts: bool = True,
        add_perms: bool = True,
        track_solution: bool = True,
    ):
        super().__init__(**{
            "num_qubits": num_qubits,
            "difficulty": difficulty,
            "gateset": gateset,
            "depth_slope": depth_slope,
            "max_depth": max_depth,
            "metrics_weights": metrics_weights,
            "add_inverts": add_inverts,
            "add_perms": add_perms,
            "track_solution": track_solution,
        })

    def get_state(self, input: QuantumCircuit | Clifford):
        if isinstance(input, QuantumCircuit):
            input = Clifford(input)
        return input.adjoint().tableau[:, :-1].T.flatten().astype(int).tolist()
    
    def post_process_synthesis(self, synth_circuit: QuantumCircuit, input):
        synth_circuit = synth_circuit.inverse()
        if isinstance(input, QuantumCircuit):
            input = Clifford(input)
        dcliff = Clifford(synth_circuit).compose(input)
        out = _solve_phases(dcliff).compose(synth_circuit).inverse()
        return out


# ------------- Linear Function -------------
from qiskit.circuit.library.generalized_gates import LinearFunction

LinearFunctionEnv = gym_adapter(qiskit_gym_rs.LinearFunctionEnv)


class LinearFunctionGym(LinearFunctionEnv, BaseSynthesisEnv):
    cls_name = "LinearFunctionEnv"
    allowed_gates = ["CX", "SWAP"]

    def __init__(
        self,
        num_qubits: int,
        gateset: List[Tuple[str, List[int]]],
        difficulty: int = 1,
        depth_slope: int = 2,
        max_depth: int = 128,
        metrics_weights: dict[str, float] | None = None,
        add_inverts: bool = True,
        add_perms: bool = True,
        track_solution: bool = True,
    ):
        super().__init__(**{
            "num_qubits": num_qubits,
            "difficulty": difficulty,
            "gateset": gateset,
            "depth_slope": depth_slope,
            "max_depth": max_depth,
            "metrics_weights": metrics_weights,
            "add_inverts": add_inverts,
            "add_perms": add_perms,
            "track_solution": track_solution,
        })
    
    def get_state(self, input: QuantumCircuit | LinearFunction):
        # This returns the inverse permutation to get the right 
        # synthesized circuit at output, instead of its inverse.
        input = LinearFunction(Clifford(input).adjoint())
        return np.array(input.linear).flatten().astype(int).tolist()
        

# ------------- Permutation -------------
from qiskit.circuit.library.generalized_gates import PermutationGate

PermutationEnv = gym_adapter(qiskit_gym_rs.PermutationEnv)


class PermutationGym(PermutationEnv, BaseSynthesisEnv):
    cls_name = "PermutationEnv"
    allowed_gates = ["SWAP"]

    def __init__(
        self,
        num_qubits: int,
        gateset: List[Tuple[str, List[int]]],
        difficulty: int = 1,
        depth_slope: int = 2,
        max_depth: int = 128,
        metrics_weights: dict[str, float] | None = None,
        add_inverts: bool = True,
        add_perms: bool = True,
        track_solution: bool = True,
    ):
        super().__init__(**{
            "num_qubits": num_qubits,
            "difficulty": difficulty,
            "gateset": gateset,
            "depth_slope": depth_slope,
            "max_depth": max_depth,
            "metrics_weights": metrics_weights,
            "add_inverts": add_inverts,
            "add_perms": add_perms,
            "track_solution": track_solution,
        })

    def get_state(self, input: QuantumCircuit | PermutationGate | Iterable[int]):
        if isinstance(input, QuantumCircuit):
            input = LinearFunction(input).permutation_pattern()
        elif isinstance(input, PermutationGate):
            input = input.pattern

        # This returns the inverse permutation to get the right
        # synthesized circuit at output, instead of its inverse.
        return np.argsort(np.array(input)).astype(int).tolist()


# ------------- Pauli Network -------------

PauliNetworkEnv = gym_adapter(qiskit_gym_rs.PauliNetworkEnv)

def _parse_pauli_circuit(circuit: QuantumCircuit) -> Tuple[Clifford, List[str]]:
    """
    Parse a QuantumCircuit into its Clifford and rotation components.
    The circuit should contain Clifford gates (h, s, sdg, sx, sxdg, cx, cz, swap)
    and Pauli rotation gates (rx, ry, rz).
    Returns:
        Tuple of (Clifford, list of rotation strings)
    """
    num_qubits = circuit.num_qubits
    clifford_gates = {'h', 's', 'sdg', 'sx', 'sxdg', 'cx', 'cz', 'swap', 'x', 'y', 'z'}
    rotation_gates = {'rx', 'ry', 'rz'}
    # Build a circuit with just Clifford gates to get the Clifford
    clifford_circuit = QuantumCircuit(num_qubits)
    rotations = []
    for instruction in circuit.data:
        gate_name = instruction.operation.name.lower()
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        if gate_name in clifford_gates:
            # Add Clifford gate to the clifford circuit
            if gate_name == 'h':
                clifford_circuit.h(qubits[0])
            elif gate_name == 's':
                clifford_circuit.s(qubits[0])
            elif gate_name == 'sdg':
                clifford_circuit.sdg(qubits[0])
            elif gate_name == 'sx':
                clifford_circuit.sx(qubits[0])
            elif gate_name == 'sxdg':
                clifford_circuit.sxdg(qubits[0])
            elif gate_name == 'cx':
                clifford_circuit.cx(qubits[0], qubits[1])
            elif gate_name == 'cz':
                clifford_circuit.cz(qubits[0], qubits[1])
            elif gate_name == 'swap':
                clifford_circuit.swap(qubits[0], qubits[1])
            elif gate_name == 'x':
                clifford_circuit.x(qubits[0])
            elif gate_name == 'y':
                clifford_circuit.y(qubits[0])
            elif gate_name == 'z':
                clifford_circuit.z(qubits[0])
        elif gate_name in rotation_gates:
            # Extract rotation as Pauli string
            qubit = qubits[0]
            pauli_chars = ['I'] * num_qubits
            if gate_name == 'rx':
                pauli_chars[num_qubits - 1 - qubit] = 'X'
            elif gate_name == 'ry':
                pauli_chars[num_qubits - 1 - qubit] = 'Y'
            elif gate_name == 'rz':
                pauli_chars[num_qubits - 1 - qubit] = 'Z'
            rotations.append(''.join(pauli_chars))
    clifford = Clifford(clifford_circuit) if clifford_circuit.data else Clifford(np.eye(2 * num_qubits, dtype=int))
    return clifford, rotations

class PauliGym(PauliNetworkEnv, BaseSynthesisEnv):
    cls_name = "PauliNetworkEnv"
    allowed_gates = ONE_Q_GATES + TWO_Q_GATES

    def __init__(
        self,
        num_qubits: int,
        gateset: List[Tuple[str, List[int]]],
        difficulty: int = 1,
        depth_slope: int = 2,
        max_depth: int = 128,
        max_rotations: int = 5,
        pauli_diff_scale: int = 16,
        num_qubits_decay: float = 0.5,
        final_pauli_layers: int | None = None,
        metrics_weights: dict[str, float] | None = None,
        add_perms: bool = True,
        pauli_layer_reward: float = 0.01,
        track_solution: bool = True,
    ):
        super().__init__(**{
            "num_qubits": num_qubits,
            "difficulty": difficulty,
            "gateset": gateset,
            "depth_slope": depth_slope,
            "max_depth": max_depth,
            "max_rotations": max_rotations,
            "pauli_diff_scale": pauli_diff_scale,
            "num_qubits_decay": num_qubits_decay,
            "final_pauli_layers": final_pauli_layers,
            "metrics_weights": metrics_weights,
            "add_perms": add_perms,
            "pauli_layer_reward": pauli_layer_reward,
            "track_solution": track_solution,
        })

    def get_state(self, input, rotations: List[str] = None):
        """
        Encode Clifford tableau and rotation labels into state.

        Args:
            input: Either:
                - A Qiskit Clifford object (with rotations as second arg)
                - A tuple (Clifford, rotations)
                - A QuantumCircuit to parse
            rotations: List of Pauli rotation labels (e.g., ["IX", "ZY"])
                      when input is a Clifford

        Returns:
            State as list of integers for set_state()
        """
        if isinstance(input, tuple):
            clifford, rotations = input
        elif isinstance(input, QuantumCircuit):
            clifford, rotations = _parse_pauli_circuit(input)
        elif isinstance(input, Clifford):
            clifford = input
            if rotations is None:
                rotations = []
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

        # State format: [rotation_count, tableau..., len1, chars1..., len2, chars2..., ...]
        tableau = clifford.adjoint().tableau[:, :-1].T.flatten().astype(int).tolist()

        state = [len(rotations)]
        state.extend(tableau)

        for rot in rotations:
            state.append(len(rot))
            state.extend([ord(c) for c in rot])

        return state
    
    # def post_process_synthesis(self, synth_circuit: QuantumCircuit, input):
    #     synth_circuit = synth_circuit.inverse()
    #     if isinstance(input, QuantumCircuit):
    #         input = Clifford(input)
    #     dcliff = Clifford(synth_circuit).compose(input)
    #     out = _solve_phases(dcliff).compose(synth_circuit).inverse()
    #     return out


# ---------------------------------------

SYNTH_ENVS = {
    "CliffordEnv": CliffordGym,
    "CliffordGym": CliffordGym,
    "LinearFunctionEnv": LinearFunctionGym,
    "LinearFunctionGym": LinearFunctionGym,
    "PermutationEnv": PermutationGym,
    "PermutationGym": PermutationGym,
    "PauliNetworkEnv": PauliGym,
    "PauliGym": PauliGym,
}
