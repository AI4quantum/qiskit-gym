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
from qiskit.exceptions import QiskitError


ONE_Q_GATES = ["H", "S", "Sdg", "SX", "SXdg"]
TWO_Q_GATES = ["CX", "CZ", "SWAP"]

# Marker for encoded rotations in solution (must match Rust ROTATION_MARKER)
ROTATION_MARKER = 0x80000000  # 2^31


def decode_pauli_solution(encoded_solution: List[int]) -> List[Tuple[str, int, int, int]]:
    """
    Decode a Pauli solution encoded as Vec<usize> into the full solution format.

    Encoding scheme:
    - Gate actions: just the action index (< ROTATION_MARKER)
    - Rotations: ROTATION_MARKER | axis(2 bits) | qubit(10 bits) | index(10 bits) | phase(1 bit)

    Returns:
        List of tuples: ("gate", action_index, 0, 0) or ("rx"/"ry"/"rz", qubit, index, phase_mult)
    """
    result = []
    axis_names = ["rx", "ry", "rz"]

    for val in encoded_solution:
        if val >= ROTATION_MARKER:
            # Decode rotation
            axis_code = (val >> 21) & 0x3
            qubit = (val >> 11) & 0x3FF
            index = (val >> 1) & 0x3FF
            phase_mult = 1 if (val & 1) else -1
            result.append((axis_names[axis_code], qubit, index, phase_mult))
        else:
            # Gate action
            result.append(("gate", val, 0, 0))

    return result


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

    @abstractmethod
    def get_state(self, input):
        pass

    def post_process_synthesis(self, synth_circuit: QuantumCircuit, _input_state):
        return synth_circuit

    def build_circuit_from_solution(self, actions: List[int], input) -> QuantumCircuit:
        """
        Build a circuit from the solution actions.
        Default implementation uses gate_list_to_circuit + post_process_synthesis.
        Subclasses can override for custom handling (e.g., decoding rotations).
        """
        from qiskit_gym.rl.synthesis import gate_list_to_circuit
        synth_circuit = gate_list_to_circuit(
            [self.config["gateset"][a] for a in actions],
            num_qubits=self.config["num_qubits"],
        )
        return self.post_process_synthesis(synth_circuit, input)


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

def _make_pauli_string(num_qubits: int, qubit: int, axis: str) -> str:
    """Create a Pauli string with a single non-identity element at the given qubit."""
    pauli_chars = ['I'] * num_qubits
    pauli_chars[num_qubits - 1 - qubit] = axis
    return ''.join(pauli_chars)


def _parse_pauli_circuit(circuit: QuantumCircuit) -> Tuple[Clifford, List[str], List]:
    """
    Parse a QuantumCircuit into its Clifford and rotation components.

    This follows the same logic as the old transpiler's PauliNetwork.from_circuit:
    - Start with identity Clifford
    - For rotations: create Pauli, evolve through current Clifford, store adjoint
    - For Clifford gates: compose onto current Clifford

    Returns:
        Tuple of (Clifford, list of evolved rotation labels, list of rotation parameters)
    """
    from qiskit.quantum_info import Pauli

    num_qubits = circuit.num_qubits
    rotation_gates = {'rx', 'ry', 'rz'}

    # Start with identity Clifford (matches old transpiler)
    clifford = Clifford(np.eye(2 * num_qubits, dtype=bool))
    rotations = []
    rotation_params = []

    for instruction in circuit.data:
        gate_name = instruction.operation.name.lower()
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]

        if gate_name in rotation_gates:
            # Create Pauli string for this rotation
            axis = gate_name[1].upper()  # 'rx' -> 'X', 'ry' -> 'Y', 'rz' -> 'Z'
            pauli_str = _make_pauli_string(num_qubits, qubits[0], axis)
            p = Pauli(pauli_str)

            # Evolve through current Clifford and store adjoint (matches old transpiler)
            p = p.evolve(clifford)
            rotations.append(p.adjoint().to_label())
            rotation_params.extend(instruction.operation.params)

        else:
            # Compose Clifford gate onto current Clifford
            try:
                clifford = clifford.compose(instruction.operation, qubits)
            except QiskitError:
                raise TypeError(
                    f"Gate {gate_name} on qubits {qubits} not supported."
                )

    # Return clifford (adjoint will be applied in get_state)
    return clifford, rotations, rotation_params


def _just_clifford(circuit: QuantumCircuit) -> QuantumCircuit:
    """Extract only Clifford gates from a circuit (removes rx, ry, rz)."""
    circuit_out = QuantumCircuit.copy_empty_like(circuit)
    for g in circuit:
        if g.operation.name not in {"rx", "ry", "rz"}:
            circuit_out.append(g.operation, g.qubits)
    return circuit_out


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
        self._rotation_params = []
        self._original_circuit = None

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
            # Tuple input: assume clifford is already in correct form (adjoint)
            clifford_for_state = clifford
            self._rotation_params = []
            self._original_circuit = None
        elif isinstance(input, QuantumCircuit):
            clifford, rotations, rotation_params = _parse_pauli_circuit(input)
            clifford_for_state = clifford.adjoint()
            self._rotation_params = rotation_params
            self._original_circuit = input
        elif isinstance(input, Clifford):
            clifford = input
            if rotations is None:
                rotations = []
            # Raw Clifford: need to take adjoint
            clifford_for_state = clifford.adjoint()
            self._rotation_params = []
            self._original_circuit = None
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

        # State format: [rotation_count, tableau..., len1, chars1..., len2, chars2..., ...]
        tableau = clifford_for_state.tableau[:, :-1].T.flatten().astype(int).tolist()

        state = [len(rotations)]
        state.extend(tableau)

        for rot in rotations:
            state.append(len(rot))
            state.extend([ord(c) for c in rot])

        return state

    def _reconstruct_circuit_from_solution(
        self,
        full_solution: List[Tuple[str, int, int, int]],
        input,
    ) -> QuantumCircuit:
        """
        Reconstruct circuit from synthesis solution with rotation parameters.

        Args:
            full_solution: List of (type, arg1, arg2, arg3) tuples from full_solution()
                - For gates: ("gate", action_index, 0, 0)
                - For rotations: ("rx"/"ry"/"rz", qubit, rotation_index, phase_mult)
            input: Original input (QuantumCircuit or tuple)

        Returns:
            Reconstructed QuantumCircuit with rotations and phase correction
        """
        num_qubits = self.config["num_qubits"]
        circuit = QuantumCircuit(num_qubits)

        for step_type, arg1, arg2, arg3 in full_solution:
            if step_type == "gate":
                gate_name, gate_args = self.config["gateset"][arg1]
                # CX qubit order: The gateset stores CX as (control, target), but the
                # Rust PauliNetwork.cnot() uses the opposite convention internally
                # (it does row[i] ^= row[j] where i,j come from CX(i,j)). To produce
                # the correct output circuit, we reverse the qubit order here.
                if gate_name.lower() == "cx":
                    circuit.cx(*gate_args[::-1])
                else:
                    getattr(circuit, gate_name.lower())(*gate_args)
            elif step_type in ("rx", "ry", "rz"):
                qubit, rotation_index, phase_mult = arg1, arg2, arg3
                if rotation_index < len(self._rotation_params):
                    angle = phase_mult * self._rotation_params[rotation_index]
                else:
                    raise Exception("Too few rotation parameters stored for synthesis!")
                getattr(circuit, step_type)(angle, qubit)

        # Phase correction - use input if it's a QuantumCircuit, otherwise stored circuit
        original_circuit = input if isinstance(input, QuantumCircuit) else self._original_circuit
        if original_circuit is not None:
            phase_correction = Clifford(
                _just_clifford(circuit.inverse().compose(original_circuit))
            ).to_circuit()
            circuit = circuit.compose(phase_correction)

        return circuit

    def build_circuit_from_solution(self, actions: List[int], input) -> QuantumCircuit:
        """
        Build circuit from encoded solution (gates + rotations).
        Decodes the solution and reconstructs the circuit.
        """
        full_solution = decode_pauli_solution(actions)
        return self._reconstruct_circuit_from_solution(full_solution, input)


# ---------------------------------------

SYNTH_ENVS = {
    "CliffordEnv": CliffordGym,
    "LinearFunctionEnv": LinearFunctionGym,
    "PermutationEnv": PermutationGym,
    "PauliNetworkEnv": PauliGym,
}
