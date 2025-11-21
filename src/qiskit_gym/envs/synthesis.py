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
        }
        return cls(**config)

    @classmethod
    def from_json(cls, env_config):
        return cls(**env_config)

    @classmethod
    @abstractmethod
    def get_state(cls, input):
        pass


# ---------------------------------------
# ------------- Env classes -------------
# ---------------------------------------

# ------------- Clifford -------------
from qiskit.quantum_info import Clifford

CliffordEnv = gym_adapter(qiskit_gym_rs.CliffordEnv)


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
    ):
        super().__init__(**{
            "num_qubits": num_qubits,
            "difficulty": difficulty,
            "gateset": gateset,
            "depth_slope": depth_slope,
            "max_depth": max_depth,
            "metrics_weights": metrics_weights,
        })

    def get_state(self, input: QuantumCircuit | Clifford):
        if isinstance(input, QuantumCircuit):
            input = Clifford(input)
        return input.adjoint().tableau[:, :-1].T.flatten().astype(int).tolist()


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
    ):
        super().__init__(**{
            "num_qubits": num_qubits,
            "difficulty": difficulty,
            "gateset": gateset,
            "depth_slope": depth_slope,
            "max_depth": max_depth,
            "metrics_weights": metrics_weights,
        })
    
    def get_state(self, input: QuantumCircuit | LinearFunction):
        if isinstance(input, QuantumCircuit):
            input = LinearFunction(input.inverse())
        elif isinstance(input, LinearFunction):
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
    ):
        super().__init__(**{
            "num_qubits": num_qubits,
            "difficulty": difficulty,
            "gateset": gateset,
            "depth_slope": depth_slope,
            "max_depth": max_depth,
            "metrics_weights": metrics_weights,
        })

    def get_state(self, input: QuantumCircuit | PermutationGate | Iterable[int]):
        if isinstance(input, QuantumCircuit):
            input = LinearFunction(input).permutation_pattern()
        elif isinstance(input, PermutationGate):
            input = input.pattern

        return np.argsort(np.array(input)).astype(int).tolist()


# ---------------------------------------

SYNTH_ENVS = {
    "CliffordEnv": CliffordGym,
    "LinearFunctionEnv": LinearFunctionGym,
    "PermutationEnv": PermutationGym,
}
