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

from typing import List, Tuple

from qiskit_gym import qiskit_gym_rs
from qiskit_gym.envs.adapters import gym_adapter
from qiskit.transpiler import CouplingMap
from qiskit import QuantumCircuit

RoutingEnv = gym_adapter(qiskit_gym_rs.RoutingEnv)


class RoutingGym(RoutingEnv):
    """Gymnasium wrapper for the routing RL environment.

    Given an input circuit with 2-qubit gates and a hardware coupling map,
    the agent inserts SWAP gates (one per step) to make all gates executable
    on the hardware topology.
    """

    cls_name = "RoutingEnv"

    def __init__(
        self,
        num_qubits: int,
        coupling_map: List[Tuple[int, int]],
        num_active_swaps: int = 16,
        horizon: int = 8,
        difficulty: int = 1,
        depth_slope: int = 2,
        max_depth: int = 128,
        obs_bins: int = 7,
        metrics_weights: dict[str, float] | None = None,
        track_solution: bool = True,
    ):
        super().__init__(
            num_qubits=num_qubits,
            coupling_map=coupling_map,
            num_active_swaps=num_active_swaps,
            horizon=horizon,
            difficulty=difficulty,
            depth_slope=depth_slope,
            max_depth=max_depth,
            obs_bins=obs_bins,
            metrics_weights=metrics_weights,
            track_solution=track_solution,
        )

    @classmethod
    def from_coupling_map(
        cls,
        coupling_map: CouplingMap | List[Tuple[int, int]],
        **kwargs,
    ):
        """Create a RoutingGym from a Qiskit CouplingMap or edge list."""
        if isinstance(coupling_map, CouplingMap):
            coupling_map = list(coupling_map.get_edges())
        coupling_map = sorted(coupling_map)
        num_qubits = max(max(q) for q in coupling_map) + 1
        return cls(num_qubits=num_qubits, coupling_map=coupling_map, **kwargs)

    def get_state(self, input_circuit: QuantumCircuit) -> List[int]:
        """Encode a QuantumCircuit as a flat list of qubit pairs for set_state().

        Only 2-qubit gates are extracted; single-qubit gates are ignored
        since routing only cares about 2-qubit connectivity.
        """
        state = []
        for instruction in input_circuit.data:
            if len(instruction.qubits) == 2:
                q1 = input_circuit.find_bit(instruction.qubits[0]).index
                q2 = input_circuit.find_bit(instruction.qubits[1]).index
                state.extend([q1, q2])
        return state

    def build_circuit_from_solution(
        self, actions: List[int], input_circuit: QuantumCircuit
    ) -> QuantumCircuit | None:
        """Reconstruct a routed circuit from the solution action sequence.

        This is a placeholder — full reconstruction requires tracking the
        SWAP insertions and original gate placements, which the Rust env
        does internally via track_solution.
        """
        # TODO: implement full circuit reconstruction from solution
        return None
