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
        max_difficulty: int = 256,
        depth_slope: int = 2,
        max_depth: int = 128,
        obs_bins: int = 7,
        layout_exponent: float = 1.0,
        metrics_weights: dict[str, float] | None = None,
        track_solution: bool = True,
    ):
        super().__init__(
            num_qubits=num_qubits,
            coupling_map=coupling_map,
            num_active_swaps=num_active_swaps,
            horizon=horizon,
            difficulty=difficulty,
            max_difficulty=max_difficulty,
            depth_slope=depth_slope,
            max_depth=max_depth,
            obs_bins=obs_bins,
            layout_exponent=layout_exponent,
            metrics_weights=metrics_weights,
            track_solution=track_solution,
        )

    @classmethod
    def from_json(cls, env_config: dict):
        """Create a RoutingGym from a config dict (supports legacy field names)."""
        cm_data = env_config["coupling_map"]
        if isinstance(cm_data, dict) and cm_data.get("__tuple_list__"):
            coupling_map = [tuple(e) for e in cm_data["list"]]
        else:
            coupling_map = [tuple(e) for e in cm_data]

        return cls(
            num_qubits=env_config["num_qubits"],
            coupling_map=coupling_map,
            num_active_swaps=env_config.get("num_active_swaps", 16),
            horizon=env_config.get("horizon", 8),
            difficulty=env_config.get("difficulty", 1),
            max_difficulty=env_config.get("max_difficulty", 256),
            depth_slope=env_config.get("depth_slope", env_config.get("len_slope", 2)),
            max_depth=env_config.get("max_depth", env_config.get("max_len", 128)),
            obs_bins=env_config.get("obs_bins", 7),
            layout_exponent=env_config.get("layout_exponent", 1.0),
            metrics_weights=env_config.get("metrics_weights"),
            track_solution=True,
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
    ) -> Tuple[QuantumCircuit, List[int], List[int]] | None:
        """Reconstruct a routed circuit from the solution (coupling-map edge indices).

        The solution records which coupling-map edge was swapped at each step.
        This method replays those SWAPs, placing original gates whenever their
        qubits become adjacent, and returns the routed circuit plus layouts.

        Returns (routed_circuit, init_layout, final_layout) or None on failure.
        """
        coupling_map = self.config["coupling_map"]
        num_qubits = self.config["num_qubits"]

        from qiskit.transpiler import CouplingMap as CM

        dists = CM(coupling_map).distance_matrix.astype(int)

        # Build gate list with virtual-qubit indices
        gates: List[Tuple[List[int], object]] = []
        for inst in input_circuit.data:
            qubits = [input_circuit.find_bit(q).index for q in inst.qubits]
            gates.append((qubits, inst))

        # Per-qubit chains: ordered gate indices touching each virtual qubit
        qubit_chains: List[List[int]] = [[] for _ in range(num_qubits)]
        for gate_idx, (qubits, _) in enumerate(gates):
            for q in qubits:
                qubit_chains[q].append(gate_idx)

        # Predecessor count for each gate (dependency = same qubit, earlier gate)
        remaining = [0] * len(gates)
        for chain in qubit_chains:
            for i in range(1, len(chain)):
                remaining[chain[i]] += 1

        front = {i for i in range(len(gates)) if remaining[i] == 0}
        placed = [False] * len(gates)

        # Layout tracking (identity start)
        locations = list(range(num_qubits))   # virtual -> physical
        qubits_map = list(range(num_qubits))  # physical -> virtual

        out = QuantumCircuit(num_qubits, input_circuit.num_clbits)

        def _activate_successors(gate_idx: int):
            qs, _ = gates[gate_idx]
            for q in qs:
                chain = qubit_chains[q]
                pos = chain.index(gate_idx)
                if pos + 1 < len(chain):
                    succ = chain[pos + 1]
                    remaining[succ] -= 1
                    if remaining[succ] == 0:
                        front.add(succ)

        def _place(gate_idx: int):
            qs, inst = gates[gate_idx]
            phys_qubits = [out.qubits[locations[q]] for q in qs]
            clbits = [out.clbits[input_circuit.find_bit(c).index] for c in inst.clbits]
            out.append(inst.operation, phys_qubits, clbits)
            placed[gate_idx] = True
            _activate_successors(gate_idx)

        def _execute_ready():
            changed = True
            while changed:
                changed = False
                to_place = []
                for gidx in list(front):
                    qs, _ = gates[gidx]
                    if len(qs) <= 1:
                        to_place.append(gidx)
                    elif len(qs) == 2:
                        if dists[locations[qs[0]]][locations[qs[1]]] <= 1:
                            to_place.append(gidx)
                for gidx in to_place:
                    front.discard(gidx)
                    _place(gidx)
                    changed = True

        # Execute initially adjacent gates
        _execute_ready()

        # Process SWAPs from the solution
        for edge_idx in actions:
            l1, l2 = coupling_map[edge_idx]
            out.swap(l1, l2)

            vq1, vq2 = qubits_map[l1], qubits_map[l2]
            locations[vq1], locations[vq2] = l2, l1
            qubits_map[l1], qubits_map[l2] = vq2, vq1

            _execute_ready()

        init_layout = list(range(num_qubits))
        final_layout = list(locations)
        return out, init_layout, final_layout
