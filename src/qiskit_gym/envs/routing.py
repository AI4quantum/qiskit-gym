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

from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import SabreLayout

from qiskit_gym import qiskit_gym_rs


def _qubit_index(qubit):
    if hasattr(qubit, "index"):
        return qubit.index
    return qubit._index


def _compute_dists(coupling_map: CouplingMap) -> List[List[int]]:
    dm = coupling_map.distance_matrix
    dists = dm() if callable(dm) else dm
    if hasattr(dists, "tolist"):
        dists = dists.tolist()
    return np.asarray(dists, dtype=int).tolist()


def _build_dist_pairs(dists: List[List[int]]) -> tuple[dict[int, list[tuple[int, int]]], list[int]]:
    dist_pairs = defaultdict(list)
    num_qubits = len(dists)
    for q1 in range(num_qubits):
        for q2 in range(q1 + 1, num_qubits):
            dist_pairs[dists[q1][q2]].append((q1, q2))
    all_dists = sorted(dist_pairs.keys())
    return dist_pairs, all_dists


def _generate_pairs(difficulty: int, dist_pairs, all_dists) -> list[tuple[int, int]]:
    out = []
    remaining = difficulty
    while remaining > 0:
        valid = [d for d in all_dists if 0 < d <= remaining]
        if not valid:
            break
        next_d = valid[np.random.randint(len(valid))]
        q1, q2 = dist_pairs[next_d][np.random.randint(len(dist_pairs[next_d]))]
        out.append((q1, q2))
        remaining -= next_d
    return out


class RoutingGym(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        num_qubits: int,
        coupling_map: CouplingMap | List[Tuple[int, int]],
        dists: List[List[int]] | None = None,
        difficulty: int = 1,
        len_slope: int = 2,
        max_len: int = 128,
        min_len: int = 1,
        depth_slope: int | None = None,
        max_depth: int | None = None,
        num_active_swaps: int = 9,
        horizon: int = 8,
        metrics_weights: dict[str, float] | None = None,
        add_inverts: bool = True,
        layout_type: str = "trivial",
        routing_ops_type: str = "cz",
        autocancel_ops: bool = False,
        auto_sabre_layout: bool = True,
    ):
        if depth_slope is not None:
            len_slope = depth_slope
        if max_depth is not None:
            max_len = max_depth

        self.config = {
            "num_qubits": num_qubits,
            "coupling_map": coupling_map,
            "difficulty": difficulty,
            "len_slope": len_slope,
            "max_len": max_len,
            "min_len": min_len,
            "depth_slope": depth_slope,
            "max_depth": max_depth,
            "num_active_swaps": num_active_swaps,
            "horizon": horizon,
            "metrics_weights": metrics_weights,
            "add_inverts": add_inverts,
            "layout_type": layout_type,
            "routing_ops_type": routing_ops_type,
            "autocancel_ops": autocancel_ops,
            "auto_sabre_layout": auto_sabre_layout,
        }
        self._auto_sabre_layout = auto_sabre_layout
        self._sabre_layout_pass = None

        if isinstance(coupling_map, CouplingMap):
            cmap = coupling_map
            edges = _normalize_edges(cmap.get_edges())
        else:
            edges = _normalize_edges(coupling_map)
            cmap = CouplingMap(edges)
        self.config["coupling_map"] = edges

        if dists is None:
            dists = _compute_dists(cmap)
        else:
            dists = np.asarray(dists, dtype=int).tolist()

        self._dist_pairs, self._all_dists = _build_dist_pairs(dists)

        self._raw_env = qiskit_gym_rs.RoutingEnv(
            num_qubits,
            edges,
            dists,
            difficulty,
            len_slope,
            max_len,
            min_len,
            num_active_swaps,
            horizon,
            metrics_weights,
            add_inverts,
            layout_type,
            routing_ops_type,
            autocancel_ops,
        )

        # Keep Python/Gym observation shape aligned with legacy dense routing obs,
        # regardless of the sparse-encoding shape used internally by TwisteRL.
        self._obs_shape = (num_active_swaps, horizon + 1)
        self.observation_space = spaces.Box(
            low=-2,
            high=2,
            shape=self._obs_shape,
            dtype=float,
        )
        self.action_space = spaces.MultiBinary(num_active_swaps)

    @classmethod
    def from_coupling_map(
        cls,
        coupling_map: CouplingMap | List[Tuple[int, int]],
        num_qubits: int | None = None,
        **kwargs,
    ):
        if isinstance(coupling_map, CouplingMap):
            edges = _normalize_edges(coupling_map.get_edges())
            if num_qubits is None:
                num_qubits = coupling_map.size()
        else:
            edges = _normalize_edges(coupling_map)
            if num_qubits is None:
                num_qubits = max(max(edge) for edge in edges) + 1

        return cls(num_qubits=num_qubits, coupling_map=edges, **kwargs)

    @classmethod
    def from_json(cls, env_config):
        normalized = dict(env_config)
        # Keep backward compatibility with older configs that serialized dists,
        # but always compute distances from the coupling map at init.
        normalized.pop("dists", None)
        return cls(**normalized)

    def _get_sabre_layout_pass(self) -> SabreLayout:
        if self._sabre_layout_pass is None:
            cmap = CouplingMap(self.config["coupling_map"])
            cmap.make_symmetric()
            self._sabre_layout_pass = SabreLayout(
                cmap, max_iterations=6, swap_trials=2
            )
        return self._sabre_layout_pass

    def _maybe_set_sabre_layout(self, qc_target: QuantumCircuit) -> None:
        if not self._auto_sabre_layout or self.config["layout_type"] != "sabre":
            return
        sl = self._get_sabre_layout_pass()
        layout = list(sl(qc_target).layout.final_index_layout())
        self._raw_env.set_sabre_layout(layout)
        self._raw_env.set_fixed_layout(layout)

    def _qc_from_pairs(self, pairs: Iterable[Tuple[int, int]]) -> QuantumCircuit:
        qc = QuantumCircuit(self.config["num_qubits"])
        for q1, q2 in pairs:
            qc.append(random_unitary(4), (q1, q2))
        return qc

    def _encode_action(self, action: Iterable[int]) -> int:
        if isinstance(action, (int, np.integer)):
            return int(action)
        bits = np.asarray(action, dtype=np.int64).ravel()
        if bits.size == 1 and bits[0] not in (0, 1):
            # TwisteRL passes routing actions as pre-encoded integer bitmasks.
            return int(bits[0])
        mask = 0
        for i, val in enumerate(bits):
            if val:
                mask |= 1 << i
        return int(mask)

    def _dense_obs(self):
        obs = np.asarray(self._raw_env.observe_float(), dtype=float)
        return obs.reshape(self._obs_shape)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._raw_env.reset()
        return self._dense_obs(), {}

    def step(self, action):
        assert not bool(self._raw_env.is_final()), (
            "Action provided when env is in final state."
        )
        mask = self._encode_action(action)
        self._raw_env.step(int(mask))
        obs = self._dense_obs()
        reward = float(self._raw_env.reward())
        terminated = bool(self._raw_env.is_final())
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def set_target(self, target):
        if isinstance(target, list):
            if _is_qubit_pair_list(target):
                pairs = [tuple(pair) for pair in target]
            else:
                pairs = _generate_pairs(
                    self.config["difficulty"], self._dist_pairs, self._all_dists
                )
            ops = [(self.config["routing_ops_type"], pair) for pair in pairs]
            self._raw_env.set_target(ops)
            self._maybe_set_sabre_layout(self._qc_from_pairs(pairs))
            return

        if isinstance(target, QuantumCircuit):
            ops = []
            for inst, qargs, _ in target.data:
                if len(qargs) != 2:
                    continue
                name = inst.name.lower()
                if name not in ("cx", "cz", "swap"):
                    name = self.config["routing_ops_type"]
                ops.append((name, (_qubit_index(qargs[0]), _qubit_index(qargs[1]))))
            self._raw_env.set_target(ops)
            self._maybe_set_sabre_layout(target)
            return

        raise TypeError("target must be a list or QuantumCircuit")

    def clear_target(self):
        self._raw_env.clear_target()

    def set_fixed_layout(self, layout: List[int]):
        self._raw_env.set_fixed_layout(layout)

    def set_sabre_layout(self, layout: List[int]):
        self._raw_env.set_sabre_layout(layout)
        self._raw_env.set_fixed_layout(layout)

    def render(self, mode="human"):
        if hasattr(self._raw_env, "render"):
            return self._raw_env.render(mode=mode)
        return self._dense_obs()

    def close(self):
        if hasattr(self._raw_env, "close"):
            self._raw_env.close()

    def __getattr__(self, name):
        return getattr(self._raw_env, name)

    def __setattr__(self, name, value):
        if name in ("difficulty",) and "_raw_env" in self.__dict__:
            setattr(self._raw_env, name, value)
        else:
            super().__setattr__(name, value)

    def to_json(self):
        return self.config


def _is_qubit_pair_list(target: Sequence) -> bool:
    if not isinstance(target, Sequence):
        return False
    if len(target) == 0:
        return False
    for pair in target:
        if not isinstance(pair, Sequence) or len(pair) != 2:
            return False
        if not all(isinstance(q, (int, np.integer)) for q in pair):
            return False
    return True


def _normalize_edges(edges: Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    out = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"Invalid coupling-map edge {edge}; each edge must have 2 entries")
        out.append((int(edge[0]), int(edge[1])))
    return out
