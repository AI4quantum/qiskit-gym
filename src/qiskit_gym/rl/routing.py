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

import json

import torch
from torch.utils.tensorboard import SummaryWriter

from twisterl.utils import dynamic_import, load_checkpoint
from qiskit_gym.rl.configs import (
    AlphaZeroConfig,
    PPOConfig,
    BasicPolicyConfig,
    Conv1dPolicyConfig,
    POLICIES,
    ALGORITHMS,
)
from qiskit_gym.envs.routing import RoutingGym

from qiskit import QuantumCircuit


class RLRouting:
    def __init__(
        self,
        env: RoutingGym,
        rl_config: AlphaZeroConfig | PPOConfig,
        model_config: BasicPolicyConfig | Conv1dPolicyConfig,
        model_path: str = None,
    ):
        self.env = env

        self.env_config = env.to_json()
        self.rl_config = rl_config
        self.model_config = model_config

        self.algorithm_cls = dynamic_import(rl_config.algorithm_cls)
        self.model_cls = dynamic_import(model_config.policy_cls)

        self.algorithm = self.init_algorithm(model_path)

    @classmethod
    def from_config_json(cls, config_path, model_path=None):
        full_config = json.load(open(config_path))

        env_cfg = full_config["env"]
        # Use diff_max from the algorithm's curriculum as max_difficulty
        algo_cfg = full_config.get("algorithm", {})
        learning_cfg = algo_cfg.get("learning", {})
        if "diff_max" in learning_cfg:
            env_cfg["max_difficulty"] = learning_cfg["diff_max"]
        env = RoutingGym.from_json(env_cfg)

        algorithm_cls = full_config["algorithm_cls"].split(".")[-1]
        assert algorithm_cls in ALGORITHMS, (
            f"Algorithm class {full_config['algorithm_cls']} not supported, "
            f"should be {list(ALGORITHMS.keys())}"
        )
        algorithm_config = ALGORITHMS[algorithm_cls].from_json(full_config["algorithm"])

        model_cls = full_config["policy_cls"].split(".")[-1]
        assert model_cls in POLICIES, (
            f"Policy class {full_config['policy_cls']} not supported, "
            f"should be {list(POLICIES.keys())}"
        )
        model_config = POLICIES[model_cls].from_json(full_config["policy"])

        return cls(env, algorithm_config, model_config, model_path)

    def init_algorithm(self, model_path=None):
        obs_perms, act_perms = self.env.twists()
        model = self.model_cls(
            self.env.obs_shape(),
            self.env.num_actions(),
            **self.model_config.to_json(),
            obs_perms=obs_perms,
            act_perms=act_perms,
        )
        if model_path is not None:
            model.load_state_dict(load_checkpoint(model_path))

        return self.algorithm_cls(
            self.env._raw_env, model, self.rl_config.to_json(), None
        )

    def route(
        self,
        circuit: QuantumCircuit,
        deterministic: bool = False,
        num_searches: int = 100,
        num_mcts_searches: int = 0,
        C: float = 2**0.5,
        max_expand_depth: int = 1,
    ):
        """Route a circuit. Returns (routed_circuit, init_layout, final_layout) or None."""
        state = self.env.get_state(circuit)
        actions = self.algorithm.solve(
            state, deterministic, num_searches, num_mcts_searches, C, max_expand_depth
        )
        if actions is not None:
            return self.env.build_circuit_from_solution(actions, circuit)

    def learn(self, initial_difficulty=1, num_iterations=int(1e10), tb_path=None):
        if tb_path is not None:
            self.algorithm.run_path = tb_path
            self.algorithm.tb_writer = SummaryWriter(tb_path)

        self.env.difficulty = initial_difficulty

        try:
            self.algorithm.learn(num_iterations)
        except KeyboardInterrupt:
            return
