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

import json

import torch
from torch.utils.tensorboard import SummaryWriter

from twisterl.utils import dynamic_import
from qiskit_gym.rl.configs import (
    AlphaZeroConfig,
    PPOConfig,
    BasicPolicyConfig,
    Conv1dPolicyConfig,
    POLICIES,
    ALGORITHMS,
)
from qiskit_gym.envs.synthesis import BaseSynthesisEnv, SYNTH_ENVS

from qiskit import QuantumCircuit


class RLSynthesis:
    def __init__(
        self,
        env: BaseSynthesisEnv,
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

        env_cls = full_config["env_cls"].split(".")[-1]
        assert env_cls in SYNTH_ENVS, (
            f"Synth env class {full_config['env_cls']} not supported, should be {list(SYNTH_ENVS.keys())}"
        )
        env = SYNTH_ENVS[env_cls].from_json(full_config["env"])

        algorithm_cls = full_config["algorithm_cls"].split(".")[-1]
        assert algorithm_cls in ALGORITHMS, (
            f"Algorithm class {full_config['algorithm_cls']} not supported, should be {list(ALGORITHMS.keys())}"
        )
        algorithm_config = ALGORITHMS[algorithm_cls].from_json(full_config["algorithm"])

        model_cls = full_config["policy_cls"].split(".")[-1]
        assert model_cls in POLICIES, (
            f"Policy class {full_config['policy_cls']} not supported, should be {list(POLICIES.keys())}"
        )
        model_config = POLICIES[model_cls].from_json(full_config["policy"])

        return cls(env, algorithm_config, model_config, model_path)

    def to_json(self):
        return {
            "env_cls": f"qiskit_gym.envs.synthesis.{self.env.cls_name}",
            "env": self.env_config,
            "policy_cls": self.model_config.policy_cls,
            "policy": self.model_config.to_json(),
            "algorithm_cls": self.rl_config.algorithm_cls,
            "algorithm": self.rl_config.to_json(),
        }

    def save(self, config_path, model_path=None):
        with open(config_path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

        if model_path is not None:
            with open(model_path, "wb") as f:
                torch.save(self.algorithm.policy.state_dict(), f)

    def init_algorithm(self, model_path=None):
        # Import policy class and make policy
        obs_perms, act_perms = self.env.twists()
        model = self.model_cls(
            self.env.obs_shape(),
            self.env.num_actions(),
            **self.model_config.to_json(),
            obs_perms=obs_perms,
            act_perms=act_perms,
        )
        if model_path is not None:
            model.load_state_dict(
                torch.load(open(model_path, "rb"), map_location=torch.device("cpu"))
            )

        return self.algorithm_cls(
            self.env._raw_env, model, self.rl_config.to_json(), None
        )

    def synth(
        self,
        input,
        deterministic: bool = False,
        num_searches: int = 100,
        num_mcts_searches: int = 0,
        C: float = (2**0.5),
        max_expand_depth: int = 1,
    ) -> QuantumCircuit:
        state = self.env.get_state(input)
        actions = self.algorithm.solve(
            state, deterministic, num_searches, num_mcts_searches, C, max_expand_depth
        )
        if actions is not None:
            return gate_list_to_circuit(
                [self.env_config["gateset"][a] for a in actions],
                num_qubits=self.env.config["num_qubits"],
            )

    def learn(self, initial_difficulty=1, num_iterations=int(1e10), tb_path=None):
        if tb_path is not None:
            self.algorithm.run_path = tb_path
            self.algorithm.tb_writer = SummaryWriter(tb_path)

        self.env.difficulty = initial_difficulty

        try:
            self.algorithm.learn(num_iterations)
        except KeyboardInterrupt:
            return


def gate_list_to_circuit(gate_list, num_qubits=None):
    if num_qubits is None:
        num_qubits = max(max(gate_args) for _, gate_args in gate_list) + 1
    qc = QuantumCircuit(num_qubits)
    for gate_name, gate_args in gate_list:
        getattr(qc, gate_name.lower())(*gate_args)
    return qc
