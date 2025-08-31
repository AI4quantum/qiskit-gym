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
from dataclasses import dataclass, field, replace
from typing import List, Mapping, Any, Dict

# ----------------------------- Shared: evaluation presets -----------------------------


@dataclass
class EvalConfig:
    """
    A single named evaluation setting.

    Semantics
    ---------
    - deterministic: if True, greedy decoding (argmax). If False, sample from policy.
    - num_searches: run N independent whole-episode rollouts and keep the best result.
    - num_mcts_searches: for each rollout, run N MCTS simulations per decision.
      This stacks with num_searches (e.g., num_searches=2 & num_mcts_searches=10
      performs two rounds of 10 MCTS sims and picks the best).
    - num_cores: parallel workers used for evaluation.
    - C: MCTS exploration constant (higher => more exploration).

    """

    num_episodes: int = 100
    deterministic: bool = True
    num_searches: int = 1
    num_mcts_searches: int = 0
    num_cores: int = 32
    C: float = 1.41

    def validate(self) -> None:
        if self.num_episodes <= 0:
            raise ValueError("EvalConfig.num_episodes must be > 0")
        if self.num_searches <= 0:
            raise ValueError("EvalConfig.num_searches must be > 0")
        if self.num_mcts_searches < 0:
            raise ValueError("EvalConfig.num_mcts_searches must be >= 0")
        if self.num_cores <= 0:
            raise ValueError("EvalConfig.num_cores must be > 0")
        if self.C <= 0:
            raise ValueError("EvalConfig.C must be > 0")

    @classmethod
    def from_partial(cls, data: Mapping[str, Any] | None) -> "EvalConfig":
        """Build from partial dict, filling unspecified fields with defaults."""
        if not data:
            return cls()
        return cls(
            num_episodes=int(data.get("num_episodes", 100)),
            deterministic=bool(data.get("deterministic", True)),
            num_searches=int(data.get("num_searches", 1)),
            num_mcts_searches=int(data.get("num_mcts_searches", 0)),
            num_cores=int(data.get("num_cores", 32)),
            C=float(data.get("C", 1.41)),
        )


# ----------------------------- PPO (flat, kwargs-first) -----------------------------


@dataclass
class PPOConfig:
    """
    PPO configuration (flat, kwargs-first).

    Collection (rollouts)
    ---------------------
    num_cores : int
        Parallel actors for experience collection.
    num_episodes : int
        Episodes collected per training batch (across all cores).
    gae_lambda : float
        GAE(lambda) parameter in [0, 1].
    gamma : float
        Discount factor gamma in [0, 1].

    Training
    --------
    num_epochs : int
        Optimization epochs per batch.
    vf_coef : float
        Value loss coefficient.
    ent_coef : float
        Entropy bonus coefficient.
    clip_ratio : float
        PPO clip epsilon (> 0).
    normalize_advantage : bool
        Normalize advantages per batch.

    Optimizer
    ---------
    lr : float
        Learning rate.

    Curriculum (difficulty progression)
    -----------------------------------
    diff_threshold : float
        Success-rate threshold in [0, 1] to advance difficulty.
    diff_max : int
        Maximum difficulty (inclusive).
    diff_metric : str
        Name of the evaluation metric used to decide advancement.
        Must be a key in `evals`.

    Evaluation presets
    ------------------
    evals : Dict[str, EvalConfig]
        Any number of named evaluation configs. Defaults:
        - "ppo_deterministic": greedy, 1 search.
        - "ppo_10": 10 searches (sampled), keep best.

    Logging
    -------
    log_freq : int
        Log every N steps/updates.
    checkpoint_freq : int
        Checkpoint every N steps/updates.
    """

    # ---- collection
    num_cores: int = 32
    num_episodes: int = 1024
    gae_lambda: float = 0.995
    gamma: float = 0.995

    # ---- training
    num_epochs: int = 10
    vf_coef: float = 0.8
    ent_coef: float = 0.01
    clip_ratio: float = 0.1
    normalize_advantage: bool = False

    # ---- optimizer
    lr: float = 3e-4

    # ---- curriculum
    diff_threshold: float = 0.85
    diff_max: int = 256
    diff_metric: str = "ppo_deterministic"

    # ---- evals & logging
    evals: Dict[str, EvalConfig] = field(
        default_factory=lambda: {
            "ppo_deterministic": EvalConfig(),
            "ppo_10": EvalConfig(deterministic=False, num_searches=10),
        }
    )
    log_freq: int = 1
    checkpoint_freq: int = 10

    # ---- constant
    algorithm_cls: str = "twisterl.rl.PPO"

    # -------------- API --------------

    def validate(self) -> None:
        if self.num_cores <= 0:
            raise ValueError("num_cores must be > 0")
        if self.num_episodes <= 0:
            raise ValueError("num_episodes must be > 0")
        if not (0.0 <= self.gae_lambda <= 1.0):
            raise ValueError("gae_lambda must be in [0, 1]")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if self.clip_ratio <= 0:
            raise ValueError("clip_ratio must be > 0")
        if not (0.0 <= self.diff_threshold <= 1.0):
            raise ValueError("diff_threshold must be in [0, 1]")
        if self.diff_max < 1:
            raise ValueError("diff_max must be >= 1")
        if self.diff_metric not in self.evals:
            raise ValueError(
                f"diff_metric '{self.diff_metric}' not found in evals: {list(self.evals.keys())}"
            )
        for name, ev in self.evals.items():
            try:
                ev.validate()
            except Exception as e:
                raise ValueError(f"Invalid eval '{name}': {e}") from e

    def with_updates(self, **kwargs) -> "PPOConfig":
        """Return a new PPOConfig with given kwargs applied (dataclasses.replace wrapper)."""
        return replace(self, **kwargs)

    # ------- Interop with nested JSON schema -------

    def to_json(self) -> dict:
        """
        Export to the nested schema expected by twisterl:

        {
          "algorithm_cls": "twisterl.rl.PPO",
          "algorithm": {
            "collecting": {...}, "training": {...}, "learning": {...},
            "optimizer": {...}, "evals": {...}, "logging": {...}
          }
        }
        """
        self.validate()
        return {
            "collecting": {
                "num_cores": self.num_cores,
                "num_episodes": self.num_episodes,
                "lambda": self.gae_lambda,  # JSON name is "lambda"
                "gamma": self.gamma,
            },
            "training": {
                "num_epochs": self.num_epochs,
                "vf_coef": self.vf_coef,
                "ent_coef": self.ent_coef,
                "clip_ratio": self.clip_ratio,
                "normalize_advantage": self.normalize_advantage,
            },
            "learning": {
                "diff_threshold": self.diff_threshold,
                "diff_max": self.diff_max,
                "diff_metric": self.diff_metric,
            },
            "optimizer": {"lr": self.lr},
            "evals": {k: vars(v) for k, v in self.evals.items()},
            "logging": {
                "log_freq": self.log_freq,
                "checkpoint_freq": self.checkpoint_freq,
            },
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "PPOConfig":
        """
        Build from either:
        - a full nested dict with keys {"algorithm_cls", "algorithm": {...}}, or
        - a flat dict using this class's field names.
        Unknown keys are ignored.
        """

        algo = data
        # Collect evals first, so diff_metric validation can succeed
        evals_raw = algo.get("evals", {})
        evals = {}
        # Seed with defaults
        evals.update(cls().evals)
        for name, partial in evals_raw.items():
            evals[name] = EvalConfig.from_partial(partial)

        flat = {
            # collecting
            "num_cores": algo.get("collecting", {}).get("num_cores", cls.num_cores),
            "num_episodes": algo.get("collecting", {}).get(
                "num_episodes", cls.num_episodes
            ),
            "gae_lambda": algo.get("collecting", {}).get("lambda", cls.gae_lambda),
            "gamma": algo.get("collecting", {}).get("gamma", cls.gamma),
            # training
            "num_epochs": algo.get("training", {}).get("num_epochs", cls.num_epochs),
            "vf_coef": algo.get("training", {}).get("vf_coef", cls.vf_coef),
            "ent_coef": algo.get("training", {}).get("ent_coef", cls.ent_coef),
            "clip_ratio": algo.get("training", {}).get("clip_ratio", cls.clip_ratio),
            "normalize_advantage": algo.get("training", {}).get(
                "normalize_advantage", cls.normalize_advantage
            ),
            # curriculum
            "diff_threshold": algo.get("learning", {}).get(
                "diff_threshold", cls.diff_threshold
            ),
            "diff_max": algo.get("learning", {}).get("diff_max", cls.diff_max),
            "diff_metric": algo.get("learning", {}).get("diff_metric", cls.diff_metric),
            # optimizer
            "lr": algo.get("optimizer", {}).get("lr", cls.lr),
            # logging
            "log_freq": algo.get("logging", {}).get("log_freq", cls.log_freq),
            "checkpoint_freq": algo.get("logging", {}).get(
                "checkpoint_freq", cls.checkpoint_freq
            ),
            # constant
            "algorithm_cls": data.get("algorithm_cls", "twisterl.rl.PPO"),
        }
        obj = cls(**flat, evals=evals)
        obj.validate()
        return obj


# ----------------------------- AlphaZero (flat, kwargs-first) -----------------------------


@dataclass
class AlphaZeroConfig:
    """
    AlphaZero configuration (flat, kwargs-first).

    Collection (self-play)
    ----------------------
    num_cores : int
        Parallel self-play workers.
    num_episodes : int
        Episodes/games per iteration.
    num_mcts_searches : int
        MCTS simulations per decision during collection.
    C : float
        MCTS exploration constant (e.g., PUCT).
    max_expand_depth : int
        Node expansion cap; 1 expands only one level initially.

    Training
    --------
    num_epochs : int
        Optimization epochs per iteration.

    Optimizer
    ---------
    lr : float
        Learning rate.

    Curriculum (difficulty progression)
    -----------------------------------
    diff_threshold : float
        Success-rate threshold in [0, 1] to advance difficulty.
    diff_max : int
        Maximum difficulty (inclusive).
    diff_metric : str
        Name of the evaluation metric used to decide advancement.
        Must be a key in `evals`.

    Evaluation presets
    ------------------
    evals : Dict[str, EvalConfig]
        Any number of named evaluation configs. Defaults:
        - "ppo_deterministic": greedy, 1 search.
        - "ppo_10": 10 searches (sampled), keep best.
        - "mcts_100": greedy with 100 MCTS sims per decision.

    Logging
    -------
    log_freq : int
        Log every N steps/updates.
    checkpoint_freq : int
        Checkpoint every N steps/updates.
    """

    # ---- collection
    num_cores: int = 32
    num_episodes: int = 128
    num_mcts_searches: int = 1000
    C: float = 1.41
    max_expand_depth: int = 1

    # ---- training
    num_epochs: int = 10

    # ---- optimizer
    lr: float = 3e-4

    # ---- curriculum
    diff_threshold: float = 0.85
    diff_max: int = 256
    diff_metric: str = "mcts_100"

    # ---- evals & logging
    evals: Dict[str, EvalConfig] = field(
        default_factory=lambda: {
            "ppo_deterministic": EvalConfig(),
            "ppo_10": EvalConfig(deterministic=False, num_searches=10),
            "mcts_100": EvalConfig(
                deterministic=True, num_searches=1, num_mcts_searches=100
            ),
        }
    )
    log_freq: int = 1
    checkpoint_freq: int = 10

    # ---- constant
    algorithm_cls: str = "twisterl.rl.AZ"

    # -------------- API --------------

    def validate(self) -> None:
        if self.num_cores <= 0:
            raise ValueError("num_cores must be > 0")
        if self.num_episodes <= 0:
            raise ValueError("num_episodes must be > 0")
        if self.num_mcts_searches <= 0:
            raise ValueError("num_mcts_searches must be > 0")
        if self.C <= 0:
            raise ValueError("C must be > 0")
        if self.max_expand_depth < 1:
            raise ValueError("max_expand_depth must be >= 1")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if not (0.0 <= self.diff_threshold <= 1.0):
            raise ValueError("diff_threshold must be in [0, 1]")
        if self.diff_max < 1:
            raise ValueError("diff_max must be >= 1")
        if self.diff_metric not in self.evals:
            raise ValueError(
                f"diff_metric '{self.diff_metric}' not found in evals: {list(self.evals.keys())}"
            )
        for name, ev in self.evals.items():
            try:
                ev.validate()
            except Exception as e:
                raise ValueError(f"Invalid eval '{name}': {e}") from e

    def with_updates(self, **kwargs) -> "AlphaZeroConfig":
        """Return a new AlphaZeroConfig with given kwargs applied (dataclasses.replace wrapper)."""
        return replace(self, **kwargs)

    # ------- Interop with nested JSON schema -------

    def to_json(self) -> dict:
        """
        Export to the nested schema expected by twisterl:

        {
          "algorithm_cls": "twisterl.rl.AZ",
          "algorithm": {
            "collecting": {...}, "training": {...}, "learning": {...},
            "optimizer": {...}, "evals": {...}, "logging": {...}
          }
        }
        """
        self.validate()
        return {
            "collecting": {
                "num_cores": self.num_cores,
                "num_episodes": self.num_episodes,
                "num_mcts_searches": self.num_mcts_searches,
                "C": self.C,
                "max_expand_depth": self.max_expand_depth,
            },
            "training": {"num_epochs": self.num_epochs},
            "learning": {
                "diff_threshold": self.diff_threshold,
                "diff_max": self.diff_max,
                "diff_metric": self.diff_metric,
            },
            "optimizer": {"lr": self.lr},
            "evals": {k: vars(v) for k, v in self.evals.items()},
            "logging": {
                "log_freq": self.log_freq,
                "checkpoint_freq": self.checkpoint_freq,
            },
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "AlphaZeroConfig":
        """
        Build from either:
        - a full nested dict with keys {"algorithm_cls", "algorithm": {...}}, or
        - a flat dict using this class's field names.
        Unknown keys are ignored.
        """

        algo = data
        # Collect evals first
        evals_raw = algo.get("evals", {})
        evals = {}
        evals.update(cls().evals)
        for name, partial in evals_raw.items():
            evals[name] = EvalConfig.from_partial(partial)

        flat = {
            # collecting
            "num_cores": algo.get("collecting", {}).get("num_cores", cls.num_cores),
            "num_episodes": algo.get("collecting", {}).get(
                "num_episodes", cls.num_episodes
            ),
            "num_mcts_searches": algo.get("collecting", {}).get(
                "num_mcts_searches", cls.num_mcts_searches
            ),
            "C": algo.get("collecting", {}).get("C", cls.C),
            "max_expand_depth": algo.get("collecting", {}).get(
                "max_expand_depth", cls.max_expand_depth
            ),
            # training
            "num_epochs": algo.get("training", {}).get("num_epochs", cls.num_epochs),
            # curriculum
            "diff_threshold": algo.get("learning", {}).get(
                "diff_threshold", cls.diff_threshold
            ),
            "diff_max": algo.get("learning", {}).get("diff_max", cls.diff_max),
            "diff_metric": algo.get("learning", {}).get("diff_metric", cls.diff_metric),
            # optimizer
            "lr": algo.get("optimizer", {}).get("lr", cls.lr),
            # logging
            "log_freq": algo.get("logging", {}).get("log_freq", cls.log_freq),
            "checkpoint_freq": algo.get("logging", {}).get(
                "checkpoint_freq", cls.checkpoint_freq
            ),
            # constant
            "algorithm_cls": data.get("algorithm_cls", "twisterl.rl.AZ"),
        }
        obj = cls(**flat, evals=evals)
        obj.validate()
        return obj


ALGORITHMS = {
    "AZ": AlphaZeroConfig,
    "PPO": PPOConfig,
}

# =============================== Policy configs ===============================


def _validate_layers(layers: List[int], name: str) -> None:
    if not isinstance(layers, list):
        raise ValueError(
            f"{name} must be a list of ints (got {type(layers).__name__})."
        )
    if any((not isinstance(x, int)) or x < 1 for x in layers):
        raise ValueError(f"Every entry in {name} must be an int >= 1 (got {layers}).")


# =============================== Fully Connected ===============================


@dataclass
class BasicPolicyConfig:
    """
    Fully-connected (MLP) policy/value architecture (flat, kwargs-first).

    Shared torso
    ------------
    embedding_size : int
        Size of the input embedding vector fed to the MLP torso.
    common_layers : List[int]
        Hidden layer sizes of the shared torso (applies before branching into heads).

    Heads
    -----
    policy_layers : List[int]
        Hidden layer sizes for the policy head (after the shared torso).
    value_layers : List[int]
        Hidden layer sizes for the value head (after the shared torso).

    Example
    -------
    >>> cfg = BasicPolicyConfig(embedding_size=512, common_layers=[256], policy_layers=[], value_layers=[])
    >>> cfg_json = cfg.to_json()  # -> {"policy_cls": "twisterl.nn.BasicPolicy", "policy": {...}}
    """

    embedding_size: int = 512
    common_layers: List[int] = field(default_factory=lambda: [256])
    policy_layers: List[int] = field(default_factory=list)
    value_layers: List[int] = field(default_factory=list)

    # constant used by to_json()/from_json()
    policy_cls: str = "twisterl.nn.BasicPolicy"

    # ---------------- API ----------------

    def validate(self) -> None:
        if self.embedding_size < 1:
            raise ValueError("embedding_size must be >= 1.")
        _validate_layers(self.common_layers, "common_layers")
        _validate_layers(self.policy_layers, "policy_layers")
        _validate_layers(self.value_layers, "value_layers")

    def with_updates(self, **kwargs) -> "BasicPolicyConfig":
        """Return a new config with given kwargs applied."""
        return replace(self, **kwargs)

    # -------- Interop with nested schema --------

    def to_json(self) -> dict:
        """Export to {'policy_cls': 'twisterl.nn.BasicPolicy', 'policy': {...}}."""
        self.validate()
        return {
            "embedding_size": self.embedding_size,
            "common_layers": list(self.common_layers),
            "policy_layers": list(self.policy_layers),
            "value_layers": list(self.value_layers),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "BasicPolicyConfig":
        """
        Build from either:
        - a nested dict with keys {'policy_cls', 'policy': {...}}, or
        - a flat dict using this class's field names.
        Unknown keys are ignored.
        """

        pol = data
        obj = cls(
            embedding_size=int(pol.get("embedding_size", cls.embedding_size)),
            common_layers=list(pol.get("common_layers", cls().common_layers)),
            policy_layers=list(pol.get("policy_layers", cls().policy_layers)),
            value_layers=list(pol.get("value_layers", cls().value_layers)),
            policy_cls=data.get("policy_cls", "twisterl.nn.BasicPolicy"),
        )
        obj.validate()
        return obj


# =============================== Conv1d ===============================


@dataclass
class Conv1dPolicyConfig:
    """
    Conv1d-based policy/value architecture (flat, kwargs-first).

    Convolutional frontend
    ----------------------
    conv_dim : int
        Axis along which the 1D convolution will happen. Bust be an integer from 0 to the shape of the obs.
        For example, for 2D inputs, this can be either 0 or 1.
    embedding_size : int
        Size of the flattened/aggregated feature vector produced by the Conv1d stack
        that is fed into the MLP torso.

    Shared torso
    ------------
    common_layers : List[int]
        Hidden layer sizes of the shared MLP torso after the Conv1d embedding.

    Heads
    -----
    policy_layers : List[int]
        Hidden layer sizes for the policy head.
    value_layers : List[int]
        Hidden layer sizes for the value head.

    Example
    -------
    >>> cfg = Conv1dPolicyConfig(conv_dim=1, embedding_size=1260, common_layers=[256])
    >>> cfg_json = cfg.to_json()  # -> {"policy_cls": "twisterl.nn.Conv1dPolicy", "policy": {...}}
    """

    conv_dim: int = 1
    embedding_size: int = 1260
    common_layers: List[int] = field(default_factory=lambda: [256])
    policy_layers: List[int] = field(default_factory=list)
    value_layers: List[int] = field(default_factory=list)

    # constant used by to_json()/from_json()
    policy_cls: str = "twisterl.nn.Conv1dPolicy"

    # ---------------- API ----------------

    def validate(self) -> None:
        if self.embedding_size < 1:
            raise ValueError("embedding_size must be >= 1.")
        _validate_layers(self.common_layers, "common_layers")
        _validate_layers(self.policy_layers, "policy_layers")
        _validate_layers(self.value_layers, "value_layers")

    def with_updates(self, **kwargs) -> "Conv1dPolicyConfig":
        """Return a new config with given kwargs applied."""
        return replace(self, **kwargs)

    # -------- Interop with nested schema --------

    def to_json(self) -> dict:
        """Export to {'policy_cls': 'twisterl.nn.Conv1dPolicy', 'policy': {...}}."""
        self.validate()
        return {
            "conv_dim": self.conv_dim,
            "embedding_size": self.embedding_size,
            "common_layers": list(self.common_layers),
            "policy_layers": list(self.policy_layers),
            "value_layers": list(self.value_layers),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Conv1dPolicyConfig":
        """
        Build from either:
        - a nested dict with keys {'policy_cls', 'policy': {...}}, or
        - a flat dict using this class's field names.
        Unknown keys are ignored.
        """

        pol = data
        obj = cls(
            conv_dim=int(pol.get("conv_dim", cls.conv_dim)),
            embedding_size=int(pol.get("embedding_size", cls.embedding_size)),
            common_layers=list(pol.get("common_layers", cls().common_layers)),
            policy_layers=list(pol.get("policy_layers", cls().policy_layers)),
            value_layers=list(pol.get("value_layers", cls().value_layers)),
            policy_cls=data.get("policy_cls", "twisterl.nn.Conv1dPolicy"),
        )
        obj.validate()
        return obj


POLICIES = {
    "BasicPolicy": BasicPolicyConfig,
    "Conv1dPolicy": Conv1dPolicyConfig,
}
