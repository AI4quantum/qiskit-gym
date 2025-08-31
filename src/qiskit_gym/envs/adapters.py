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

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def gym_adapter(cls):
    """
    Decorator that converts a TwisteRL env into a Gymnasium env.
    ------------------------------------------------------
    Expected raw-env API
        obs_shape()        -> list[int]
        observe()          -> list[int]           # sparse indices of 1-bits
        reward()           -> float
        is_final()         -> bool
        num_actions()      -> int
        reset()            -> None
        step(action:int)   -> None
    Optional / forwarded
        difficulty (property)
        set_state(state)   -> None
        render(), close(), etc. (if they exist)
    """

    class GymWrapper(gym.Env):
        metadata = {"render_modes": ["human"], "render_fps": 4}

        def __init__(self, *args, **kwargs):
            # build the original env
            self.config = kwargs.copy()
            self._raw_env = cls(*args, **kwargs)

            # observation & action spaces
            self._obs_shape = tuple(self._raw_env.obs_shape())
            self.observation_space = spaces.MultiBinary(self._obs_shape)
            self.action_space = spaces.Discrete(self._raw_env.num_actions())

        # ------------- helpers ------------------------------------------------
        def _full_obs(self):
            """Return dense binary observation array"""
            full = np.zeros(np.prod(self._obs_shape), dtype=np.int8)
            full[self._raw_env.observe()] = 1
            return full.reshape(self._obs_shape)

        # ------------- gym-required methods -----------------------------------
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._raw_env.reset()
            return self._full_obs(), {}

        def step(self, action):
            assert not bool(self._raw_env.is_final()), (
                "Action provided when env is in final state."
            )
            self._raw_env.step(int(action))
            obs = self._full_obs()
            reward = float(self._raw_env.reward())
            terminated = bool(self._raw_env.is_final())
            truncated = False  # no time-limit truncation here
            info = {}
            return obs, reward, terminated, truncated, info

        # ------------- convenience passthroughs -------------------------------
        def render(self, mode="human"):
            if hasattr(self._raw_env, "render"):
                return self._raw_env.render(mode=mode)
            elif hasattr(self._raw_env, "get_state"):
                # fallback: print state
                print(self._raw_env.get_state())
            else:
                # fallback: crude text render
                print(self._full_obs())

        def close(self):
            if hasattr(self._raw_env, "close"):
                self._raw_env.close()

        # forward all other attributes / methods transparently
        def __getattr__(self, name):
            return getattr(self._raw_env, name)

        # allow `env.difficulty` assignment to propagate
        def __setattr__(self, name, value):
            if name in ("difficulty",) and "_raw_env" in self.__dict__:
                setattr(self._raw_env, name, value)
            else:
                super().__setattr__(name, value)

        def to_json(self):
            """Returns config parameters as a dict."""
            return self.config

    GymWrapper.__name__ = f"{cls.__name__}Gym"
    return GymWrapper
