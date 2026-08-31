# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import torch
import numpy as np

def restore_tuples(obj):
    return [tuple(x) for x in obj["list"]] if obj.get("__tuple_list__") else obj


class CliffordTSolver:
    def __init__(self, env, policy, device="cpu"):
        self.env = env
        self.policy = policy
        self.device = device

    def current_observation(self):
        obs = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        obs.flat[self.env.observe()] = 1.0
        return obs

    def actions_to_channels(self, actions):
        return [a+(1 if a < (self.env.num_actions()//2) else 2) for a in actions]

    @torch.inference_mode()
    def choose_action(self, obs, sample=False):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.policy(x)
        valid = torch.as_tensor(self.env.masks(), dtype=torch.bool, device=self.device)
        probs = logits[0].masked_fill(~valid, -torch.inf).softmax(dim=0)
        act = int(torch.multinomial(probs, 1).item() if sample else probs.argmax().item())
        return act, probs[act].item()

    def choose_action(self, obs, temperature=0.0):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.policy(x)
        valid = torch.as_tensor(self.env.masks(), dtype=torch.bool, device=self.device)
        masked_logits = logits[0].masked_fill(~valid, -torch.inf)

        if temperature == 0:
            act = int(masked_logits.argmax())
        else:
            probs = (masked_logits / temperature).softmax(dim=0)
            act = int(torch.multinomial(probs, 1).item())

        return act

    def rollout(self, obs, temperature=0.0):
        actions = []
        total_reward = 0.0
        while not self.env.is_final():
            action = self.choose_action(obs, temperature)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            actions.append(action)
            if terminated or truncated:
                break
        return actions, self.env.reward() == 1.0, float(total_reward)


    def solve1(self, sequence, temperature=0.0):
        self.env.set_state_from_actions(sequence)
        #self.env.set_state(self.env.get_state())  # Give the rollout config['env']['max_steps'] steps
        return self.rollout(self.current_observation(), temperature)


    def solve(self, sequence, n, temperature=1.0):
        samples = [self.solve1(sequence, temperature=temperature) for _ in range(n)]
        return max(samples, key=lambda rw: rw[2])