// -*- coding: utf-8 -*-
/* 
(C) Copyright 2025 IBM. All Rights Reserved.

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory
of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.
*/

use pyo3::prelude::*;

use rand::distributions::{Distribution, Uniform};
use rand::Rng;

use twisterl::rl::env::Env;
use twisterl::python_interface::env::PyBaseEnv;

use crate::envs::common::Gate;
use crate::envs::metrics::{MetricsCounts, MetricsTracker, MetricsWeights};
use crate::envs::symmetry::compute_twists_square;
use std::collections::HashMap;


// This is the Env definition
#[derive(Clone)]
pub struct Permutation {
    pub state: Vec<usize>,
    pub depth: usize,
    pub success: bool,

    pub num_qubits: usize,
    pub difficulty: usize,
    pub gateset: Vec<Gate>,
    pub depth_slope: usize,
    pub max_depth: usize,
    pub obs_perms: Vec<Vec<usize>>,
    pub act_perms: Vec<Vec<usize>>,
    metrics: MetricsTracker,
    metrics_values: MetricsCounts,
    metrics_weights: MetricsWeights,
    reward_value: f32,
    pub add_inverts: bool,
    track_solution: bool,
    solution: Vec<usize>,
    solution_inv: Vec<usize>,    
    inverted: bool,
}


impl Permutation {
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        gateset: Vec<Gate>,
        depth_slope: usize,
        max_depth: usize,
        metrics_weights: MetricsWeights,
        add_inverts: bool,
        add_perms: bool,
        track_solution: bool,
    ) -> Self {
        // Only compute symmetries if enabled
        let (obs_perms, act_perms) = if add_perms {
            compute_twists_square(num_qubits, &gateset)
        } else {
            (Vec::new(), Vec::new())
        };
        
        let metrics = MetricsTracker::new(num_qubits);
        let metrics_values = metrics.snapshot();
        let success = true;
        Permutation {
            state:(0..num_qubits).collect(),
            depth:1,
            success,
            num_qubits,
            difficulty,
            gateset,
            depth_slope,
            max_depth,
            obs_perms,
            act_perms,
            metrics,
            metrics_values,
            metrics_weights,
            reward_value: 1.0,
            add_inverts,
            track_solution,
            solution: Vec::new(),
            solution_inv: Vec::new(),
            inverted: false,
        }
    }

    /// Compute the inverse of a permutation
    /// For a permutation perm, returns inv such that perm[inv[i]] = i for all i
    fn invert_perm(perm: &[usize]) -> Vec<usize> {
        let mut inv = vec![0; perm.len()];
        for (i, &val) in perm.iter().enumerate() {
            inv[val] = i;
        }
        inv
    }

    /// Randomly invert the permutation with 50% probability when enabled.
    fn maybe_random_invert(&mut self) {
        if !self.add_inverts {
            return;
        }

        let mut rng = rand::thread_rng();
        if rng.gen_bool(0.5) {
            self.state = Self::invert_perm(&self.state);
            self.inverted = !self.inverted;
        }
    }

    pub fn solved(&self) -> bool {
        for i in 0..self.state.len() {
            if self.state[i] != i {return false}
        }

        true
    }

    pub fn get_state(&self) -> Vec<usize> {
        self.state.clone()
    }
}

// This implements the necessary functions for the environment
impl Env for Permutation {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn num_actions(&self) -> usize {
        self.gateset.len()
    }

    fn obs_shape(&self) -> Vec<usize> {
        vec![self.state.len(), self.state.len()]
    }

    fn set_difficulty(&mut self, difficulty: usize) {
        self.difficulty = difficulty;
    }

    fn get_difficulty(&self) -> usize {
        self.difficulty
    }

    fn set_state(&mut self, state: Vec<i64>) {
        self.state = state.iter().map(|&x| x as usize).collect();

        self.depth = self.max_depth;  
        self.success = self.solved();
        self.metrics.reset();
        self.metrics_values = self.metrics.snapshot();
        self.reward_value = if self.success { 1.0 } else { 0.0 };
        self.inverted = false;
    }

    fn reset(&mut self) {
        // Reset the state to the target
        self.state = (0..self.num_qubits).collect();
        let mut rng = rand::thread_rng();
        let action_range = Uniform::new(0, self.num_actions());

        // Apply random actions based on the difficulty
        for _ in 0..self.difficulty {
            let action = action_range.sample(&mut rng);
            let gate = &self.gateset[action];
            match gate {
                Gate::SWAP(q1, q2) => (self.state[*q2], self.state[*q1]) = (self.state[*q1], self.state[*q2]),
                _ => {}
            }
        }
        self.depth = (self.depth_slope * self.difficulty).min(self.max_depth);
        self.success = self.solved();
        self.metrics.reset();
        self.metrics_values = self.metrics.snapshot();
        self.reward_value = if self.success { 1.0 } else { 0.0 };
        self.inverted = false;
    }

    fn step(&mut self, action: usize)  {
        let mut penalty = 0.0f32;

        if action < self.gateset.len() {
            let gate = &self.gateset[action];
            let previous = self.metrics_values.clone();
            self.metrics.apply_gate(gate);
            let new_metrics = self.metrics.snapshot();
            penalty = new_metrics.weighted_delta(&previous, &self.metrics_weights);
            self.metrics_values = new_metrics;

            match gate {
                Gate::SWAP(q1, q2) => (self.state[*q2], self.state[*q1]) = (self.state[*q1], self.state[*q2]),
                _ => {}
            }

            if self.track_solution {
               if self.inverted {
                   self.solution_inv.push(action);
                } else {
                    self.solution.push(action);
                }
            }
        }

        self.maybe_random_invert();

        self.depth = self.depth.saturating_sub(1); // Prevent underflow
        self.success = self.solved();
        let achieved = if self.success { 1.0 } else { 0.0 };
        self.reward_value = achieved - penalty;
    }
    
    fn masks(&self) -> Vec<bool> {
        vec![!self.success; self.num_actions()]
    }

    fn is_final(&self) -> bool {
        self.depth == 0 || self.success
    }

    fn reward(&self) -> f32 { self.reward_value }

    fn success(&self) -> bool {
        self.success
    }
    
    fn observe(&self,) -> Vec<usize> {
        self.state.iter().enumerate().map(|(i, v)| i * self.num_qubits + v ).collect()  
    }

    fn twists(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        (self.obs_perms.clone(), self.act_perms.clone())
    }

    fn track_solution(&self) -> bool { self.track_solution }

    fn solution(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.solution.len() + self.solution_inv.len());
        out.extend_from_slice(&self.solution);
        out.extend(self.solution_inv.iter().rev().copied());
        out
    }
}


#[pyclass(name="PermutationEnv", extends=PyBaseEnv)]
pub struct PyPermutationEnv;

#[pymethods]
impl PyPermutationEnv {
    #[new]
    #[pyo3(signature = (
        num_qubits,
        difficulty,
        gateset,
        depth_slope,
        max_depth,
        metrics_weights=None,
        add_inverts=None,
        add_perms=None,
        track_solution=None,
    ))]
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        gateset: Vec<Gate>,
        depth_slope: usize,
        max_depth: usize,
        metrics_weights: Option<HashMap<String, f32>>,
        add_inverts: Option<bool>,
        add_perms: Option<bool>,
        track_solution: Option<bool>,
    ) -> (Self, PyBaseEnv) {
        let weights = MetricsWeights::from_hashmap(metrics_weights);
        let env = Permutation::new(
            num_qubits,
            difficulty,
            gateset,
            depth_slope,
            max_depth,
            weights,
            add_inverts.unwrap_or(true),
            add_perms.unwrap_or(true),
            track_solution.unwrap_or(true)
        );
        let env = Box::new(env);
        (PyPermutationEnv, PyBaseEnv { env })
    }
}
