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

use twisterl::rl::env::Env;
use twisterl::python_interface::env::PyBaseEnv;

use crate::envs::common::Gate;

use crate::envs::symmetry::compute_twists_square;
use std::collections::{HashMap, HashSet};

// Define some internal representation
#[derive(Clone)]
pub struct LFState {
    pub data: Vec<bool>,
    size: usize,
}

// Here are some functions to manipulate the internal representation
impl LFState {
    // Constructor to create a new LinearFunction
    fn new(size: usize) -> Self {
        let mut lf = LFState {
            data: vec![false; size * size],
            size,
        };
        for i in 0..size {
            lf.set(i, i, true);
        }
        lf
    }

    // Method to set a value in the LinearFunction
    fn set(&mut self, row: usize, column: usize, value: bool) {
        let index = self.index(row, column);
        self.data[index] = value;
    }

    // Method to get a value from the LinearFunction
    fn get(&self, row: usize, column: usize) -> bool {
        let index = self.index(row, column);
        self.data[index]
    }

    // Method to perform cx between q1 and q2
    fn cx(&mut self, q1: usize, q2: usize) {
        if q1 == q2 {
            return;
        }
        for column in 0..self.size {
            let a_val = self.get(q1, column);
            let b_val = self.get(q2, column);
            self.set(q2, column, a_val ^ b_val);
        }
    }

    fn swap(&mut self, q1: usize, q2: usize) {
        if q1 == q2 {
            return;
        }
        for column in 0..self.size {
            let a_val = self.get(q1, column);
            let b_val = self.get(q2, column);
            self.set(q1, column, b_val);
            self.set(q2, column, a_val);
        }
    }

    // Private helper method to calculate the linear index from row and column
    fn index(&self, row: usize, column: usize) -> usize {
        row * self.size + column
    }

    // Check if it is identity
    fn solved(&self) -> bool {
        for i in 0..self.size {
            for j in 0..self.size {
                if ((i == j) && (self.get(i, j) != true)) || ((i != j) && (self.get(i, j) != false)) {
                    return false;
                }
            }
        }
        true
    }
}

// This is the Env definition
#[derive(Clone)]
pub struct LinearFunction {
    pub lf: LFState,
    pub depth: usize,
    pub success: bool,

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
}


impl LinearFunction {
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        gateset: Vec<Gate>,
        depth_slope: usize,
        max_depth: usize,
        metrics_weights: MetricsWeights,
    ) -> Self {
        let lf = LFState::new(num_qubits);
        let success = lf.solved();
        let (obs_perms, act_perms) = compute_twists_square(num_qubits, &gateset);
        let metrics = MetricsTracker::new(num_qubits);
        let metrics_values = metrics.snapshot();
        LinearFunction {
            lf,
            depth: 1,
            success,
            difficulty,
            gateset,
            depth_slope,
            max_depth,
            obs_perms,
            act_perms,
            metrics,
            metrics_values,
            metrics_weights,
            reward_value: if success { 1.0 } else { 0.0 },
        }
    }
    pub fn solved(&self) -> bool {
        self.lf.solved()
    }

}

// This implements the necessary functions for the environment
impl Env for LinearFunction {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn num_actions(&self) -> usize {
        self.gateset.len()
    }

    fn obs_shape(&self) -> Vec<usize> {
        vec![self.lf.size, self.lf.size]
    }

    fn set_difficulty(&mut self, difficulty: usize) {
        self.difficulty = difficulty;
    }

    fn get_difficulty(&self) -> usize {
        self.difficulty
    }

    fn set_state(&mut self, state: Vec<i64>) {
        self.lf.data = state.iter().map(|&x| x>0).collect();
        self.depth = self.max_depth;
        self.success = self.solved();
        self.metrics.reset();
        self.metrics_values = self.metrics.snapshot();
        //self.reward_value = if self.success { 1.0 } else { 0.0 };
    }

    fn reset(&mut self) {
        // Create an identity matrix for the initial 'lf' state
        self.lf = LFState::new(self.lf.size);
        self.depth = self.max_depth;
        self.success = self.solved();
        self.metrics.reset();
        self.metrics_values = self.metrics.snapshot();
        self.reward_value = if self.success { 1.0 } else { 0.0 };

        let mut rng = rand::thread_rng();
        let action_range = Uniform::new(0, self.num_actions());

        // Apply random actions based on the difficulty
        for _ in 0..self.difficulty {
            let action = action_range.sample(&mut rng);
            self.step(action);
        }
        self.depth = (self.depth_slope * self.difficulty).min(self.max_depth);
        self.success = self.solved();
        self.metrics.reset();
        self.metrics_values = self.metrics.snapshot();
        self.reward_value = if self.success { 1.0 } else { 0.0 };
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
                &Gate::CX(q1, q2) => self.lf.cx(q1, q2),
                &Gate::SWAP(q1, q2) => self.lf.swap(q1, q2),
                _ => {}
            }
        }

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

    fn reward(&self) -> f32 {
        self.reward_value
    }

    fn success(&self) -> bool {
        self.success
    }

    fn observe(&self,) -> Vec<usize> {
        self.lf.data.iter()
        .enumerate() // Iterate over the Vec with indices
        .filter_map(|(index, &value)| if value { Some(index) } else { None }) // Collect indices where the value is true
        .collect()    
    }

    fn twists(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        (self.obs_perms.clone(), self.act_perms.clone())
    }
}

#[derive(Clone)]
struct MetricsTracker {
    num_qubits: usize,
    n_cnots: usize,
    n_gates: usize,
    cnot_layers: HashSet<usize>,
    layers: HashSet<usize>,
    last_gates: Vec<isize>,
    last_cxs: Vec<isize>,
}

impl MetricsTracker {
    fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            n_cnots: 0,
            n_gates: 0,
            cnot_layers: HashSet::new(),
            layers: HashSet::new(),
            last_gates: vec![-1; num_qubits],
            last_cxs: vec![-1; num_qubits],
        }
    }

    fn reset(&mut self) {
        self.n_cnots = 0;
        self.n_gates = 0;
        self.cnot_layers.clear();
        self.layers.clear();
        for val in self.last_gates.iter_mut() {
            *val = -1;
        }
        for val in self.last_cxs.iter_mut() {
            *val = -1;
        }
    }

    fn snapshot(&self) -> MetricsCounts {
        MetricsCounts {
            n_cnots: self.n_cnots,
            n_layers_cnots: self.cnot_layers.len(),
            n_layers: self.layers.len(),
            n_gates: self.n_gates,
        }
    }

    fn apply_gate(&mut self, gate: &Gate) {
        match gate {
            Gate::CX(c, t) => self.cx(*c, *t),
            Gate::SWAP(c, t) => {
                self.cx(*c, *t);
                self.cx(*t, *c);
                self.cx(*c, *t);
            }
            Gate::CZ(c, t) => {
                self.single_qubit(*t);
                self.cx(*c, *t);
                self.single_qubit(*t);
            }
            Gate::H(q) | Gate::S(q) | Gate::Sdg(q) | Gate::SX(q) | Gate::SXdg(q) => {
                self.single_qubit(*q);
            }
        }
    }

    fn single_qubit(&mut self, q: usize) {
        if q >= self.num_qubits {
            return;
        }
        self.n_gates += 1;
        let gate_layer = self.last_gates[q] + 1;
        self.last_gates[q] = gate_layer;
        if gate_layer >= 0 {
            self.layers.insert(gate_layer as usize);
        }
    }

    fn cx(&mut self, control: usize, target: usize) {
        if control >= self.num_qubits || target >= self.num_qubits {
            return;
        }
        self.n_cnots += 1;
        self.n_gates += 1;

        let gate_layer = (self.last_gates[control].max(self.last_gates[target])) + 1;
        self.last_gates[control] = gate_layer;
        self.last_gates[target] = gate_layer;

        if gate_layer >= 0 {
            self.layers.insert(gate_layer as usize);
        }

        let cx_layer = (self.last_cxs[control].max(self.last_cxs[target])) + 1;
        self.last_cxs[control] = cx_layer;
        self.last_cxs[target] = cx_layer;

        if cx_layer >= 0 {
            self.cnot_layers.insert(cx_layer as usize);
        }
    }
}

#[derive(Clone)]
struct MetricsCounts {
    n_cnots: usize,
    n_layers_cnots: usize,
    n_layers: usize,
    n_gates: usize,
}

impl MetricsCounts {
    fn weighted_delta(&self, previous: &Self, weights: &MetricsWeights) -> f32 {
        let delta_cnots = self.n_cnots.saturating_sub(previous.n_cnots) as f32;
        let delta_layers_cnots =
            self.n_layers_cnots.saturating_sub(previous.n_layers_cnots) as f32;
        let delta_layers = self.n_layers.saturating_sub(previous.n_layers) as f32;
        let delta_gates = self.n_gates.saturating_sub(previous.n_gates) as f32;

        weights.n_cnots * delta_cnots
            + weights.n_layers_cnots * delta_layers_cnots
            + weights.n_layers * delta_layers
            + weights.n_gates * delta_gates
    }
}

#[derive(Clone)]
pub struct MetricsWeights {
    n_cnots: f32,
    n_layers_cnots: f32,
    n_layers: f32,
    n_gates: f32,
}

impl Default for MetricsWeights {
    fn default() -> Self {
        Self {
            n_cnots: 0.01,
            n_layers_cnots: 0.0,
            n_layers: 0.0,
            n_gates: 0.0001,
        }
    }
}

impl MetricsWeights {
    fn from_hashmap(map: Option<HashMap<String, f32>>) -> Self {
        let mut weights = Self::default();
        if let Some(values) = map {
            for (key, value) in values {
                match key.as_str() {
                    "n_cnots" => weights.n_cnots = value,
                    "n_layers_cnots" => weights.n_layers_cnots = value,
                    "n_layers" => weights.n_layers = value,
                    "n_gates" => weights.n_gates = value,
                    _ => {}
                }
            }
        }
        weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cx_gate_is_self_inverse() {
        let gateset = vec![Gate::CX(0, 1)];
        let metrics_weights = MetricsWeights::default();
        let mut env = LinearFunction::new(2, 1, gateset, 2, 8, metrics_weights);
        env.depth = env.max_depth;

        env.step(0);
        assert!(!env.solved());

        env.step(0);
        assert!(env.solved());
        assert!(env.reward() <= 1.0);
    }
}


#[pyclass(name="LinearFunctionEnv", extends=PyBaseEnv)]
pub struct PyLinearFunctionEnv;

#[pymethods]
impl PyLinearFunctionEnv {
    #[new]
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        gateset: Vec<Gate>,
        depth_slope: usize,
        max_depth: usize,
        metrics_weights: Option<HashMap<String, f32>>
    ) -> (Self, PyBaseEnv) {
        let weights = MetricsWeights::from_hashmap(metrics_weights);
        let env = LinearFunction::new(
            num_qubits,
            difficulty,
            gateset,
            depth_slope,
            max_depth,
            weights,
        );
        let env = Box::new(env);
        (PyLinearFunctionEnv, PyBaseEnv { env })
    }
}
