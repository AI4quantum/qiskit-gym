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

use twisterl::python_interface::env::PyBaseEnv;
use twisterl::rl::env::Env;

use crate::envs::common::Gate;
use crate::envs::metrics::{MetricsCounts, MetricsTracker, MetricsWeights};
use crate::envs::symmetry::compute_twists_pauli;
use crate::pauli::pauli_network::PauliNetwork;
use nalgebra::DMatrix;
use std::collections::HashMap;

fn identity_tableau(num_qubits: usize) -> Vec<u8> {
    let dim = 2 * num_qubits;
    let mut data = vec![0u8; dim * dim];
    for i in 0..dim {
        data[i * dim + i] = 1;
    }
    data
}

fn encode_pauli_label(num_qubits: usize, qubit: usize, axis: char) -> String {
    let mut chars = vec!['I'; num_qubits];
    if qubit < num_qubits {
        chars[num_qubits - 1 - qubit] = axis;
    }
    chars.into_iter().collect()
}

fn random_rotations(num_qubits: usize, max_rotations: usize) -> Vec<String> {
    if max_rotations == 0 || num_qubits == 0 {
        return Vec::new();
    }
    let mut rng = rand::thread_rng();
    let axes = ['X', 'Y', 'Z'];
    let rotation_count = rng.gen_range(1..=max_rotations);
    (0..rotation_count)
        .map(|_| {
            let q = rng.gen_range(0..num_qubits);
            let axis = axes[rng.gen_range(0..axes.len())];
            encode_pauli_label(num_qubits, q, axis)
        })
        .collect()
}

fn pad_state(data: &DMatrix<u8>, max_cols: usize) -> Vec<u8> {
    let rows = data.nrows();
    let mut dense = vec![0u8; rows * max_cols];
    for c in 0..data.ncols().min(max_cols) {
        for r in 0..rows {
            dense[r * max_cols + c] = data[(r, c)];
        }
    }
    dense
}

#[derive(Clone)]
pub struct PauliEnv {
    pub network: PauliNetwork,
    pub depth: usize,
    pub success: bool,

    pub num_qubits: usize,
    pub max_rotations: usize,
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
    pauli_layer_reward: f32,
    track_solution: bool,
    solution: Vec<usize>,
}

impl PauliEnv {
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        gateset: Vec<Gate>,
        depth_slope: usize,
        max_depth: usize,
        max_rotations: usize,
        metrics_weights: MetricsWeights,
        add_perms: bool,
        pauli_layer_reward: f32,
        track_solution: bool,
    ) -> Self {
        let tableau = identity_tableau(num_qubits);
        let network = PauliNetwork::new(tableau, Vec::new());
        let success = network.solved();

        // Only compute symmetries if enabled
        let (obs_perms, act_perms) = if add_perms {
            compute_twists_pauli(num_qubits, &gateset, max_rotations)
        } else {
            (Vec::new(), Vec::new())
        };

        let metrics = MetricsTracker::new(num_qubits);
        let metrics_values = metrics.snapshot();

        PauliEnv {
            network,
            depth: 1,
            success,
            num_qubits,
            max_rotations: max_rotations.max(1),
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
            pauli_layer_reward,
            track_solution,
            solution: Vec::new(),
        }
    }

    fn pad_and_collect(&self) -> Vec<u8> {
        let rows = 2 * self.num_qubits;
        let max_cols = 2 * self.num_qubits + self.max_rotations;
        let mut dense = vec![0u8; rows * max_cols];

        // Copy the Clifford tableau (first 2*num_qubits columns)
        for c in 0..(2 * self.num_qubits) {
            for r in 0..rows {
                dense[r * max_cols + c] = self.network.data[(r, c)];
            }
        }

        // Copy only the active rotation columns (those still in the DAG)
        let active_indices = self.network.active_rotation_indices();
        for (i, &rot_idx) in active_indices.iter().enumerate() {
            if i >= self.max_rotations {
                break;
            }
            let src_col = 2 * self.num_qubits + rot_idx;
            let dst_col = 2 * self.num_qubits + i;
            for r in 0..rows {
                dense[r * max_cols + dst_col] = self.network.data[(r, src_col)];
            }
        }

        dense
    }

    fn gate_count(&self) -> usize {
        self.gateset.len()
    }

    fn rebuild_network(&mut self, tableau: Vec<u8>, rotations: Vec<String>) {
        self.network = PauliNetwork::new(tableau, rotations);
    }
}

impl Env for PauliEnv {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn num_actions(&self) -> usize {
        self.gate_count()
    }

    fn obs_shape(&self) -> Vec<usize> {
        vec![2 * self.num_qubits, 2 * self.num_qubits + self.max_rotations]
    }

    fn set_difficulty(&mut self, difficulty: usize) {
        self.difficulty = difficulty;
    }

    fn get_difficulty(&self) -> usize {
        self.difficulty
    }

    fn set_state(&mut self, state: Vec<i64>) {
        if state.is_empty() {
            return;
        }
        let mut iter = state.into_iter();
        let rotation_count = iter.next().unwrap_or(0).max(0) as usize;
        let tableau_len = 4 * self.num_qubits * self.num_qubits;
        let mut tableau = Vec::with_capacity(tableau_len);
        for _ in 0..tableau_len {
            let val = iter.next().unwrap_or(0);
            tableau.push(if val > 0 { 1 } else { 0 });
        }
        let mut rotations = Vec::with_capacity(rotation_count);
        for idx in 0..rotation_count {
            let len = iter.next().unwrap_or(0).max(0) as usize;
            let mut chars = Vec::with_capacity(len);
            for _ in 0..len {
                let ch = iter.next().unwrap_or(73); // 'I'
                let c = char::from_u32(ch as u32).unwrap_or('I');
                chars.push(c);
            }
            if idx < self.max_rotations {
                rotations.push(chars.into_iter().collect());
            }
        }

        self.rebuild_network(tableau, rotations);
        self.depth = self.max_depth;
        self.success = self.network.solved();
        self.metrics.reset();
        self.metrics_values = self.metrics.snapshot();
        self.reward_value = if self.success { 1.0 } else { 0.0 };
    }

    fn reset(&mut self) {
        let rotations = random_rotations(self.num_qubits, self.max_rotations);
        let tableau = identity_tableau(self.num_qubits);
        self.rebuild_network(tableau, rotations);

        // Clean initially trivial rotations (like Python does)
        self.network.clean_and_return_with_phases();

        self.depth = self.max_depth;
        self.success = self.network.solved();
        self.metrics.reset();
        self.metrics_values = self.metrics.snapshot();
        self.reward_value = if self.success { 1.0 } else { 0.0 };

        if self.gate_count() == 0 {
            return;
        }

        let mut rng = rand::thread_rng();
        let action_range = Uniform::new(0, self.num_actions());
        for _ in 0..self.difficulty {
            let action = action_range.sample(&mut rng);
            self.step(action);
        }

        self.depth = (self.depth_slope * self.difficulty).min(self.max_depth);
        self.success = self.network.solved();
        self.metrics.reset();
        self.metrics_values = self.metrics.snapshot();
        self.reward_value = if self.success { 1.0 } else { 0.0 };
        if self.track_solution {
            self.solution = Vec::new();
        }
    }

    fn step(&mut self, action: usize) {
        let mut penalty = 0.0f32;
        let mut new_rotations = 0usize;

        if let Some(gate) = self.gateset.get(action).cloned() {
            let previous = self.metrics_values.clone();
            self.metrics.apply_gate(&gate);
            let new_metrics = self.metrics.snapshot();
            penalty = new_metrics.weighted_delta(&previous, &self.metrics_weights);
            self.metrics_values = new_metrics;

            // act() returns rotations that became trivial
            let applied_rotations = self.network.act(&gate);
            new_rotations = applied_rotations.len();
        }

        if self.track_solution {
            self.solution.push(action);
        }

        self.depth = self.depth.saturating_sub(1);
        self.success = self.network.solved();
        let achieved = if self.success { 1.0 } else { 0.0 };
        // Add pauli_layer_reward for each rotation that became trivial
        self.reward_value = achieved - penalty + self.pauli_layer_reward * (new_rotations as f32);
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

    fn observe(&self) -> Vec<usize> {
        self.pad_and_collect()
            .iter()
            .enumerate()
            .filter_map(|(idx, &val)| if val > 0 { Some(idx) } else { None })
            .collect()
    }

    fn twists(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        (self.obs_perms.clone(), self.act_perms.clone())
    }

    fn track_solution(&self) -> bool {
        self.track_solution
    }

    fn solution(&self) -> Vec<usize> {
        self.solution.clone()
    }
}

#[pyclass(name = "PauliNetworkEnv", extends = PyBaseEnv)]
pub struct PyPauliEnv;

#[pymethods]
impl PyPauliEnv {
    #[new]
    #[pyo3(signature = (
        num_qubits,
        difficulty,
        gateset,
        depth_slope,
        max_depth,
        max_rotations,
        metrics_weights=None,
        add_perms=None,
        pauli_layer_reward=None,
        track_solution=None,
    ))]
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        gateset: Vec<Gate>,
        depth_slope: usize,
        max_depth: usize,
        max_rotations: usize,
        metrics_weights: Option<HashMap<String, f32>>,
        add_perms: Option<bool>,
        pauli_layer_reward: Option<f32>,
        track_solution: Option<bool>,
    ) -> (Self, PyBaseEnv) {
        let weights = MetricsWeights::from_hashmap(metrics_weights);
        let env = PauliEnv::new(
            num_qubits,
            difficulty,
            gateset,
            depth_slope,
            max_depth,
            max_rotations,
            weights,
            add_perms.unwrap_or(true),
            pauli_layer_reward.unwrap_or(0.01),
            track_solution.unwrap_or(true),
        );
        let env = Box::new(env);
        (PyPauliEnv, PyBaseEnv { env })
    }
}
