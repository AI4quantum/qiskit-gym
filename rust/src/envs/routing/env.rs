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
use std::collections::HashMap;

use twisterl::python_interface::env::PyBaseEnv;
use twisterl::rl::env::Env;

use super::config::{DistType, RoutingConfig};
use super::dag::TwoQubitDAG;
use super::generator::RoutingGenerator;
use crate::envs::metrics::{MetricsCounts, MetricsTracker, MetricsWeights};

use crate::envs::common::Gate;

/// Routing environment: insert SWAPs to make a circuit executable on a coupling map.
///
/// Actions: Discrete — pick one active swap per step.
/// Observations: one-hot encoded distance improvements per (swap, layer) cell.
#[derive(Clone)]
pub struct RoutingEnv {
    // Config
    config: RoutingConfig,

    // Runtime state
    in_dag: TwoQubitDAG,
    locations: Vec<usize>,  // qubit -> physical location
    qubits: Vec<usize>,     // physical location -> qubit
    active_swaps: Vec<usize>, // indices into coupling_map
    cached_obs: Vec<usize>, // pre-computed sparse observation indices
    depth: usize,
    success: bool,

    // Metrics tracking
    metrics: MetricsTracker,
    metrics_values: MetricsCounts,
    metrics_weights: MetricsWeights,
    reward_value: f32,

    // Generator
    generator: RoutingGenerator,

    // Solution tracking
    track_solution: bool,
    solution: Vec<usize>,
}

impl RoutingEnv {
    pub fn new(
        num_qubits: usize,
        coupling_map: Vec<(usize, usize)>,
        num_active_swaps: usize,
        horizon: usize,
        difficulty: usize,
        depth_slope: usize,
        max_depth: usize,
        obs_bins: usize,
        metrics_weights: MetricsWeights,
        track_solution: bool,
    ) -> Self {
        // Compute all-pairs shortest path distances via BFS
        let dists = compute_all_pairs_distances(num_qubits, &coupling_map);

        let generator = RoutingGenerator::new(num_qubits, &dists);

        let config = RoutingConfig {
            num_qubits,
            num_active_swaps,
            horizon,
            obs_bins,
            coupling_map,
            dists,
            difficulty,
            depth_slope,
            max_depth,
        };

        let in_dag = TwoQubitDAG::new(&config.coupling_map, num_qubits);
        let metrics = MetricsTracker::new(num_qubits);
        let metrics_values = metrics.snapshot();

        RoutingEnv {
            config,
            in_dag,
            locations: (0..num_qubits).collect(),
            qubits: (0..num_qubits).collect(),
            active_swaps: Vec::new(),
            cached_obs: Vec::new(),
            depth: 1,
            success: true,
            metrics,
            metrics_values,
            metrics_weights,
            reward_value: 1.0,
            generator,
            track_solution,
            solution: Vec::new(),
        }
    }

    fn build_dag_from_pairs(&mut self, pairs: &[(usize, usize)]) {
        self.in_dag = TwoQubitDAG::new(&self.config.coupling_map, self.config.num_qubits);
        for &(q1, q2) in pairs {
            self.in_dag.push(q1, q2);
        }
        self.in_dag.create_topgens();
    }

    fn reset_internals(&mut self) {
        self.locations = (0..self.config.num_qubits).collect();
        self.qubits = (0..self.config.num_qubits).collect();
        self.active_swaps = Vec::new();
        self.cached_obs = Vec::new();
        self.success = self.in_dag.len() == 0;
        self.metrics.reset();
        self.metrics_values = self.metrics.snapshot();
        self.reward_value = if self.success { 1.0 } else { 0.0 };
        self.solution.clear();
    }

    /// Compute and cache the observation and active swaps. Called after
    /// update_gens in reset/step/set_state.
    fn cache_observation(&mut self) {
        let num_bins = self.config.num_bins();
        let k = self.config.obs_bins as DistType;

        let (raw_obs, active_swaps) = self.in_dag.get_obs(
            &self.locations,
            &self.config.coupling_map,
            &self.config.dists,
            self.config.num_active_swaps,
        );
        self.active_swaps = active_swaps;

        self.cached_obs = Vec::with_capacity(raw_obs.len());
        for (i, &val) in raw_obs.iter().enumerate() {
            let clamped = val.clamp(-k, k);
            let bin = (clamped + k) as usize;
            let idx = i * num_bins + bin;
            self.cached_obs.push(idx);
        }
    }
}

impl Env for RoutingEnv {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn num_actions(&self) -> usize {
        self.config.num_active_swaps
    }

    fn obs_shape(&self) -> Vec<usize> {
        vec![
            self.config.num_active_swaps * self.config.horizon,
            self.config.num_bins(),
        ]
    }

    fn set_difficulty(&mut self, difficulty: usize) {
        self.config.difficulty = difficulty;
    }

    fn get_difficulty(&self) -> usize {
        self.config.difficulty
    }

    fn set_state(&mut self, state: Vec<i64>) {
        // Decode flat list of qubit pairs: [q1, q2, q1, q2, ...]
        let pairs: Vec<(usize, usize)> = state
            .chunks(2)
            .filter(|c| c.len() == 2)
            .map(|c| (c[0] as usize, c[1] as usize))
            .collect();

        self.build_dag_from_pairs(&pairs);
        self.depth = self.config.max_depth;
        self.reset_internals();

        // Resolve initially executable operations
        self.in_dag.execute_front(&self.locations, &self.config.dists);
        self.success = self.in_dag.len() == 0;
        self.reward_value = if self.success { 1.0 } else { 0.0 };

        // Compute and cache observation + active swaps
        self.in_dag.update_gens(self.config.horizon);
        self.cache_observation();
    }

    fn reset(&mut self) {
        // Generate random target circuit
        let pairs = self.generator.generate(self.config.difficulty);
        self.build_dag_from_pairs(&pairs);
        self.depth = (self.config.depth_slope * self.config.difficulty).min(self.config.max_depth);
        self.reset_internals();

        // Resolve initially executable operations
        self.in_dag.execute_front(&self.locations, &self.config.dists);
        self.success = self.in_dag.len() == 0;
        self.reward_value = if self.success { 1.0 } else { 0.0 };

        // Compute and cache observation + active swaps
        self.in_dag.update_gens(self.config.horizon);
        self.cache_observation();
    }

    fn step(&mut self, action: usize) {
        let mut penalty = 0.0f32;

        if action < self.active_swaps.len() {
            let swap_idx = self.active_swaps[action];
            let (l1, l2) = self.config.coupling_map[swap_idx];

            // Track metrics (SWAP = 3 CX)
            let previous = self.metrics_values.clone();
            self.metrics.apply_gate(&Gate::SWAP(l1, l2));
            let new_metrics = self.metrics.snapshot();
            penalty = new_metrics.weighted_delta(&previous, &self.metrics_weights);
            self.metrics_values = new_metrics;

            // Apply swap to location tracking
            self.qubits.swap(l1, l2);
            let q1 = self.qubits[l1];
            let q2 = self.qubits[l2];
            self.locations[q1] = l1;
            self.locations[q2] = l2;

            // Track solution
            if self.track_solution {
                self.solution.push(action);
            }
        }

        // Resolve newly executable operations
        self.in_dag
            .execute_front(&self.locations, &self.config.dists);

        self.depth = self.depth.saturating_sub(1);
        self.success = self.in_dag.len() == 0;
        let achieved = if self.success { 1.0 } else { 0.0 };
        self.reward_value = achieved - penalty;

        // Compute and cache observation + active swaps for next step
        self.in_dag.update_gens(self.config.horizon);
        self.cache_observation();
    }

    fn masks(&self) -> Vec<bool> {
        let n = self.config.num_active_swaps;
        if self.success {
            return vec![false; n];
        }
        let mut m = vec![false; n];
        for i in 0..self.active_swaps.len().min(n) {
            m[i] = true;
        }
        m
    }

    fn is_final(&self) -> bool {
        self.success || self.depth == 0
    }

    fn success(&self) -> bool {
        self.success
    }

    fn reward(&self) -> f32 {
        self.reward_value
    }

    fn observe(&self) -> Vec<usize> {
        self.cached_obs.clone()
    }

    fn track_solution(&self) -> bool {
        self.track_solution
    }

    fn solution(&self) -> Vec<usize> {
        self.solution.clone()
    }
}

// --------------- PyO3 bindings ---------------

#[pyclass(name = "RoutingEnv", extends = PyBaseEnv)]
pub struct PyRoutingEnv;

#[pymethods]
impl PyRoutingEnv {
    #[new]
    #[pyo3(signature = (
        num_qubits,
        coupling_map,
        num_active_swaps = 16,
        horizon = 8,
        difficulty = 1,
        depth_slope = 2,
        max_depth = 128,
        obs_bins = None,
        metrics_weights = None,
        track_solution = None,
    ))]
    pub fn new(
        num_qubits: usize,
        coupling_map: Vec<(usize, usize)>,
        num_active_swaps: usize,
        horizon: usize,
        difficulty: usize,
        depth_slope: usize,
        max_depth: usize,
        obs_bins: Option<usize>,
        metrics_weights: Option<HashMap<String, f32>>,
        track_solution: Option<bool>,
    ) -> (Self, PyBaseEnv) {
        let weights = MetricsWeights::from_hashmap(metrics_weights);
        let env = RoutingEnv::new(
            num_qubits,
            coupling_map,
            num_active_swaps,
            horizon,
            difficulty,
            depth_slope,
            max_depth,
            obs_bins.unwrap_or(7),
            weights,
            track_solution.unwrap_or(true),
        );
        (PyRoutingEnv, PyBaseEnv { env: Box::new(env) })
    }
}

// --------------- Utility: BFS all-pairs shortest paths ---------------

fn compute_all_pairs_distances(
    num_qubits: usize,
    coupling_map: &[(usize, usize)],
) -> Vec<Vec<DistType>> {
    use std::collections::VecDeque;

    // Build adjacency list
    let mut adj = vec![Vec::new(); num_qubits];
    for &(a, b) in coupling_map {
        adj[a].push(b);
        adj[b].push(a);
    }

    let mut dists = vec![vec![DistType::MAX; num_qubits]; num_qubits];

    for src in 0..num_qubits {
        dists[src][src] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(src);
        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if dists[src][v] == DistType::MAX {
                    dists[src][v] = dists[src][u] + 1;
                    queue.push_back(v);
                }
            }
        }
    }

    dists
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_coupling(n: usize) -> Vec<(usize, usize)> {
        (0..n - 1).map(|i| (i, i + 1)).collect()
    }

    #[test]
    fn test_bfs_distances() {
        let cmap = line_coupling(5);
        let dists = compute_all_pairs_distances(5, &cmap);
        assert_eq!(dists[0][0], 0);
        assert_eq!(dists[0][1], 1);
        assert_eq!(dists[0][4], 4);
        assert_eq!(dists[1][3], 2);
    }

    #[test]
    fn test_routing_env_basic() {
        let cmap = line_coupling(5);
        let weights = MetricsWeights::default();
        let mut env = RoutingEnv::new(5, cmap, 8, 4, 2, 2, 64, 5, weights, false);

        env.reset();

        let shape = env.obs_shape();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0], 8 * 4); // num_active_swaps * horizon
        assert_eq!(shape[1], 11); // 2*5+1

        let obs = env.observe();
        assert_eq!(obs.len(), 8 * 4); // one index per cell

        let masks = env.masks();
        assert_eq!(masks.len(), 8);
    }

    #[test]
    fn test_routing_env_trivial() {
        // Difficulty 0 should produce an immediately solved env
        let cmap = line_coupling(3);
        let weights = MetricsWeights::default();
        let mut env = RoutingEnv::new(3, cmap, 4, 4, 0, 2, 64, 5, weights, false);

        env.reset();
        assert!(env.success());
        assert!(env.is_final());
        assert_eq!(env.reward(), 1.0);
    }

    #[test]
    fn test_routing_env_step() {
        let cmap = line_coupling(5);
        let weights = MetricsWeights::default();
        let mut env = RoutingEnv::new(5, cmap, 8, 4, 5, 4, 128, 5, weights, true);

        env.reset();
        if !env.is_final() {
            let masks = env.masks();
            // Find first valid action
            if let Some(action) = masks.iter().position(|&m| m) {
                env.step(action);
                // Should still be functional
                let _obs = env.observe();
                assert!(env.solution().len() <= 1);
            }
        }
    }

    #[test]
    fn test_set_state() {
        let cmap = line_coupling(5);
        let weights = MetricsWeights::default();
        let mut env = RoutingEnv::new(5, cmap, 8, 4, 1, 2, 64, 5, weights, false);

        // Set state with a single gate between qubits 0 and 4 (distance 4)
        env.set_state(vec![0, 4]);
        assert!(!env.success()); // not adjacent, needs routing
        assert!(!env.is_final());
    }
}
