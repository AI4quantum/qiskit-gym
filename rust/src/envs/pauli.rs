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
use crate::pauli::pauli_network::{Axis, PauliNetwork};
use nalgebra::DMatrix;
use std::collections::{HashMap, HashSet, VecDeque, BTreeMap};

/// Represents a step in the Pauli network synthesis solution.
/// Can be either a gate action or a rotation that became trivial.
#[derive(Clone, Debug)]
pub enum SolutionStep {
    /// A gate action (index into gateset)
    Gate(usize),
    /// A rotation that became trivial: (axis, qubit, rotation_index, phase_multiplier)
    /// phase_multiplier is 1 or -1 depending on the phase
    Rotation {
        axis: Axis,
        qubit: usize,
        index: usize,
        phase_mult: i32,
    },
}

fn identity_tableau(num_qubits: usize) -> Vec<u8> {
    let dim = 2 * num_qubits;
    let mut data = vec![0u8; dim * dim];
    for i in 0..dim {
        data[i * dim + i] = 1;
    }
    data
}

/// Compute shortest path distances between all pairs of qubits using BFS
/// Returns a map from (q1, q2) -> distance
fn compute_graph_distances(num_qubits: usize, edges: &[(usize, usize)]) -> HashMap<(usize, usize), usize> {
    // Build adjacency list
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); num_qubits];
    for &(q1, q2) in edges {
        if !adj[q1].contains(&q2) {
            adj[q1].push(q2);
        }
        if !adj[q2].contains(&q1) {
            adj[q2].push(q1);
        }
    }

    let mut distances = HashMap::new();

    // BFS from each node
    for start in 0..num_qubits {
        let mut visited = vec![false; num_qubits];
        let mut queue = VecDeque::new();
        queue.push_back((start, 0usize));
        visited[start] = true;

        while let Some((node, dist)) = queue.pop_front() {
            distances.insert((start, node), dist);
            distances.insert((node, start), dist);

            for &neighbor in &adj[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back((neighbor, dist + 1));
                }
            }
        }
    }

    distances
}

/// Build dist_pairs: map from distance -> list of qubit pairs at that distance
/// Also returns all_dists: sorted unique distances
fn build_dist_pairs(
    num_qubits: usize,
    distances: &HashMap<(usize, usize), usize>,
) -> (BTreeMap<usize, Vec<(usize, usize)>>, Vec<usize>) {
    let mut dist_pairs: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();

    for q1 in 0..num_qubits {
        for q2 in (q1 + 1)..num_qubits {
            if let Some(&dist) = distances.get(&(q1, q2)) {
                dist_pairs.entry(dist).or_default().push((q1, q2));
            }
        }
    }

    let all_dists: Vec<usize> = dist_pairs.keys().copied().collect();
    (dist_pairs, all_dists)
}

/// Generate a single Pauli rotation using difficulty budget (matches Python _get_pauli_under_diff)
/// Returns (pauli_string, cost) or None if can't generate
fn get_pauli_under_diff(
    num_qubits: usize,
    difficulty: usize,
    dist_pairs: &BTreeMap<usize, Vec<(usize, usize)>>,
    all_dists: &[usize],
    num_qubits_decay: f32,
) -> Option<(String, usize)> {
    let mut rng = rand::thread_rng();
    let axes = ['X', 'Y', 'Z'];

    // Find valid distances <= difficulty
    let valid_dists: Vec<usize> = all_dists.iter().copied().filter(|&d| d <= difficulty).collect();
    if valid_dists.is_empty() {
        return None;
    }

    let mut qubits: HashSet<usize> = HashSet::new();
    let mut pauli_diff = difficulty;

    // Generate the first pair
    let valid_for_first: Vec<usize> = valid_dists.iter().copied().filter(|&d| d <= pauli_diff).collect();
    if valid_for_first.is_empty() {
        return None;
    }

    let next_dif = valid_for_first[rng.gen_range(0..valid_for_first.len())];
    let pairs = &dist_pairs[&next_dif];
    let (q1, q2) = pairs[rng.gen_range(0..pairs.len())];
    qubits.insert(q1);
    qubits.insert(q2);
    pauli_diff = pauli_diff.saturating_sub(next_dif);

    // Add more qubits with decay probability (matches Python's num_qubits_decay)
    loop {
        let valid_diffs: Vec<usize> = valid_dists.iter().copied().filter(|&d| d <= pauli_diff).collect();
        let remaining_qs: Vec<usize> = (0..num_qubits).filter(|q| !qubits.contains(q)).collect();

        if pauli_diff == 0 || valid_diffs.is_empty() || remaining_qs.is_empty() {
            break;
        }

        // Continue with probability (1 - num_qubits_decay)
        if rng.gen::<f32>() <= num_qubits_decay {
            break;
        }

        let next_dif = valid_diffs[rng.gen_range(0..valid_diffs.len())];

        // Find pairs at this distance that connect to existing qubits
        let valid_pairs: Vec<(usize, usize)> = dist_pairs[&next_dif]
            .iter()
            .copied()
            .filter(|(a, b)| qubits.contains(a) || qubits.contains(b))
            .collect();

        if valid_pairs.is_empty() {
            continue;
        }

        let (q1, q2) = valid_pairs[rng.gen_range(0..valid_pairs.len())];
        qubits.insert(q1);
        qubits.insert(q2);
        pauli_diff = pauli_diff.saturating_sub(next_dif);
    }

    // Build the pauli string
    let mut pauli_layer = vec!['I'; num_qubits];
    for q in &qubits {
        pauli_layer[*q] = axes[rng.gen_range(0..axes.len())];
    }

    let cost = difficulty - pauli_diff;
    Some((pauli_layer.into_iter().collect(), cost))
}

/// Generate random Pauli rotations with random weights (for pauli_difficulty=None case)
/// This matches the Python PauliNetworkGenerator behavior when difficulty is None
fn random_rotations(num_qubits: usize, rotation_count: usize) -> Vec<String> {
    if rotation_count == 0 || num_qubits == 0 {
        return Vec::new();
    }
    let mut rng = rand::thread_rng();
    let axes = ['X', 'Y', 'Z'];

    (0..rotation_count)
        .map(|_| {
            // Generate Pauli label with random weight (1 to num_qubits-1)
            // Matches Python: pauli_weight = np.random.randint(1, self.num_qubits)
            let mut pauli_layer = vec!['I'; num_qubits];
            let pauli_weight = if num_qubits > 1 {
                rng.gen_range(1..num_qubits)
            } else {
                1
            };

            // Choose random positions without replacement (partial Fisher-Yates)
            let mut positions: Vec<usize> = (0..num_qubits).collect();
            for i in 0..pauli_weight {
                let j = rng.gen_range(i..num_qubits);
                positions.swap(i, j);
            }

            // Assign random Pauli types to selected positions
            for &pos in positions.iter().take(pauli_weight) {
                pauli_layer[pos] = axes[rng.gen_range(0..axes.len())];
            }

            pauli_layer.into_iter().collect()
        })
        .collect()
}

/// Generate paulis based on difficulty budget (matches Python's generate() method)
fn generate_paulis_with_difficulty(
    num_qubits: usize,
    pauli_difficulty: usize,
    max_paulis: usize,
    dist_pairs: &BTreeMap<usize, Vec<(usize, usize)>>,
    all_dists: &[usize],
    num_qubits_decay: f32,
) -> Vec<String> {
    let mut paulis = Vec::new();
    let mut remaining_diff = pauli_difficulty;

    while remaining_diff > 0 && paulis.len() < max_paulis {
        match get_pauli_under_diff(num_qubits, remaining_diff, dist_pairs, all_dists, num_qubits_decay) {
            Some((pauli, cost)) => {
                paulis.push(pauli);
                remaining_diff = remaining_diff.saturating_sub(cost.max(1));
            }
            None => break,
        }
    }

    paulis
}

/// Generate a random Clifford tableau by applying random H/S/CX gates to identity
/// This matches the Python PauliNetworkGenerator.generate() behavior:
/// - 70% probability of CX
/// - 15% probability of H
/// - 15% probability of S
fn random_clifford_tableau(num_qubits: usize, difficulty: usize, valid_pairs: &[(usize, usize)]) -> Vec<u8> {
    let dim = 2 * num_qubits;
    let mut data = vec![0u8; dim * dim];

    // Start with identity
    for i in 0..dim {
        data[i * dim + i] = 1;
    }

    if difficulty == 0 || valid_pairs.is_empty() {
        return data;
    }

    let mut rng = rand::thread_rng();

    // Helper to XOR row_a with row_b (row_a ^= row_b)
    let xor_rows = |data: &mut Vec<u8>, dim: usize, row_a: usize, row_b: usize| {
        for col in 0..dim {
            data[row_a * dim + col] ^= data[row_b * dim + col];
        }
    };

    // Helper to swap rows
    let swap_rows = |data: &mut Vec<u8>, dim: usize, row_a: usize, row_b: usize| {
        for col in 0..dim {
            let tmp = data[row_a * dim + col];
            data[row_a * dim + col] = data[row_b * dim + col];
            data[row_b * dim + col] = tmp;
        }
    };

    for _ in 0..difficulty {
        let r = rng.gen::<f32>();
        if r > 0.3 {
            // 70% CX
            let (q0, q1) = valid_pairs[rng.gen_range(0..valid_pairs.len())];
            // CX conjugation: X_target += X_control, Z_control += Z_target
            xor_rows(&mut data, dim, q1, q0); // target row += control row
            xor_rows(&mut data, dim, num_qubits + q0, num_qubits + q1); // control+n += target+n
        } else if r > 0.15 {
            // 15% H
            let q = rng.gen_range(0..num_qubits);
            swap_rows(&mut data, dim, q, num_qubits + q);
        } else {
            // 15% S
            let q = rng.gen_range(0..num_qubits);
            xor_rows(&mut data, dim, num_qubits + q, q);
        }
    }

    data
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
    pub pauli_diff_scale: usize,
    pub num_qubits_decay: f32,
    pub final_pauli_layers: usize,
    // Graph distance data for pauli generation
    dist_pairs: BTreeMap<usize, Vec<(usize, usize)>>,
    all_dists: Vec<usize>,
    valid_pairs: Vec<(usize, usize)>,
    pub obs_perms: Vec<Vec<usize>>,
    pub act_perms: Vec<Vec<usize>>,
    metrics: MetricsTracker,
    metrics_values: MetricsCounts,
    metrics_weights: MetricsWeights,
    reward_value: f32,
    pauli_layer_reward: f32,
    track_solution: bool,
    solution: Vec<SolutionStep>,
}

impl PauliEnv {
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        gateset: Vec<Gate>,
        depth_slope: usize,
        max_depth: usize,
        max_rotations: usize,
        pauli_diff_scale: usize,
        num_qubits_decay: f32,
        final_pauli_layers: usize,
        metrics_weights: MetricsWeights,
        add_perms: bool,
        pauli_layer_reward: f32,
        track_solution: bool,
    ) -> Self {
        let tableau = identity_tableau(num_qubits);
        let network = PauliNetwork::new(tableau, Vec::new());
        let success = network.solved();

        // Extract valid CX pairs from gateset (matches Python's valid_pairs)
        let valid_pairs: Vec<(usize, usize)> = gateset
            .iter()
            .filter_map(|gate| match gate {
                Gate::CX(q0, q1) => Some((*q0, *q1)),
                _ => None,
            })
            .collect();

        // Compute graph distances from coupling map
        let distances = compute_graph_distances(num_qubits, &valid_pairs);
        let (dist_pairs, all_dists) = build_dist_pairs(num_qubits, &distances);

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
            pauli_diff_scale: pauli_diff_scale.max(1),
            num_qubits_decay,
            final_pauli_layers,
            dist_pairs,
            all_dists,
            valid_pairs,
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
        if self.track_solution {
            self.solution = Vec::new();
        }
    }

    fn reset(&mut self) {
        // Compute pauli_difficulty (matches Python's lf_diff_scale logic)
        // pauli_difficulty = difficulty // pauli_diff_scale
        let pauli_difficulty = self.difficulty / self.pauli_diff_scale;

        // Generate paulis based on difficulty budget (matches Python exactly)
        let rotations = generate_paulis_with_difficulty(
            self.num_qubits,
            pauli_difficulty,
            self.final_pauli_layers,
            &self.dist_pairs,
            &self.all_dists,
            self.num_qubits_decay,
        );

        // Generate random Clifford tableau directly (matches Python's generator.generate())
        // This applies random H/S/CX to identity, not via step()
        let tableau = random_clifford_tableau(self.num_qubits, self.difficulty, &self.valid_pairs);

        self.rebuild_network(tableau, rotations);

        // Clean initially trivial rotations (like Python does)
        self.network.clean_and_return_with_phases();

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

            // act() returns rotations that became trivial: (Axis, qubit, rotation_index)
            let applied_rotations = self.network.act(&gate);
            new_rotations = applied_rotations.len();

            if self.track_solution {
                // Record the gate action
                self.solution.push(SolutionStep::Gate(action));

                // Record each rotation that became trivial with its phase
                for (axis, qubit, rot_idx) in applied_rotations.iter() {
                    let phase = self.network.rotation_qk[*rot_idx].phase();
                    let phase_mult = if phase == 2 { -1 } else { 1 };
                    self.solution.push(SolutionStep::Rotation {
                        axis: axis.clone(),
                        qubit: *qubit,
                        index: *rot_idx,
                        phase_mult,
                    });
                }
            }
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
        // Encode both gates and rotations in Vec<usize> using a marker scheme:
        // - Gate actions: just the action index (< ROTATION_MARKER)
        // - Rotations: ROTATION_MARKER | axis(2 bits) | qubit(10 bits) | index(10 bits) | phase(1 bit)
        const ROTATION_MARKER: usize = 0x80000000; // 2^31

        self.solution
            .iter()
            .map(|step| match step {
                SolutionStep::Gate(action) => *action,
                SolutionStep::Rotation { axis, qubit, index, phase_mult } => {
                    let axis_code: usize = match axis {
                        Axis::X => 0,
                        Axis::Y => 1,
                        Axis::Z => 2,
                    };
                    let phase_code: usize = if *phase_mult == 1 { 1 } else { 0 };
                    ROTATION_MARKER
                        | (axis_code << 21)  // bits 21-22
                        | (*qubit << 11)     // bits 11-20
                        | (*index << 1)      // bits 1-10
                        | phase_code         // bit 0
                }
            })
            .collect()
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
        pauli_diff_scale=None,
        num_qubits_decay=None,
        final_pauli_layers=None,
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
        pauli_diff_scale: Option<usize>,
        num_qubits_decay: Option<f32>,
        final_pauli_layers: Option<usize>,
        metrics_weights: Option<HashMap<String, f32>>,
        add_perms: Option<bool>,
        pauli_layer_reward: Option<f32>,
        track_solution: Option<bool>,
    ) -> (Self, PyBaseEnv) {
        let weights = MetricsWeights::from_hashmap(metrics_weights);
        // Default final_pauli_layers to max_rotations + 2 (like Python's horizon + 2)
        let final_layers = final_pauli_layers.unwrap_or(max_rotations + 2);
        let env = PauliEnv::new(
            num_qubits,
            difficulty,
            gateset,
            depth_slope,
            max_depth,
            max_rotations,
            pauli_diff_scale.unwrap_or(8), // Default matches Python's lf_diff_scale
            num_qubits_decay.unwrap_or(0.5), // Default matches Python
            final_layers,
            weights,
            add_perms.unwrap_or(true),
            pauli_layer_reward.unwrap_or(0.01),
            track_solution.unwrap_or(true),
        );
        let env = Box::new(env);
        (PyPauliEnv, PyBaseEnv { env })
    }
}
