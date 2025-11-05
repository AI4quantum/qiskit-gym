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

    fn row_xor(&mut self, dest: usize, src: usize) {
        if dest == src {
            return;
        }
        for col in 0..self.size {
            let dest_idx = self.index(dest, col);
            let src_idx = self.index(src, col);
            self.data[dest_idx] ^= self.data[src_idx];
        }
    }

    fn swap_rows(&mut self, r1: usize, r2: usize) {
        if r1 == r2 {
            return;
        }
        for col in 0..self.size {
            let i1 = self.index(r1, col);
            let i2 = self.index(r2, col);
            self.data.swap(i1, i2);
        }
    }

    fn inverse(&self) -> Self {
        let mut mat = self.clone();
        let mut inv = LFState::new(self.size);

        for col in 0..self.size {
            if !mat.get(col, col) {
                let pivot = ((col + 1)..self.size).find(|&row| mat.get(row, col));
                let pivot = pivot.expect("LFState is singular; cannot invert");
                mat.swap_rows(col, pivot);
                inv.swap_rows(col, pivot);
            }

            for row in 0..self.size {
                if row != col && mat.get(row, col) {
                    mat.row_xor(row, col);
                    inv.row_xor(row, col);
                }
            }
        }

        debug_assert!(mat.solved(), "LFState inverse computation failed");
        inv
    }

    fn invert(&mut self) {
        *self = self.inverse();
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
    add_inverts: bool,
    track_solution: bool,
    solution: Vec<usize>,
    solution_inv: Vec<usize>,
    inverted: bool,
}


impl LinearFunction {
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
        let lf = LFState::new(num_qubits);
        let success = lf.solved();

        // Only compute symmetries if enabled
        let (obs_perms, act_perms) = if add_perms {
            compute_twists_square(num_qubits, &gateset)
        } else {
            (Vec::new(), Vec::new())
        };

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
            add_inverts,
            track_solution,
            solution: Vec::new(),
            solution_inv: Vec::new(),
            inverted: false,
        }
    }
    pub fn solved(&self) -> bool {
        self.lf.solved()
    }

    fn maybe_random_invert(&mut self) {
        if !self.add_inverts {
            return;
        }
        if rand::thread_rng().gen_bool(0.5) {
            self.lf.invert();
            self.inverted = !self.inverted;
        }
    }

    fn apply_gate_to_state(&mut self, gate: &Gate) {
        match gate {
            &Gate::CX(q1, q2) => self.lf.cx(q1, q2),
            &Gate::SWAP(q1, q2) => self.lf.swap(q1, q2),
            _ => {}
        }
    }

    fn reset_internals(&mut self) {
        self.success = self.solved();
        self.metrics.reset();
        self.metrics_values = self.metrics.snapshot();
        self.reward_value = if self.success { 1.0 } else { 0.0 };
        self.inverted = false;
        if self.track_solution {
            self.solution_inv = Vec::new();
            self.solution = Vec::new();
        }
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
        self.reset_internals();
    }

    fn reset(&mut self) {
        // Create an identity matrix for the initial 'lf' state
        self.lf = LFState::new(self.lf.size);
        let mut rng = rand::thread_rng();
        let action_range = Uniform::new(0, self.num_actions());

        // Apply random actions based on the difficulty
        for _ in 0..self.difficulty {
            let action = action_range.sample(&mut rng);
            if let Some(gate) = self.gateset.get(action).cloned() {
                self.apply_gate_to_state(&gate);
            }
        }
        self.depth = (self.depth_slope * self.difficulty).min(self.max_depth); 
        self.reset_internals();
    }

    fn step(&mut self, action: usize)  {
        let mut penalty = 0.0f32;

         if let Some(gate) = self.gateset.get(action).cloned() {
            let previous = self.metrics_values.clone();
            self.metrics.apply_gate(&gate);
            let new_metrics = self.metrics.snapshot();
            penalty = new_metrics.weighted_delta(&previous, &self.metrics_weights);
            self.metrics_values = new_metrics;

            self.apply_gate_to_state(&gate);
        }

        if self.track_solution {
           if self.inverted {
               self.solution_inv.push(action);
            } else {
                self.solution.push(action);
            }
        }

        self.depth = self.depth.saturating_sub(1); // Prevent underflow
        self.maybe_random_invert();
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

#[pyclass(name="LinearFunctionEnv", extends=PyBaseEnv)]
pub struct PyLinearFunctionEnv;

#[pymethods]
impl PyLinearFunctionEnv {
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
        let env = LinearFunction::new(
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
        (PyLinearFunctionEnv, PyBaseEnv { env })
    }
}

#[derive(Hash, Eq, PartialEq, Clone, Copy)]
enum GateKind {
    H,
    S,
    Sdg,
    SX,
    SXdg,
    CX,
    CZ,
    Swap,
}

#[derive(Hash, Eq, PartialEq, Clone)]
struct GateKey {
    kind: GateKind,
    qubits: Vec<usize>,
}

fn gate_kind(gate: &Gate) -> GateKind {
    match gate {
        Gate::H(_) => GateKind::H,
        Gate::S(_) => GateKind::S,
        Gate::Sdg(_) => GateKind::Sdg,
        Gate::SX(_) => GateKind::SX,
        Gate::SXdg(_) => GateKind::SXdg,
        Gate::CX(_, _) => GateKind::CX,
        Gate::CZ(_, _) => GateKind::CZ,
        Gate::SWAP(_, _) => GateKind::Swap,
    }
}

fn gate_qubits(gate: &Gate) -> Vec<usize> {
    match gate {
        Gate::H(q)
        | Gate::S(q)
        | Gate::Sdg(q)
        | Gate::SX(q)
        | Gate::SXdg(q) => vec![*q],
        Gate::CX(q1, q2)
        | Gate::CZ(q1, q2)
        | Gate::SWAP(q1, q2) => vec![*q1, *q2],
    }
}

fn canonical_key(kind: GateKind, mut qubits: Vec<usize>) -> GateKey {
    if matches!(kind, GateKind::Swap) {
        qubits.sort_unstable();
    }
    GateKey { kind, qubits }
}

fn two_qubit_targets(gate: &Gate) -> Option<(usize, usize)> {
    match gate {
        Gate::CX(q1, q2) | Gate::CZ(q1, q2) | Gate::SWAP(q1, q2) => Some((*q1, *q2)),
        _ => None,
    }
}

fn identity_obs_perm(num_qubits: usize) -> Vec<usize> {
    let mut obs_perm = Vec::with_capacity(num_qubits * num_qubits);
    for row in 0..num_qubits {
        for col in 0..num_qubits {
            obs_perm.push(row * num_qubits + col);
        }
    }
    obs_perm
}

fn compute_twists(num_qubits: usize, gateset: &[Gate]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    if num_qubits == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut gate_index: HashMap<GateKey, usize> = HashMap::new();
    for (idx, gate) in gateset.iter().enumerate() {
        let kind = gate_kind(gate);
        let qubits = gate_qubits(gate);
        let key = canonical_key(kind, qubits);
        gate_index.insert(key, idx);
    }

    let mut adjacency = vec![vec![false; num_qubits]; num_qubits];
    let mut has_edge = false;

    for gate in gateset {
        if let Some((q1, q2)) = two_qubit_targets(gate) {
            if q1 != q2 {
                adjacency[q1][q2] = true;
                adjacency[q2][q1] = true;
                has_edge = true;
            }
        }
    }

    if !has_edge {
        let obs_perm = identity_obs_perm(num_qubits);
        let act_perm: Vec<usize> = (0..gateset.len()).collect();
        return (vec![obs_perm], vec![act_perm]);
    }

    let degrees: Vec<usize> = adjacency
        .iter()
        .map(|row| row.iter().filter(|&&edge| edge).count())
        .collect();

    let signatures: Vec<NodeSignature> = (0..num_qubits)
        .map(|idx| NodeSignature::new(idx, &degrees, &adjacency))
        .collect();

    let automorphisms = enumerate_automorphisms(&adjacency, &signatures);

    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    let mut obs_perms: Vec<Vec<usize>> = Vec::new();
    let mut act_perms: Vec<Vec<usize>> = Vec::new();

    for mapping in automorphisms {
        if mapping.len() != num_qubits {
            continue;
        }
        if !seen.insert(mapping.clone()) {
            continue;
        }

        let mut obs_perm = vec![0usize; num_qubits * num_qubits];
        for row in 0..num_qubits {
            for col in 0..num_qubits {
                let idx_old = row * num_qubits + col;
                obs_perm[idx_old] = mapping[row] * num_qubits + mapping[col];
            }
        }

        let mut act_perm: Vec<usize> = Vec::with_capacity(gateset.len());
        let mut valid = true;
        for gate in gateset {
            let kind = gate_kind(gate);
            let mut qubits = gate_qubits(gate);
            for q in qubits.iter_mut() {
                *q = mapping[*q];
            }
            let key = canonical_key(kind, qubits);
            if let Some(idx) = gate_index.get(&key) {
                act_perm.push(*idx);
            } else {
                valid = false;
                break;
            }
        }

        if valid {
            obs_perms.push(obs_perm);
            act_perms.push(act_perm);
        }
    }

    if obs_perms.is_empty() {
        let obs_perm = identity_obs_perm(num_qubits);
        let act_perm: Vec<usize> = (0..gateset.len()).collect();
        obs_perms.push(obs_perm);
        act_perms.push(act_perm);
    }

    (obs_perms, act_perms)
}

#[derive(Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct NodeSignature {
    degree: usize,
    neighbor_degrees: Vec<usize>,
}

impl NodeSignature {
    fn new(node: usize, degrees: &[usize], adjacency: &[Vec<bool>]) -> Self {
        let mut neighbor_degrees: Vec<usize> = adjacency[node]
            .iter()
            .enumerate()
            .filter_map(|(idx, &connected)| if connected { Some(degrees[idx]) } else { None })
            .collect();
        neighbor_degrees.sort_unstable();
        Self {
            degree: degrees[node],
            neighbor_degrees,
        }
    }
}

fn enumerate_automorphisms(
    adjacency: &[Vec<bool>],
    signatures: &[NodeSignature],
) -> Vec<Vec<usize>> {
    let n = adjacency.len();
    let mut nodes_order: Vec<usize> = (0..n).collect();
    nodes_order.sort_by(|&a, &b| signatures[a].cmp(&signatures[b]));

    let mut perm = vec![usize::MAX; n];
    let mut used = vec![false; n];
    let mut results = Vec::new();

    backtrack_automorphisms(
        0,
        &nodes_order,
        signatures,
        adjacency,
        &mut perm,
        &mut used,
        &mut results,
    );

    results
}

fn backtrack_automorphisms(
    idx: usize,
    nodes_order: &[usize],
    signatures: &[NodeSignature],
    adjacency: &[Vec<bool>],
    perm: &mut Vec<usize>,
    used: &mut Vec<bool>,
    results: &mut Vec<Vec<usize>>,
) {
    if idx == nodes_order.len() {
        results.push(perm.clone());
        return;
    }

    let node_from = nodes_order[idx];
    let target_signature = &signatures[node_from];
    let n = adjacency.len();

    for node_to in 0..n {
        if used[node_to] || &signatures[node_to] != target_signature {
            continue;
        }

        let mut consistent = true;
        for prev_idx in 0..idx {
            let prev_from = nodes_order[prev_idx];
            let prev_to = perm[prev_from];
            if prev_to == usize::MAX {
                continue;
            }
            if adjacency[node_from][prev_from] != adjacency[node_to][prev_to] {
                consistent = false;
                break;
            }
        }

        if !consistent {
            continue;
        }

        perm[node_from] = node_to;
        used[node_to] = true;
        backtrack_automorphisms(
            idx + 1,
            nodes_order,
            signatures,
            adjacency,
            perm,
            used,
            results,
        );
        used[node_to] = false;
        perm[node_from] = usize::MAX;
    }
}
