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
use crate::envs::symmetry::compute_twists_clifford;
use std::collections::HashMap;


#[derive(Clone)]
pub struct CFState {
    pub data: Vec<bool>, // flattened 2N x 2N symplectic matrix, row-major
    n: usize,            // number of qubits
}

impl CFState {
    fn new(n: usize) -> Self {
        let dim = 2 * n;
        let mut data = vec![false; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = true; // identity
        }
        Self { data, n }
    }

    #[inline]
    fn dim(&self) -> usize { 2 * self.n }

    #[inline]
    fn index(&self, row: usize, col: usize) -> usize {
        row * self.dim() + col
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> bool {
        self.data[self.index(row, col)]
    }

    #[inline]
    fn set(&mut self, row: usize, col: usize, val: bool) {
        let idx = self.index(row, col);
        self.data[idx] = val;
    }

    // Row ops over GF(2)
    fn row_xor(&mut self, dest: usize, src: usize) {
        if dest == src { return; }
        let dim = self.dim();
        let d_off = dest * dim;
        let s_off = src * dim;
        for c in 0..dim {
            self.data[d_off + c] ^= self.data[s_off + c];
        }
    }

    fn swap_rows(&mut self, r1: usize, r2: usize) {
        if r1 == r2 { return; }
        let dim = self.dim();
        for c in 0..dim {
            let i1 = r1 * dim + c;
            let i2 = r2 * dim + c;
            self.data.swap(i1, i2);
        }
    }

    // --- Clifford generators on the tableau (phase ignored) ---
    // We left-multiply the tableau M by each gate's symplectic matrix,
    // which corresponds to these row operations.

    // H(i): (x,z) -> (z,x)
    fn h(&mut self, q: usize) {
        let n = self.n;
        self.swap_rows(q, n + q);
    }

    // S(i): (x,z) -> (x, x ⊕ z); Sdg is identical mod global phases.
    fn s(&mut self, q: usize) {
        let n = self.n;
        self.row_xor(n + q, q);
    }
    fn sdg(&mut self, q: usize) { self.s(q); }

    // SX(i) = H S H (ignoring phase): (x,z) -> (x ⊕ z, z); SXdg identical when phases ignored.
    fn sx(&mut self, q: usize) {
        let n = self.n;
        self.row_xor(q, n + q);
    }
    fn sxdg(&mut self, q: usize) { self.sx(q); }

    // CX(c, t):
    // x_t' = x_t ⊕ x_c   => row_X_t ^= row_X_c
    // z_c' = z_c ⊕ z_t   => row_Z_c ^= row_Z_t
    fn cx(&mut self, c: usize, t: usize) {
        if c == t { return; }
        let n = self.n;
        self.row_xor(t, c);         // X-rows
        self.row_xor(n + c, n + t); // Z-rows
    }

    // CZ(a, b):
    // z_a' = z_a ⊕ x_b ; z_b' = z_b ⊕ x_a
    fn cz(&mut self, a: usize, b: usize) {
        if a == b { return; }
        let n = self.n;
        self.row_xor(n + a, b);
        self.row_xor(n + b, a);
    }

    // SWAP(a, b): swap both X and Z row pairs
    fn swap(&mut self, a: usize, b: usize) {
        if a == b { return; }
        let n = self.n;
        self.swap_rows(a, b);
        self.swap_rows(n + a, n + b);
    }

    // Identity check
    fn solved(&self) -> bool {
        let dim = self.dim();
        for i in 0..dim {
            for j in 0..dim {
                let want = i == j;
                if self.get(i, j) != want { return false; }
            }
        }
        true
    }

    fn inverse(&self) -> Self {
        let dim = self.dim();
        let mut mat = self.clone();
        let mut inv = CFState::new(self.n);

        for col in 0..dim {
            if !mat.get(col, col) {
                let pivot = ((col + 1)..dim).find(|&row| mat.get(row, col));
                let pivot = pivot.expect("CFState is singular; cannot invert");
                mat.swap_rows(col, pivot);
                inv.swap_rows(col, pivot);
            }

            for row in 0..dim {
                if row != col && mat.get(row, col) {
                    mat.row_xor(row, col);
                    inv.row_xor(row, col);
                }
            }
        }

        debug_assert!(mat.solved(), "CFState inverse computation failed");
        inv
    }

    fn invert(&mut self) {
        *self = self.inverse();
    }
}

// -------- Env: Clifford synthesis over the symplectic tableau (phase ignored) --------

#[derive(Clone)]
pub struct Clifford {
    pub cf: CFState,
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

impl Clifford {
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
        let cf = CFState::new(num_qubits);
        let success = cf.solved();

        // Only compute symmetries if enabled
        let (obs_perms, act_perms) = if add_perms {
            compute_twists_clifford(num_qubits, &gateset)
        } else {
            (Vec::new(), Vec::new())
        };

        let metrics = MetricsTracker::new(num_qubits);
        let metrics_values = metrics.snapshot();
        Clifford {
            cf,
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
    pub fn solved(&self) -> bool { self.cf.solved() }

    fn apply_gate_to_state(&mut self, gate: &Gate) {
        match gate {
            Gate::H(q) => self.cf.h(*q),
            Gate::S(q) => self.cf.s(*q),
            Gate::Sdg(q) => self.cf.sdg(*q), // identical to S modulo global phase (ignored)
            Gate::SX(q) => self.cf.sx(*q),
            Gate::SXdg(q) => self.cf.sxdg(*q), // identical to SX modulo global phase (ignored)
            Gate::CX(c, t) => self.cf.cx(*c, *t),
            Gate::CZ(a, b) => self.cf.cz(*a, *b),
            Gate::SWAP(a, b) => self.cf.swap(*a, *b),
        }
    }

    fn maybe_random_invert(&mut self) {
        if !self.add_inverts {
            return;
        }
        if rand::thread_rng().gen_bool(0.5) {
            self.cf.invert();
            self.inverted = !self.inverted;
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

impl Env for Clifford {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn num_actions(&self) -> usize { self.gateset.len() }

    fn obs_shape(&self) -> Vec<usize> {
        let d = self.cf.dim();
        vec![d, d] // 2N x 2N tableau (phase ignored)
    }

    fn set_difficulty(&mut self, difficulty: usize) { self.difficulty = difficulty; }
    fn get_difficulty(&self) -> usize { self.difficulty }

    fn set_state(&mut self, state: Vec<i64>) {
        // Expecting a flattened 2N x 2N boolean matrix encoded as i64s (>0 => true)
        self.cf.data = state.iter().map(|&x| x > 0).collect();
        self.depth = self.max_depth;
        self.reset_internals();
    }

    fn reset(&mut self) {
        self.cf = CFState::new(self.cf.n);
        let mut rng = rand::thread_rng();
        let action_range = Uniform::new(0, self.num_actions());

        for _ in 0..self.difficulty {
            let action = action_range.sample(&mut rng);
            if let Some(gate) = self.gateset.get(action).cloned() {
                self.apply_gate_to_state(&gate);
            }
        }
        self.depth = (self.depth_slope * self.difficulty).min(self.max_depth);
        self.reset_internals();
    }

    fn step(&mut self, action: usize) {
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

        self.depth = self.depth.saturating_sub(1);
        self.maybe_random_invert();
        self.success = self.solved();
        let achieved = if self.success { 1.0 } else { 0.0 };
        self.reward_value = achieved - penalty;
    }

    fn masks(&self) -> Vec<bool> {
        vec![!self.success; self.num_actions()]
    }

    fn is_final(&self) -> bool { self.depth == 0 || self.success }

    fn reward(&self) -> f32 { self.reward_value }

    fn success(&self) -> bool {
        self.success
    }

    fn success(&self) -> bool {
        self.success
    }

    fn observe(&self) -> Vec<usize> {
        self.cf
            .data
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i) } else { None })
            .collect()
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

#[pyclass(name="CliffordEnv", extends=PyBaseEnv)]
pub struct PyCliffordEnv;

#[pymethods]
impl PyCliffordEnv {
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
        let env = Clifford::new(
            num_qubits,
            difficulty,
            gateset,
            depth_slope,
            max_depth,
            weights,
            add_inverts.unwrap_or(true),
            add_perms.unwrap_or(true),
            track_solution.unwrap_or(true),
        );
        let env = Box::new(env);
        (PyCliffordEnv, PyBaseEnv { env })
    }
}
