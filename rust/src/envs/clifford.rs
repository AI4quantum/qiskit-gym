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
use twisterl::python_interface::env::{PyBaseEnv, get_env_ref, get_env_mut};

use crate::envs::common::Gate;


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
}

impl Clifford {
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        gateset: Vec<Gate>,
        depth_slope: usize,
        max_depth: usize,
    ) -> Self {
        let cf = CFState::new(num_qubits);
        let success = cf.solved();
        Clifford { cf, depth: 1, success, difficulty, gateset, depth_slope, max_depth }
    }
    pub fn solved(&self) -> bool { self.cf.solved() }
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
        self.success = self.solved();
    }

    fn reset(&mut self) {
        self.cf = CFState::new(self.cf.n);
        self.depth = self.max_depth;
        self.success = self.solved();

        let mut rng = rand::thread_rng();
        let action_range = Uniform::new(0, self.num_actions());

        for _ in 0..self.difficulty {
            let action = action_range.sample(&mut rng);
            self.step(action);
        }
        self.depth = (self.depth_slope * self.difficulty).min(self.max_depth);
        self.success = self.solved();
    }

    fn step(&mut self, action: usize) {
        match self.gateset[action] {
            Gate::H(q)      => self.cf.h(q),
            Gate::S(q)      => self.cf.s(q),
            Gate::Sdg(q)    => self.cf.sdg(q),   // identical to S modulo global phase (ignored)
            Gate::SX(q)     => self.cf.sx(q),
            Gate::SXdg(q)   => self.cf.sxdg(q),  // identical to SX modulo global phase (ignored)
            Gate::CX(c, t)  => self.cf.cx(c, t),
            Gate::CZ(a, b)  => self.cf.cz(a, b),
            Gate::SWAP(a,b) => self.cf.swap(a, b),
        }
        self.depth = self.depth.saturating_sub(1);
        self.success = self.solved();
    }

    fn masks(&self) -> Vec<bool> {
        vec![!self.success; self.num_actions()]
    }

    fn is_final(&self) -> bool { self.depth == 0 || self.success }

    fn reward(&self) -> f32 {
        if self.success {
            1.0
        } else if self.depth == 0 {
            -0.5
        } else {
            -0.5 / (self.max_depth as f32)
        }
    }

    fn observe(&self) -> Vec<usize> {
        self.cf
            .data
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i) } else { None })
            .collect()
    }
}

#[pyclass(name="CliffordEnv", extends=PyBaseEnv)]
pub struct PyCliffordEnv;

#[pymethods]
impl PyCliffordEnv {
    #[new]
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        gateset: Vec<Gate>,
        depth_slope: usize,
        max_depth: usize
    ) -> (Self, PyBaseEnv) {
        let env = Clifford::new(num_qubits, difficulty, gateset, depth_slope, max_depth);
        let env = Box::new(env);
        (PyCliffordEnv, PyBaseEnv { env })
    }
}