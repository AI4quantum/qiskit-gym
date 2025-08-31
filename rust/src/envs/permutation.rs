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
    pub max_depth: usize
}


impl Permutation {
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        gateset: Vec<Gate>,
        depth_slope: usize,
        max_depth: usize,
    ) -> Self {
        Permutation {state:(0..num_qubits).collect(), depth:1, success:true, num_qubits:num_qubits, difficulty:difficulty, gateset:gateset, depth_slope:depth_slope, max_depth:max_depth}
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
    }

    fn reset(&mut self) {
        // Reset the state to the target
        self.state = (0..self.num_qubits).collect();

        let mut rng = rand::thread_rng();
        let action_range = Uniform::new(0, self.num_actions());

        // Apply random actions based on the difficulty
        for _ in 0..self.difficulty {
            let action = action_range.sample(&mut rng);
            self.step(action);
        }
        self.depth = (self.depth_slope * self.difficulty).min(self.max_depth);
        self.success = self.solved();
    }

    fn step(&mut self, action: usize)  {
        match self.gateset[action] {
            Gate::SWAP(q1, q2) => (self.state[q2], self.state[q1]) = (self.state[q1], self.state[q2]),
            _ => {}
        }
        self.depth = self.depth.saturating_sub(1); // Prevent underflow
        self.success = self.solved();
    }
    
    fn masks(&self) -> Vec<bool> {
        vec![!self.success; self.num_actions()]
    }

    fn is_final(&self) -> bool {
        self.depth == 0 || self.success
    }

    fn reward(&self) -> f32 {
        if self.success {
            1.0
        } else {
            if self.depth == 0 { -0.5 } else { -0.5/(self.max_depth as f32) }
        }
    }

    fn observe(&self,) -> Vec<usize> {
        self.state.iter().enumerate().map(|(i, v)| i * self.num_qubits + v ).collect()  
    }
}


#[pyclass(name="PermutationEnv", extends=PyBaseEnv)]
pub struct PyPermutationEnv;

#[pymethods]
impl PyPermutationEnv {
    #[new]
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        gateset: Vec<Gate>,
        depth_slope: usize,
        max_depth: usize
    ) -> (Self, PyBaseEnv) {
        let env = Permutation::new(num_qubits, difficulty, gateset, depth_slope, max_depth);
        let env = Box::new(env);
        (PyPermutationEnv, PyBaseEnv { env })
    }
}