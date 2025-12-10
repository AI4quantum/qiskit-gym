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

use std::{collections::{HashMap, HashSet}, i128, usize};

use pyo3::prelude::*;

use rand::distributions::{Distribution, Uniform};

use twisterl::rl::env::Env;
use twisterl::python_interface::env::{PyBaseEnv, get_env_ref, get_env_mut};


// Some helper functions
fn simplify(mut a: i128, mut b: i128, mut k: i128) -> (i128, i128, i128) {
    while (k > 0) && ((a % 2) == 0) {
        (a, b) = (b, a / 2);
        k -= 1;
    }

    (a, b, k)
}

fn vec2bin(v: &Vec<i128>, max_depth: usize) -> Vec<usize> {
    let mut result: Vec<usize> = Vec::new();
    
    // The total number of bits for each number is max_depth + 1 (for sign bit)
    let num_bits = max_depth + 1;
    let size = v.len()/2;

    for i in 0..size {
        let num_a = v[i].clone();
        let num_k = v[i+size].clone();

        for (j, &num) in [num_a, num_k].iter().enumerate(){
            // Extract the sign bit and push its index if it's set (negative number)
            let sign_bit_index = (2*i + j) * num_bits;
            if num < 0 {
                result.push(sign_bit_index);
            }

            // Check the absolute value of the number, and examine its binary form
            let abs_num = num.abs();
            for bit_pos in 0..max_depth {
                if (abs_num & (1 << bit_pos)) != 0 {
                    result.push(sign_bit_index + 1 + bit_pos);
                }
            }
        }
    }

    result
}

fn pow2(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        2 << (n - 1)
    }
}

fn pow2i(n: i128) -> i128 {
    if n <= 0 {
        1
    } else {
        2 << (n - 1)
    }
}

fn increases_sde(pauli: &Vec<(i128, i128)>, max_sde_rows: &Vec<Vec<usize>>) -> bool {
    for &(i, j) in pauli.iter() {
        let abs_j = j.abs();
        for row_array in max_sde_rows.iter() {
            let mut increases = false;

            for &a in row_array.iter() {
                if i == (a as i128) {
                    increases = !increases;
                } else if abs_j == (a as i128) {
                    increases = !increases;
                }
            }

            if increases {return true};
        }
    }
    false
}

pub fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| &data[i]);
    indices
}

pub fn to_full(abk: &(i128, i128, i128)) -> f64 {
    ((abk.0 as f64) + (abk.1 as f64) * (2.0_f64.sqrt())) / (pow2(abk.2 as usize) as f64)
}

pub fn sorted_row_order(
    rows: &[Vec<(i128, i128, i128)>],
) -> Vec<usize> {
    let keys: Vec<Vec<f64>> = rows
        .iter()
        .map(|row| {
            let mut k: Vec<f64> = row.iter().map(|abk| to_full(abk)).collect();
            k.sort_by(|a, b| a.partial_cmp(b).unwrap());
            k
        })
        .collect();


    let mut idx: Vec<usize> = (0..rows.len()).collect();
    idx.sort_by(|&i, &j| keys[i].partial_cmp(&keys[j]).unwrap());
    idx
}


// Define some internal representation
#[derive(Clone)]
pub struct Channel3Data {
    pub data: Vec<i128>,
    pub num_qubits: usize,
    size: usize,
}

// Some functions to manipulate the internal representation
impl Channel3Data {
    fn col_sign(&self, col: usize) -> i128 {
        let mut largest: f64 = 0.0;
        for row in 0..self.size {
            let abk = self.get3(row, col);
            let fv = to_full(&abk);
            if fv.abs() > largest.abs() {
                largest = fv;
            }
        }

        if largest < 0.0 {-1} else {1}
    }

    fn canonical(&self) -> Channel3Data {
        // Take the sings of the "a" for the first nonzero column
        let signs: Vec<i128> = (0..self.size).map(|col| self.col_sign(col)).collect();
        
        let mut cc = self.clone();
        // Normalize the column signs so that first nonozero is positive
        for col in 0..self.size {
            for row in 0..self.size {
                cc.set(row, col, 0, signs[col] * cc.get(row, col, 0));
                cc.set(row, col, 1, signs[col] * cc.get(row, col, 1));
            }
        }

        // Sort columns
        let mut sorted_cols = (0..self.size).map(|col| (0..self.size).map(|row| self.get3(row, col)).collect::<Vec<(i128,i128,i128)>>()).collect::<Vec<Vec<(i128,i128,i128)>>>();
        let col_order = sorted_row_order(&sorted_cols);
        sorted_cols = col_order.iter().map(|&i| sorted_cols[i].clone()).collect();

        for (col, col_data) in sorted_cols.iter().enumerate() {
            for (row, &abk) in col_data.iter().enumerate() {
                cc.set3(row, col, abk);
            }
        }
        
        // Return canonical form
        cc
    }

    fn adjoint(&self) -> Channel3Data {
        let mut cc = self.clone();
        for col in 0..self.size {
            for row in 0..self.size {
                cc.set3(col, row, self.get3(row, col));
            }
        }

        cc
    }

    // Constructor to create a new Channel3Data
    fn new(num_qubits: usize) -> Self {
        let mut ch = Channel3Data {
            data: vec![0; pow2(4*num_qubits) * 3],
            num_qubits,
            size: pow2(2*num_qubits),
        };
        for i in 0..ch.size {
            ch.set(i, i, 0, 1);
        }
        ch
    }

    // Method to set a value in the Channel3Data
    fn set(&mut self, row: usize, column: usize, ch: usize, value: i128) {
        let index = self.index(row, column, ch);
        self.data[index] = value;
    }

    // Method to get a value from the Channel3Data
    fn get(&self, row: usize, column: usize, ch: usize) -> i128 {
        let index = self.index(row, column, ch);
        self.data[index]
    }

    // Method to set a value in the Channel3Data
    fn set3(&mut self, row: usize, column: usize, value: (i128, i128, i128)) {
        let (ia, ib, ik) = self.index3(row, column);
        (self.data[ia], self.data[ib], self.data[ik]) = value;
    }

    // Method to get a value from the Channel3Data
    fn get3(&self, row: usize, column: usize) -> (i128, i128, i128) {
        let (ia, ib, ik) = self.index3(row, column);
        (self.data[ia], self.data[ib], self.data[ik])
    }

    fn add(&self, v1: (usize, usize), v2: (usize, usize)) -> (i128, i128, i128) {
        let (a1, b1, k1) = self.get3(v1.0, v1.1);
        let (a2, b2, k2) = self.get3(v2.0, v2.1);

        let (a, b, k) = if k1 > k2 {(a1, b1, k1)} else {(a2, b2, k2)};
        let (c, d, l) = if k1 > k2 {(a2, b2, k2)} else {(a1, b1, k1)};

        if ((k - l) % 2) == 0 {
            simplify(a + c * pow2i((k-l)/2), b + d * pow2i((k-l)/2), k)
        } else {
            simplify(a + d * pow2i((k-l+1)/2), b + c * pow2i((k-l-1)/2), k)
        }
    }

    fn sub(&self, v1: (usize, usize), v2: (usize, usize)) -> (i128, i128, i128) {
        let (a1, b1, k1) = self.get3(v1.0, v1.1);
        let (a2, b2, k2) = self.get3(v2.0, v2.1);

        let (a, b, k) = if k1 > k2 {(a1, b1, k1)} else {(a2, b2, k2)};
        let (c, d, l) = if k1 > k2 {(a2, b2, k2)} else {(a1, b1, k1)};
        let sign: i128 = if k1 > k2 {1} else {-1};

        let (a3, b3, k3) = if ((k - l) % 2) == 0 {
            simplify(a - c * pow2i((k-l)/2), b - d * pow2i((k-l)/2), k)
        } else {
            simplify(a - d * pow2i((k-l+1)/2), b - c * pow2i((k-l-1)/2), k)
        };

        (sign * a3, sign * b3, k3)
    }

    fn mult(&self, v1: (usize, usize), v2: (usize, usize), sign: i128) -> ((i128, i128, i128), (i128, i128, i128)) {
        // v1 and v2 must be on the same column (that is, v1.1 == v2.1)
        if sign > 0 {
            let (a1, b1, k1) = self.add(v2, v1);
            let (a2, b2, k2) = self.sub(v2, v1);

            (simplify(a1, b1, k1 + 1), simplify(a2, b2, k2 + 1))
        } else {
            let (a1, b1, k1) = self.sub(v1, v2);
            let (a2, b2, k2) = self.add(v1, v2);

            (simplify(a1, b1, k1 + 1), simplify(a2, b2, k2 + 1))
        }
    }

    // Method to perform a gate on the Channel3Data
    fn gate(&mut self, paulis: &Vec<(i128, i128)>, inverse: bool) {
        for &(i, l) in paulis.iter() {
            let li = if inverse {-l} else {l};

            for c in 0..self.size {
                let (vi, vl) = self.mult((i as usize, c), (li.abs() as usize, c), if li < 0 {-1} else {1});
                self.set3(i as usize, c, vi);
                self.set3(li.abs() as usize, c, vl);
            }
        }
    }

    // Private helper method to calculate the index from row and column
    fn index(&self, row: usize, column: usize, ch: usize) -> usize {
        ch * self.size * self.size + row * self.size + column
     }

    // Private helper method to calculate the index from row and column
    fn index3(&self, row: usize, column: usize) -> (usize, usize, usize) {
        (row * self.size + column, 
         self.size * self.size + row * self.size + column, 
         2 * self.size * self.size + row * self.size + column)
    }

    // Check if it is identity
    fn solved(&self) -> bool {
        for i in 0..self.size {
            for j in 0..self.size {
                let (a, b, k) = self.get3(i, j);
                if (b != 0) || (k != 0) || (a.abs() > 1) {
                    return false;
                }
            }
        }
        true
    }

    fn sde(&self) -> i128 {
        self.data[(2*self.size*self.size)..].iter().max().unwrap().clone()
    }

    fn max_sde_rows(&self) -> Vec<Vec<usize>> {
        let sde = self.sde();
        if sde == 0 {return vec![]};
        
        let mut out: HashSet<Vec<usize>> = HashSet::new();
        for col in 0..self.size {
            let mut out_row: Vec<usize> = vec![];
            for row in 0..self.size {
                if self.get(row, col, 2) == sde {
                    out_row.push(row);
                }
            }
            if out_row.len() > 0 {
                out.insert(out_row);
            }
        }

        out.into_iter().collect()
    }

    fn masks(&self, actions: &Vec<Vec<(i128, i128)>>) -> Vec<bool> {
        let num_actions = actions.len();
        let mut masks = vec![true; num_actions]; 

        let mut num_increases = 0;
        let max_sde_rows = self.max_sde_rows();
        if max_sde_rows.len() == 0 {return masks;}

        for action in 0..num_actions {
            masks[action] = increases_sde(&actions[action], &max_sde_rows);
            if masks[action] {num_increases += 1;};
        }
        if num_increases == num_actions {return masks;}

        if num_increases > (num_actions - num_increases) {
            // We should allow actions on the min cardinality set
            // If we are here this means the actions with True (the ones that increase SDE) are majority, so they must be flipped
            for action in 0..num_actions {
                masks[action] = !masks[action];
            }
        }

        if masks.iter().map(|m| if *m {1} else {0}).sum::<usize>() == 0 {
            vec![true; num_actions]
        } else {
            masks
        }
    }
}


// This is the Env definition
#[pyclass]
#[derive(Clone)]
pub struct Channel3 {
    pub ch: Channel3Data,
    pub depth: usize,
    pub success: bool,

    pub difficulty: usize,
    actions: Vec<Vec<(i128, i128)>>,
    pub depth_slope: usize,
    max_obs_depth: usize,
    max_steps: usize,
    use_mask: bool,
    canonical_obs: bool,
}


// This specifies how we initialize new envs from python
#[pymethods]
impl Channel3 {
    #[new]
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        actions: Vec<Vec<(i128, i128)>>,
        depth_slope: usize,
        max_obs_depth: usize,
        max_steps: usize,
        use_mask: bool,
        canonical_obs: bool,
    ) -> Self {
        let ch = Channel3Data::new(num_qubits);
        let success = ch.solved();
        Channel3 {ch, depth:1, success, difficulty, actions, depth_slope, max_obs_depth, max_steps, use_mask, canonical_obs }
    }

    pub fn set_state_from_actions(&mut self, acts: Vec<i128>) {
        self.ch = Channel3Data::new(self.ch.num_qubits);
        self.depth = 0;
        for a in acts.iter().rev() {
            if *a > 0 {
                let corrected_a = (*a as usize) - 1;
                self.ch.gate(&self.actions[corrected_a], true);
                //println!("inv {},{}", a, corrected_a);
            } else if *a < 0 {
                let corrected_a = ((*a).abs() as usize) - 1;
                //println!("dir {},{}", a, corrected_a);
                self.ch.gate(&self.actions[corrected_a], false);
            }
            self.depth += 1; 
            self.success = self.ch.solved();
        }
        self.depth = acts.len() * self.depth_slope;
    }

    pub fn set_max_steps(&mut self, max_steps: usize) {
        self.max_steps = max_steps;
    }

    pub fn get_state(&self) -> Vec<i128> {
        self.ch.data.clone()
    }

    pub fn get_canonical_state(&self) -> Vec<i128> {
        self.ch.canonical().data
    }

    fn prev(&mut self, action: usize) {
        if action < self.actions.len(){
            self.ch.gate(&self.actions[action], true);
        } else {
            self.ch = self.ch.adjoint();
            self.ch.gate(&self.actions[action - self.actions.len()], false);
            self.ch = self.ch.adjoint();
        }
        self.depth += 1; 
        self.success = self.ch.solved();
    }
}

// This implements the necessary functions for the environment
impl Env for Channel3 {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn num_actions(&self) -> usize {
        self.actions.len() * 2
    }

    fn obs_shape(&self) -> Vec<usize> {
        vec![3*(self.max_obs_depth + 1) , 2 * pow2(4*self.ch.num_qubits)]
        //3*(self.max_obs_depth + 1) , pow2(4*self.ch.num_qubits)
    }

    fn set_difficulty(&mut self, difficulty: usize) {
        self.difficulty = difficulty;
    }

    fn get_difficulty(&self) -> usize {
        self.difficulty
    }

    fn set_state(&mut self, state: Vec<i64>) {
        self.ch.data = state.iter().map(|&v| v as i128).collect();
        self.depth = self.max_steps;  
        self.success = self.ch.solved();
    }

    fn reset(&mut self) {
        // Init the channel to the identity
        self.ch = Channel3Data::new(self.ch.num_qubits);
        self.depth = 0;
        self.success = self.ch.solved();

        let mut rng = rand::thread_rng();
        let action_range = Uniform::new(0, self.num_actions());

        // Apply random actions based on the difficulty
        for _ in 0..self.difficulty {
            let action = action_range.sample(&mut rng);
            self.prev(action);
        }
        self.depth = self.depth_slope * self.difficulty;
        self.success = self.ch.solved();
    }

    fn step(&mut self, action: usize)  {
        if action < self.actions.len(){
            self.ch.gate(&self.actions[action], false);
        } else {
            self.ch = self.ch.adjoint();
            self.ch.gate(&self.actions[action - self.actions.len()], true);
            self.ch = self.ch.adjoint();
        }
        self.depth = self.depth.saturating_sub(1); // Prevent underflow
        self.success = self.ch.solved();
    }

    fn masks(&self) -> Vec<bool> {
        // Masking for SDE increase vs non-increase
        if self.success || !self.use_mask {return vec![true; self.num_actions()];}

        [self.ch.masks(&self.actions), self.ch.adjoint().masks(&self.actions)].concat()
    }

    fn is_final(&self) -> bool {
        self.depth == 0 || self.success
    }

    fn reward(&self) -> f32 {
        if self.success {
            1.0
        } else {
            if self.depth == 0 { -0.5 } else { -0.5/(self.max_steps as f32) }
        }
    }

    fn observe(&self,) -> Vec<usize> {
        let (ch_dir, ch_inv) = if self.canonical_obs {
            (&self.ch.canonical().data, &self.ch.adjoint().canonical().data)
        } else {
            (&self.ch.data, &self.ch.adjoint().data)
        };

        let v_dir = vec2bin(ch_dir, self.max_obs_depth);
        let mut v_inv = vec2bin(ch_inv, self.max_obs_depth);
        let n = 3*(self.max_obs_depth + 1) * pow2(4*self.ch.num_qubits);
        v_inv = v_inv.iter().map(|&vv| vv + n).collect();

        [v_dir, v_inv].concat()
    }
}



#[pyclass(name="Channel3Env", extends=PyBaseEnv)]
pub struct PyChannel3Env;

#[pymethods]
impl PyChannel3Env {
    #[new]
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        actions: Vec<Vec<(i128, i128)>>,
        depth_slope: usize,
        max_obs_depth: usize,
        max_steps: usize,
        use_mask: bool,
        canonical_obs: bool,
    ) -> (Self, PyBaseEnv) {
        let env = Channel3::new(num_qubits, difficulty, actions, depth_slope, max_obs_depth, max_steps, use_mask, canonical_obs);
        let env = Box::new(env);
        (PyChannel3Env, PyBaseEnv { env })
    }
}