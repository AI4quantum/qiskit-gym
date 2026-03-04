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

//! Random circuit generator for routing environments.
//!
//! Ported from ai-quantum-compiling/circuit/generators.py RoutingGenerator.
//! Generates random 2-qubit gate sequences where each gate's qubit pair has
//! a distance that fits within the remaining difficulty budget.

use super::config::DistType;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::collections::BTreeMap;

/// Generates random 2-qubit gate sequences for routing.
#[derive(Clone)]
pub struct RoutingGenerator {
    /// Qubit pairs grouped by their shortest-path distance.
    dist_pairs: BTreeMap<DistType, Vec<(usize, usize)>>,
    /// All distinct distances (sorted).
    all_dists: Vec<DistType>,
}

impl RoutingGenerator {
    /// Build a generator from a coupling map and its all-pairs shortest distances.
    pub fn new(num_qubits: usize, dists: &[Vec<DistType>]) -> Self {
        let mut dist_pairs: BTreeMap<DistType, Vec<(usize, usize)>> = BTreeMap::new();

        for q1 in 0..num_qubits {
            for q2 in (q1 + 1)..num_qubits {
                let d = dists[q1][q2];
                if d < DistType::MAX {
                    dist_pairs.entry(d).or_default().push((q1, q2));
                }
            }
        }

        let all_dists: Vec<DistType> = dist_pairs.keys().copied().collect();

        RoutingGenerator {
            dist_pairs,
            all_dists,
        }
    }

    /// Generate a random sequence of 2-qubit gate pairs consuming `difficulty` distance budget.
    pub fn generate(&self, mut difficulty: usize) -> Vec<(usize, usize)> {
        let mut rng = rand::thread_rng();
        let mut out = Vec::new();

        while difficulty > 0 {
            // Find distances that fit in the remaining budget
            let valid: Vec<DistType> = self
                .all_dists
                .iter()
                .copied()
                .filter(|&d| (d as usize) <= difficulty)
                .collect();

            if valid.is_empty() {
                break;
            }

            let dist_idx = Uniform::new(0, valid.len()).sample(&mut rng);
            let next_dist = valid[dist_idx];
            let pairs = &self.dist_pairs[&next_dist];
            let pair_idx = rng.gen_range(0..pairs.len());
            out.push(pairs[pair_idx]);
            difficulty -= next_dist as usize;
        }

        out
    }
}
