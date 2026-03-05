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

/// Distance type for shortest-path distances on the coupling map.
pub type DistType = i32;

/// Configuration for the routing environment.
#[derive(Clone)]
pub struct RoutingConfig {
    pub num_qubits: usize,
    pub num_active_swaps: usize,
    pub horizon: usize,
    pub obs_bins: usize, // K: values in [-K, K], total bins = 2K+1
    pub coupling_map: Vec<(usize, usize)>,
    pub dists: Vec<Vec<DistType>>,
    pub difficulty: usize,
    pub max_difficulty: usize,
    pub depth_slope: usize,
    pub max_depth: usize,
    pub layout_exponent: f32,
}

impl RoutingConfig {
    pub fn num_bins(&self) -> usize {
        2 * self.obs_bins + 1
    }
}
