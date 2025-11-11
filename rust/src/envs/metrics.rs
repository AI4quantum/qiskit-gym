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

use std::collections::{HashMap, HashSet};

use crate::envs::common::Gate;

#[derive(Clone)]
pub struct MetricsTracker {
    num_qubits: usize,
    n_cnots: usize,
    n_gates: usize,
    cnot_layers: HashSet<usize>,
    layers: HashSet<usize>,
    last_gates: Vec<isize>,
    last_cxs: Vec<isize>,
}

impl MetricsTracker {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            n_cnots: 0,
            n_gates: 0,
            cnot_layers: HashSet::new(),
            layers: HashSet::new(),
            last_gates: vec![-1; num_qubits],
            last_cxs: vec![-1; num_qubits],
        }
    }

    pub fn reset(&mut self) {
        self.n_cnots = 0;
        self.n_gates = 0;
        self.cnot_layers.clear();
        self.layers.clear();
        for val in self.last_gates.iter_mut() {
            *val = -1;
        }
        for val in self.last_cxs.iter_mut() {
            *val = -1;
        }
    }

    pub fn snapshot(&self) -> MetricsCounts {
        MetricsCounts {
            n_cnots: self.n_cnots,
            n_layers_cnots: self.cnot_layers.len(),
            n_layers: self.layers.len(),
            n_gates: self.n_gates,
        }
    }

    pub fn apply_gate(&mut self, gate: &Gate) {
        match gate {
            Gate::CX(c, t) => self.cx(*c, *t),
            Gate::SWAP(c, t) => {
                self.cx(*c, *t);
                self.cx(*t, *c);
                self.cx(*c, *t);
            }
            Gate::CZ(c, t) => {
                self.single_qubit(*t);
                self.cx(*c, *t);
                self.single_qubit(*t);
            }
            Gate::H(q) | Gate::S(q) | Gate::Sdg(q) | Gate::SX(q) | Gate::SXdg(q) => {
                self.single_qubit(*q);
            }
        }
    }

    fn single_qubit(&mut self, target: usize) {
        if target >= self.num_qubits {
            return;
        }

        self.n_gates += 1;

        if self.last_gates[target] >= 0 {
            self.layers.insert(self.last_gates[target] as usize);
        }
        self.last_gates[target] += 1;
    }

    fn cx(&mut self, control: usize, target: usize) {
        if control == target
            || control >= self.num_qubits
            || target >= self.num_qubits
        {
            return;
        }

        self.n_cnots += 1;
        self.n_gates += 1;

        let gate_layer = (self.last_gates[control].max(self.last_gates[target])) + 1;
        self.last_gates[control] = gate_layer;
        self.last_gates[target] = gate_layer;

        if gate_layer >= 0 {
            self.layers.insert(gate_layer as usize);
        }

        let cx_layer = (self.last_cxs[control].max(self.last_cxs[target])) + 1;
        self.last_cxs[control] = cx_layer;
        self.last_cxs[target] = cx_layer;

        if cx_layer >= 0 {
            self.cnot_layers.insert(cx_layer as usize);
        }
    }
}

#[derive(Clone)]
pub struct MetricsCounts {
    n_cnots: usize,
    n_layers_cnots: usize,
    n_layers: usize,
    n_gates: usize,
}

impl MetricsCounts {
    pub fn weighted_delta(&self, previous: &Self, weights: &MetricsWeights) -> f32 {
        let delta_cnots = self.n_cnots.saturating_sub(previous.n_cnots) as f32;
        let delta_layers_cnots =
            self.n_layers_cnots.saturating_sub(previous.n_layers_cnots) as f32;
        let delta_layers = self.n_layers.saturating_sub(previous.n_layers) as f32;
        let delta_gates = self.n_gates.saturating_sub(previous.n_gates) as f32;

        weights.n_cnots * delta_cnots
            + weights.n_layers_cnots * delta_layers_cnots
            + weights.n_layers * delta_layers
            + weights.n_gates * delta_gates
    }
}

#[derive(Clone)]
pub struct MetricsWeights {
    pub n_cnots: f32,
    pub n_layers_cnots: f32,
    pub n_layers: f32,
    pub n_gates: f32,
}

impl Default for MetricsWeights {
    fn default() -> Self {
        Self {
            n_cnots: 0.01,
            n_layers_cnots: 0.0,
            n_layers: 0.0,
            n_gates: 0.0001,
        }
    }
}

impl MetricsWeights {
    pub fn from_hashmap(map: Option<HashMap<String, f32>>) -> Self {
        let mut weights = Self::default();
        if let Some(values) = map {
            for (key, value) in values {
                match key.as_str() {
                    "n_cnots" => weights.n_cnots = value,
                    "n_layers_cnots" => weights.n_layers_cnots = value,
                    "n_layers" => weights.n_layers = value,
                    "n_gates" => weights.n_gates = value,
                    _ => {}
                }
            }
        }
        weights
    }
}

