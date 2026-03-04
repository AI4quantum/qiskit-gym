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

//! Input-side DAG for routing: tracks which 2-qubit gates still need to be routed.
//!
//! Ported and simplified from qiskit-ai-transpiler-local/rust/src/routing/dag.rs.
//! Only the input DAG portion is kept (no output circuit building or autocancellation).

use super::config::DistType;
use super::graph::{RoutingOp, TopologicalGenerations};
use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::Direction::Incoming;
use rand::seq::SliceRandom;
use std::collections::HashMap;

/// A DAG of 2-qubit operations to be routed onto a coupling map.
#[derive(Clone)]
pub struct TwoQubitDAG {
    pub dag: StableDiGraph<RoutingOp, ()>,
    front: HashMap<usize, NodeIndex>,
    total_gates: usize,
    pub topgens: TopologicalGenerations,
    gens: Vec<Vec<NodeIndex>>,
    /// For each physical location, the list of coupling-map edge indices involving it.
    loc_to_act: Vec<Vec<usize>>,
}

impl TwoQubitDAG {
    pub fn new(coupling_map: &[(usize, usize)], num_qubits: usize) -> Self {
        let dag = StableDiGraph::new();
        let mut loc_to_act = vec![Vec::new(); num_qubits];

        for (i, &(l1, l2)) in coupling_map.iter().enumerate() {
            loc_to_act[l1].push(i);
            loc_to_act[l2].push(i);
        }

        TwoQubitDAG {
            dag,
            front: HashMap::new(),
            total_gates: 0,
            topgens: TopologicalGenerations::new(&StableDiGraph::new()),
            gens: Vec::new(),
            loc_to_act,
        }
    }

    /// Add a 2-qubit operation to the DAG.
    pub fn push(&mut self, q1: usize, q2: usize) {
        let op = RoutingOp {
            id: self.total_gates,
            qubits: (q1, q2),
        };
        self.total_gates += 1;

        let idx = self.dag.add_node(op);

        // Add edges from the front nodes on each qubit
        if let Some(&f1) = self.front.get(&q1) {
            self.dag.add_edge(f1, idx, ());
        }
        if let Some(&f2) = self.front.get(&q2) {
            if self.front.get(&q1) != Some(&f2) {
                self.dag.add_edge(f2, idx, ());
            }
        }

        self.front.insert(q1, idx);
        self.front.insert(q2, idx);
    }

    /// Initialize topological generations after all gates have been pushed.
    pub fn create_topgens(&mut self) {
        self.topgens = TopologicalGenerations::new(&self.dag);
    }

    /// Update the horizon layers (shuffled within each layer).
    pub fn update_gens(&mut self, horizon: usize) {
        let mut gens = self.topgens.get_layers(horizon, &self.dag);
        let mut rng = rand::thread_rng();
        for layer in &mut gens {
            layer.shuffle(&mut rng);
        }
        self.gens = gens;
    }

    /// Remove a node from the DAG and update topological tracking.
    pub fn remove_node(&mut self, idx: NodeIndex) {
        self.topgens.pop(idx, &self.dag);
        self.dag.remove_node(idx);
    }

    /// Number of remaining operations in the DAG.
    pub fn len(&self) -> usize {
        self.dag.node_count()
    }

    /// Compute active swaps: coupling-map edge indices near the front of the DAG.
    pub fn get_active_swaps(
        &self,
        locations: &[usize],
        num_active_swaps: usize,
    ) -> Vec<usize> {
        let mut active_swaps: Vec<usize> = Vec::with_capacity(num_active_swaps);
        let mut rng = rand::thread_rng();

        for gen in &self.gens {
            if active_swaps.len() >= num_active_swaps {
                break;
            }
            for &idx in gen {
                if active_swaps.len() >= num_active_swaps {
                    break;
                }
                let (q1, q2) = self.dag[idx].qubits;
                for q in [q1, q2] {
                    for &a in &self.loc_to_act[locations[q]] {
                        if active_swaps.len() >= num_active_swaps {
                            active_swaps.shuffle(&mut rng);
                            return active_swaps;
                        }
                        if !active_swaps.contains(&a) {
                            active_swaps.push(a);
                        }
                    }
                }
            }
        }

        active_swaps.shuffle(&mut rng);
        active_swaps
    }

    /// Compute the distance-improvement observation matrix and return active swaps.
    ///
    /// Returns (flat_obs, active_swaps) where flat_obs has length num_active_swaps * horizon,
    /// with values representing the distance improvement for each (swap, layer) pair.
    pub fn get_obs(
        &self,
        locations: &[usize],
        coupling_map: &[(usize, usize)],
        dists: &[Vec<DistType>],
        num_active_swaps: usize,
    ) -> (Vec<DistType>, Vec<usize>) {
        let horizon = self.gens.len();
        let mut out = vec![0 as DistType; num_active_swaps * horizon];

        let active_swaps = self.get_active_swaps(locations, num_active_swaps);

        for (d, gen) in self.gens.iter().enumerate() {
            for (isw, &sw) in active_swaps.iter().enumerate() {
                let (a, b) = coupling_map[sw];
                for &idx in gen {
                    let (aq, bq) = self.dag[idx].qubits;
                    let (al, bl) = (locations[aq], locations[bq]);
                    if al == a && bl != b {
                        out[d + isw * horizon] += dists[a][bl] - dists[b][bl];
                    } else if al == b && bl != a {
                        out[d + isw * horizon] += dists[b][bl] - dists[a][bl];
                    } else if bl == a && al != b {
                        out[d + isw * horizon] += dists[a][al] - dists[b][al];
                    } else if bl == b && al != a {
                        out[d + isw * horizon] += dists[b][al] - dists[a][al];
                    }
                }
            }
        }

        (out, active_swaps)
    }

    /// Resolve all executable operations at the front of the DAG.
    /// An operation (q1, q2) is executable when locations[q1] and locations[q2]
    /// are adjacent on the coupling map (distance <= 1).
    ///
    /// Returns the number of operations resolved.
    pub fn execute_front(
        &mut self,
        locations: &[usize],
        dists: &[Vec<DistType>],
    ) -> usize {
        let mut num_solved = 0;
        let mut roots: Vec<NodeIndex> = self.topgens.zero_indegree.iter().copied().collect();

        let mut i = 0;
        while i < roots.len() {
            let node = roots[i];
            i += 1;

            // Node may have been removed already
            if self.dag.node_weight(node).is_none() {
                continue;
            }

            let (q1, q2) = self.dag[node].qubits;
            let l1 = locations[q1];
            let l2 = locations[q2];

            if dists[l1][l2] <= 1 {
                num_solved += 1;

                // Collect successors before removing
                let successors: Vec<NodeIndex> =
                    self.dag.neighbors(node).collect();
                self.remove_node(node);

                for successor in successors {
                    if self
                        .dag
                        .neighbors_directed(successor, Incoming)
                        .count()
                        == 0
                    {
                        roots.push(successor);
                    }
                }
            }
        }

        num_solved
    }
}
