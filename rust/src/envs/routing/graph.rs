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

//! Incremental topological generation tracker for stable directed graphs.
//!
//! Ported from qiskit-ai-transpiler-local/rust/src/routing/graph.rs.

use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::visit::IntoNodeIdentifiers;
use std::collections::{HashMap, VecDeque};

/// A 2-qubit operation in the routing DAG.
#[derive(Clone, Debug)]
pub struct RoutingOp {
    pub id: usize,
    pub qubits: (usize, usize),
}

/// Tracks the topological front (zero-indegree nodes) of a DAG incrementally,
/// and supports extracting horizon layers without modifying the underlying structure.
#[derive(Clone)]
pub struct TopologicalGenerations {
    indegree_map: HashMap<NodeIndex, usize>,
    pub zero_indegree: VecDeque<NodeIndex>,
}

impl TopologicalGenerations {
    pub fn new(graph: &StableDiGraph<RoutingOp, ()>) -> Self {
        let mut indegree_map = HashMap::new();
        let mut zero_indegree = VecDeque::new();

        for node in graph.node_identifiers() {
            let indegree = graph
                .edges_directed(node, petgraph::Direction::Incoming)
                .count();
            if indegree == 0 {
                zero_indegree.push_back(node);
            } else {
                indegree_map.insert(node, indegree);
            }
        }

        TopologicalGenerations {
            indegree_map,
            zero_indegree,
        }
    }

    /// Remove a node from the topological front and promote its successors.
    pub fn pop(&mut self, node: NodeIndex, graph: &StableDiGraph<RoutingOp, ()>) {
        if let Some(pos) = self.zero_indegree.iter().position(|&n| n == node) {
            self.zero_indegree.remove(pos);
        }

        for neighbor in graph.neighbors_directed(node, petgraph::Direction::Outgoing) {
            if let Some(indegree) = self.indegree_map.get_mut(&neighbor) {
                *indegree -= 1;
                if *indegree == 0 {
                    self.indegree_map.remove(&neighbor);
                    self.zero_indegree.push_back(neighbor);
                }
            }
        }
    }

    /// Extract up to `num_layers` topological generations without mutating state.
    /// Returns layers of node indices; the internal state is restored afterwards.
    pub fn get_layers(
        &mut self,
        num_layers: usize,
        graph: &StableDiGraph<RoutingOp, ()>,
    ) -> Vec<Vec<NodeIndex>> {
        let mut layers = Vec::new();
        let original_zero_indegree = self.zero_indegree.clone();
        let mut indegree_changes = Vec::new();

        for _ in 0..num_layers {
            if self.zero_indegree.is_empty() {
                break;
            }

            let this_generation = std::mem::take(&mut self.zero_indegree);
            let mut next_generation = VecDeque::new();

            for &node in &this_generation {
                for neighbor in graph.neighbors_directed(node, petgraph::Direction::Outgoing) {
                    let entry = self.indegree_map.entry(neighbor).or_insert(0);
                    indegree_changes.push((neighbor, *entry));
                    *entry -= 1;

                    if *entry == 0 {
                        self.indegree_map.remove(&neighbor);
                        next_generation.push_back(neighbor);
                    }
                }
            }

            layers.push(this_generation.into_iter().collect());
            self.zero_indegree = next_generation;
        }

        // Revert indegree changes
        for (node, original_indegree) in indegree_changes.into_iter().rev() {
            *self.indegree_map.entry(node).or_insert(0) = original_indegree;
        }

        // Restore the original zero_indegree queue
        self.zero_indegree = original_zero_indegree;

        layers
    }
}
