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

use crate::pauli::pauli::Pauli;
use petgraph::graph::{DiGraph, NodeIndex};
use std::ops::{Deref, DerefMut};

#[derive(Clone)]
pub struct PauliDag {
    graph: DiGraph<usize, ()>,
}

impl PauliDag {
    // Constructor for creating a PauliDag from a sequence of Pauli objects
    pub fn new(rotation_sequence: &[Pauli]) -> Self {
        let mut graph = DiGraph::new();
        let mut node_indices = Vec::new();

        // Add nodes to the graph representing each Pauli object
        for (i, _) in rotation_sequence.iter().enumerate() {
            node_indices.push(graph.add_node(i));
        }

        // Add edges based on non-commutation
        for (i1, p1) in rotation_sequence.iter().enumerate() {
            for (i2, p2) in rotation_sequence.iter().take(i1).enumerate() {
                if !p1.commutes_with(p2) {
                    graph.add_edge(node_indices[i1], node_indices[i2], ());
                }
            }
        }

        PauliDag { graph }
    }

    // Method to get the front layer of the DAG (nodes with no outgoing edges)
    pub fn get_front_layer(&self) -> Vec<NodeIndex> {
        self.graph
            .node_indices()
            .filter(|&node| {
                self.graph
                    .edges_directed(node, petgraph::Direction::Outgoing)
                    .count()
                    == 0
            })
            .collect()
    }
}

impl Deref for PauliDag {
    type Target = DiGraph<usize, ()>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for PauliDag {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}
