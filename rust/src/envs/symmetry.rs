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
use petgraph::algo::isomorphism::subgraph_isomorphisms_iter;
use petgraph::graph::Graph;
use petgraph::visit::NodeIndexable;
use petgraph::Directed;

#[derive(Hash, Eq, PartialEq, Clone, Copy)]
enum GateKind {
    H,
    S,
    Sdg,
    SX,
    SXdg,
    CX,
    CZ,
    Swap,
}

#[derive(Hash, Eq, PartialEq, Clone)]
struct GateKey {
    kind: GateKind,
    qubits: Vec<usize>,
}

fn gate_kind(gate: &Gate) -> GateKind {
    match gate {
        Gate::H(_) => GateKind::H,
        Gate::S(_) => GateKind::S,
        Gate::Sdg(_) => GateKind::Sdg,
        Gate::SX(_) => GateKind::SX,
        Gate::SXdg(_) => GateKind::SXdg,
        Gate::CX(_, _) => GateKind::CX,
        Gate::CZ(_, _) => GateKind::CZ,
        Gate::SWAP(_, _) => GateKind::Swap,
    }
}

fn gate_qubits(gate: &Gate) -> Vec<usize> {
    match gate {
        Gate::H(q)
        | Gate::S(q)
        | Gate::Sdg(q)
        | Gate::SX(q)
        | Gate::SXdg(q) => vec![*q],
        Gate::CX(q1, q2)
        | Gate::CZ(q1, q2)
        | Gate::SWAP(q1, q2) => vec![*q1, *q2],
    }
}

fn canonical_key(kind: GateKind, mut qubits: Vec<usize>) -> GateKey {
    if matches!(kind, GateKind::Swap) {
        qubits.sort_unstable();
    }
    GateKey { kind, qubits }
}

fn two_qubit_targets(gate: &Gate) -> Option<(usize, usize)> {
    match gate {
        Gate::CX(q1, q2) | Gate::CZ(q1, q2) | Gate::SWAP(q1, q2) => Some((*q1, *q2)),
        _ => None,
    }
}

fn identity_perm(num_qubits: usize) -> Vec<usize> {
    (0..num_qubits).collect()
}

fn all_permutations(num_qubits: usize) -> Vec<Vec<usize>> {
    let mut perm: Vec<usize> = (0..num_qubits).collect();
    let mut results = Vec::new();

    fn heap_permute(k: usize, perm: &mut Vec<usize>, results: &mut Vec<Vec<usize>>) {
        if k == 1 {
            results.push(perm.clone());
            return;
        }

        heap_permute(k - 1, perm, results);

        for i in 0..(k - 1) {
            if k % 2 == 0 {
                perm.swap(i, k - 1);
            } else {
                perm.swap(0, k - 1);
            }
            heap_permute(k - 1, perm, results);
        }
    }

    if num_qubits == 0 {
        results.push(Vec::new());
    } else {
        heap_permute(num_qubits, &mut perm, &mut results);
    }

    results
}

fn compute_automorphisms(adjacency: &[Vec<bool>], has_edge: bool) -> Vec<Vec<usize>> {
    let n = adjacency.len();
    if n == 0 {
        return vec![Vec::new()];
    }

    if !has_edge {
        return all_permutations(n);
    }

    // Build a directed graph with symmetric edges and use petgraph's VF2 enumerator.
    let mut graph = Graph::<usize, (), Directed>::new();
    let mut nodes = Vec::with_capacity(n);
    for node in 0..n {
        nodes.push(graph.add_node(node));
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if adjacency[i][j] {
                graph.add_edge(nodes[i], nodes[j], ());
                graph.add_edge(nodes[j], nodes[i], ());
            }
        }
    }

    let mut results: Vec<Vec<usize>> = Vec::new();
    let mut node_match = |_: &usize, _: &usize| true;
    let mut edge_match = |_: &(), _: &()| true;

    // Use &&graph so G0/G1 are `&Graph`, which implement the required traits for VF2.
    let graph_ref = &graph;
    if let Some(iter) =
        subgraph_isomorphisms_iter(&graph_ref, &graph_ref, &mut node_match, &mut edge_match)
    {
        for mapping in iter {
            if mapping.len() != n {
                continue;
            }
            // mapping indices are compact node indices; translate back to node labels
            let mut perm = vec![usize::MAX; n];
            for (from_idx, to_idx) in mapping.into_iter().enumerate() {
                let from_node = graph.from_index(from_idx);
                let to_node = graph.from_index(to_idx);
                let from_label = graph.node_weight(from_node).copied().unwrap_or(0);
                let to_label = graph.node_weight(to_node).copied().unwrap_or(0);
                perm[from_label] = to_label;
            }
            if perm.iter().any(|&v| v == usize::MAX) {
                continue;
            }
            results.push(perm);
        }
    }

    if results.is_empty() {
        results.push(identity_perm(n));
    }

    results.sort();
    results.dedup();
    results
}

fn build_action_perm(
    gateset: &[Gate],
    gate_index: &HashMap<GateKey, usize>,
    perm: &[usize],
) -> Option<Vec<usize>> {
    let mut act_perm: Vec<usize> = Vec::with_capacity(gateset.len());

    for gate in gateset {
        let kind = gate_kind(gate);
        let mut qubits = gate_qubits(gate);
        for q in qubits.iter_mut() {
            if *q >= perm.len() {
                return None;
            }
            *q = perm[*q];
        }
        let key = canonical_key(kind, qubits);
        if let Some(idx) = gate_index.get(&key) {
            act_perm.push(*idx);
        } else {
            return None;
        }
    }

    Some(act_perm)
}

fn compute_twists_with_builder<F>(
    num_qubits: usize,
    gateset: &[Gate],
    mut build_obs_perm: F,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>)
where
    F: FnMut(&[usize]) -> Vec<usize>,
{
    if num_qubits == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut gate_index: HashMap<GateKey, usize> = HashMap::new();
    for (idx, gate) in gateset.iter().enumerate() {
        let kind = gate_kind(gate);
        let qubits = gate_qubits(gate);
        let key = canonical_key(kind, qubits);
        gate_index.insert(key, idx);
    }

    let mut adjacency = vec![vec![false; num_qubits]; num_qubits];
    let mut has_edge = false;

    for gate in gateset {
        if let Some((q1, q2)) = two_qubit_targets(gate) {
            if q1 != q2 {
                adjacency[q1][q2] = true;
                adjacency[q2][q1] = true;
                has_edge = true;
            }
        }
    }

    let automorphisms = compute_automorphisms(&adjacency, has_edge);

    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    let mut obs_perms: Vec<Vec<usize>> = Vec::new();
    let mut act_perms: Vec<Vec<usize>> = Vec::new();

    for perm in automorphisms {
        if !seen.insert(perm.clone()) {
            continue;
        }
        if let Some(act_perm) = build_action_perm(gateset, &gate_index, &perm) {
            obs_perms.push(build_obs_perm(&perm));
            act_perms.push(act_perm);
        }
    }

    if obs_perms.is_empty() {
        let identity = identity_perm(num_qubits);
        if let Some(act_perm) = build_action_perm(gateset, &gate_index, &identity) {
            obs_perms.push(build_obs_perm(&identity));
            act_perms.push(act_perm);
        }
    }

    (obs_perms, act_perms)
}

fn obs_perm_square(num_qubits: usize, perm: &[usize]) -> Vec<usize> {
    let mut obs_perm = vec![0usize; num_qubits * num_qubits];
    for row in 0..num_qubits {
        for col in 0..num_qubits {
            let idx_old = row * num_qubits + col;
            obs_perm[idx_old] = perm[row] * num_qubits + perm[col];
        }
    }
    obs_perm
}

fn obs_perm_clifford(num_qubits: usize, perm: &[usize]) -> Vec<usize> {
    let dim = 2 * num_qubits;
    let mut obs_perm = vec![0usize; dim * dim];
    for row in 0..dim {
        let mapped_row = if row < num_qubits {
            perm[row]
        } else {
            num_qubits + perm[row - num_qubits]
        };
        for col in 0..dim {
            let mapped_col = if col < num_qubits {
                perm[col]
            } else {
                num_qubits + perm[col - num_qubits]
            };
            obs_perm[row * dim + col] = mapped_row * dim + mapped_col;
        }
    }
    obs_perm
}

pub fn compute_twists_square(num_qubits: usize, gateset: &[Gate]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    compute_twists_with_builder(num_qubits, gateset, |perm| obs_perm_square(num_qubits, perm))
}

pub fn compute_twists_clifford(num_qubits: usize, gateset: &[Gate]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    compute_twists_with_builder(num_qubits, gateset, |perm| obs_perm_clifford(num_qubits, perm))
}

fn obs_perm_pauli(num_qubits: usize, max_rotations: usize, perm: &[usize]) -> Vec<usize> {
    let dim = 2 * num_qubits;
    let total_cols = dim + max_rotations;
    let mut obs_perm = vec![0usize; dim * total_cols];

    // Permute the Clifford tableau part (2N x 2N)
    for row in 0..dim {
        let mapped_row = if row < num_qubits {
            perm[row]
        } else {
            num_qubits + perm[row - num_qubits]
        };
        for col in 0..dim {
            let mapped_col = if col < num_qubits {
                perm[col]
            } else {
                num_qubits + perm[col - num_qubits]
            };
            obs_perm[row * total_cols + col] = mapped_row * total_cols + mapped_col;
        }
    }

    // Permute the rotation columns
    // Each rotation column has X bits in rows 0..num_qubits and Z bits in rows num_qubits..2*num_qubits
    for rot_col in 0..max_rotations {
        let col = dim + rot_col;
        for row in 0..dim {
            let mapped_row = if row < num_qubits {
                perm[row]
            } else {
                num_qubits + perm[row - num_qubits]
            };
            obs_perm[row * total_cols + col] = mapped_row * total_cols + col;
        }
    }

    obs_perm
}

pub fn compute_twists_pauli(
    num_qubits: usize,
    gateset: &[Gate],
    max_rotations: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    compute_twists_with_builder(num_qubits, gateset, |perm| {
        obs_perm_pauli(num_qubits, max_rotations, perm)
    })
}

/// Compute qubit permutations and action permutations for PauliEnv internal symmetry handling.
/// Returns (qubit_perms, act_perms) where qubit_perms are the raw automorphisms (size num_qubits).
pub fn compute_qubit_perms(
    num_qubits: usize,
    gateset: &[Gate],
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    if num_qubits == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut gate_index: HashMap<GateKey, usize> = HashMap::new();
    for (idx, gate) in gateset.iter().enumerate() {
        let kind = gate_kind(gate);
        let qubits = gate_qubits(gate);
        let key = canonical_key(kind, qubits);
        gate_index.insert(key, idx);
    }

    let mut adjacency = vec![vec![false; num_qubits]; num_qubits];
    let mut has_edge = false;

    for gate in gateset {
        if let Some((q1, q2)) = two_qubit_targets(gate) {
            if q1 != q2 {
                adjacency[q1][q2] = true;
                adjacency[q2][q1] = true;
                has_edge = true;
            }
        }
    }

    let automorphisms = compute_automorphisms(&adjacency, has_edge);

    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    let mut qubit_perms: Vec<Vec<usize>> = Vec::new();
    let mut act_perms: Vec<Vec<usize>> = Vec::new();

    for perm in automorphisms {
        if !seen.insert(perm.clone()) {
            continue;
        }
        if let Some(act_perm) = build_action_perm(gateset, &gate_index, &perm) {
            qubit_perms.push(perm);
            act_perms.push(act_perm);
        }
    }

    if qubit_perms.is_empty() {
        let identity = identity_perm(num_qubits);
        if let Some(act_perm) = build_action_perm(gateset, &gate_index, &identity) {
            qubit_perms.push(identity);
            act_perms.push(act_perm);
        }
    }

    (qubit_perms, act_perms)
}
