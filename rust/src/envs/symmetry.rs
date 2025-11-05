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

#[derive(Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct NodeSignature {
    degree: usize,
    neighbor_degrees: Vec<usize>,
}

impl NodeSignature {
    fn new(node: usize, degrees: &[usize], adjacency: &[Vec<bool>]) -> Self {
        let mut neighbor_degrees: Vec<usize> = adjacency[node]
            .iter()
            .enumerate()
            .filter_map(|(idx, &connected)| if connected { Some(degrees[idx]) } else { None })
            .collect();
        neighbor_degrees.sort_unstable();
        Self {
            degree: degrees[node],
            neighbor_degrees,
        }
    }
}

fn enumerate_automorphisms(
    adjacency: &[Vec<bool>],
    has_edge: bool,
    signatures: &[NodeSignature],
) -> Vec<Vec<usize>> {
    let n = adjacency.len();
    if n == 0 {
        return vec![Vec::new()];
    }

    if !has_edge {
        return vec![identity_perm(n)];
    }

    let mut nodes_order: Vec<usize> = (0..n).collect();
    nodes_order.sort_by(|&a, &b| signatures[a].cmp(&signatures[b]));

    let mut perm = vec![usize::MAX; n];
    let mut used = vec![false; n];
    let mut results = Vec::new();

    backtrack_automorphisms(
        0,
        &nodes_order,
        signatures,
        adjacency,
        &mut perm,
        &mut used,
        &mut results,
    );

    if results.is_empty() {
        results.push(identity_perm(n));
    }

    results
}

fn backtrack_automorphisms(
    idx: usize,
    nodes_order: &[usize],
    signatures: &[NodeSignature],
    adjacency: &[Vec<bool>],
    perm: &mut Vec<usize>,
    used: &mut Vec<bool>,
    results: &mut Vec<Vec<usize>>,
) {
    if idx == nodes_order.len() {
        results.push(perm.clone());
        return;
    }

    let node_from = nodes_order[idx];
    let target_signature = &signatures[node_from];
    let n = adjacency.len();

    for node_to in 0..n {
        if used[node_to] || &signatures[node_to] != target_signature {
            continue;
        }

        let mut consistent = true;
        for prev_idx in 0..idx {
            let prev_from = nodes_order[prev_idx];
            let prev_to = perm[prev_from];
            if prev_to == usize::MAX {
                continue;
            }
            if adjacency[node_from][prev_from] != adjacency[node_to][prev_to] {
                consistent = false;
                break;
            }
        }

        if !consistent {
            continue;
        }

        perm[node_from] = node_to;
        used[node_to] = true;
        backtrack_automorphisms(
            idx + 1,
            nodes_order,
            signatures,
            adjacency,
            perm,
            used,
            results,
        );
        used[node_to] = false;
        perm[node_from] = usize::MAX;
    }
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

    let degrees: Vec<usize> = adjacency
        .iter()
        .map(|row| row.iter().filter(|&&edge| edge).count())
        .collect();
    let signatures: Vec<NodeSignature> = (0..num_qubits)
        .map(|idx| NodeSignature::new(idx, &degrees, &adjacency))
        .collect();

    let automorphisms = enumerate_automorphisms(&adjacency, has_edge, &signatures);

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
