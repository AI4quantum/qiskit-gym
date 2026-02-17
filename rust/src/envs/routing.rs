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

use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::Direction::{Incoming, Outgoing};
use rand::seq::SliceRandom;
use rand::Rng;

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::RwLock;

use twisterl::python_interface::env::PyBaseEnv;
use twisterl::rl::env::Env;

use crate::envs::metrics::MetricsWeights;

type DistType = i32;

const DIST_INF: DistType = DistType::MAX / 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpType {
    Cx,
    Cz,
    Swap,
}

impl OpType {
    fn from_name(name: &str, fallback: OpType) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "cx" => Self::Cx,
            "cz" => Self::Cz,
            "swap" => Self::Swap,
            _ => fallback,
        }
    }

    fn as_name(self) -> &'static str {
        match self {
            Self::Cx => "cx",
            Self::Cz => "cz",
            Self::Swap => "swap",
        }
    }
}

#[derive(Clone, Debug)]
struct TargetTemplate {
    op_type: OpType,
    qubits: (usize, usize),
}

#[derive(Clone, Debug)]
struct DagOp {
    op_type: OpType,
    qubits: (usize, usize),
}

#[derive(Clone, Debug)]
struct CircuitOp {
    id: usize,
    op_type: OpType,
    qubits: (usize, usize),
}

#[derive(Clone)]
struct CircuitDag {
    dag: StableDiGraph<CircuitOp, ()>,
    front: HashMap<usize, NodeIndex>,
    layout: Vec<usize>,
    next_id: usize,
}

#[derive(Clone, Default)]
struct RoutingMetricsSnapshot {
    n_cnots: usize,
    n_layers_cnots: usize,
    n_layers: usize,
    n_gates: usize,
}

#[derive(Default)]
struct ActiveSwaps {
    values: RwLock<Vec<usize>>,
}

impl ActiveSwaps {
    fn new() -> Self {
        Self {
            values: RwLock::new(Vec::new()),
        }
    }

    fn snapshot(&self) -> Vec<usize> {
        match self.values.read() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    fn set(&self, active_swaps: Vec<usize>) {
        match self.values.write() {
            Ok(mut guard) => {
                *guard = active_swaps;
            }
            Err(poisoned) => {
                *poisoned.into_inner() = active_swaps;
            }
        }
    }

    fn clear(&self) {
        self.set(Vec::new());
    }
}

impl Clone for ActiveSwaps {
    fn clone(&self) -> Self {
        Self {
            values: RwLock::new(self.snapshot()),
        }
    }
}

impl RoutingMetricsSnapshot {
    fn weighted_delta(&self, previous: &Self, weights: &MetricsWeights) -> f32 {
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

impl CircuitDag {
    fn new(num_qubits: usize) -> Self {
        Self {
            dag: StableDiGraph::new(),
            front: HashMap::new(),
            layout: (0..num_qubits).collect(),
            next_id: 0,
        }
    }

    fn get_unique_front(&self, qubits: (usize, usize)) -> Option<NodeIndex> {
        let (q1, q2) = qubits;
        match (self.front.get(&q1), self.front.get(&q2)) {
            (Some(a), Some(b)) if *a == *b => Some(*a),
            _ => None,
        }
    }

    fn get_previous_front(&self, q: usize, idx: NodeIndex) -> Option<NodeIndex> {
        let mut pred_front: Option<NodeIndex> = None;
        for pred in self.dag.neighbors_directed(idx, Incoming) {
            let pop = &self.dag[pred];
            if pop.qubits.0 == q || pop.qubits.1 == q {
                if let Some(prev_idx) = pred_front {
                    if pop.id > self.dag[prev_idx].id {
                        pred_front = Some(pred);
                    }
                } else {
                    pred_front = Some(pred);
                }
            }
        }
        pred_front
    }

    fn remove_node(&mut self, idx: NodeIndex) {
        if self.dag.node_weight(idx).is_none() {
            return;
        }
        let op = self.dag[idx].clone();
        for q in [op.qubits.0, op.qubits.1] {
            if let Some(prev) = self.get_previous_front(q, idx) {
                self.front.insert(q, prev);
            } else {
                self.front.remove(&q);
            }
        }
        self.dag.remove_node(idx);
    }

    fn append(&mut self, op_type: OpType, qubits: (usize, usize), autocancel: bool, virtual_swap: bool) {
        let (q1, q2) = qubits;
        if virtual_swap && op_type == OpType::Swap {
            if !self.front.contains_key(&q1) && !self.front.contains_key(&q2) {
                if q1 < self.layout.len() && q2 < self.layout.len() {
                    self.layout.swap(q1, q2);
                }
                return;
            }
        }

        if autocancel {
            if let Some(fidx) = self.get_unique_front(qubits) {
                let fop = self.dag[fidx].clone();
                match (fop.op_type, op_type) {
                    (OpType::Swap, OpType::Swap) | (OpType::Cz, OpType::Cz) => {
                        self.remove_node(fidx);
                        return;
                    }
                    (OpType::Swap, OpType::Cx) => {
                        self.remove_node(fidx);
                        self.append(OpType::Cx, qubits, autocancel, virtual_swap);
                        self.append(
                            OpType::Cx,
                            (qubits.1, qubits.0),
                            autocancel,
                            virtual_swap,
                        );
                        return;
                    }
                    (OpType::Cx, OpType::Swap) => {
                        self.remove_node(fidx);
                        self.append(OpType::Swap, qubits, autocancel, virtual_swap);
                        self.append(
                            OpType::Cx,
                            (fop.qubits.1, fop.qubits.0),
                            autocancel,
                            virtual_swap,
                        );
                        return;
                    }
                    (OpType::Cx, OpType::Cx) if fop.qubits == qubits => {
                        self.remove_node(fidx);
                        return;
                    }
                    _ => {}
                }
            }
        }

        let op = CircuitOp {
            id: self.next_id,
            op_type,
            qubits,
        };
        self.next_id = self.next_id.saturating_add(1);
        let idx = self.dag.add_node(op);

        if let Some(prev) = self.front.get(&q1).copied() {
            self.dag.add_edge(prev, idx, ());
        }
        if let Some(prev) = self.front.get(&q2).copied() {
            if self.front.get(&q1).copied() != Some(prev) {
                self.dag.add_edge(prev, idx, ());
            }
        }
        self.front.insert(q1, idx);
        self.front.insert(q2, idx);
    }

    fn to_gate_list(&self) -> Vec<(OpType, (usize, usize))> {
        let mut indegree = HashMap::<NodeIndex, usize>::new();
        let mut ready = Vec::<NodeIndex>::new();
        for node in self.dag.node_indices() {
            let d = self.dag.neighbors_directed(node, Incoming).count();
            if d == 0 {
                ready.push(node);
            } else {
                indegree.insert(node, d);
            }
        }

        let mut out = Vec::<(OpType, (usize, usize))>::new();
        while !ready.is_empty() {
            ready.sort_by_key(|idx| self.dag[*idx].id);
            let node = ready.remove(0);
            let op = &self.dag[node];
            out.push((op.op_type, op.qubits));
            for succ in self.dag.neighbors_directed(node, Outgoing) {
                if let Some(d) = indegree.get_mut(&succ) {
                    *d = d.saturating_sub(1);
                    if *d == 0 {
                        indegree.remove(&succ);
                        ready.push(succ);
                    }
                }
            }
        }
        out
    }
}

#[derive(Clone)]
pub struct RoutingEnv {
    num_qubits: usize,
    coupling_map: Vec<(usize, usize)>,
    edge_set: HashSet<(usize, usize)>,
    loc_to_actions: Vec<Vec<usize>>,
    dists: Vec<Vec<DistType>>,
    dist_pairs: HashMap<DistType, Vec<(usize, usize)>>,
    all_dists: Vec<DistType>,

    difficulty: usize,
    len_slope: usize,
    _max_len: usize,
    min_len: usize,
    current_max_len: usize,

    num_active_swaps: usize,
    horizon: usize,
    metrics_weights: MetricsWeights,
    add_inverts: bool,
    layout_type: String,
    routing_ops_type: OpType,
    autocancel_ops: bool,

    target: Option<Vec<TargetTemplate>>,
    fixed_layout: Option<Vec<usize>>,
    sabre_layout: Option<Vec<usize>>,

    in_dag: StableDiGraph<DagOp, ()>,
    out_circuit: CircuitDag,
    qubits: Vec<usize>,
    locations: Vec<usize>,
    active_swaps: ActiveSwaps,
    qc_front: Vec<f32>,

    metrics_values: RoutingMetricsSnapshot,
    reward_value: f32,
    success: bool,
    is_final: bool,
    num_steps: usize,
}

impl RoutingEnv {
    pub fn new(
        num_qubits: usize,
        coupling_map: Vec<(usize, usize)>,
        dists: Vec<Vec<DistType>>,
        difficulty: usize,
        len_slope: usize,
        max_len: usize,
        min_len: usize,
        num_active_swaps: usize,
        horizon: usize,
        metrics_weights: MetricsWeights,
        add_inverts: bool,
        layout_type: String,
        routing_ops_type: String,
        autocancel_ops: bool,
    ) -> Self {
        let coupling_map = canonicalize_edges(num_qubits, coupling_map);
        let edge_set = coupling_map.iter().copied().collect::<HashSet<_>>();
        let loc_to_actions = build_loc_to_actions(num_qubits, &coupling_map);

        let dists = normalize_dists(num_qubits, dists, &coupling_map);
        let (dist_pairs, all_dists) = build_dist_pairs(num_qubits, &dists);

        let layout_type = layout_type.to_ascii_lowercase();
        let routing_ops_type = OpType::from_name(&routing_ops_type, OpType::Cz);

        let metrics_values = RoutingMetricsSnapshot::default();
        let mut env = Self {
            num_qubits,
            coupling_map,
            edge_set,
            loc_to_actions,
            dists,
            dist_pairs,
            all_dists,
            difficulty,
            len_slope,
            _max_len: max_len,
            min_len,
            current_max_len: max_len.max(min_len),
            num_active_swaps,
            horizon,
            metrics_weights,
            add_inverts,
            layout_type,
            routing_ops_type,
            autocancel_ops,
            target: None,
            fixed_layout: None,
            sabre_layout: None,
            in_dag: StableDiGraph::new(),
            out_circuit: CircuitDag::new(num_qubits),
            qubits: (0..num_qubits).collect(),
            locations: (0..num_qubits).collect(),
            active_swaps: ActiveSwaps::new(),
            qc_front: vec![0.0; 0],
            metrics_values,
            reward_value: 0.0,
            success: false,
            is_final: false,
            num_steps: 0,
        };
        env.set_difficulty(difficulty);
        env.reset();
        env
    }

    fn current_qc_front_value(&self) -> f32 {
        if self.layout_type == "trivial" {
            1.0
        } else {
            0.0
        }
    }

    fn edge_exists(&self, q1: usize, q2: usize) -> bool {
        let edge = canonical_edge(q1, q2);
        self.edge_set.contains(&edge)
    }

    fn get_dist(&self, q1: usize, q2: usize) -> DistType {
        self.dists
            .get(q1)
            .and_then(|row| row.get(q2))
            .copied()
            .unwrap_or(DIST_INF)
    }

    fn generate_random_target(&self) -> Vec<TargetTemplate> {
        let mut out = Vec::new();
        let mut remaining = self.difficulty as DistType;
        let mut rng = rand::thread_rng();

        while remaining > 0 {
            let valid_dists: Vec<DistType> = self
                .all_dists
                .iter()
                .copied()
                .filter(|d| *d > 0 && *d <= remaining)
                .collect();
            if valid_dists.is_empty() {
                break;
            }

            let next_dist = valid_dists[rng.gen_range(0..valid_dists.len())];
            if let Some(pairs) = self.dist_pairs.get(&next_dist) {
                let pair = pairs[rng.gen_range(0..pairs.len())];
                out.push(TargetTemplate {
                    op_type: self.routing_ops_type,
                    qubits: pair,
                });
                remaining -= next_dist;
            } else {
                break;
            }
        }

        out
    }

    fn apply_layout_to_target(&self, ops: &[TargetTemplate], layout: &[usize]) -> Vec<TargetTemplate> {
        ops.iter()
            .map(|op| TargetTemplate {
                op_type: op.op_type,
                qubits: (
                    layout.get(op.qubits.0).copied().unwrap_or(op.qubits.0),
                    layout.get(op.qubits.1).copied().unwrap_or(op.qubits.1),
                ),
            })
            .collect()
    }

    fn target_for_reset(&mut self) -> Vec<TargetTemplate> {
        let mut ops = if let Some(target) = &self.target {
            target.clone()
        } else {
            self.generate_random_target()
        };

        if self.target.is_some() {
            if self.layout_type == "random" {
                let mut layout: Vec<usize> = (0..self.num_qubits).collect();
                layout.shuffle(&mut rand::thread_rng());
                self.fixed_layout = Some(layout);
            } else if self.layout_type == "sabre" {
                if let Some(layout) = &self.sabre_layout {
                    self.fixed_layout = Some(layout.clone());
                }
            }
        }

        if let Some(layout) = &self.fixed_layout {
            if layout.len() == self.num_qubits {
                ops = self.apply_layout_to_target(&ops, layout);
            }
        }

        ops
    }

    fn build_dag(&self, ops: &[TargetTemplate]) -> StableDiGraph<DagOp, ()> {
        let mut dag = StableDiGraph::<DagOp, ()>::new();
        let mut front: Vec<Option<NodeIndex>> = vec![None; self.num_qubits];

        for op in ops {
            let idx = dag.add_node(DagOp {
                op_type: op.op_type,
                qubits: op.qubits,
            });
            let (q1, q2) = op.qubits;

            if let Some(prev) = front[q1] {
                dag.add_edge(prev, idx, ());
            }
            if let Some(prev) = front[q2] {
                if Some(prev) != front[q1] {
                    dag.add_edge(prev, idx, ());
                }
            }

            front[q1] = Some(idx);
            front[q2] = Some(idx);
        }

        dag
    }

    fn solve_targets(&mut self) -> usize {
        let mut solved = 0usize;
        let mut roots: VecDeque<NodeIndex> = self
            .in_dag
            .node_indices()
            .filter(|node| self.in_dag.neighbors_directed(*node, Incoming).next().is_none())
            .collect();

        while let Some(node) = roots.pop_front() {
            if self.in_dag.node_weight(node).is_none() {
                continue;
            }
            let op = self.in_dag[node].clone();
            let (q1, q2) = op.qubits;
            let l1 = self.locations[q1];
            let l2 = self.locations[q2];

            if self.edge_exists(l1, l2) {
                solved += 1;
                self.out_circuit.append(
                    op.op_type,
                    (l1, l2),
                    self.autocancel_ops,
                    false,
                );
                let successors: Vec<NodeIndex> =
                    self.in_dag.neighbors_directed(node, Outgoing).collect();
                self.in_dag.remove_node(node);
                for succ in successors {
                    if self.in_dag.node_weight(succ).is_none() {
                        continue;
                    }
                    if self.in_dag.neighbors_directed(succ, Incoming).next().is_none() {
                        roots.push_back(succ);
                    }
                }
            }
        }

        solved
    }

    fn compute_topological_layers(&self) -> Vec<Vec<NodeIndex>> {
        let mut indegrees = HashMap::<NodeIndex, usize>::new();
        let mut queue = VecDeque::<NodeIndex>::new();

        for node in self.in_dag.node_indices() {
            let indeg = self.in_dag.neighbors_directed(node, Incoming).count();
            if indeg == 0 {
                queue.push_back(node);
            } else {
                indegrees.insert(node, indeg);
            }
        }

        let mut layers: Vec<Vec<NodeIndex>> = Vec::new();
        while !queue.is_empty() && layers.len() < self.horizon {
            let mut this_layer: Vec<NodeIndex> = queue.drain(..).collect();
            if self.add_inverts {
                this_layer.shuffle(&mut rand::thread_rng());
            }

            let mut next_layer = VecDeque::<NodeIndex>::new();
            for node in &this_layer {
                for succ in self.in_dag.neighbors_directed(*node, Outgoing) {
                    if let Some(v) = indegrees.get_mut(&succ) {
                        *v -= 1;
                        if *v == 0 {
                            indegrees.remove(&succ);
                            next_layer.push_back(succ);
                        }
                    }
                }
            }
            layers.push(this_layer);
            queue = next_layer;
        }

        layers
    }

    fn choose_active_swaps(&self, layers: &[Vec<NodeIndex>]) -> Vec<usize> {
        let mut active_swaps = Vec::<usize>::new();
        let mut seen = HashSet::<usize>::new();

        for layer in layers {
            if active_swaps.len() >= self.num_active_swaps {
                break;
            }
            for node in layer {
                if active_swaps.len() >= self.num_active_swaps {
                    break;
                }
                let op = &self.in_dag[*node];
                let (q1, q2) = op.qubits;
                let l1 = self.locations[q1];
                let l2 = self.locations[q2];

                for l in [l1, l2] {
                    if l >= self.loc_to_actions.len() {
                        continue;
                    }
                    for action in &self.loc_to_actions[l] {
                        if seen.insert(*action) {
                            active_swaps.push(*action);
                            if active_swaps.len() >= self.num_active_swaps {
                                break;
                            }
                        }
                    }
                    if active_swaps.len() >= self.num_active_swaps {
                        break;
                    }
                }
            }
        }

        if self.add_inverts {
            active_swaps.shuffle(&mut rand::thread_rng());
        }
        active_swaps.truncate(self.num_active_swaps);
        active_swaps
    }

    fn get_obs_and_active_swaps(&self) -> (Vec<f32>, Vec<usize>) {
        let mut obs = vec![0.0f32; self.num_active_swaps * (self.horizon + 1)];
        let layers = self.compute_topological_layers();
        let active_swaps = self.choose_active_swaps(&layers);

        for (n, action_idx) in active_swaps.iter().enumerate() {
            let (l1, l2) = self.coupling_map[*action_idx];
            let last_col_idx = n * (self.horizon + 1) + self.horizon;
            obs[last_col_idx] = self.qc_front.get(*action_idx).copied().unwrap_or(0.0);

            for (h, layer) in layers.iter().enumerate() {
                if h >= self.horizon {
                    break;
                }
                let mut improve = 0.0f32;
                for node in layer {
                    let op = &self.in_dag[*node];
                    let (q1, q2) = op.qubits;
                    let ql1 = self.locations[q1];
                    let ql2 = self.locations[q2];

                    if ql1 == l1 && ql2 != l2 {
                        improve += (self.get_dist(l1, ql2) - self.get_dist(l2, ql2)) as f32;
                    } else if ql1 == l2 && ql2 != l1 {
                        improve += (self.get_dist(l2, ql2) - self.get_dist(l1, ql2)) as f32;
                    } else if ql2 == l1 && ql1 != l2 {
                        improve += (self.get_dist(l1, ql1) - self.get_dist(l2, ql1)) as f32;
                    } else if ql2 == l2 && ql1 != l1 {
                        improve += (self.get_dist(l2, ql1) - self.get_dist(l1, ql1)) as f32;
                    }
                }
                let idx = n * (self.horizon + 1) + h;
                obs[idx] = improve;
            }
        }

        (obs, active_swaps)
    }

    fn apply_swap(&mut self, action_idx: usize) {
        if action_idx >= self.coupling_map.len() {
            return;
        }
        let (l1, l2) = self.coupling_map[action_idx];
        if l1 >= self.qubits.len() || l2 >= self.qubits.len() {
            return;
        }

        self.qubits.swap(l1, l2);
        let q1 = self.qubits[l1];
        let q2 = self.qubits[l2];
        self.locations[q1] = l1;
        self.locations[q2] = l2;

        self.out_circuit.append(
            OpType::Swap,
            (l1, l2),
            true,
            self.layout_type != "trivial",
        );
    }

    fn recompute_metrics(&self) -> RoutingMetricsSnapshot {
        let mut n_cnots = 0usize;
        let mut n_gates = 0usize;
        let mut cnot_layers = HashSet::<usize>::new();
        let mut layers = HashSet::<usize>::new();
        let mut last_gates = vec![-1isize; self.num_qubits];
        let mut last_cxs = vec![-1isize; self.num_qubits];

        let cx = |q0: usize,
                  q1: usize,
                  n_cnots: &mut usize,
                  n_gates: &mut usize,
                  cnot_layers: &mut HashSet<usize>,
                  layers: &mut HashSet<usize>,
                  last_gates: &mut Vec<isize>,
                  last_cxs: &mut Vec<isize>| {
            *n_cnots = n_cnots.saturating_add(1);
            *n_gates = n_gates.saturating_add(1);
            let gate_layer = (last_gates[q0].max(last_gates[q1])) + 1;
            let cx_layer = (last_cxs[q0].max(last_cxs[q1])) + 1;
            last_gates[q0] = gate_layer;
            last_gates[q1] = gate_layer;
            last_cxs[q0] = cx_layer;
            last_cxs[q1] = cx_layer;
            cnot_layers.insert(cx_layer as usize);
            layers.insert(gate_layer as usize);
        };

        let gate = |q: usize,
                    n_gates: &mut usize,
                    layers: &mut HashSet<usize>,
                    last_gates: &mut Vec<isize>| {
            *n_gates = n_gates.saturating_add(1);
            let gate_layer = last_gates[q] + 1;
            last_gates[q] = gate_layer;
            layers.insert(gate_layer as usize);
        };

        for (op_type, qubits) in self.out_circuit.to_gate_list() {
            match op_type {
                OpType::Cx => cx(
                    qubits.0,
                    qubits.1,
                    &mut n_cnots,
                    &mut n_gates,
                    &mut cnot_layers,
                    &mut layers,
                    &mut last_gates,
                    &mut last_cxs,
                ),
                OpType::Swap => {
                    // Legacy routing_qc.py metric model: 3x same-direction CX.
                    for _ in 0..3 {
                        cx(
                            qubits.0,
                            qubits.1,
                            &mut n_cnots,
                            &mut n_gates,
                            &mut cnot_layers,
                            &mut layers,
                            &mut last_gates,
                            &mut last_cxs,
                        );
                    }
                }
                OpType::Cz => {
                    gate(qubits.1, &mut n_gates, &mut layers, &mut last_gates);
                    cx(
                        qubits.0,
                        qubits.1,
                        &mut n_cnots,
                        &mut n_gates,
                        &mut cnot_layers,
                        &mut layers,
                        &mut last_gates,
                        &mut last_cxs,
                    );
                    gate(qubits.1, &mut n_gates, &mut layers, &mut last_gates);
                }
            }
        }
        RoutingMetricsSnapshot {
            n_cnots,
            n_layers_cnots: cnot_layers.len(),
            n_layers: layers.len(),
            n_gates,
        }
    }

    fn routed(&self) -> bool {
        self.in_dag.node_count() == 0
    }

    fn reset_internal(&mut self) {
        self.in_dag = StableDiGraph::new();
        self.out_circuit = CircuitDag::new(self.num_qubits);
        self.qubits = (0..self.num_qubits).collect();
        self.locations = (0..self.num_qubits).collect();
        self.active_swaps.clear();
        self.qc_front = vec![self.current_qc_front_value(); self.coupling_map.len()];
        self.metrics_values = RoutingMetricsSnapshot::default();
        self.reward_value = 0.0;
        self.success = false;
        self.is_final = false;
        self.num_steps = 0;

        let target_ops = self.target_for_reset();
        self.in_dag = self.build_dag(&target_ops);

        self.solve_targets();
        self.success = self.routed();
        // Match legacy routing_qc.py behavior: reset never marks final, even if solved.
        self.is_final = false;
        self.reward_value = 0.0;
    }

    pub fn set_target_ops(&mut self, ops: Vec<(String, (usize, usize))>) {
        let target = ops
            .into_iter()
            .map(|(name, qubits)| TargetTemplate {
                op_type: OpType::from_name(&name, self.routing_ops_type),
                qubits,
            })
            .collect();
        self.target = Some(target);
    }

    pub fn clear_target_ops(&mut self) {
        self.target = None;
    }

    pub fn set_fixed_layout(&mut self, layout: Vec<usize>) {
        if layout.len() == self.num_qubits {
            self.fixed_layout = Some(layout);
        }
    }

    pub fn set_sabre_layout(&mut self, layout: Vec<usize>) {
        if layout.len() == self.num_qubits {
            self.sabre_layout = Some(layout.clone());
            self.fixed_layout = Some(layout);
        }
    }

    pub fn observe_float(&mut self) -> Vec<f32> {
        let (obs, active_swaps) = self.get_obs_and_active_swaps();
        self.active_swaps.set(active_swaps);
        obs
    }

    pub fn get_locations(&self) -> Vec<usize> {
        self.locations.clone()
    }

    pub fn get_qubits(&self) -> Vec<usize> {
        self.qubits.clone()
    }

    pub fn get_circuit(&self) -> Vec<(String, (usize, usize))> {
        self.out_circuit
            .to_gate_list()
            .iter()
            .map(|(op_type, qubits)| (op_type.as_name().to_string(), *qubits))
            .collect()
    }
}

impl Env for RoutingEnv {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn num_actions(&self) -> usize {
        let bits = self.num_active_swaps.min(20);
        1usize << bits
    }

    fn obs_shape(&self) -> Vec<usize> {
        // Sparse observation encoding for RL policies:
        // - first block: positive entries
        // - second block: negative entries
        vec![self.num_active_swaps.saturating_mul(2), self.horizon + 1]
    }

    fn set_difficulty(&mut self, difficulty: usize) {
        self.difficulty = difficulty;
        let scaled = self.difficulty.saturating_mul(self.len_slope);
        self.current_max_len = scaled.max(self.min_len).max(1);
    }

    fn get_difficulty(&self) -> usize {
        self.difficulty
    }

    fn set_state(&mut self, _state: Vec<i64>) {}

    fn reset(&mut self) {
        self.reset_internal();
    }

    fn step(&mut self, action: usize) {
        if self.is_final {
            return;
        }

        self.num_steps = self.num_steps.saturating_add(1);
        let active_swaps = self.active_swaps.snapshot();

        for base_action in 0..self.num_active_swaps {
            if base_action >= usize::BITS as usize {
                break;
            }
            if ((action >> base_action) & 1usize) == 0 {
                continue;
            }
            if base_action >= active_swaps.len() {
                continue;
            }
            let action_idx = active_swaps[base_action];
            self.apply_swap(action_idx);
        }

        self.solve_targets();
        let achieved_goal = self.routed();

        let mut penalty = 0.0f32;
        if self.target.is_none() || achieved_goal {
            let new_metrics = self.recompute_metrics();
            penalty = new_metrics.weighted_delta(&self.metrics_values, &self.metrics_weights);
            self.metrics_values = new_metrics;
        }

        let truncated = self.num_steps >= self.current_max_len;
        self.success = achieved_goal;
        self.is_final = achieved_goal || truncated;
        self.reward_value = if achieved_goal { 1.0 } else { 0.0 } - penalty;
    }

    fn masks(&self) -> Vec<bool> {
        vec![!self.is_final; self.num_actions()]
    }

    fn is_final(&self) -> bool {
        self.is_final
    }

    fn success(&self) -> bool {
        self.success
    }

    fn reward(&self) -> f32 {
        self.reward_value
    }

    fn observe(&self) -> Vec<usize> {
        let (obs, active_swaps) = self.get_obs_and_active_swaps();
        self.active_swaps.set(active_swaps);
        let base = self.num_active_swaps.saturating_mul(self.horizon + 1);
        let mut out = Vec::<usize>::new();

        for (idx, value) in obs.iter().enumerate() {
            if *value == 0.0 {
                continue;
            }
            let reps = value.abs().round().max(1.0) as usize;
            let mapped_idx = if *value > 0.0 {
                idx
            } else {
                idx.saturating_add(base)
            };
            out.extend(std::iter::repeat(mapped_idx).take(reps));
        }
        out
    }
}

#[pyclass(name = "RoutingEnv", extends = PyBaseEnv)]
pub struct PyRoutingEnv;

#[pymethods]
impl PyRoutingEnv {
    #[new]
    #[pyo3(signature = (
        num_qubits,
        coupling_map,
        dists,
        difficulty = 1,
        len_slope = 2,
        max_len = 128,
        min_len = 1,
        num_active_swaps = 9,
        horizon = 8,
        metrics_weights = None,
        add_inverts = true,
        layout_type = "trivial",
        routing_ops_type = "cz",
        autocancel_ops = false,
    ))]
    pub fn new(
        num_qubits: usize,
        coupling_map: Vec<(usize, usize)>,
        dists: Vec<Vec<DistType>>,
        difficulty: usize,
        len_slope: usize,
        max_len: usize,
        min_len: usize,
        num_active_swaps: usize,
        horizon: usize,
        metrics_weights: Option<HashMap<String, f32>>,
        add_inverts: bool,
        layout_type: &str,
        routing_ops_type: &str,
        autocancel_ops: bool,
    ) -> (Self, PyBaseEnv) {
        let env = RoutingEnv::new(
            num_qubits,
            coupling_map,
            dists,
            difficulty,
            len_slope,
            max_len,
            min_len,
            num_active_swaps,
            horizon,
            routing_metrics_weights(metrics_weights),
            add_inverts,
            layout_type.to_string(),
            routing_ops_type.to_string(),
            autocancel_ops,
        );
        (PyRoutingEnv, PyBaseEnv { env: Box::new(env) })
    }

    pub fn set_target(mut slf: PyRefMut<'_, Self>, ops: Vec<(String, (usize, usize))>) {
        if let Some(env) = slf.as_mut().env.as_any_mut().downcast_mut::<RoutingEnv>() {
            env.set_target_ops(ops);
        }
    }

    pub fn clear_target(mut slf: PyRefMut<'_, Self>) {
        if let Some(env) = slf.as_mut().env.as_any_mut().downcast_mut::<RoutingEnv>() {
            env.clear_target_ops();
        }
    }

    pub fn set_fixed_layout(mut slf: PyRefMut<'_, Self>, layout: Vec<usize>) {
        if let Some(env) = slf.as_mut().env.as_any_mut().downcast_mut::<RoutingEnv>() {
            env.set_fixed_layout(layout);
        }
    }

    pub fn set_sabre_layout(mut slf: PyRefMut<'_, Self>, layout: Vec<usize>) {
        if let Some(env) = slf.as_mut().env.as_any_mut().downcast_mut::<RoutingEnv>() {
            env.set_sabre_layout(layout);
        }
    }

    pub fn observe_float(mut slf: PyRefMut<'_, Self>) -> Vec<f32> {
        if let Some(env) = slf.as_mut().env.as_any_mut().downcast_mut::<RoutingEnv>() {
            env.observe_float()
        } else {
            Vec::new()
        }
    }

    pub fn get_locations(slf: PyRef<'_, Self>) -> Vec<usize> {
        if let Some(env) = slf.as_ref().env.as_any().downcast_ref::<RoutingEnv>() {
            env.get_locations()
        } else {
            Vec::new()
        }
    }

    pub fn get_qubits(slf: PyRef<'_, Self>) -> Vec<usize> {
        if let Some(env) = slf.as_ref().env.as_any().downcast_ref::<RoutingEnv>() {
            env.get_qubits()
        } else {
            Vec::new()
        }
    }

    pub fn get_circuit(slf: PyRef<'_, Self>) -> Vec<(String, (usize, usize))> {
        if let Some(env) = slf.as_ref().env.as_any().downcast_ref::<RoutingEnv>() {
            env.get_circuit()
        } else {
            Vec::new()
        }
    }
}

fn canonical_edge(q1: usize, q2: usize) -> (usize, usize) {
    if q1 < q2 {
        (q1, q2)
    } else {
        (q2, q1)
    }
}

fn canonicalize_edges(num_qubits: usize, edges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut out = Vec::<(usize, usize)>::new();
    let mut seen = HashSet::<(usize, usize)>::new();
    for (q1, q2) in edges {
        if q1 == q2 || q1 >= num_qubits || q2 >= num_qubits {
            continue;
        }
        let edge = canonical_edge(q1, q2);
        if seen.insert(edge) {
            out.push(edge);
        }
    }
    out
}

fn build_loc_to_actions(num_qubits: usize, coupling_map: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut loc_to_actions = vec![Vec::<usize>::new(); num_qubits];
    for (idx, (q1, q2)) in coupling_map.iter().enumerate() {
        loc_to_actions[*q1].push(idx);
        loc_to_actions[*q2].push(idx);
    }
    loc_to_actions
}

fn compute_dists_from_edges(num_qubits: usize, coupling_map: &[(usize, usize)]) -> Vec<Vec<DistType>> {
    let mut adjacency = vec![Vec::<usize>::new(); num_qubits];
    for (q1, q2) in coupling_map {
        adjacency[*q1].push(*q2);
        adjacency[*q2].push(*q1);
    }

    let mut dists = vec![vec![DIST_INF; num_qubits]; num_qubits];
    for src in 0..num_qubits {
        let mut queue = VecDeque::<usize>::new();
        dists[src][src] = 0;
        queue.push_back(src);

        while let Some(node) = queue.pop_front() {
            let next_dist = dists[src][node].saturating_add(1);
            for nb in &adjacency[node] {
                if dists[src][*nb] == DIST_INF {
                    dists[src][*nb] = next_dist;
                    queue.push_back(*nb);
                }
            }
        }
    }
    dists
}

fn normalize_dists(
    num_qubits: usize,
    dists: Vec<Vec<DistType>>,
    coupling_map: &[(usize, usize)],
) -> Vec<Vec<DistType>> {
    let valid_shape = dists.len() == num_qubits
        && dists
            .iter()
            .all(|row| row.len() == num_qubits);
    if !valid_shape {
        return compute_dists_from_edges(num_qubits, coupling_map);
    }

    let mut out = dists;
    for i in 0..num_qubits {
        out[i][i] = 0;
    }
    for i in 0..num_qubits {
        for j in (i + 1)..num_qubits {
            let a = out[i][j];
            let b = out[j][i];
            let v = if a >= 0 && b >= 0 {
                a.min(b)
            } else if a >= 0 {
                a
            } else if b >= 0 {
                b
            } else {
                DIST_INF
            };
            out[i][j] = v;
            out[j][i] = v;
        }
    }
    out
}

fn build_dist_pairs(
    num_qubits: usize,
    dists: &[Vec<DistType>],
) -> (HashMap<DistType, Vec<(usize, usize)>>, Vec<DistType>) {
    let mut dist_pairs: HashMap<DistType, Vec<(usize, usize)>> = HashMap::new();

    for q1 in 0..num_qubits {
        for q2 in (q1 + 1)..num_qubits {
            let d = dists[q1][q2];
            if d > 0 && d < DIST_INF {
                dist_pairs.entry(d).or_default().push((q1, q2));
            }
        }
    }

    let mut all_dists = dist_pairs.keys().copied().collect::<Vec<_>>();
    all_dists.sort_unstable();
    (dist_pairs, all_dists)
}

fn routing_metrics_weights(map: Option<HashMap<String, f32>>) -> MetricsWeights {
    let mut weights = MetricsWeights {
        n_cnots: 0.01,
        n_layers_cnots: 0.0,
        n_layers: 0.0,
        n_gates: 0.0,
    };
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

#[cfg(test)]
mod tests {
    use super::*;
    use twisterl::rl::env::Env;

    fn make_test_env(add_inverts: bool) -> RoutingEnv {
        RoutingEnv::new(
            3,
            vec![(0, 1), (1, 2)],
            vec![],
            1,
            2,
            16,
            1,
            2,
            2,
            routing_metrics_weights(None),
            add_inverts,
            "trivial".to_string(),
            "cz".to_string(),
            false,
        )
    }

    #[test]
    fn observe_updates_active_swap_mapping_used_by_step() {
        let mut env = make_test_env(false);
        env.set_target_ops(vec![("cz".to_string(), (0, 2))]);
        Env::reset(&mut env);
        assert!(!Env::success(&env));

        // Reset does not build action mapping; acting now should be a no-op.
        Env::step(&mut env, 1);
        assert!(!Env::success(&env));

        Env::reset(&mut env);
        let sparse_obs = Env::observe(&env);
        assert!(!sparse_obs.is_empty());

        // First bit maps to swap (0,1) for this setup, which solves target (0,2).
        Env::step(&mut env, 1);
        assert!(Env::success(&env));
    }

    #[test]
    fn clone_keeps_last_observed_active_swap_mapping() {
        let mut env = make_test_env(false);
        env.set_target_ops(vec![("cz".to_string(), (0, 2))]);
        Env::reset(&mut env);
        let _ = Env::observe(&env);

        let mut cloned = env.clone();
        Env::step(&mut cloned, 1);
        assert!(Env::success(&cloned));
    }
}
