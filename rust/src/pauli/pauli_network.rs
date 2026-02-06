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

use crate::envs::common::Gate;
use crate::pauli::pauli::Pauli;
use crate::pauli::pauli_dag::PauliDag;
use nalgebra::{DMatrix, DVector};
use std::error::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

#[derive(Clone)]
pub struct PauliNetwork {
    pub num_qubits: usize,
    pub rotation_qk: Vec<Pauli>,
    pub data: DMatrix<u8>,
    pub dag: PauliDag,
    goal: DMatrix<u8>,
}

impl PauliNetwork {
    pub fn new(target_clifford: Vec<u8>, rotations: Vec<String>) -> Self {
        let num_qubits = ((target_clifford.len() as f64).sqrt() as usize) / 2;
        let rotation_qk: Vec<Pauli> = rotations
            .iter()
            .map(|r| Pauli::from_label(&r.chars().collect::<String>()).unwrap())
            .collect();

        let tmp_data =
            DMatrix::from_vec(2 * num_qubits, 2 * num_qubits, target_clifford).transpose();
        let mut data = DMatrix::<u8>::zeros(2 * num_qubits, 2 * num_qubits + rotations.len());

        data.view_mut((0, 0), (2 * num_qubits, 2 * num_qubits))
            .copy_from(&tmp_data);

        for (i, pauli) in rotation_qk.iter().enumerate() {
            if pauli.base_x.len() != num_qubits {
                panic!(
                    "Number of qubits differ for Clifford ({} qubits) and Paulis ({} qubits)",
                    num_qubits,
                    pauli.base_x.len()
                );
            }

            let base_x = vec_to_dvector(&pauli.base_x);
            let base_z = vec_to_dvector(&pauli.base_z);
            data.view_mut((0, 2 * num_qubits + i), (num_qubits, 1))
                .copy_from(&base_x);
            data.view_mut((num_qubits, 2 * num_qubits + i), (num_qubits, 1))
                .copy_from(&base_z);
        }
        let dag = PauliDag::new(&rotation_qk);
        let goal = DMatrix::<u8>::identity(num_qubits * 2, num_qubits * 2);

        PauliNetwork {
            num_qubits,
            rotation_qk,
            data,
            dag,
            goal,
        }
    }

    fn is_rotation_trivial(&self, rotation_index: usize) -> bool {
        self.data
            .column(rotation_index + 2 * self.num_qubits)
            .rows(0, self.num_qubits)
            .iter()
            .zip(
                self.data
                    .column(rotation_index + 2 * self.num_qubits)
                    .rows(self.num_qubits, self.num_qubits)
                    .iter(),
            )
            .map(|(&x, &z)| x | z)
            .sum::<u8>()
            <= 1
    }

    fn which_qubit(&self, rotation_index: usize) -> usize {
        self.data
            .view(
                (0, rotation_index + 2 * self.num_qubits),
                (self.num_qubits, 1),
            )
            .iter()
            .zip(
                self.data
                    .view(
                        (self.num_qubits, rotation_index + 2 * self.num_qubits),
                        (self.num_qubits, 1),
                    )
                    .iter(),
            )
            .enumerate()
            .filter(|(_, (&x, &z))| x | z != 0)
            .map(|(i, _)| i)
            .next()
            .unwrap()
    }

    fn which_axis(&self, rotation_index: usize, qubit: usize) -> Result<Axis, Box<dyn Error>> {
        if self.data[(qubit, rotation_index + 2 * self.num_qubits)] == 1 {
            if self.data[(
                self.num_qubits + qubit,
                rotation_index + 2 * self.num_qubits,
            )] == 1
            {
                Ok(Axis::Y)
            } else {
                Ok(Axis::X)
            }
        } else if self.data[(
            self.num_qubits + qubit,
            rotation_index + 2 * self.num_qubits,
        )] == 1
        {
            Ok(Axis::Z)
        } else {
            Err("Invalid axis".into())
        }
    }

    pub fn clean_and_return_with_phases(&mut self) -> Vec<(Axis, usize, usize)> {
        let mut r_qubits: Vec<(Axis, usize, usize)> = Vec::new();
        let mut removed = true;

        while removed {
            removed = false;
            let mut nodes_to_remove = vec![];
            for &node_index in self.dag.get_front_layer().iter() {
                let rindex = *self.dag.node_weight(node_index).unwrap();
                if self.is_rotation_trivial(rindex) {
                    let qno = self.which_qubit(rindex);
                    let raxis = self.which_axis(rindex, qno).unwrap();
                    r_qubits.push((raxis, qno, rindex));
                    nodes_to_remove.push(node_index);
                    self.data.set_column(
                        rindex + 2 * self.num_qubits,
                        &DVector::zeros(2 * self.num_qubits),
                    );
                    removed = true;
                }
            }
            self.dag
                .retain_nodes(|_, node_index| !nodes_to_remove.contains(&node_index));
        }

        r_qubits
    }

    pub fn solved(&self) -> bool {
        self.dag.node_count() == 0
            && self
                .data
                .view((0, 0), (2 * self.num_qubits, 2 * self.num_qubits))
                == self.goal
    }

    /// Returns the rotation indices that are still in the DAG (non-trivial)
    pub fn active_rotation_indices(&self) -> Vec<usize> {
        self.dag
            .node_indices()
            .filter_map(|node_idx| self.dag.node_weight(node_idx).copied())
            .collect()
    }

    fn xor_rows(&mut self, row_a: usize, row_b: usize) {
        for col in 0..self.data.ncols() {
            self.data[(row_a, col)] ^= self.data[(row_b, col)];
        }
    }

    fn h(&mut self, i: usize) {
        self.data.swap_rows(i, self.num_qubits + i);
        for rotation in &mut self.rotation_qk {
            rotation.evolve_h(i);
        }
    }

    fn cnot(&mut self, i: usize, j: usize) -> Vec<(Axis, usize, usize)> {
        // Match transpiler's convention exactly:
        // row[i] ^= row[j], row[n+j] ^= row[n+i], evolve_cx(j, i)
        self.xor_rows(i, j);
        self.xor_rows(self.num_qubits + j, self.num_qubits + i);

        for rotation in &mut self.rotation_qk {
            rotation.evolve_cx(j, i);
        }

        self.clean_and_return_with_phases()
    }

    fn s(&mut self, i: usize) {
        self.xor_rows(self.num_qubits + i, i);

        for rotation in &mut self.rotation_qk {
            rotation.evolve_s(i);
        }
    }

    fn sx(&mut self, i: usize) {
        self.xor_rows(i, self.num_qubits + i);

        for rotation in &mut self.rotation_qk {
            rotation.evolve_sx(i);
        }
    }

    pub fn act(&mut self, gate: &Gate) -> Vec<(Axis, usize, usize)> {
        match gate {
            Gate::H(q) => self.h(*q),
            Gate::S(q) => self.s(*q),
            Gate::Sdg(q) => {
                // Sdg = S^3
                self.s(*q);
                self.s(*q);
                self.s(*q);
            }
            Gate::SX(q) => self.sx(*q),
            Gate::SXdg(q) => {
                // SXdg = SX^3
                self.sx(*q);
                self.sx(*q);
                self.sx(*q);
            }
            Gate::CX(q0, q1) => return self.cnot(*q0, *q1),
            Gate::CZ(q0, q1) => {
                // CZ = H[1] CX H[1]
                self.h(*q1);
                let result = self.cnot(*q0, *q1);
                self.h(*q1);
                return result;
            }
            Gate::SWAP(q0, q1) => {
                // SWAP = CX CX CX (alternating directions)
                let mut rotations = Vec::new();
                rotations.extend(self.cnot(*q0, *q1));
                rotations.extend(self.cnot(*q1, *q0));
                rotations.extend(self.cnot(*q0, *q1));
                return rotations;
            }
        }
        Vec::new()
    }
}

fn vec_to_dvector(vec: &Vec<bool>) -> DVector<u8> {
    DVector::from_vec(vec.iter().map(|&b| if b { 1 } else { 0 }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_pauli_network_creation() {
        let target_clifford =
            DMatrix::<u8>::from_row_slice(4, 4, &[1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1])
                .transpose();
        let rotations = vec!["XY".to_string()];
        let pauli_network = PauliNetwork::new(target_clifford.data.as_slice().to_vec(), rotations);

        assert_eq!(pauli_network.num_qubits, 2);
        assert_eq!(pauli_network.num_qubits, 2);
        assert_eq!(pauli_network.data.shape(), (4, 5));

        let expected_vec = vec![1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1];
        assert_eq!(pauli_network.data.as_slice(), expected_vec.as_slice());
    }

    #[test]
    fn test_is_rotation_trivial() {
        let target_clifford = DMatrix::<u8>::identity(4, 4);
        let rotations = vec!["XY".to_string(), "YZ".to_string(), "YI".to_string()];
        let network = PauliNetwork::new(target_clifford.data.as_slice().to_vec(), rotations);

        assert_eq!(network.is_rotation_trivial(0), false);
        assert_eq!(network.is_rotation_trivial(1), false);
        assert_eq!(network.is_rotation_trivial(2), true);
    }

    #[test]
    fn test_which_qubit() {
        let target_clifford = DMatrix::<u8>::identity(8, 8);
        let rotations = vec!["IIYX".to_string(), "IXYI".to_string(), "XZYX".to_string()];
        let network = PauliNetwork::new(target_clifford.data.as_slice().to_vec(), rotations);

        let mut qubit = network.which_qubit(0);
        assert_eq!(qubit, 2);
        qubit = network.which_qubit(1);
        assert_eq!(qubit, 1);
        qubit = network.which_qubit(2);
        assert_eq!(qubit, 0);
    }

    #[test]
    fn test_which_axis() {
        let target_clifford = DMatrix::<u8>::identity(8, 8);
        let rotations = vec!["IIZX".to_string(), "IYYX".to_string(), "XZYX".to_string()];
        let network = PauliNetwork::new(target_clifford.data.as_slice().to_vec(), rotations);
        let axis = network.which_axis(0, 2);
        assert_eq!(axis.unwrap(), Axis::Z);
        let axis = network.which_axis(1, 1);
        assert_eq!(axis.unwrap(), Axis::Y);
        let axis = network.which_axis(2, 3);
        assert_eq!(axis.unwrap(), Axis::X);
        let axis = network.which_axis(1, 0);
        assert!(axis.is_err());
    }

    #[test]
    fn test_clean_and_return_with_phases() {
        let target_clifford = DMatrix::<u8>::identity(4, 4);
        let rotations = vec!["IX".to_string(), "YZ".to_string()];
        let mut network = PauliNetwork::new(target_clifford.data.as_slice().to_vec(), rotations);
        assert_eq!(network.which_axis(0, 1).unwrap(), Axis::X);

        let r_qubits = network.clean_and_return_with_phases();
        assert_eq!(r_qubits.len(), 1);
        assert_eq!(r_qubits[0].0, Axis::X);
        assert_eq!(r_qubits[0].1, 1);
        assert_eq!(r_qubits[0].2, 0);

        let r_qubits = network.clean_and_return_with_phases();
        assert_eq!(r_qubits.len(), 0);
    }

    #[test]
    fn test_clean_and_return_with_phases_node_removal() {
        let target_clifford = DMatrix::<u8>::identity(8, 8);
        let rotations = vec!["YIII".to_string(), "YIII".to_string()];
        let mut network = PauliNetwork::new(target_clifford.data.as_slice().to_vec(), rotations);

        let r_qubits = network.clean_and_return_with_phases();
        assert_eq!(r_qubits.len(), 2);
        assert_eq!(r_qubits[0].0, Axis::Y);
        assert_eq!(r_qubits[0].1, 0);
        assert_eq!(r_qubits[0].2, 0);
    }

    #[test]
    fn test_is_over() {
        let target_clifford = DMatrix::<u8>::identity(4, 4);
        let rotations = vec!["XI".to_string()];
        let mut network = PauliNetwork::new(target_clifford.data.as_slice().to_vec(), rotations);

        assert_eq!(network.solved(), false);

        network.clean_and_return_with_phases();
        assert_eq!(network.solved(), true);
    }

    #[test]
    fn test_act_h_gate() {
        let target_clifford = DMatrix::<u8>::identity(4, 4);
        let rotations = vec!["XY".to_string()];
        let mut network = PauliNetwork::new(target_clifford.data.as_slice().to_vec(), rotations);

        network.act(&Gate::H(0));
        assert_eq!(network.data[(0, 0)], 0);
    }

    #[test]
    fn test_act_cnot_gate() {
        let target_clifford = DMatrix::<u8>::identity(4, 4);
        let rotations = vec!["XY".to_string()];
        let mut network = PauliNetwork::new(target_clifford.data.as_slice().to_vec(), rotations);

        let r_qubits = network.act(&Gate::CX(0, 1));
        assert_eq!(r_qubits.len(), 1);
        assert_eq!(r_qubits[0].0, Axis::Y);
        assert_eq!(r_qubits[0].1, 1);
        assert_eq!(r_qubits[0].2, 0);
    }

    #[test]
    fn test_swap_accumulates_trivial_rotations() {
        let target_clifford = DMatrix::<u8>::identity(4, 4);
        let rotations = vec!["XI".to_string()];
        let base = PauliNetwork::new(target_clifford.data.as_slice().to_vec(), rotations);

        let mut sequential = base.clone();
        let mut expected = Vec::new();
        expected.extend(sequential.act(&Gate::CX(0, 1)));
        expected.extend(sequential.act(&Gate::CX(1, 0)));
        expected.extend(sequential.act(&Gate::CX(0, 1)));
        assert!(!expected.is_empty());

        let mut swapped = base.clone();
        let got = swapped.act(&Gate::SWAP(0, 1));

        assert_eq!(got, expected);
    }
}
