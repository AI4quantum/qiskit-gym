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

use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;
use std::error::Error;

lazy_static! {
    static ref VALID_LABEL_PATTERN: Regex =
        Regex::new(r"^(?P<coeff>[+-]?[ij1]?)(?P<pauli>[IXYZ]*)$").unwrap();
    static ref CANONICAL_PHASE_LABEL: HashMap<&'static str, i32> = {
        let mut m = HashMap::new();
        m.insert("", 0);
        m.insert("-i", 1);
        m.insert("-", 2);
        m.insert("i", 3);
        m
    };
}

#[derive(Clone, Debug)]
pub struct Pauli {
    pub base_z: Vec<bool>,
    pub base_x: Vec<bool>,
    pub base_phase: i32,
    pub init_phase: i32,
}

impl Pauli {
    pub fn from_label(label: &str) -> Result<Pauli, Box<dyn Error>> {
        let captures = VALID_LABEL_PATTERN
            .captures(label)
            .ok_or_else(|| "Pauli string label is not valid.".to_string())?;

        let coeff = captures.name("coeff").map_or("", |m| m.as_str());
        let canonical_coeff = coeff.replace('1', "").replace('+', "").replace('j', "i");
        let phase = *CANONICAL_PHASE_LABEL
            .get(canonical_coeff.as_str())
            .ok_or_else(|| "Invalid Pauli phase".to_string())?;

        let pauli_str = captures.name("pauli").map_or("", |m| m.as_str());
        let pauli_bytes: Vec<u8> = pauli_str.bytes().rev().collect();

        let ys: Vec<bool> = pauli_bytes.iter().map(|&b| b == b'Y').collect();
        let base_x: Vec<bool> =
            pauli_bytes.iter().map(|&b| matches!(b, b'X' | b'Y')).collect();
        let base_z: Vec<bool> =
            pauli_bytes.iter().map(|&b| matches!(b, b'Z' | b'Y')).collect();
        let base_phase: i32 = (phase + ys.iter().filter(|&&y| y).count() as i32) % 4;

        Ok(Pauli {
            base_z,
            base_x,
            base_phase,
            init_phase: phase,
        })
    }

    pub fn evolve_h(&mut self, qubit: usize) {
        let x = self.base_x[qubit];
        let z = self.base_z[qubit];
        self.base_x[qubit] = z;
        self.base_z[qubit] = x;
        self.base_phase = (self.base_phase + 2 * ((x && z) as i32)) % 4;
    }

    pub fn evolve_s(&mut self, qubit: usize) {
        let x = self.base_x[qubit];
        self.base_z[qubit] ^= x;
        self.base_phase = (self.base_phase + (x as i32)) % 4;
    }

    pub fn evolve_cx(&mut self, qctrl: usize, qtrgt: usize) {
        self.base_x[qtrgt] ^= self.base_x[qctrl];
        self.base_z[qctrl] ^= self.base_z[qtrgt];
    }

    pub fn evolve_sx(&mut self, qubit: usize) {
        self.evolve_h(qubit);
        self.evolve_s(qubit);
        self.evolve_h(qubit);
    }

    pub fn commutes_with(&self, other: &Pauli) -> bool {
        let commutation = self
            .base_x
            .iter()
            .zip(&self.base_z)
            .zip(&other.base_x)
            .zip(&other.base_z)
            .fold(0, |acc, (((&x1, &z1), &x2), &z2)| {
                acc + ((x1 && z2) as i32 + (z1 && x2) as i32) % 2
            });
        commutation % 2 == 0
    }

    pub fn phase(&self) -> i32 {
        let mut num_ys = 0;
        for i in 0..self.base_z.len() {
            if self.base_z[i] && self.base_x[i] {
                num_ys += 1;
            }
        }
        (self.base_phase + (4 * self.base_z.len() as i32 - num_ys)) % 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_label() {
        let mut pauli = Pauli::from_label("iXYZ").unwrap();
        assert_eq!(pauli.base_z, vec![true, true, false]);
        assert_eq!(pauli.base_x, vec![false, true, true]);
        assert_eq!(pauli.base_phase, 0);

        pauli = Pauli::from_label("XYZ").unwrap();
        assert_eq!(pauli.base_z, vec![true, true, false]);
        assert_eq!(pauli.base_x, vec![false, true, true]);
        assert_eq!(pauli.base_phase, 1);

        pauli = Pauli::from_label("XYYIXIZ").unwrap();
        let expected_x = vec![false, false, true, false, true, true, true];
        let expected_z = vec![true, false, false, false, true, true, false];

        assert_eq!(pauli.base_z, expected_z);
        assert_eq!(pauli.base_x, expected_x);
        assert_eq!(pauli.base_phase, 2);

        pauli = Pauli::from_label("iXYYIXIZ").unwrap();
        assert_eq!(pauli.base_z, expected_z);
        assert_eq!(pauli.base_x, expected_x);
        assert_eq!(pauli.base_phase, 1);

        pauli = Pauli::from_label("-XYYIXIZ").unwrap();
        assert_eq!(pauli.base_z, expected_z);
        assert_eq!(pauli.base_x, expected_x);
        assert_eq!(pauli.base_phase, 0);

        pauli = Pauli::from_label("-iXYYIXIZ").unwrap();
        assert_eq!(pauli.base_z, expected_z);
        assert_eq!(pauli.base_x, expected_x);
        assert_eq!(pauli.base_phase, 3);
    }
}
