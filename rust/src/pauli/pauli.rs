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

use regex::Regex;
use std::collections::HashMap;
use std::error::Error;
use std::sync::OnceLock;

static VALID_LABEL_PATTERN: OnceLock<Regex> = OnceLock::new();
static CANONICAL_PHASE_LABEL: OnceLock<HashMap<&'static str, i32>> = OnceLock::new();

fn get_valid_label_pattern() -> &'static Regex {
    VALID_LABEL_PATTERN.get_or_init(|| {
        Regex::new(r"^(?P<coeff>[+-]?[ij1]?)(?P<pauli>[IXYZ]*)$").unwrap()
    })
}

fn get_canonical_phase_label() -> &'static HashMap<&'static str, i32> {
    CANONICAL_PHASE_LABEL.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("", 0);
        m.insert("-i", 1);
        m.insert("-", 2);
        m.insert("i", 3);
        m
    })
}

#[derive(Clone)]
pub struct Pauli {
    pub base_z: Vec<bool>,
    pub base_x: Vec<bool>,
    pub base_phase: i32,
    pub init_phase: i32,
}

impl Pauli {
    pub fn from_label(label: &str) -> Result<Pauli, Box<dyn Error>> {
        let captures = match get_valid_label_pattern().captures(label) {
            Some(caps) => caps,
            None => return Err("Pauli string label is not valid.".into()),
        };

        let coeff = captures.name("coeff").map_or("", |m| m.as_str());
        let canonical_coeff = coeff.replace("1", "").replace("+", "").replace("j", "i");
        let phase = *get_canonical_phase_label()
            .get(&canonical_coeff as &str)
            .ok_or("unknown phase coefficient")?;

        // Convert to Symplectic representation
        let pauli_str = captures.name("pauli").map_or("", |m| m.as_str());
        let pauli_bytes: Vec<u8> = pauli_str.bytes().rev().collect();

        let ys: Vec<bool> = pauli_bytes.iter().map(|&b| b == b'Y').collect();
        let base_x: Vec<bool> = pauli_bytes
            .iter()
            .map(|&b| b == b'X' || b == b'Y')
            .collect();
        let base_z: Vec<bool> = pauli_bytes
            .iter()
            .map(|&b| b == b'Z' || b == b'Y')
            .collect();
        let base_phase: i32 = (phase + ys.iter().filter(|&&y| y).count() as i32) % 4;

        Ok(Pauli {
            base_z,
            base_x,
            base_phase,
            init_phase: phase,
        })
    }

    pub fn evolve_h(&mut self, qubit: usize) {
        // Update P -> H.P.H
        let x = self.base_x[qubit];
        let z = self.base_z[qubit];
        self.base_x[qubit] = z;
        self.base_z[qubit] = x;
        self.base_phase = (self.base_phase + 2 * ((x && z) as i32)) % 4;
    }

    pub fn evolve_s(&mut self, qubit: usize) {
        // Update P -> S.P.Sdg
        let x = self.base_x[qubit];
        self.base_z[qubit] ^= x;
        self.base_phase = (self.base_phase + (x as i32)) % 4;
    }

    pub fn evolve_cx(&mut self, qctrl: usize, qtrgt: usize) {
        // Update P -> CX.P.CX
        self.base_x[qtrgt] ^= self.base_x[qctrl];
        self.base_z[qctrl] ^= self.base_z[qtrgt];
    }

    pub fn evolve_sx(&mut self, qubit: usize) {
        // Update P -> SX.P.SX
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
