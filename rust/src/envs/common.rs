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
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::types::PySequence;

#[derive(Clone, Debug)]
pub enum Gate {
    H(usize),
    S(usize),
    Sdg(usize),
    SX(usize),
    SXdg(usize),
    CX(usize, usize),
    CZ(usize, usize),
    SWAP(usize, usize),
}

impl std::fmt::Display for Gate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Gate::H(i) => write!(f, r#"("H", [{}])"#, i),
            Gate::S(i) => write!(f, r#"("S", [{}])"#, i),
            Gate::Sdg(i) => write!(f, r#"("Sdg", [{}])"#, i),
            Gate::SX(i) => write!(f, r#"("SX", [{}])"#, i),
            Gate::SXdg(i) => write!(f, r#"("SXdg", [{}])"#, i),
            Gate::CX(i, j) => write!(f, r#"("CX", [{}, {}])"#, i, j),
            Gate::CZ(i, j) => write!(f, r#"("CZ", [{}, {}])"#, i, j),
            Gate::SWAP(i, j) => write!(f, r#"("SWAP", [{}, {}])"#, i, j),
        }
    }
}

impl<'py> FromPyObject<'py> for Gate {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        // Expect (name, indices)
        let pair = ob
            .downcast::<PySequence>()
            .map_err(|_| PyTypeError::new_err("Each gate must be a 2-item sequence: (name, indices)"))?;
        if pair.len()? != 2 {
            return Err(PyValueError::new_err(
                "Each gate must have exactly 2 items: (name, indices)",
            ));
        }

        let name_obj = pair.get_item(0)?;
        let mut name: String = name_obj
            .extract()
            .map_err(|_| PyTypeError::new_err("Gate name must be a string"))?;
        name = name.trim().to_string();
        let key = name.to_ascii_lowercase();

        let idx_obj = pair.get_item(1)?;
        let idx_seq = idx_obj
            .downcast::<PySequence>()
            .map_err(|_| PyTypeError::new_err("Gate indices must be a list/tuple of integers"))?;
        let n = idx_seq.len()?;
        let mut idx = Vec::with_capacity(n);
        for i in 0..n {
            idx.push(idx_seq.get_item(i)?.extract::<usize>().map_err(|_| {
                PyTypeError::new_err("Gate indices must be non-negative integers (usize)")
            })?);
        }

        match (key.as_str(), idx.as_slice()) {
            ("h",    [i])     => Ok(Gate::H(*i)),
            ("s",    [i])     => Ok(Gate::S(*i)),
            ("sdg",  [i])     => Ok(Gate::Sdg(*i)),
            ("sx",   [i])     => Ok(Gate::SX(*i)),
            ("sxdg", [i])     => Ok(Gate::SXdg(*i)),
            ("cx",   [i, j])  => Ok(Gate::CX(*i, *j)),
            ("cz",   [i, j])  => Ok(Gate::CZ(*i, *j)),
            ("swap", [i, j])  => Ok(Gate::SWAP(*i, *j)),
            ("h" | "s" | "sdg" | "sx" | "sxdg", _) => Err(PyValueError::new_err(format!(
                "Gate `{}` expects 1 index, got {}", name, idx.len()
            ))),
            ("cx" | "cz" | "swap", _) => Err(PyValueError::new_err(format!(
                "Gate `{}` expects 2 indices, got {}", name, idx.len()
            ))),
            _ => Err(PyValueError::new_err(format!(
                "Unknown gate name `{}`. Allowed: H, S, Sdg, SX, SXdg, CX, CZ, SWAP",
                name
            ))),
        }
    }
}


