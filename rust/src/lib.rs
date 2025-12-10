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
pub mod envs;

use crate::envs::channel3::PyChannel3Env;
use crate::envs::clifford::PyCliffordEnv;
use crate::envs::linear_function::PyLinearFunctionEnv;
use crate::envs::permutation::PyPermutationEnv;


#[pymodule]
fn qiskit_gym_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCliffordEnv>()?;
    m.add_class::<PyLinearFunctionEnv>()?;
    m.add_class::<PyPermutationEnv>()?;
    m.add_class::<PyChannel3Env>()?;
    Ok(())
}