# -*- coding: utf-8 -*-

# (C) Copyright 2025 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

try:
    import qiskit_gym_rs as _qiskit_gym_rs
except ImportError:
    from . import qiskit_gym_rs as _qiskit_gym_rs

qiskit_gym_rs = _qiskit_gym_rs

__all__ = ["qiskit_gym_rs"]
