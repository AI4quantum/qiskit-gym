# Qiskit Gym

Gymnasium-compatible environments for quantum circuit synthesis, powered by Qiskit and TwisteRL.

## Project Overview

- **Environments**: `PermutationGym`, `LinearFunctionGym`, `CliffordGym`.
- **Backend**: Uses `twisteRL` for RL algorithms.
- **Hardware-Aware**: Environments respect `CouplingMap` constraints.

## Setup & Build Commands

```bash
# Install (editable mode, compiles Rust extension via maturin)
pip install -e .

# Verify installation
python -c "import qiskit_gym; print('OK')"
```

## Code Style & Conventions

- **Python**: PEP 8. Type hints required for public APIs.
- **Gymnasium**: Strictly adhere to `step() -> (obs, reward, terminated, truncated, info)`.
- **Qiskit**: Use `QuantumCircuit` and `CouplingMap` from Qiskit.

## Testing Instructions

```bash
# Verify imports
python -c "import qiskit_gym"

# Run example notebook (manual)
jupyter notebook examples/intro.ipynb
```

## Directory Structure

```
qiskit-gym/
├── rust/              # Rust accelerators (maturin)
├── src/qiskit_gym/    # Python package
│   ├── envs/          # Gym environments
│   └── rl/            # RL configs and synthesis
├── examples/          # Notebooks and scripts
└── pyproject.toml     # Build config (maturin)
```

## Dev Environment Tips

- **Dependency**: Ensure `twisterl` version matches (`~=0.5.1` or local).
- **Rewards**: 1.0 for success minus gate penalties; configure via `MetricsWeights`.
- **New Envs**: Subclass `gymnasium.Env` and follow existing patterns in `envs/`.
