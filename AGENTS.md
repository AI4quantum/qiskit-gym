# Guidance for coding agents

Read [README.md](README.md) for the project overview, installation, and example usage. This file covers code entry points and how to check changes.

## Setup and dependencies

- Use Python 3.10 or newer, as required by `pyproject.toml`.
- Install a Rust toolchain (for example, via [rustup](https://rustup.rs/)) to build the PyO3 extension.
- The Python package requires `twisterl~=0.5.1`, and the Rust crate requires `twisterl = "~0.5.1"`. Check both `pyproject.toml` and `rust/Cargo.toml` when changing this dependency.

For local `twisterl` development, install its Python checkout in editable mode. Rust changes also require a matching local path dependency in `rust/Cargo.toml`.

Follow the installation steps in the README. `pip install -e .` builds the Rust extension through maturin. For iterative Rust changes, activate a virtual environment, install maturin, and run this from the repository root:

```bash
maturin develop --manifest-path rust/Cargo.toml
```

## Code map

- `rust/src/envs/` implements the synthesis environments; `rust/src/lib.rs` exports their PyO3 classes in `qiskit_gym_rs`.
- `src/qiskit_gym/envs/adapters.py` wraps a raw Rust environment as a Gymnasium environment. Its `reset()` returns `(obs, info)`, and `step()` returns `(obs, reward, terminated, truncated, info)`.
- `src/qiskit_gym/envs/synthesis.py` defines `BaseSynthesisEnv` and the Qiskit-facing `PermutationGym`, `LinearFunctionGym`, and `CliffordGym` classes. These convert Qiskit inputs to backend states and, where needed, post-process the resulting circuit.
- `src/qiskit_gym/rl/synthesis.py` defines `RLSynthesis` for training and circuit synthesis. `src/qiskit_gym/rl/configs.py` defines `PPOConfig`, `AlphaZeroConfig`, and policy configs used with `twisterl`.

## Adding an environment

1. Add the Rust environment under `rust/src/envs/`, following an existing implementation, and expose its PyO3 class through `rust/src/envs/mod.rs` and `rust/src/lib.rs`.
2. In `src/qiskit_gym/envs/synthesis.py`, wrap the PyO3 class with `gym_adapter`, then define a `BaseSynthesisEnv` subclass with its allowed gates and `get_state()` conversion. Add `post_process_synthesis()` if the returned circuit needs adjustment.
3. Export the class from `src/qiskit_gym/envs/__init__.py`. Add it to `SYNTH_ENVS` so `RLSynthesis.from_config_json()` can load saved configurations.
4. Rebuild the extension and manually check `reset()`, `step()`, and a small synthesis example with the new environment.

Follow PEP 8 in Python code. Type hints are preferred for public APIs; the repository does not enforce them with a linter or type checker.

## Verification

There is no automated test suite or test CI workflow at present. After installing or rebuilding the package, verify imports with:

```bash
python -c "import qiskit_gym; from qiskit_gym.envs import PermutationGym, LinearFunctionGym, CliffordGym"
```

For changes affecting the documented workflows, open `examples/intro.ipynb` with `jupyter notebook examples/intro.ipynb` and run the relevant cells manually. Jupyter is needed for this optional check.
