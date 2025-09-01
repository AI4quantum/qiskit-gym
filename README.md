# Qiskit Gym 🏋️‍♀️

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.1+-purple.svg)](https://qiskit.org/)

**Qiskit Gym** is a collection of quantum information science problems formulated as reinforcement learning environments. 

## 🚀 Key Features
Train AI agents to discover efficient and near-optimal quantum circuit implementations for various quantum computing tasks.

- **🎯 Purpose-Built RL Environments**: Specialized Gymnasium-compatible environments for quantum circuit synthesis problems
- **🔧 Hardware-Aware Environments**: Design environments matching real quantum hardware coupling maps
- **🔄 Flexible Gate Sets**: Define custom gate sets as part of your environments
- **🧠 State-of-the-Art RL Integration**: Built-in support for PPO, Alpha Zero, and custom policy configurations
- **⚡ High-Performance Backend**: Rust-powered core for lightning-fast quantum circuit operations  
- **💾 Model Persistence**: Save and load trained models for deployment
- **⚖️ Scalable Training**: Efficient parallel environment execution
- **📊 Built-in TensorBoard Integration**: For training visualization and performance tracking
- **🔌 Qiskit integration**: You can use Qiksit seamlessly to define your problem and use your trained models.

## 🎲 Supported Problem Types

### 🔄 **Permutation Synthesis**
Learn to implement arbitrary qubit permutations using minimal SWAP gates on constrained coupling maps.

### 🔢 **Linear Function Synthesis** 
Decompose linear Boolean functions into efficient quantum circuits using CNOT gates.

### 🌊 **Clifford Synthesis**
Generate optimal implementations of Clifford group elements with customizable gate sets.

## 🔭 Roadmap and Vision
There are lots of interesting quantum computing problems that can be addressed with RL. We hope to centralize the problem formulations here to make it easier for Quantum and AI researchers to collaborate in taclikng them.

For future releases we plan to include envs for:

- Pauli Network synthesis
- Qubit routing
- Clifford+T synthesis

We are also open to env requests and contributions!

Stay tuned!

## 🏃‍♂️ Quick Start

Get started with quantum circuit synthesis RL in minutes! Check out our comprehensive tutorial:

**👉 [examples/intro.ipynb](examples/intro.ipynb) - Interactive Introduction Notebook**

This Jupyter notebook walks you through:
- Setting up different synthesis environments
- Training RL agents from scratch  
- Evaluating and using trained models
- Visualizing quantum circuits and training progress

## 🛠️ Installation

```bash
git clone https://github.com/AI4quantum/qiskit-gym.git
cd qiskit-gym
pip install -e .
```

## 💡 Example Usage

Here's how to set up a permutation synthesis environment and train an RL agent:

```python
from qiskit_gym.envs import PermutationGym
from qiskit_gym.rl import RLSynthesis, PPOConfig, BasicPolicyConfig
from qiskit.transpiler import CouplingMap

# Create a 3x3 grid coupling map environment
cmap = CouplingMap.from_grid(3, 3, bidirectional=False)
env = PermutationGym.from_coupling_map(cmap)

# Set up RL synthesis with PPO
rls = RLSynthesis(env, PPOConfig(), BasicPolicyConfig())

# Train the agent
rls.learn(num_iterations=100, tb_path="runs/my_experiment/")

# Use the trained model to synthesize circuits
import numpy as np
random_permutation = np.random.permutation(9)
optimized_circuit = rls.synth(random_permutation, num_searches=1000)
```

## 🤝 Contributing

We welcome contributions! Whether you're adding new synthesis problems, improving RL algorithms, or enhancing documentation - every contribution helps advance quantum computing research.

## 📝 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE.txt) for details.


## 🧪 Relevant Papers

- Kremer, D., Villar, V., Paik, H., Duran, I., Faro, I., & Cruz-Benito, J. (2024). Practical and efficient quantum circuit synthesis and transpiling with reinforcement learning. arXiv preprint [arXiv:2405.13196](https://arxiv.org/abs/2405.13196).

- Dubal, A., Kremer, D., Martiel, S., Villar, V., Wang, D., & Cruz-Benito, J. (2025). Pauli Network Circuit Synthesis with Reinforcement Learning. arXiv preprint [arXiv:2503.14448](https://arxiv.org/abs/2503.14448). 