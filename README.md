# Parsex Compiler

A HPC circuit solving compiler under development designed to batch hundreds/thousands of circuits across CPU/GPU cores into single HPC computations. Currently includes a C++23 reference MNA solver.

Current Version: **0.1.0**

## 🪐 Features

The compiler features a core solver built on custom Gaussian Elimination, eliminating the need for external linear algebra dependencies. Its MNA engine handles linear components including ideal voltage sources, current sources, and resistors. 

The system calculates comprehensive simulation metrics such as node voltages, branch currents, and component power dissipation, utilizing a JSON-based data format for seamless input and output parsing.

## 🌌 Usage

Requires a C++ compiler with C++23 support (e.g., Clang 15+, GCC 13+).

```bash
g++ -std=c++2b -Iinclude src/main.cpp src/MNASolver.cpp -o circuit_solver
```

Run the solver on a circuit JSON file:

```bash
./circuit_solver circuits/voltage_divider.json
```

Results are saved to `results/<name>_sol.json`:

```json
{
  "circuit_name": "voltage_divider",
  "results": {
    "voltages": { "1": 12.0, "2": 8.0 },
    "currents": { "R1": 0.004 },
    "power": { "R1": 0.016 }
  }
}
```
