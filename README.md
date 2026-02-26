# Parsex

An HPC circuit simulation engine under development designed to simulate and solve hundreds/thousands of circuits across CPU/GPU cores in massive parallel batches.

Current Version: **0.3.0**

## 🪐 Features

1.  **Unified SoA Architecture**: Implements Structure-of-Arrays (SoA) memory layout (`A[row][col][batch]`) across both backends, ensuring contiguous memory access for batch coefficients.
    
2.  **SIMD-Batched CPU Solver**: Utilizes OpenMP SIMD to map batch indices to vector lanes (AVX2/AVX-512), processing multiple circuits per instruction.

3.  **Custom Kernel GPU Solver**: Uses a custom CUDA kernel with one thread per circuit. Includes independent circuit partial pivoting for numerical stability. SoA layout enables coalesced global memory access.

4.  **Zero-Copy Python ML Bindings**: Uses `pybind11` to expose the C++ HPC batching arrays directly to Python, allowing instantaneous data generation throughput for Machine Learning and PyTorch.

The system calculates comprehensive simulation metrics such as node voltages, branch currents, and component power dissipation, utilizing a JSON-based data format for seamless input and output parsing.

## 🌌 Usage

Requires a C++ compiler with C++23 support (e.g., Clang 15+, GCC 13+). To enable GPU support, the CUDA Toolkit is required.

### Build

We highly recommend using our tracked Conda environment to instantly resolve `pybind11` and `cmake` for compiling the Python module.

```bash
conda env create -f parsexenv.yml
conda activate parsexenv

mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

### Run Batch Solver
Run the solver on a directory of circuit JSON files:

```bash
# CPU Solver (OpenMP)
./parsex_batch --cpu circuits/

# GPU Solver (CUDA)
./parsex_batch --gpu circuits/
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
