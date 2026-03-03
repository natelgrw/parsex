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
# IMPORTANT: compile in release mode and point CMake to the Conda Python executable
cmake .. -DPython_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

### Run Batch Solver
Use the standalone C++ solver to test simulations. You can limit the batch size and specify the topology file:

```bash
# CPU solver (OpenMP) executing 1 Million parallel variations of the voltage divider
./parsex_batch --cpu --batch=1000000 --topo=voltage_divider.json

# GPU solver (CUDA) executing 500k variations
./parsex_batch --gpu --batch=500000 --topo=voltage_divider.json
```

### PyTorch Python Bindings (Troubleshooting)
If streaming data directly from Parsex into a PyTorch neural network inside a Jupyter Notebook on macOS, you may experience a silent kernel hang. This happens because Parsex's C++ OpenMP thread pool collides with PyTorch's `Accelerate` backend.

To resolve this:
1. **Force PyTorch to use Apple Silicon (MPS):** Move your PyTorch model and tensors to the GPU (`.to("mps")`).
2. **Restrict PyTorch to 1 CPU thread:** Add this to the absolute top of your Python file to break the deadlock:
   ```python
   import os
   os.environ["OMP_NUM_THREADS"] = "1"
   import torch
   torch.set_num_threads(1)
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
