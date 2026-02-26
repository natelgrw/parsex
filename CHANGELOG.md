Version **0.3.0**
Date Released: **02/25/2026**

- Integrated Pybind11 `.so` module compilation to expose C++ structures securely to Python.
- Adopted Conda environments (`parsexenv.yml`) for seamless deployment and PyTorch binding tests.
- Implemented independent partial pivoting and host memory extraction for the CUDA GPU Kernel algorithm.
- Standardized C++ preprocessor syntax block-alignment and finalized uniform `snake_case` taxonomy changes across header and source files.
Version **0.2.0**

Date Released: **02/16/2026**

- Unified SoA architecture for both CPU and GPU solvers
- SIMD-batched CPU solver using OpenMP SIMD (AVX2/AVX-512)
- Custom kernel GPU solver using CUDA with one thread per circuit

Version **0.1.1**

Date Released: **02/04/2026**

- Parallel MNA processing for batch circuit simulation

Version **0.1.0**

Date Released: **12/04/2025**

- Initial release of basic MNA digitial circuit solver
- Circuits simulated one at a time
- Components restricted to resistors, ideal voltage sources, and ideal current sources