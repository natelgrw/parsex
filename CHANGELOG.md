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