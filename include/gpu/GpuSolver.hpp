#ifndef GPUSOLVER_HPP
#define GPUSOLVER_HPP

#include "MNASolver.hpp"
#include <vector>

namespace GpuSolver {

// Batched solver function using CUDA
// Solves all circuit graphs in the vector using GPU acceleration
// Returns a vector of SimulationResults corresponding to the input graphs
std::vector<SimulationResult> solve_batch(const std::vector<CircuitGraph>& graphs);

} // namespace GpuSolver

#endif // GPUSOLVER_HPP
