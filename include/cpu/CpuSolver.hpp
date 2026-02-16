#ifndef CPUSOLVER_HPP
#define CPUSOLVER_HPP

#include "MNASolver.hpp"
#include <vector>

namespace CpuSolver {

// Batched solver function
// Solves all circuit graphs in the vector using OpenMP parallelization
// Returns a vector of SimulationResults corresponding to the input graphs
std::vector<SimulationResult> solve_batch(const std::vector<CircuitGraph>& graphs);

} // namespace CpuSolver

#endif // CPUSOLVER_HPP
