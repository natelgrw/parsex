#ifndef BATCHSOLVER_HPP
#define BATCHSOLVER_HPP

#include "MNASolver.hpp"
#include <vector>

// Batched solver function
// Solves all circuit graphs in the vector using OpenMP parallelization
// Returns a vector of SimulationResults corresponding to the input graphs
std::vector<SimulationResult> solve_batch(const std::vector<CircuitGraph>& graphs);

#endif // BATCHSOLVER_HPP
