/*
 * cpu_solver.hpp
 * 
 * Author: natelgrw
 * Last Edited: 02/25/2026
 * 
 * A header file for the CPU solver namespace.
*/

#ifndef CPUSOLVER_HPP
#define CPUSOLVER_HPP

#include "mna_solver.hpp"
#include <vector>

namespace CpuSolver {
// batched solver function
std::vector<SimulationResult> solve_batch(const CircuitTopology& topo, const ParameterBatch& params);
}

#endif
