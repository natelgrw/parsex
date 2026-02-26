/*
 * gpu_solver.hpp
 * 
 * Author: natelgrw
 * Last Edited: 02/25/2026
 * 
 * a header file for the GPU solver namespace.
*/

#ifndef GPUSOLVER_HPP
#define GPUSOLVER_HPP

#include "mna_solver.hpp"
#include <vector>

namespace GpuSolver {
// batched solver function using CUDA
std::vector<SimulationResult> solve_batch(const CircuitTopology& topo, const ParameterBatch& params);
}

#endif
