#include "BatchSolver.hpp"
#include <iostream>
#include <omp.h>

std::vector<SimulationResult> solve_batch(const std::vector<CircuitGraph>& graphs) {
    std::vector<SimulationResult> results(graphs.size());

    // Embarrassingly Parallel Loop
    // Each thread gets its own MNASolver instance (stack allocated)
    // No shared state, no locks needed (writing to distinct indices)
    #pragma omp parallel for
    for (size_t i = 0; i < graphs.size(); ++i) {
        // Instantiate the existing serial solver
        MNASolver solver(graphs[i]);
        
        try {
            results[i] = solver.solve();
        } catch (const std::exception& e) {
            std::cerr << "Error solving circuit " << graphs[i].name << ": " << e.what() << std::endl;
            // Result will be default constructed (empty maps)
        }
    }

    return results;
}
