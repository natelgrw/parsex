#ifndef MNASOLVER_HPP
#define MNASOLVER_HPP

#include "CircuitGraph.hpp"
#include "LinearSolver.hpp"
#include <map>
#include <string>

struct SimulationResult {
    std::map<std::string, double> node_voltages;
    std::map<std::string, double> currents;
    std::map<std::string, double> power;
};

class MNASolver {
public:
    MNASolver(const CircuitGraph& graph);
    SimulationResult solve();

private:
    const CircuitGraph& graph_;
    int num_nodes_;
    int num_v_sources_;
    int matrix_size_;
    
    std::map<std::string, int> voltage_source_indices_; // Map voltage source name to row index (M+N)
    
    // MNA build helpers can be added here if needed
};

#endif // MNASOLVER_HPP
