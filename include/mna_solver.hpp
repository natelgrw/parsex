/*
 * mna_solver.hpp
 * 
 * Author: natelgrw
 * Last Edited: 02/25/2026
 * 
 * A header file for the MNA solver class.
*/

#ifndef MNASOLVER_HPP
#define MNASOLVER_HPP

#include "circuit_topology.hpp"
#include "linear_solver.hpp"
#include <map>
#include <string>

struct SimulationResult {
    std::map<std::string, double> node_voltages;
    std::map<std::string, double> currents;
    std::map<std::string, double> power;
};

class MNASolver {
public:
    // create a solver for a fixed topology
    MNASolver(const CircuitTopology& topo);
    
    // original legacy solve
    SimulationResult solve(const ParameterBatch& params, int batch_idx);

    // direct SoA assembly
    void assemble_system_soa(double* A_soa, double* b_soa, const ParameterBatch& params, int batch_idx, int max_n, int max_batch_size);

    int get_matrix_size() const { return matrix_size_; }

private:
    const CircuitTopology& topo_;
    int num_nodes_;
    int num_v_sources_;
    int matrix_size_;
    std::map<std::string, int> voltage_source_indices_;
};

#endif
