/*
 * mna_solver.cpp
 * 
 * Author: natelgrw
 * Last Edited: 02/25/2026
 * 
 * implementation file for the MNA solver class.
*/

#include "mna_solver.hpp"
#include <iostream>

MNASolver::MNASolver(const CircuitTopology& topo) : topo_(topo) {
    num_nodes_ = topo_.get_num_nodes();
    num_v_sources_ = topo_.num_voltage_sources;
    
    // size = N-1 (if ground exists) + M
    int n_vars = num_nodes_;
    if (topo_.ground_idx != -1) n_vars -= 1;
    
    matrix_size_ = n_vars + num_v_sources_; 
    if (matrix_size_ < 0) matrix_size_ = 0;
}

// original solve method
SimulationResult MNASolver::solve(const ParameterBatch& params, int batch_idx) {
    if (topo_.ground_idx == -1) {
        throw std::runtime_error("Circuit must contain a ground node named '0'");
    }

    LinearSolver::Matrix A(matrix_size_, LinearSolver::Vector(matrix_size_, 0.0));
    LinearSolver::Vector b(matrix_size_, 0.0);

    // call the original assemble style
    auto get_matrix_idx = [&](int topo_idx) -> int {
        if (topo_idx == topo_.ground_idx) return -1;
        if (topo_idx > topo_.ground_idx) return topo_idx - 1;
        return topo_idx;
    };

    for (const auto& comp : topo_.components) {
        int n1 = comp.nodes[0];
        int n2 = comp.nodes[1];
        
        int idx1 = get_matrix_idx(n1);
        int idx2 = get_matrix_idx(n2);
        
        if (comp.type == "R") {
            double value = params.r_values[comp.param_index * params.batch_size + batch_idx];
            double g = 1.0 / value;
            
            if (idx1 != -1) {
                A[idx1][idx1] += g;
                if (idx2 != -1) A[idx1][idx2] -= g;
            }
            if (idx2 != -1) {
                A[idx2][idx2] += g;
                if (idx1 != -1) A[idx2][idx1] -= g;
            }
        }
        else if (comp.type == "I") {
            double value = params.i_values[comp.param_index * params.batch_size + batch_idx];
            if (idx1 != -1) b[idx1] -= value;
            if (idx2 != -1) b[idx2] += value;
        }
        else if (comp.type == "V") {
            double value = params.v_values[comp.param_index * params.batch_size + batch_idx];
            int v_idx = (num_nodes_ - 1) + comp.param_index;
            
            if (idx1 != -1) {
                A[v_idx][idx1] = 1.0;
                A[idx1][v_idx] = 1.0;
            }
            if (idx2 != -1) {
                A[v_idx][idx2] = -1.0;
                A[idx2][v_idx] = -1.0;
            }
            b[v_idx] = value;
        }
    }

    // solve
    LinearSolver::Vector x = LinearSolver::solve_linear_system(A, b);
    
    // process results
    SimulationResult result;
    result.node_voltages["0"] = 0.0;
    
    // extract node voltages
    auto get_node_voltage = [&](int topo_idx) -> double {
        if (topo_idx == topo_.ground_idx) return 0.0;
        int m_idx = get_matrix_idx(topo_idx);
        return x[m_idx];
    };

    for (int i = 0; i < num_nodes_; ++i) {
        if (i == topo_.ground_idx) continue;
        int m_idx = get_matrix_idx(i);
        result.node_voltages[topo_.node_names[i]] = x[m_idx];
    }
    
    // calculate currents and power
    for (const auto& comp : topo_.components) {
        double current = 0.0;
        double voltage_drop = get_node_voltage(comp.nodes[0]) - get_node_voltage(comp.nodes[1]);
        
        if (comp.type == "R") {
            double value = params.r_values[comp.param_index * params.batch_size + batch_idx];
            current = voltage_drop / value; 
        }
        else if (comp.type == "I") {
            double value = params.i_values[comp.param_index * params.batch_size + batch_idx];
            current = value;
        }
        else if (comp.type == "V") {
            int v_idx = (num_nodes_ - 1) + comp.param_index;
            current = x[v_idx]; 
        }
        
        result.currents[comp.name] = current;
        result.power[comp.name] = voltage_drop * current; 
    }
    
    return result;
}

// direct SoA assembly
void MNASolver::assemble_system_soa(double* A_soa, double* b_soa, const ParameterBatch& params, int batch_idx, int max_n, int max_batch_size) {
    if (topo_.ground_idx == -1) return;

    auto get_matrix_idx = [&](int topo_idx) -> int {
        if (topo_idx == topo_.ground_idx) return -1;
        if (topo_idx > topo_.ground_idx) return topo_idx - 1;
        return topo_idx;
    };

    // SoA index helper
    #define SOA_IDX(r, c) (((r) * max_n + (c)) * max_batch_size + batch_idx)

    for (const auto& comp : topo_.components) {
        int n1 = comp.nodes[0];
        int n2 = comp.nodes[1];
        
        int idx1 = get_matrix_idx(n1);
        int idx2 = get_matrix_idx(n2);
        
        if (comp.type == "R") {
            double g = 1.0 / params.r_values[comp.param_index * params.batch_size + batch_idx];
            
            if (idx1 != -1) {
                A_soa[SOA_IDX(idx1, idx1)] += g;
                if (idx2 != -1) A_soa[SOA_IDX(idx1, idx2)] -= g;
            }
            if (idx2 != -1) {
                A_soa[SOA_IDX(idx2, idx2)] += g;
                if (idx1 != -1) A_soa[SOA_IDX(idx2, idx1)] -= g;
            }
        }
        else if (comp.type == "I") {
            double val = params.i_values[comp.param_index * params.batch_size + batch_idx];
            if (idx1 != -1) b_soa[idx1 * max_batch_size + batch_idx] -= val;
            if (idx2 != -1) b_soa[idx2 * max_batch_size + batch_idx] += val;
        }
        else if (comp.type == "V") {
            double val = params.v_values[comp.param_index * params.batch_size + batch_idx];
            int v_idx = (num_nodes_ - 1) + comp.param_index;
            
            if (idx1 != -1) {
                A_soa[SOA_IDX(v_idx, idx1)] = 1.0;
                A_soa[SOA_IDX(idx1, v_idx)] = 1.0;
            }
            if (idx2 != -1) {
                A_soa[SOA_IDX(v_idx, idx2)] = -1.0;
                A_soa[SOA_IDX(idx2, v_idx)] = -1.0;
            }
            b_soa[v_idx * max_batch_size + batch_idx] = val;
        }
    }
}

