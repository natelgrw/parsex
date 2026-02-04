#include "MNASolver.hpp"
#include <iostream>

MNASolver::MNASolver(const CircuitGraph& graph) : graph_(graph) {
    num_nodes_ = graph_.nodes.size();
    
    // Count voltage sources to determine matrix size
    num_v_sources_ = 0;
    for (const auto& comp : graph_.components) {
        if (comp.type == "V") {
            voltage_source_indices_[comp.name] = num_v_sources_;
            num_v_sources_++;
        }
    }
    
    // MNA Matrix size: (N nodes) + (M voltage sources)
    // Note: MNA usually excludes the reference node (ground) from the KCL equations.
    // However, for simplicity, if we include ground in the nodes list, we must force its voltage to 0.
    // A standard approach is to remove the ground row/col.
    // Here we will use the approach:
    // Size = N + M. We will then set the row for ground to be explicitly V_ground = 0 (or remove it).
    // Let's stick to the "remove ground" or "set ground to 0" method.
    // For this implementation, we will treat node "0" as ground and REMOVE it from the unknown vector
    // if it exists. If it doesn't exist, we pick the first node as reference?
    // SPICE standard is "0" is ground.
    
    // Let's assume node "0" exists and is index 0. We will solve for N-1 nodes + M sources.
    // The matrix dimension will be (N-1) + M.
    
    matrix_size_ = (num_nodes_ - 1) + num_v_sources_; 
    if (matrix_size_ < 0) matrix_size_ = 0; // Should not happen for valid circuits
}

SimulationResult MNASolver::solve() {
    // Check if node "0" exists
    int ground_idx = -1;
    try {
        ground_idx = graph_.get_node_index("0");
    } catch (...) {
        // No ground defined. This is an issue for simulation stability usually, 
        // but for a pure graph we might need to arbitrarily pick one.
        // For now, let's assume we REQUIRE node "0".
        // If not found, throw.
        throw std::runtime_error("Circuit must contain a ground node named '0'");
    }

    // Initialize A matrix and b vector (z)
    // Dimension D = (NumNodes - 1) + NumVoltageSources
    // We map graph node indices to matrix indices.
    // Node mappings:
    // graph_index -> matrix_index
    // ground_idx -> -1 (ignore)
    // other -> if index < ground_idx, index. if > ground_idx, index - 1.
    
    auto get_matrix_idx = [&](int graph_idx) -> int {
        if (graph_idx == ground_idx) return -1;
        if (graph_idx > ground_idx) return graph_idx - 1;
        return graph_idx;
    };

    LinearSolver::Matrix A(matrix_size_, LinearSolver::Vector(matrix_size_, 0.0));
    LinearSolver::Vector b(matrix_size_, 0.0);

    // 1. Fill conductance part (N-1 x N-1)
    // 2. Fill source incidence part
    
    for (const auto& comp : graph_.components) {
        int n1 = graph_.get_node_index(comp.nodes[0]);
        int n2 = graph_.get_node_index(comp.nodes[1]);
        
        int idx1 = get_matrix_idx(n1);
        int idx2 = get_matrix_idx(n2);
        
        if (comp.type == "R") {
            double g = 1.0 / comp.value;
            
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
            // Current source flowing FROM n1 TO n2 means LEAVING n1, ENTERING n2.
            // KCL: Sum of currents LEAVING = 0.
            // If I leaves n1, it adds to the KCL at n1 (positive).
            // Actually standard MNA RHS vector 'b' usually contains known currents entering the node.
            // If I flows n1->n2, it leaves n1 ( enters n2).
            // RHS[n1] -= I (current leaving) -> Wait, convention:
            // Eq: G*v = I_in
            // Current source FROM n1 TO n2 -> Leaves n1 (-I on LHS, or +I on RHS? No, I leaves n1.)
            // Current entering is positive on RHS.
            // So if I goes n1 -> n2, it LEAVES n1 (entering is -I) and ENTERS n2.
            // b[idx1] -= I (since it leaves n1)
            // b[idx2] += I (since it enters n2)
            
            if (idx1 != -1) b[idx1] -= comp.value;
            if (idx2 != -1) b[idx2] += comp.value;
        }
        else if (comp.type == "V") {
            // Voltage source connected between n1 (+) and n2 (-) with value V.
            // Adds a row at the end.
            int v_idx = (num_nodes_ - 1) + voltage_source_indices_[comp.name];
            
            // Constraint equation: v_n1 - v_n2 = Value
            if (idx1 != -1) {
                A[v_idx][idx1] = 1.0;
                A[idx1][v_idx] = 1.0; // Symmetric stamp for MNA
            }
            if (idx2 != -1) {
                A[v_idx][idx2] = -1.0;
                A[idx2][v_idx] = -1.0; // Symmetric stamp
            }
            
            b[v_idx] = comp.value;
        }
    }

    // Solve
    LinearSolver::Vector x = LinearSolver::solve_linear_system(A, b);
    
    // Process Results
    SimulationResult result;
    result.node_voltages["0"] = 0.0; // Ground
    
    // 1. Extract Node Voltages
    auto get_node_voltage = [&](const std::string& node_name) -> double {
        if (node_name == "0") return 0.0;
        int idx = graph_.get_node_index(node_name);
        if (idx == ground_idx) return 0.0; // Should be redundant if name is "0"
        
        int m_idx = get_matrix_idx(idx);
        return x[m_idx];
    };

    for (int i = 0; i < num_nodes_; ++i) {
        if (i == ground_idx) continue;
        int m_idx = get_matrix_idx(i);
        result.node_voltages[graph_.nodes[i]] = x[m_idx];
    }
    
    // 2. Calculate Currents and Power
    for (const auto& comp : graph_.components) {
        double current = 0.0;
        double voltage_drop = get_node_voltage(comp.nodes[0]) - get_node_voltage(comp.nodes[1]);
        
        if (comp.type == "R") {
            // I = V/R
            current = voltage_drop / comp.value; // Current flows n1 -> n2
        }
        else if (comp.type == "I") {
            // Current is known: Value. Flows n1 -> n2?
            // SPICE Convention: I n1 n2 Value => Current flows from n1 to n2 within the source?
            // Actually usually independent source n1 n2 val means current flows from n1 to n2 *through the source*?
            // NO, standard SPICE "I1 n1 n2 1A" means current flows from n1 to n2 THROUGH the source?
            // Actually, convention usually is n1 is positive node.
            // If we assume I1 n1 n2 1A means current enters n2 from n1?
            // Wait, for MNA stamp we did: leaves n1, enters n2. So, yes, n1 -> n2.
            current = comp.value;
        }
        else if (comp.type == "V") {
            // Current is in the solution vector.
            // Variable was "current through voltage source".
            // MNA variable usually is current flowing from + to - (high to low potential)?
            // Convention: Current flows from positive node to negative node...
            // Actually, MNA variable for Voltage source is often defined as current flowing *into* the positive terminal.
            // Let's check our stamp:
            // n1 (+), n2 (-)
            // Eq: v1 - v2 = E
            // KCL n1: ... + I_v = 0 -> I_v leaves N1?
            // Matrix row n1: ... + 1*I_v ...
            // If 1*I_v is in KCL for n1 (sum of currents leaving), then I_v is leaving n1.
            // So I_v is current flowing OUT of n1 (into n2).
            // So it flows n1 -> n2.
            
            int v_idx = (num_nodes_ - 1) + voltage_source_indices_[comp.name];
            current = x[v_idx]; 
        }
        
        result.currents[comp.name] = current;
        result.power[comp.name] = voltage_drop * current; // P = V*I (Passive sign convention). 
        // If P > 0, absorbing power. If P < 0, supplying power.
    }
    
    return result;
}
