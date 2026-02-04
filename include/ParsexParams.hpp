#ifndef PARSEXPARAMS_HPP
#define PARSEXPARAMS_HPP

#include <vector>
#include <string>

// Lightweight structs for batch processing
// Unlike CircuitGraph, these are optimized for the solver loop

struct Resistor {
    int node1;
    int node2;
    double resistance;
};

struct VoltageSource {
    int node_pos;
    int node_neg; // "node2" usually negative terminal
    double voltage;
    // We might track its index in the MNA matrix for current recovery
};

struct CurrentSource {
    int node_in;  // Current enters this node
    int node_out; // Current leaves this node
    double current;
};

struct Circuit {
    std::string name;
    int num_nodes; // 0 is always ground. Total nodes = num_nodes (indices 0..num_nodes-1)
    
    std::vector<Resistor> resistors;
    std::vector<VoltageSource> voltage_sources;
    std::vector<CurrentSource> current_sources;
};

#endif // PARSEXPARAMS_HPP
