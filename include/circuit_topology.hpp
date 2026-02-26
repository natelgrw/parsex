/*
 * circuit_topology.hpp
 * 
 * Author: natelgrw
 * Last Edited: 02/25/2026
 * 
 * A header file for the circuit topology class.
*/

#ifndef CIRCUIT_TOPOLOGY_HPP
#define CIRCUIT_TOPOLOGY_HPP

#include "json.hpp"
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

using json = nlohmann::json;

struct TopologicalComponent {
    std::string name;
    // "R", "V", "I"
    std::string type;
    // int IDs instead of strings
    std::vector<int> nodes;
    int param_index; 
};

class CircuitTopology {
public:
    std::string name;
    std::vector<std::string> node_names;
    std::map<std::string, int> node_map;
    
    std::vector<TopologicalComponent> components;
    
    // counts for parameter arrays
    int num_resistors = 0;
    int num_voltage_sources = 0;
    int num_current_sources = 0;
    
    int ground_idx = -1;

    bool load_from_json(const std::string& filepath) {
        std::ifstream f(filepath);
        if (!f.is_open()) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return false;
        }

        try {
            json data = json::parse(f);
            name = data.value("name", "circuit");
            
            // build node map
            if (data.contains("nodes")) {
                node_names = data["nodes"].get<std::vector<std::string>>();
            }

            // ensure ground is mapped correctly
            auto ground_it = std::find(node_names.begin(), node_names.end(), "0");
            if (ground_it != node_names.end()) {
                ground_idx = std::distance(node_names.begin(), ground_it);
            }

            for (size_t i = 0; i < node_names.size(); ++i) {
                node_map[node_names[i]] = i;
            }

            // build topological components
            if (data.contains("components")) {
                for (const auto& item : data["components"]) {
                    TopologicalComponent c;
                    c.name = item.value("name", "unknown");
                    c.type = item.value("type", "");
                    
                    std::vector<std::string> str_nodes = item["nodes"].get<std::vector<std::string>>();
                    for (const auto& n : str_nodes) {
                        c.nodes.push_back(get_node_index(n));
                    }
                    
                    if (c.type == "R") {
                        c.param_index = num_resistors++;
                    } else if (c.type == "V") {
                        c.param_index = num_voltage_sources++;
                    } else if (c.type == "I") {
                        c.param_index = num_current_sources++;
                    }
                    
                    components.push_back(c);
                }
            }

        } catch (const json::parse_error& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    int get_node_index(const std::string& node_name) const {
        auto it = node_map.find(node_name);
        if (it != node_map.end()) {
            return it->second;
        }
        throw std::runtime_error("Node not found: " + node_name);
    }
    
    int get_num_nodes() const {
        return node_names.size();
    }
};

// represents a batch of parameters for a specific topology
struct ParameterBatch {
    size_t batch_size;
    
    // SoA layout
    std::vector<double> r_values;
    std::vector<double> v_values;
    std::vector<double> i_values;
    
    ParameterBatch(size_t size, int num_r, int num_v, int num_i) : batch_size(size) {
        r_values.resize(num_r * size, 0.0);
        v_values.resize(num_v * size, 0.0);
        i_values.resize(num_i * size, 0.0);
    }
};

#endif
