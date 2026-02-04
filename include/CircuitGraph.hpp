#ifndef CIRCUIT_GRAPH_HPP
#define CIRCUIT_GRAPH_HPP

#include "json.hpp"
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <algorithm>

using json = nlohmann::json;

struct Component {
    std::string name;
    std::string type; // "R", "V", "I"
    std::vector<std::string> nodes;
    double value;
};

class CircuitGraph {
public:
    std::string name;
    std::vector<std::string> nodes;
    std::vector<Component> components;
    std::map<std::string, int> node_map; // Map node name to matrix index

    bool load_from_json(const std::string& filepath) {
        std::ifstream f(filepath);
        if (!f.is_open()) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return false;
        }

        try {
            json data = json::parse(f);
            name = data.value("name", "circuit");
            
            // Allow explicit node list or infer from components if desired
            if (data.contains("nodes")) {
                nodes = data["nodes"].get<std::vector<std::string>>();
            }

            if (data.contains("components")) {
                for (const auto& item : data["components"]) {
                    Component c;
                    c.name = item.value("name", "unknown");
                    c.type = item.value("type", "");
                    c.nodes = item["nodes"].get<std::vector<std::string>>();
                    c.value = item.value("value", 0.0);
                    components.push_back(c);
                }
            }

            // Map nodes to indices. "0" should usually be last or handled specifically as ground if we were using 
            // a different solver, but for MNA we often set one node as ref.
            // Let's ensure "0" is present and mapped to reference if possible. 
            // For simplicity, we just map 0..N. The solver will need to know which index is ground.
            
            // Re-sort nodes to ensure "0" is at index 0 if it exists
            if (std::find(nodes.begin(), nodes.end(), "0") != nodes.end()) {
                // If "0" is present, move it to front or ensure it's index 0
                // Our JSON from python script sorts nodes, so "0" should be first if present.
            }
            
            for (size_t i = 0; i < nodes.size(); ++i) {
                node_map[nodes[i]] = i;
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
};

#endif // CIRCUIT_GRAPH_HPP
