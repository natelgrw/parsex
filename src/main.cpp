/*
 * main.cpp
 * 
 * Author: natelgrw
 * Last Edited: 02/25/2026
 * 
 * Main C++ implementation file for Parsex.
*/

#include "mna_solver.hpp"
#include "circuit_topology.hpp"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <cmath>
#include "json.hpp"

using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <circuit_json_file>" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    CircuitTopology topo;
    
    if (!topo.load_from_json(filepath)) {
        return 1;
    }
    
    // for scalar simulation, we must extract the actual values from the JSON
    // because CircuitTopology doesn't load component values (it only maps topology)
    std::ifstream f(filepath);
    json data;
    try {
        data = json::parse(f);
    } catch (...) {
        std::cerr << "Failed to parse JSON for parameter extraction." << std::endl;
        return 1;
    }

    ParameterBatch params(1, topo.num_resistors, topo.num_voltage_sources, topo.num_current_sources);
    
    // quick loop to back-fill scalar values
    for (const auto& item : data["components"]) {
        std::string ctype = item.value("type", "");
        std::string cname = item.value("name", "");
        double cval = item.value("value", 0.0);
        
        for (const auto& topoc : topo.components) {
            if (topoc.name == cname) {
                if (ctype == "R") params.r_values[topoc.param_index] = cval;
                if (ctype == "V") params.v_values[topoc.param_index] = cval;
                if (ctype == "I") params.i_values[topoc.param_index] = cval;
                break;
            }
        }
    }

    try {
        MNASolver solver(topo);
        auto results = solver.solve(params, 0);

        std::cout << "Simulation Results for " << topo.name << ":" << std::endl;
        
        // create JSON object
        json solution_json;
        solution_json["circuit_name"] = topo.name;
        solution_json["results"] = json::object();
        solution_json["results"]["voltages"] = json::object();
        solution_json["results"]["currents"] = json::object();
        solution_json["results"]["power"] = json::object();
        
        // helper to round small floating point errors
        auto round_val = [](double val) {
            const double multiplier = 1e9;
            return std::round(val * multiplier) / multiplier;
        };

        // output and JSON population
        std::cout << "  Voltages:" << std::endl;
        for (const auto& pair : results.node_voltages) {
            std::cout << "    Node " << pair.first << ": " 
                      << std::fixed << std::setprecision(4) << pair.second << " V" << std::endl;
            solution_json["results"]["voltages"][pair.first] = round_val(pair.second);
        }

        std::cout << "  Currents:" << std::endl;
        for (const auto& pair : results.currents) {
             std::cout << "    " << pair.first << ": " 
                      << std::fixed << std::setprecision(4) << pair.second << " A" << std::endl;
             solution_json["results"]["currents"][pair.first] = round_val(pair.second);
        }

        std::cout << "  Power:" << std::endl;
        for (const auto& pair : results.power) {
             std::cout << "    " << pair.first << ": " 
                      << std::fixed << std::setprecision(4) << pair.second << " W" << std::endl;
             solution_json["results"]["power"][pair.first] = round_val(pair.second);
        }

        std::filesystem::path input_path(filepath);
        std::string filename = input_path.stem().string();
        std::filesystem::path output_path = "circuit_results";
        output_path /= (filename + "_sol.json");
        
        std::cout << "Exporting solution to: " << output_path << std::endl;
        
        std::ofstream out_file(output_path);
        if (out_file.is_open()) {
            out_file << solution_json.dump(2) << std::endl;
        } else {
            std::cerr << "Failed to write solution file to " << output_path << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Solver Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
