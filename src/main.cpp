#include "MNASolver.hpp"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <cmath>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <circuit_json_file>" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    CircuitGraph graph;
    
    if (!graph.load_from_json(filepath)) {
        return 1;
    }
    
    try {
        MNASolver solver(graph);
        auto results = solver.solve(); // Now returns SimulationResult struct
        
        std::cout << "Simulation Results for " << graph.name << ":" << std::endl;
        
        // Create solution JSON object
        json solution_json;
        solution_json["circuit_name"] = graph.name;
        solution_json["results"] = json::object();
        solution_json["results"]["voltages"] = json::object();
        solution_json["results"]["currents"] = json::object();
        solution_json["results"]["power"] = json::object();
        
        // Helper to round small floating point errors
        auto round_val = [](double val) {
            // Round to 9 decimal places to avoid e.g. -0.0050000000000001
            const double multiplier = 1e9;
            return std::round(val * multiplier) / multiplier;
        };

        // Output and JSON population
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

        
        // Write to output file in circuit_results directory
        // Input: path/to/filename.json -> Output: circuit_results/filename.sol.json
        std::filesystem::path input_path(filepath);
        std::string filename = input_path.stem().string();
        std::filesystem::path output_path = "circuit_results";
        output_path /= (filename + "_sol.json");
        
        std::cout << "Exporting solution to: " << output_path << std::endl;
        
        std::ofstream out_file(output_path);
        if (out_file.is_open()) {
            out_file << solution_json.dump(2) << std::endl; // Indent with 2 spaces
        } else {
            std::cerr << "Failed to write solution file to " << output_path << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Solver Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
