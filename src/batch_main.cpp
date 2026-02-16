#include "cpu/CpuSolver.hpp"
#ifndef NO_CUDA
#include "gpu/GpuSolver.hpp"
#endif
#include "CircuitGraph.hpp" 
#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <iomanip>

#include "json.hpp"
#include <fstream>
using json = nlohmann::json;

namespace fs = std::filesystem;

void save_results(const std::vector<SimulationResult>& results, const std::vector<std::string>& circuit_names, const std::string& output_dir) {
    if (!fs::exists(output_dir)) {
        fs::create_directory(output_dir);
    }
    
    // We iterate by index because SimulationResult structure doesn't store the name currently, 
    // but the results vector matches the input vector order (guaranteed by parallel loop).
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& res = results[i];
        std::string name = circuit_names[i];
        
        json j;
        j["name"] = name;
        j["node_voltages"] = res.node_voltages;
        j["currents"] = res.currents;
        j["power"] = res.power;
        
        std::string filename = output_dir + "/" + name + "_sol.json";
        std::ofstream out(filename);
        if (out.is_open()) {
            out << j.dump(4);
        } else {
            std::cerr << "Failed to write " << filename << std::endl;
        }
    }
    std::cout << "Saved " << results.size() << " results to " << output_dir << "/" << std::endl;
}

int main(int argc, char** argv) {
    bool use_gpu = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--cpu") {
            use_gpu = false;
        }
    }

    std::vector<CircuitGraph> batch;
    std::vector<std::string> circuit_names;
    
    std::string circuits_dir = "circuits";
    if (!fs::exists(circuits_dir)) {
        circuits_dir = "../circuits";
        if (!fs::exists(circuits_dir)) {
            std::cerr << "Error: 'circuits' directory not found." << std::endl;
            return 1;
        }
    }
    
    std::cout << "Loading circuits from " << circuits_dir << "..." << std::endl;
    for (const auto& entry : fs::directory_iterator(circuits_dir)) {
        if (entry.path().extension() == ".json") {
            CircuitGraph graph;
            if (graph.load_from_json(entry.path().string())) {
                batch.push_back(graph);
                circuit_names.push_back(graph.name);
                std::cout << "  Loaded " << graph.name << std::endl;
            }
        }
    }

    std::cout << "Solving batch of " << batch.size() << " circuits..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<SimulationResult> results;
    if (use_gpu) {
#ifndef NO_CUDA
        std::cout << "Using GPU Solver (CUDA)..." << std::endl;
        results = GpuSolver::solve_batch(batch);
#else
        std::cerr << "Error: CUDA support not enabled in build." << std::endl;
        return 1;
#endif
    } else {
        std::cout << "Using CPU Solver (OpenMP)..." << std::endl;
        results = CpuSolver::solve_batch(batch);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

    std::cout << "Batch solved in " << elapsed.count() << " ms" << std::endl;

    std::cout << "[Verification]" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "  " << circuit_names[i] << ": " << results[i].node_voltages.size() << " nodes solved." << std::endl;
    }

    // Save Results
    save_results(results, circuit_names, "../results");

    return 0;
}
