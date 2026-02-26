/*
 * batch_main.cpp
 * 
 * Author: natelgrw
 * Last Edited: 02/25/2026
 * 
 * main entry point for the batched simulation execution.
*/

#include "cpu/cpu_solver.hpp"
#ifndef NO_CUDA
#include "gpu/gpu_solver.hpp"
#endif
#include "circuit_topology.hpp" 
#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <random>

#include "json.hpp"
#include <fstream>
using json = nlohmann::json;

namespace fs = std::filesystem;

void save_results(const std::vector<SimulationResult>& results, const std::string& circuit_name, const std::string& output_dir) {
    if (!fs::exists(output_dir)) {
        fs::create_directory(output_dir);
    }
    
    // save the first 5 results to verify correctness of sweep
    int num_to_save = std::min((int)results.size(), 5);
    
    for (int i = 0; i < num_to_save; ++i) {
        const auto& res = results[i];
        
        json j;
        j["name"] = circuit_name + "_sweep_" + std::to_string(i);
        j["node_voltages"] = res.node_voltages;
        j["currents"] = res.currents;
        j["power"] = res.power;
        
        std::string filename = output_dir + "/" + circuit_name + "_sweep_" + std::to_string(i) + "_sol.json";
        std::ofstream out(filename);
        if (out.is_open()) {
            out << j.dump(4);
        } else {
            std::cerr << "Failed to write " << filename << std::endl;
        }
    }
    std::cout << "Saved first " << num_to_save << " results out of " << results.size() << " to " << output_dir << "/" << std::endl;
}

int main(int argc, char** argv) {
    // NUMA-aware thread affinity
    setenv("OMP_PLACES", "cores", 0);
    setenv("OMP_PROC_BIND", "close", 0);

    bool use_gpu = false;
    size_t batch_size = 10000;
    std::string target_topology = "voltage_divider.json";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--cpu") {
            use_gpu = false;
        } else if (arg.find("--batch=") == 0) {
            batch_size = std::stoull(arg.substr(8));
        } else if (arg.find("--topo=") == 0) {
            target_topology = arg.substr(7);
        }
    }

    std::string circuits_dir = "circuits";
    if (!fs::exists(circuits_dir)) {
        circuits_dir = "../circuits";
        if (!fs::exists(circuits_dir)) {
            std::cerr << "Error: 'circuits' directory not found." << std::endl;
            return 1;
        }
    }
    
    std::string topo_path = circuits_dir + "/" + target_topology;
    std::cout << "Loading Base Topology: " << topo_path << "..." << std::endl;
    
    CircuitTopology topo;
    if (!topo.load_from_json(topo_path)) {
        std::cerr << "Failed to load topology." << std::endl;
        return 1;
    }
    
    std::cout << "  Nodes: " << topo.get_num_nodes() 
              << " | Resistors: " << topo.num_resistors 
              << " | V-Sources: " << topo.num_voltage_sources 
              << " | I-Sources: " << topo.num_current_sources << std::endl;

    std::cout << "Generating Parameter Sweep for Batch Size: " << batch_size << "..." << std::endl;
    
    // instantiate one massive contiguous parameter batch
    ParameterBatch params(batch_size, topo.num_resistors, topo.num_voltage_sources, topo.num_current_sources);
    
    std::mt19937 gen(42);
    // let's assume nominal components around 1k Ohm, 5V, 1mA
    std::uniform_real_distribution<> r_dist(500.0, 5000.0);
    std::uniform_real_distribution<> v_dist(1.0, 12.0);
    std::uniform_real_distribution<> i_dist(0.001, 0.05);

    // fill SoA layout: param_array[param_index * batch_size + batch_idx]
    for (int p = 0; p < topo.num_resistors; ++p) {
        for (size_t b = 0; b < batch_size; ++b) {
            params.r_values[p * batch_size + b] = r_dist(gen);
        }
    }
    for (int p = 0; p < topo.num_voltage_sources; ++p) {
        for (size_t b = 0; b < batch_size; ++b) {
            params.v_values[p * batch_size + b] = v_dist(gen);
        }
    }
    for (int p = 0; p < topo.num_current_sources; ++p) {
        for (size_t b = 0; b < batch_size; ++b) {
            params.i_values[p * batch_size + b] = i_dist(gen);
        }
    }

    std::cout << "Solving ML Data Batch of " << batch_size << " circuits..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<SimulationResult> results;
    if (use_gpu) {
#ifndef NO_CUDA
        std::cout << "Using GPU Solver (CUDA)... [WARNING: Phase 6 pending update]" << std::endl;
        // results = GpuSolver::solve_batch(batch); // disabled until Phase 6
#else
        std::cerr << "Error: CUDA support not enabled in build." << std::endl;
        return 1;
#endif
    } else {
        std::cout << "Using CPU Solver (OpenMP direct SoA)..." << std::endl;
        results = CpuSolver::solve_batch(topo, params);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

    double throughput = (batch_size / (elapsed.count() / 1000.0));
    std::cout << "\\n--------------------------------------------------" << std::endl;
    std::cout << "Batch solved in " << elapsed.count() << " ms" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << throughput << " circuits/sec" << std::endl;
    std::cout << "--------------------------------------------------\\n" << std::endl;

    // save results
    save_results(results, topo.name, "../results");

    return 0;
}

