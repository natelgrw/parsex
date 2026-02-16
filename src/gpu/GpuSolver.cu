#include "gpu/GpuSolver.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error at line " << __LINE__ << ": " << cudaGetErrorString(status) << std::endl; \
    } \
}

namespace GpuSolver {

// Helper to get index in SoA layout (Device Side)
// Layout: (Row * MaxN + Col) * BatchSize + BatchIdx
// This ensures coalesced access: adjacent threads (BatchIdx) read adjacent floats.
__device__ inline int get_soa_idx(int row, int col, int batch_idx, int max_n, int batch_size) {
    return (row * max_n + col) * batch_size + batch_idx;
}

// Custom Gaussian Elimination Kernel
// Strategy: One Thread Per Circuit
// Limitations: Max N is limited by registers/shared mem if we optimized further, but for global mem approach, limits are high.
// No pivoting implemented (Naive GE), consistent with SIMD CPU approach.
__global__ void solve_batch_kernel(double* A_soa, double* b_soa, int max_n, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    // Forward Elimination
    for (int k = 0; k < max_n; ++k) {
        // Pivot A[k][k]
        int idx_kk = get_soa_idx(k, k, batch_idx, max_n, batch_size);
        double pivot = A_soa[idx_kk];
        
        // Regularization
        if (abs(pivot) < 1e-12) pivot = 1e-12;
        double inv_pivot = 1.0 / pivot;

        for (int i = k + 1; i < max_n; ++i) {
            int idx_ik = get_soa_idx(i, k, batch_idx, max_n, batch_size);
            double factor = A_soa[idx_ik] * inv_pivot;

            // Update Row i using Row k
            // A[i][j] -= factor * A[k][j]
            // We only need to update for j >= k+1. A[i][k] becomes 0 (implicitly)
            for (int j = k + 1; j < max_n; ++j) {
                int idx_ij = get_soa_idx(i, j, batch_idx, max_n, batch_size);
                int idx_kj = get_soa_idx(k, j, batch_idx, max_n, batch_size);
                A_soa[idx_ij] -= factor * A_soa[idx_kj];
            }

            // Update RHS
            // b[i] -= factor * b[k]
            // b_soa is (rows * batch_size) + batch_idx
            // Actually layout of b_soa:
            // To be coalesced, b[row, batch] = row * batch_size + batch_idx.
            // b[i] is at i * batch_size + batch_idx.
            b_soa[i * batch_size + batch_idx] -= factor * b_soa[k * batch_size + batch_idx];
        }
    }

    // Back Substitution
    for (int i = max_n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < max_n; ++j) {
            // x[j] is computed and stored in b[j]
            int idx_ij = get_soa_idx(i, j, batch_idx, max_n, batch_size);
            sum += A_soa[idx_ij] * b_soa[j * batch_size + batch_idx];
        }
        
        int idx_ii = get_soa_idx(i, i, batch_idx, max_n, batch_size);
        double diag = A_soa[idx_ii];
        if (abs(diag) < 1e-12) diag = 1e-12;
        
        b_soa[i * batch_size + batch_idx] = (b_soa[i * batch_size + batch_idx] - sum) / diag;
    }
}

std::vector<SimulationResult> solve_batch(const std::vector<CircuitGraph>& graphs) {
    size_t batch_size = graphs.size();
    if (batch_size == 0) return {};

    // 1. Determine sizes
    int max_n = 0;
    std::vector<MNASolver> solvers;
    solvers.reserve(batch_size);
    std::vector<int> original_sizes(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        solvers.emplace_back(graphs[i]);
        int n = solvers.back().get_matrix_size();
        original_sizes[i] = n;
        if (n > max_n) max_n = n;
    }

    // 2. Host SoA Layout
    // A: max_n * max_n * batch_size
    // b: max_n * batch_size
    std::vector<double> h_A_soa(max_n * max_n * batch_size, 0.0);
    std::vector<double> h_b_soa(max_n * batch_size, 0.0);

    // Assemble and Pack (SoA) - Same logic as CPU SIMD
    for (size_t i = 0; i < batch_size; ++i) {
        int n = original_sizes[i];
        LinearSolver::Matrix A_local(n, LinearSolver::Vector(n, 0.0));
        LinearSolver::Vector b_local(n, 0.0);
        
        solvers[i].assemble_system(A_local, b_local);

        for (int r = 0; r < max_n; ++r) {
            for (int c = 0; c < max_n; ++c) {
                double val = 0.0;
                if (r < n && c < n) val = A_local[r][c];
                else if (r == c) val = 1.0; // Identity padding

                // Index: (r * max_n + c) * batch_size + i
                h_A_soa[(r * max_n + c) * batch_size + i] = val;
            }
            
            double b_val = 0.0;
            if (r < n) b_val = b_local[r];
            h_b_soa[r * batch_size + i] = b_val;
        }
    }

    // 3. Device Transfer
    double *d_A_soa, *d_b_soa;
    CHECK_CUDA(cudaMalloc(&d_A_soa, h_A_soa.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b_soa, h_b_soa.size() * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A_soa, h_A_soa.data(), h_A_soa.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_soa, h_b_soa.data(), h_b_soa.size() * sizeof(double), cudaMemcpyHostToDevice));

    // 4. Kernel Launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch Kernel
    solve_batch_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A_soa, d_b_soa, max_n, batch_size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. Retrieve Results
    CHECK_CUDA(cudaMemcpy(h_b_soa.data(), d_b_soa, h_b_soa.size() * sizeof(double), cudaMemcpyDeviceToHost));

    // 6. Unpack
    std::vector<SimulationResult> results(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        int n = original_sizes[i];
        MNASolver& solver = solvers[i];
        
        // Extract x from b_soa
        LinearSolver::Vector x(n);
        for (int j = 0; j < n; ++j) {
            x[j] = h_b_soa[j * batch_size + i];
        }

        // --- Result Extraction Logic (Duplicated from MNASolver/CpuSolver) ---
        SimulationResult& result = results[i];
        result.node_voltages["0"] = 0.0;
        
        int ground_idx = -1;
        try { ground_idx = graphs[i].get_node_index("0"); } catch(...) {} 

        auto get_matrix_idx = [&](int graph_idx) -> int {
            if (graph_idx == ground_idx) return -1;
            if (graph_idx > ground_idx) return graph_idx - 1;
            return graph_idx;
        };

        for (int node_idx = 0; node_idx < graphs[i].nodes.size(); ++node_idx) {
             if (node_idx == ground_idx) continue;
             int m_idx = get_matrix_idx(node_idx);
             result.node_voltages[graphs[i].nodes[node_idx]] = x[m_idx];
        }

        auto get_node_v = [&](const std::string& name) {
            if (name == "0") return 0.0;
            auto it = result.node_voltages.find(name);
            if (it != result.node_voltages.end()) return it->second;
            return 0.0;
        };

        for (const auto& comp : graphs[i].components) {
            double current = 0.0;
            double v_drop = get_node_v(comp.nodes[0]) - get_node_v(comp.nodes[1]);
            
            if (comp.type == "R") {
                current = v_drop / comp.value; 
            }
            else if (comp.type == "I") {
                current = comp.value;
            }
            else if (comp.type == "V") {
                int v_src_index = 0;
                for(const auto& c : graphs[i].components) {
                     if(c.type == "V") {
                         if(c.name == comp.name) break;
                         v_src_index++;
                     }
                }
                int v_idx = (graphs[i].nodes.size() - 1) + v_src_index;
                current = x[v_idx];
            }
            
            result.currents[comp.name] = current;
            result.power[comp.name] = v_drop * current; 
        }
    }
    
    cudaFree(d_A_soa);
    cudaFree(d_b_soa);
    
    return results;
}

} // namespace GpuSolver
