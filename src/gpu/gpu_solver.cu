/*
 * gpu_solver.cu
 * 
 * Author: natelgrw
 * Last Edited: 02/25/2026
 * 
 * implementation file for the GPU CUDA hardware solver.
*/

#include "gpu/gpu_solver.hpp"
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

// helper to get index in SoA layout
__device__ inline int get_soa_idx(int row, int col, int batch_idx, int max_n, int batch_size) {
    return (row * max_n + col) * batch_size + batch_idx;
}

// custom Gaussian elimination kernel
__global__ void solve_batch_kernel(double* A_soa, double* b_soa, int max_n, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    // forward elimination
    for (int k = 0; k < max_n; ++k) {
        // pivot selection
        int pivot_row = k;
        double max_val = abs(A_soa[get_soa_idx(k, k, batch_idx, max_n, batch_size)]);
        for (int p = k + 1; p < max_n; ++p) {
            double val = abs(A_soa[get_soa_idx(p, k, batch_idx, max_n, batch_size)]);
            if (val > max_val) {
                max_val = val;
                pivot_row = p;
            }
        }
        
        // swap rows if a better pivot was found
        if (pivot_row != k) {
            for (int j = k; j < max_n; ++j) {
                int idx_kj = get_soa_idx(k, j, batch_idx, max_n, batch_size);
                int idx_pj = get_soa_idx(pivot_row, j, batch_idx, max_n, batch_size);
                double tmp = A_soa[idx_kj];
                A_soa[idx_kj] = A_soa[idx_pj];
                A_soa[idx_pj] = tmp;
            }
            int idx_k = k * batch_size + batch_idx;
            int idx_p = pivot_row * batch_size + batch_idx;
            double tmp_b = b_soa[idx_k];
            b_soa[idx_k] = b_soa[idx_p];
            b_soa[idx_p] = tmp_b;
        }

        // fetch chosen pivot A[k][k]
        int idx_kk = get_soa_idx(k, k, batch_idx, max_n, batch_size);
        double pivot = A_soa[idx_kk];
        
        // regularization for numerical stability
        if (abs(pivot) < 1e-12) pivot = 1e-12 * (pivot < 0 ? -1.0 : 1.0);
        double inv_pivot = 1.0 / pivot;

        for (int i = k + 1; i < max_n; ++i) {
            int idx_ik = get_soa_idx(i, k, batch_idx, max_n, batch_size);
            double factor = A_soa[idx_ik] * inv_pivot;

            // update row i using row k
            for (int j = k + 1; j < max_n; ++j) {
                int idx_ij = get_soa_idx(i, j, batch_idx, max_n, batch_size);
                int idx_kj = get_soa_idx(k, j, batch_idx, max_n, batch_size);
                A_soa[idx_ij] -= factor * A_soa[idx_kj];
            }

            // update RHS
            b_soa[i * batch_size + batch_idx] -= factor * b_soa[k * batch_size + batch_idx];
        }
    }

    // back substitution
    for (int i = max_n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < max_n; ++j) {
            int idx_ij = get_soa_idx(i, j, batch_idx, max_n, batch_size);
            sum += A_soa[idx_ij] * b_soa[j * batch_size + batch_idx];
        }
        
        int idx_ii = get_soa_idx(i, i, batch_idx, max_n, batch_size);
        double diag = A_soa[idx_ii];
        if (abs(diag) < 1e-12) diag = 1e-12;
        
        b_soa[i * batch_size + batch_idx] = (b_soa[i * batch_size + batch_idx] - sum) / diag;
    }
}

std::vector<SimulationResult> solve_batch(const CircuitTopology& topo, const ParameterBatch& params) {
    size_t batch_size = params.batch_size;
    if (batch_size == 0) return {};

    int max_n = topo.get_num_nodes();

    // host SoA layout
    std::vector<double> h_A_soa(max_n * max_n * batch_size, 0.0);
    std::vector<double> h_b_soa(max_n * batch_size, 0.0);

    // assemble and pack
    MNASolver solver(topo);
    for (size_t b = 0; b < batch_size; ++b) {
        solver.assemble_system_soa(h_A_soa.data(), h_b_soa.data(), params, (int)b, max_n, (int)batch_size);
    }
    
    // matrix fully assembled on host
    double *d_A_soa, *d_b_soa;
    CHECK_CUDA(cudaMalloc(&d_A_soa, h_A_soa.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b_soa, h_b_soa.size() * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A_soa, h_A_soa.data(), h_A_soa.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_soa, h_b_soa.data(), h_b_soa.size() * sizeof(double), cudaMemcpyHostToDevice));

    // launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    
    solve_batch_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A_soa, d_b_soa, max_n, batch_size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // retrieve results
    CHECK_CUDA(cudaMemcpy(h_b_soa.data(), d_b_soa, h_b_soa.size() * sizeof(double), cudaMemcpyDeviceToHost));

    // unpack
    std::vector<SimulationResult> results(batch_size);
    
    auto get_matrix_idx = [&](int topo_idx) -> int {
        if (topo_idx == topo.ground_idx) return -1;
        if (topo_idx > topo.ground_idx) return topo_idx - 1;
        return topo_idx;
    };

    for (size_t b = 0; b < batch_size; ++b) {
        std::vector<double> x(max_n);
        for (int j = 0; j < max_n; ++j) {
            x[j] = h_b_soa[j * batch_size + b];
        }

        SimulationResult& result = results[b];
        result.node_voltages["0"] = 0.0;

        for (int node_idx = 0; node_idx < topo.get_num_nodes(); ++node_idx) {
             if (node_idx == topo.ground_idx) continue;
             int m_idx = get_matrix_idx(node_idx);
             result.node_voltages[topo.node_names[node_idx]] = x[m_idx];
        }

        auto get_node_v = [&](int topo_idx) {
            if (topo_idx == topo.ground_idx) return 0.0;
            auto it = result.node_voltages.find(topo.node_names[topo_idx]);
            if (it != result.node_voltages.end()) return it->second;
            return 0.0;
        };

        for (const auto& comp : topo.components) {
            double current = 0.0;
            double v_drop = get_node_v(comp.nodes[0]) - get_node_v(comp.nodes[1]);
            
            if (comp.type == "R") {
                double value = params.r_values[comp.param_index * params.batch_size + b];
                current = v_drop / value; 
            }
            else if (comp.type == "I") {
                double value = params.i_values[comp.param_index * params.batch_size + b];
                current = value;
            }
            else if (comp.type == "V") {
                int v_idx = (max_n - 1) + comp.param_index;
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

}
