#include "cpu/CpuSolver.hpp"
#include <iostream>
#include <omp.h>
#include <cmath>
#include <algorithm>

namespace CpuSolver {

// Helper to get index in SoA layout
// Layout: (Row * MaxN + Col) * BatchSize + BatchIdx
// This ensures that for a fixed (Row, Col), the values for all circuits are contiguous (Simd-friendly)
inline size_t get_soa_idx(int row, int col, int batch_idx, int max_n, size_t batch_size) {
    return (size_t)((row * max_n + col) * batch_size + batch_idx);
}

std::vector<SimulationResult> solve_batch(const std::vector<CircuitGraph>& graphs) {
    size_t batch_size = graphs.size();
    if (batch_size == 0) return {};

    // 1. Determine maximum matrix size for padding
    int max_n = 0;
    // Also keep solvers to extract results later (allocating them on heap via vector)
    // Actually constructing MNASolver here is necessary to get the matrix size and assemble.
    // We'll store them to avoid re-parsing.
    std::vector<MNASolver> solvers;
    solvers.reserve(batch_size);
    std::vector<int> original_sizes(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        solvers.emplace_back(graphs[i]);
        int n = solvers.back().get_matrix_size();
        original_sizes[i] = n;
        if (n > max_n) max_n = n;
    }

    // 2. Allocate SoA Memory
    // separate A and b
    // A_soa size: max_n * max_n * batch_size
    // b_soa size: max_n * batch_size
    std::vector<double> A_soa(max_n * max_n * batch_size, 0.0);
    std::vector<double> b_soa(max_n * batch_size, 0.0);

    // 3. Assemble and Pack (SoA Transpose)
    // We can parallelize assembly over circuits, but packing into SoA might have false sharing if not careful.
    // Since each batch_idx is distinct, writing to ... + batch_idx is safe if cache lines aren't thrashing too much.
    // But since batch_size can be large, it's fine.
    
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size; ++i) {
        // Local matrix assembly
        int n = original_sizes[i];
        LinearSolver::Matrix A_local(n, LinearSolver::Vector(n, 0.0));
        LinearSolver::Vector b_local(n, 0.0);
        
        solvers[i].assemble_system(A_local, b_local);

        // Pack into SoA
        // Pad with Identity for rows/cols >= n
        for (int r = 0; r < max_n; ++r) {
            for (int c = 0; c < max_n; ++c) {
                double val = 0.0;
                if (r < n && c < n) {
                    val = A_local[r][c];
                } else if (r == c) {
                    val = 1.0; // Identity padding
                }
                
                A_soa[get_soa_idx(r, c, i, max_n, batch_size)] = val;
            }
            
            double b_val = 0.0;
            if (r < n) {
                b_val = b_local[r];
            }
            b_soa[r * batch_size + i] = b_val;
        }
    }

    // 4. SIMD Batched Gaussian Elimination (No Pivoting)
    // We process the entire batch "in a single forward pass" (vectorized).
    // Note: This naive GE assumes the matrix is solvable without row swaps.
    // MNA matrices often work, but singular pivots are possible.
    
    for (int k = 0; k < max_n; ++k) {
        // Compute inverse of pivot diagonal for all circuits
        // 1 / A[k][k]
        std::vector<double> inv_pivot(batch_size); // Stack/Heap alloc? Vector is safer for large batch.

        #pragma omp parallel for simd
        for (size_t b = 0; b < batch_size; ++b) {
            size_t idx_kk = get_soa_idx(k, k, b, max_n, batch_size);
            double pivot = A_soa[idx_kk];
            // Simple regularization to avoid div-by-zero if singular (should trigger pivoting in robust solver)
            if (std::abs(pivot) < 1e-12) pivot = 1e-12; 
            inv_pivot[b] = 1.0 / pivot;
        }

        // Elimination
        for (int i = k + 1; i < max_n; ++i) {
            // Factor = A[i][k] / A[k][k]
            std::vector<double> factor(batch_size);

            #pragma omp parallel for simd
            for (size_t b = 0; b < batch_size; ++b) {
                size_t idx_ik = get_soa_idx(i, k, b, max_n, batch_size);
                factor[b] = A_soa[idx_ik] * inv_pivot[b];
                
                // Update RHS: b[i] -= factor * b[k]
                b_soa[i * batch_size + b] -= factor[b] * b_soa[k * batch_size + b];
            }

            // Update Row: A[i][j] -= factor * A[k][j]
            // Inner loop over columns j > k
            // For CPU, we might want to block this or just run it.
            // Since max_n is usually small for circuits (10s-100s), simple loop is okay.
            for (int j = k; j < max_n; ++j) { // Optimization: Start from k or k+1? A[i][k] becomes 0, so k+1. But standard is inclusive k sometimes. k+1 is fine.
                 #pragma omp parallel for simd
                 for (size_t b = 0; b < batch_size; ++b) {
                     size_t idx_ij = get_soa_idx(i, j, b, max_n, batch_size);
                     size_t idx_kj = get_soa_idx(k, j, b, max_n, batch_size);
                     A_soa[idx_ij] -= factor[b] * A_soa[idx_kj];
                 }
            }
        }
    }

    // 5. Back Substitution (SIMD)
    // x is stored in b_soa
    for (int i = max_n - 1; i >= 0; --i) {
        #pragma omp parallel for simd
        for (size_t b = 0; b < batch_size; ++b) {
            double sum = 0.0; // Accumulate A[i][j] * x[j]
            // Since we need x[j], which is already computed for j > i
            // We can't vector-reduce easily across j inside the simd loop over b without re-ordering loops.
            // Wait, loop order:
            // for j = i+1 to max_n: sum += ...
            // But we want to vectorize over 'b'.
            // So we execute standard reduction logic per-lane.
            // But we can't easily iterate j inside the SIMD loop efficiently unless unrolled?
            // Actually, just loop j outside.
            // x[i] = (b[i] - sum(A[i][j]*x[j])) / A[i][i]
            // We can compute 'sum' incrementally before this line.
            // Actually, b_soa is modified in-place during elimination?
            // Standard GE modifies b, so b[i] holds the update.
            // Backsub: x[i] = (b[i] - sum...) / A[i][i]
        }
        
        // Let's perform sum accumulation by looping j
        std::vector<double> sum(batch_size, 0.0);
        for (int j = i + 1; j < max_n; ++j) {
            #pragma omp parallel for simd
            for (size_t b = 0; b < batch_size; ++b) {
                 size_t idx_ij = get_soa_idx(i, j, b, max_n, batch_size);
                 // x[j] is currently stored in b_soa[j * batch_size + b]
                 sum[b] += A_soa[idx_ij] * b_soa[j * batch_size + b];
            }
        }

        #pragma omp parallel for simd
        for (size_t b = 0; b < batch_size; ++b) {
            double val = b_soa[i * batch_size + b] - sum[b];
            size_t idx_ii = get_soa_idx(i, i, b, max_n, batch_size);
            double diag = A_soa[idx_ii];
            // Regularize diag
            if (std::abs(diag) < 1e-12) diag = 1e-12; // Should prevent NaN propagation
            b_soa[i * batch_size + b] = val / diag;
        }
    }

    // 6. Unpack Results
    std::vector<SimulationResult> results(batch_size);
    // Reuse specific post-processing logic (duplicated from MNASolver::solve unfortunately, or refactored)
    // We already have `solvers` vector. Can we inject x?
    // CpuSolver can't easily access private members of MNASolver unless we change it.
    // For now, duplicate the result extraction logic similar to GpuSolver.
    
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size; ++i) {
        int n = original_sizes[i];
        MNASolver& solver = solvers[i];
        
        // Extract x
        LinearSolver::Vector x(n);
        for (int j = 0; j < n; ++j) {
            x[j] = b_soa[j * batch_size + i];
        }

        // --- Result Extraction Logic (Copy from MNASolver/GpuSolver) ---
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

        // Helper lambda for voltage
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
                // Find index again (inefficient but safe)
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

    return results;
}

} // namespace CpuSolver
