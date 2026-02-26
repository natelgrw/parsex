/*
 * cpu_solver.cpp
 * 
 * Author: natelgrw
 * Last Edited: 02/25/2026
 * 
 * Implementation file for the CPU batched SIMD solver.
*/

#include "cpu/cpu_solver.hpp"
#include <iostream>
#include <omp.h>
#include <cmath>
#include <algorithm>

// phase 5 intrinsics
#if defined(__AVX2__)
    #include <immintrin.h>
#elif defined(__aarch64__)
    #include <arm_neon.h>
#endif

namespace CpuSolver {

inline size_t get_soa_idx(int row, int col, int batch_idx, int max_n, size_t batch_size) {
    return (size_t)((row * max_n + col) * batch_size + batch_idx);
}

std::vector<SimulationResult> solve_batch(const CircuitTopology& topo, const ParameterBatch& params) {
    size_t batch_size = params.batch_size;
    if (batch_size == 0) return {};

    MNASolver solver(topo);
    int n = solver.get_matrix_size();
    int max_n = n;

    // phase 2: global soa layout contiguous by batch dimension
    std::vector<double> A_soa(max_n * max_n * batch_size);
    std::vector<double> b_soa(max_n * batch_size);
    std::vector<double> inv_pivot(batch_size);
    std::vector<SimulationResult> results(batch_size);
    
    // phase 4: cache blocking size (fits l1/l2 limits for vector lanes)
    size_t BLOCK_SIZE = 256;
    size_t num_blocks = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // phase 3: thread-level numa-aware parallelization using openmp
    #pragma omp parallel
    {
        auto get_matrix_idx = [&](int topo_idx) -> int {
            if (topo_idx == topo.ground_idx) return -1;
            if (topo_idx > topo.ground_idx) return topo_idx - 1;
            return topo_idx;
        };

        #pragma omp for schedule(static)
        for (size_t block = 0; block < num_blocks; ++block) {
            size_t b_start = block * BLOCK_SIZE;
            size_t b_end = std::min(b_start + BLOCK_SIZE, batch_size);

            for (size_t b = b_start; b < b_end; ++b) {
                for (int i = 0; i < max_n; ++i) {
                    b_soa[i * batch_size + b] = 0.0;
                    for (int j = 0; j < max_n; ++j) {
                        A_soa[get_soa_idx(i, j, (int)b, max_n, batch_size)] = 0.0;
                    }
                }
            }

            for (size_t b = b_start; b < b_end; ++b) {
                solver.assemble_system_soa(A_soa.data(), b_soa.data(), params, (int)b, max_n, (int)batch_size);
            }

            // cross platform simd arrays and loops
            for (int k = 0; k < max_n; ++k) {
                // all circuits in the block share the same topology
                // find the pivot row for circuit 0 and swap the entire SoA row
                int pivot_row = k;
                double max_val = std::abs(A_soa[get_soa_idx(k, k, (int)b_start, max_n, batch_size)]);
                for (int p = k + 1; p < max_n; ++p) {
                    double val = std::abs(A_soa[get_soa_idx(p, k, (int)b_start, max_n, batch_size)]);
                    if (val > max_val) {
                        max_val = val;
                        pivot_row = p;
                    }
                }
                
                if (pivot_row != k) {
                    for (int j = k; j < max_n; ++j) {
                        size_t row_k_off = (k * max_n + j) * batch_size;
                        size_t row_p_off = (pivot_row * max_n + j) * batch_size;
                        for (size_t b = b_start; b < b_end; ++b) {
                            std::swap(A_soa[row_k_off + b], A_soa[row_p_off + b]);
                        }
                    }
                    size_t rhs_k_off = k * batch_size;
                    size_t rhs_p_off = pivot_row * batch_size;
                    for (size_t b = b_start; b < b_end; ++b) {
                        std::swap(b_soa[rhs_k_off + b], b_soa[rhs_p_off + b]);
                    }
                }

                size_t b = b_start;
                // avx2 pivot inversion vectorization
            #if defined(__AVX2__)
                // AVX2 4-wide loop
                __m256d v_ones = _mm256_set1_pd(1.0);
                __m256d v_eps = _mm256_set1_pd(1e-12);
                __m256d v_neg_eps = _mm256_set1_pd(-1e-12);
                for (; b + 3 < b_end; b += 4) {
                    __m256d v_pivots = _mm256_loadu_pd(&A_soa[get_soa_idx(k, k, (int)b, max_n, batch_size)]);
                    __m256d mask_pos = _mm256_cmp_pd(v_pivots, _mm256_setzero_pd(), _CMP_GE_OS);
                    __m256d mask_lt_eps = _mm256_cmp_pd(v_pivots, v_eps, _CMP_LT_OS);
                    v_pivots = _mm256_blendv_pd(v_pivots, v_eps, _mm256_and_pd(mask_pos, mask_lt_eps));
                    __m256d mask_neg = _mm256_cmp_pd(v_pivots, _mm256_setzero_pd(), _CMP_LT_OS);
                    __m256d mask_gt_neg_eps = _mm256_cmp_pd(v_pivots, v_neg_eps, _CMP_GT_OS);
                    v_pivots = _mm256_blendv_pd(v_pivots, v_neg_eps, _mm256_and_pd(mask_neg, mask_gt_neg_eps));
                    _mm256_storeu_pd(&inv_pivot[b], _mm256_div_pd(v_ones, v_pivots));
                }
                // neon pivot inversion vectorization
            #elif defined(__aarch64__)
                // ARM NEON 2-wide loop
                float64x2_t v_ones = vdupq_n_f64(1.0);
                float64x2_t v_eps = vdupq_n_f64(1e-12);
                float64x2_t v_neg_eps = vdupq_n_f64(-1e-12);
                float64x2_t v_zero = vdupq_n_f64(0.0);
                for (; b + 1 < b_end; b += 2) {
                    float64x2_t v_pivots = vld1q_f64(&A_soa[get_soa_idx(k, k, (int)b, max_n, batch_size)]);
                    uint64x2_t mask_pos = vcgeq_f64(v_pivots, v_zero);
                    uint64x2_t mask_lt_eps = vcltq_f64(v_pivots, v_eps);
                    v_pivots = vbslq_f64(vandq_u64(mask_pos, mask_lt_eps), v_eps, v_pivots);
                    uint64x2_t mask_neg = vcltq_f64(v_pivots, v_zero);
                    uint64x2_t mask_gt_neg_eps = vcgtq_f64(v_pivots, v_neg_eps);
                    v_pivots = vbslq_f64(vandq_u64(mask_neg, mask_gt_neg_eps), v_neg_eps, v_pivots);
                    vst1q_f64(&inv_pivot[b], vdivq_f64(v_ones, v_pivots));
                }
            #endif
                for (; b < b_end; ++b) {
                    double pivot = A_soa[get_soa_idx(k, k, (int)b, max_n, batch_size)];
                    if (std::abs(pivot) < 1e-12) pivot = 1e-12 * (pivot < 0 ? -1.0 : 1.0);
                    inv_pivot[b] = 1.0 / pivot;
                }
                
                for (int i = k + 1; i < max_n; ++i) {
                    b = b_start;
                    size_t ik_off = (i * max_n + k) * batch_size;
                    size_t rhsk_off = k * batch_size;
                    size_t rhsi_off = i * batch_size;
                    // avx2 forward substitution vectorization
                #if defined(__AVX2__)
                    for (; b + 3 < b_end; b += 4) {
                        __m256d v_factor = _mm256_mul_pd(_mm256_loadu_pd(&A_soa[ik_off + b]), _mm256_loadu_pd(&inv_pivot[b]));
                        __m256d v_bk = _mm256_loadu_pd(&b_soa[rhsk_off + b]);
                        __m256d v_bi = _mm256_loadu_pd(&b_soa[rhsi_off + b]);
                    #ifdef __FMA__
                        _mm256_storeu_pd(&b_soa[rhsi_off + b], _mm256_fnmadd_pd(v_factor, v_bk, v_bi));
                    #else
                        _mm256_storeu_pd(&b_soa[rhsi_off + b], _mm256_sub_pd(v_bi, _mm256_mul_pd(v_factor, v_bk)));
                    #endif
                    }
                    // neon forward substitution vectorization
                #elif defined(__aarch64__)
                    for (; b + 1 < b_end; b += 2) {
                        float64x2_t v_factor = vmulq_f64(vld1q_f64(&A_soa[ik_off + b]), vld1q_f64(&inv_pivot[b]));
                        float64x2_t v_bk = vld1q_f64(&b_soa[rhsk_off + b]);
                        float64x2_t v_bi = vld1q_f64(&b_soa[rhsi_off + b]);
                    #if defined(__ARM_FEATURE_FMA)
                        vst1q_f64(&b_soa[rhsi_off + b], vfmsq_f64(v_bi, v_factor, v_bk));
                    #else
                        vst1q_f64(&b_soa[rhsi_off + b], vsubq_f64(v_bi, vmulq_f64(v_factor, v_bk)));
                    #endif
                    }
                #endif
                    for (; b < b_end; ++b) {
                        double factor = A_soa[ik_off + b] * inv_pivot[b];
                        b_soa[rhsi_off + b] -= factor * b_soa[rhsk_off + b];
                    }
                    
                    for (int j = k; j < max_n; ++j) {
                        b = b_start;
                        size_t ij_off = (i * max_n + j) * batch_size;
                        size_t kj_off = (k * max_n + j) * batch_size;
                        // avx2 matrix elimination sequence
                #if defined(__AVX2__)
                        for (; b + 3 < b_end; b += 4) {
                            __m256d v_factor = _mm256_mul_pd(_mm256_loadu_pd(&A_soa[ik_off + b]), _mm256_loadu_pd(&inv_pivot[b]));
                            __m256d v_kj = _mm256_loadu_pd(&A_soa[kj_off + b]);
                            __m256d v_ij = _mm256_loadu_pd(&A_soa[ij_off + b]);
                    #ifdef __FMA__
                            _mm256_storeu_pd(&A_soa[ij_off + b], _mm256_fnmadd_pd(v_factor, v_kj, v_ij));
                    #else
                            _mm256_storeu_pd(&A_soa[ij_off + b], _mm256_sub_pd(v_ij, _mm256_mul_pd(v_factor, v_kj)));
                    #endif
                        }
                        // neon matrix elimination sequence
                #elif defined(__aarch64__)
                        for (; b + 1 < b_end; b += 2) {
                            float64x2_t v_factor = vmulq_f64(vld1q_f64(&A_soa[ik_off + b]), vld1q_f64(&inv_pivot[b]));
                            float64x2_t v_kj = vld1q_f64(&A_soa[kj_off + b]);
                            float64x2_t v_ij = vld1q_f64(&A_soa[ij_off + b]);
                    #if defined(__ARM_FEATURE_FMA)
                            vst1q_f64(&A_soa[ij_off + b], vfmsq_f64(v_ij, v_factor, v_kj));
                    #else
                            vst1q_f64(&A_soa[ij_off + b], vsubq_f64(v_ij, vmulq_f64(v_factor, v_kj)));
                    #endif
                        }
                #endif
                        for (; b < b_end; ++b) {
                            double factor = A_soa[ik_off + b] * inv_pivot[b];
                            A_soa[ij_off + b] -= factor * A_soa[kj_off + b];
                        }
                    }
                }
            }

            for (int i = max_n - 1; i >= 0; --i) {
                size_t b = b_start;
                size_t rhsi_off = i * batch_size;
                size_t ii_off = (i * max_n + i) * batch_size;
                #if defined(__AVX2__)
                __m256d v_eps = _mm256_set1_pd(1e-12);
                __m256d v_neg_eps = _mm256_set1_pd(-1e-12);
                for (; b + 3 < b_end; b += 4) {
                    __m256d v_sum = _mm256_setzero_pd();
                    for (int j = i + 1; j < max_n; ++j) {
                        __m256d v_ij = _mm256_loadu_pd(&A_soa[(i * max_n + j) * batch_size + b]);
                        __m256d v_xj = _mm256_loadu_pd(&b_soa[j * batch_size + b]);
                        #ifdef __FMA__
                        v_sum = _mm256_fmadd_pd(v_ij, v_xj, v_sum);
                        #else
                        v_sum = _mm256_add_pd(v_sum, _mm256_mul_pd(v_ij, v_xj));
                        #endif
                    }
                    __m256d v_val = _mm256_sub_pd(_mm256_loadu_pd(&b_soa[rhsi_off + b]), v_sum);
                    __m256d v_diag = _mm256_loadu_pd(&A_soa[ii_off + b]);
                    __m256d mask_pos = _mm256_cmp_pd(v_diag, _mm256_setzero_pd(), _CMP_GE_OS);
                    __m256d mask_lt_eps = _mm256_cmp_pd(v_diag, v_eps, _CMP_LT_OS);
                    v_diag = _mm256_blendv_pd(v_diag, v_eps, _mm256_and_pd(mask_pos, mask_lt_eps));
                    __m256d mask_neg = _mm256_cmp_pd(v_diag, _mm256_setzero_pd(), _CMP_LT_OS);
                    __m256d mask_gt_neg_eps = _mm256_cmp_pd(v_diag, v_neg_eps, _CMP_GT_OS);
                    v_diag = _mm256_blendv_pd(v_diag, v_neg_eps, _mm256_and_pd(mask_neg, mask_gt_neg_eps));
                    _mm256_storeu_pd(&b_soa[rhsi_off + b], _mm256_div_pd(v_val, v_diag));
                }
                #elif defined(__aarch64__)
                float64x2_t v_eps = vdupq_n_f64(1e-12);
                float64x2_t v_neg_eps = vdupq_n_f64(-1e-12);
                float64x2_t v_zero = vdupq_n_f64(0.0);
                for (; b + 1 < b_end; b += 2) {
                    float64x2_t v_sum = vdupq_n_f64(0.0);
                    for (int j = i + 1; j < max_n; ++j) {
                        float64x2_t v_ij = vld1q_f64(&A_soa[(i * max_n + j) * batch_size + b]);
                        float64x2_t v_xj = vld1q_f64(&b_soa[j * batch_size + b]);
                        #if defined(__ARM_FEATURE_FMA)
                        v_sum = vfmaq_f64(v_sum, v_ij, v_xj);
                        #else
                        v_sum = vaddq_f64(v_sum, vmulq_f64(v_ij, v_xj));
                        #endif
                    }
                    float64x2_t v_val = vsubq_f64(vld1q_f64(&b_soa[rhsi_off + b]), v_sum);
                    float64x2_t v_diag = vld1q_f64(&A_soa[ii_off + b]);
                    uint64x2_t mask_pos = vcgeq_f64(v_diag, v_zero);
                    uint64x2_t mask_lt_eps = vcltq_f64(v_diag, v_eps);
                    v_diag = vbslq_f64(vandq_u64(mask_pos, mask_lt_eps), v_eps, v_diag);
                    uint64x2_t mask_neg = vcltq_f64(v_diag, v_zero);
                    uint64x2_t mask_gt_neg_eps = vcgtq_f64(v_diag, v_neg_eps);
                    v_diag = vbslq_f64(vandq_u64(mask_neg, mask_gt_neg_eps), v_neg_eps, v_diag);
                    vst1q_f64(&b_soa[rhsi_off + b], vdivq_f64(v_val, v_diag));
                }
                #endif
                for (; b < b_end; ++b) {
                    double sum = 0.0;
                    for (int j = i + 1; j < max_n; ++j) {
                        sum += A_soa[(i * max_n + j) * batch_size + b] * b_soa[j * batch_size + b];
                    }
                    double val = b_soa[rhsi_off + b] - sum;
                    double diag = A_soa[ii_off + b];
                    if (std::abs(diag) < 1e-12) diag = 1e-12 * (diag < 0 ? -1.0 : 1.0);
                    b_soa[rhsi_off + b] = val / diag;
                }
            }

            for (size_t b = b_start; b < b_end; ++b) {
                LinearSolver::Vector x(n);
                for (int j = 0; j < n; ++j) {
                    x[j] = b_soa[j * batch_size + b];
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
                        int v_idx = (n - 1) + comp.param_index;
                        current = x[v_idx];
                    }
                    
                    result.currents[comp.name] = current;
                    result.power[comp.name] = v_drop * current; 
                }
            }
        } 
    } 
    return results;
}

} // namespace CpuSolver
