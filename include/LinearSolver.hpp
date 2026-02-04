#ifndef LINEAR_SOLVER_HPP
#define LINEAR_SOLVER_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>

class LinearSolver {
public:
    using Matrix = std::vector<std::vector<double>>;
    using Vector = std::vector<double>;

    static Vector solve_linear_system(Matrix A, Vector b) {
        int n = A.size();
        if (n == 0) return {};
        if (A[0].size() != n) throw std::invalid_argument("Matrix must be square");
        if (b.size() != n) throw std::invalid_argument("Vector dimension mismatch");

        // Forward elimination with partial pivoting
        for (int i = 0; i < n; i++) {
            // Pivot selection
            int pivot = i;
            for (int j = i + 1; j < n; j++) {
                if (std::abs(A[j][i]) > std::abs(A[pivot][i])) {
                    pivot = j;
                }
            }

            // Swap rows
            std::swap(A[i], A[pivot]);
            std::swap(b[i], b[pivot]);

            if (std::abs(A[i][i]) < 1e-12) {
                // Determine if consistent or inconsistent (infinite solutions vs no solution)
                // For circuit simulation, singular matrix usually means floating nodes or loops of voltage sources.
                throw std::runtime_error("Matrix is singular or near-singular");
            }

            // Eliminate
            for (int j = i + 1; j < n; j++) {
                double factor = A[j][i] / A[i][i];
                b[j] -= factor * b[i];
                for (int k = i; k < n; k++) {
                    A[j][k] -= factor * A[i][k];
                }
            }
        }

        // Back substitution
        Vector x(n);
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0;
            for (int j = i + 1; j < n; j++) {
                sum += A[i][j] * x[j];
            }
            x[i] = (b[i] - sum) / A[i][i];
        }

        return x;
    }
};

#endif // LINEAR_SOLVER_HPP
