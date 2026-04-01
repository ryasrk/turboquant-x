#pragma once
// Walsh-Hadamard Transform with random sign flips for TurboQuant rotation.
//
// The WHT-based random rotation spreads outlier magnitudes uniformly across all
// coordinates, making every entry approximately N(0, ||x||²/d).

#include <cstdint>
#include <vector>

namespace turboquant {

/// Validate that d is a positive power of 2. Throws std::invalid_argument if not.
void validate_power_of_two(int d);

/// Return the smallest power of 2 >= n.
int next_power_of_two(int n);

/// Generate d random ±1 values from a deterministic seed.
std::vector<double> random_sign_diagonal(int d, int seed);

/// Construct a normalized Hadamard matrix (d×d), stored row-major.
/// d must be a power of 2. Result is normalized by 1/sqrt(d).
std::vector<double> hadamard_matrix(int d);

/// Apply in-place Walsh-Hadamard Transform: y = H @ diag(signs) @ x.
/// x is modified in-place. shape: (..., d), but here we operate on a flat
/// vector of length d.
void rotate(double* data, int d, int seed);

/// Apply in-place inverse WHT: x = diag(signs) @ H^T @ y.
void inverse_rotate(double* data, int d, int seed);

/// Batched rotate for 2D data: n_rows vectors of length d each.
void rotate_batch(double* data, int n_rows, int d, int seed);

/// Batched inverse rotate for 2D data.
void inverse_rotate_batch(double* data, int n_rows, int d, int seed);

}  // namespace turboquant
