#pragma once
// Walsh-Hadamard Transform (WHT) rotation with AVX2 SIMD acceleration.
// Uses float (32-bit) for 2x SIMD throughput and half memory bandwidth.

#include <vector>

namespace turboquant {

void validate_power_of_two(int d);
int next_power_of_two(int n);

/// Random ±1 diagonal for WHT rotation (float precision).
std::vector<float> random_sign_diagonal(int d, int seed);

/// Full Hadamard matrix (kept for backward compat, not used in hot path).
std::vector<float> hadamard_matrix(int d);

/// In-place rotation: y = H @ diag(signs) @ x.  AVX2-accelerated.
void rotate(float* data, int d, int seed);

/// In-place inverse rotation: x = diag(signs) @ H^T @ y.
void inverse_rotate(float* data, int d, int seed);

/// Batch rotation with OpenMP parallelism.
void rotate_batch(float* data, int n_rows, int d, int seed);
void inverse_rotate_batch(float* data, int n_rows, int d, int seed);

}  // namespace turboquant
