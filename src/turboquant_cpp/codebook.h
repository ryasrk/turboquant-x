#pragma once
// Lloyd-Max optimal scalar quantization for TurboQuant.
//
// After Walsh-Hadamard rotation, KV cache values follow N(0, σ²).
// Lloyd-Max codebooks are provably optimal for this case.

#include <cstdint>
#include <vector>

namespace turboquant {

/// Precomputed Lloyd-Max codebook with centroids and decision boundaries.
struct Codebook {
    int n_bits;
    std::vector<double> centroids;   // size = 2^n_bits
    std::vector<double> boundaries;  // size = 2^n_bits - 1
};

/// Compute Lloyd-Max optimal codebook for N(0,1) distribution.
Codebook lloyd_max_codebook(int n_bits, int n_iterations = 50, double tol = 1e-8);

/// Get precomputed codebook for 2, 3, or 4 bits.
const Codebook& get_codebook(int n_bits);

/// Quantize continuous values to nearest centroid index.
/// Uses binary search on boundaries for O(n log k).
void quantize_scalar(const double* x, uint8_t* indices, int n, const Codebook& cb);

/// Reconstruct values from centroid indices (lookup).
void dequantize_scalar(const uint8_t* indices, double* out, int n, const Codebook& cb);

}  // namespace turboquant
