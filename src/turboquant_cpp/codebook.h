#pragma once
// Lloyd-Max optimal scalar quantization codebooks (float precision).
// Branchless quantization for 2/3/4-bit modes with AVX2 acceleration.

#include <cstdint>
#include <vector>

namespace turboquant {

struct Codebook {
    int n_bits;
    std::vector<float> centroids;
    std::vector<float> boundaries;
};

/// Compute Lloyd-Max codebook for N(0,1).  Internal computation uses double.
Codebook lloyd_max_codebook(int n_bits, int n_iterations = 50, double tol = 1e-8);

/// Get precomputed codebook (2, 3, or 4-bit).
const Codebook& get_codebook(int n_bits);

/// Branchless scalar quantization — AVX2 accelerated for TURBO4.
void quantize_scalar(const float* x, uint8_t* indices, int n, const Codebook& cb);

/// Dequantize: look up centroid values from indices.
void dequantize_scalar(const uint8_t* indices, float* out, int n, const Codebook& cb);

}  // namespace turboquant
