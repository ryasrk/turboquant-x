#pragma once
// PolarQuant compression pipeline (float precision).
// Uses thread-local scratch buffers to eliminate per-block allocations.

#include <cstdint>
#include <vector>

namespace turboquant {

/// Compressed block: indices + norm + metadata.
struct CompressedBlock {
    std::vector<uint8_t> indices;  // Quantized centroid indices
    float norm;                    // L2 norm of original block
    int n_bits;                    // Quantization bit-width
    int block_size;                // Original (unpadded) block length
    int seed;                      // Rotation seed
};

/// Compressed tensor: collection of blocks.
struct CompressedTensor {
    std::vector<CompressedBlock> blocks;
    std::vector<int> original_shape;
    int n_bits;
    int block_size;
};

/// Compress a single block.  Uses thread-local scratch buffers.
CompressedBlock polar_quantize_block(const float* data, int len,
                                     int n_bits, int seed);

/// Decompress a single block.  Uses thread-local scratch buffers.
std::vector<float> polar_dequantize_block(const CompressedBlock& block);

/// Compress full tensor with blocking + OpenMP.
CompressedTensor polar_quantize(const float* data, int total_elements,
                                const int* shape, int ndim,
                                int n_bits, int seed, int block_size);

/// Decompress full tensor.
std::vector<float> polar_dequantize(const CompressedTensor& ct);

/// Compression ratio vs float16.
double compression_ratio(int n_bits, int block_size);

}  // namespace turboquant
