#pragma once
// PolarQuant block-level and tensor-level compress/decompress pipeline.
//
// Pipeline: norm extraction → WHT rotation → scalar quantization → packing
// Polar decomposition separates magnitude (norm) from direction (unit vector).

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
    std::vector<int> original_shape;  // Shape before compression
    int n_bits;
    int block_size;
};

/// Compress a single block using PolarQuant pipeline.
/// Pipeline: norm → normalize → pad → rotate → scale → quantize
CompressedBlock polar_quantize_block(const double* data, int len,
                                     int n_bits, int seed);

/// Decompress a single CompressedBlock.
/// Pipeline: dequantize → unscale → inverse_rotate → trim → rescale
std::vector<double> polar_dequantize_block(const CompressedBlock& block);

/// Compress a full tensor (flat array) with blocking.
/// Block i uses seed = base_seed + i.
CompressedTensor polar_quantize(const double* data, int total_elements,
                                const int* shape, int ndim,
                                int n_bits, int seed, int block_size);

/// Decompress a CompressedTensor back to flat array.
std::vector<double> polar_dequantize(const CompressedTensor& ct);

/// Compression ratio vs float16 storage.
double compression_ratio(int n_bits, int block_size);

}  // namespace turboquant
