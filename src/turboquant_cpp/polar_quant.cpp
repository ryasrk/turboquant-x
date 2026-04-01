#include "polar_quant.h"
#include "codebook.h"
#include "rotation.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

namespace turboquant {

CompressedBlock polar_quantize_block(const double* data, int len,
                                     int n_bits, int seed) {
    if (n_bits != 2 && n_bits != 3 && n_bits != 4) {
        throw std::invalid_argument(
            "n_bits must be 2, 3, or 4, got " + std::to_string(n_bits));
    }
    if (len <= 0) {
        throw std::invalid_argument("Cannot compress an empty block");
    }

    const Codebook& cb = get_codebook(n_bits);
    int padded_len = next_power_of_two(len);

    // 1. Compute L2 norm
    double norm_sq = 0.0;
    for (int i = 0; i < len; ++i) {
        norm_sq += data[i] * data[i];
    }
    float norm_val = static_cast<float>(std::sqrt(norm_sq));

    // 2. Zero-vector fast-path
    if (norm_val == 0.0f) {
        std::vector<double> zeros(padded_len, 0.0);
        std::vector<uint8_t> indices(padded_len);
        quantize_scalar(zeros.data(), indices.data(), padded_len, cb);
        return CompressedBlock{
            std::move(indices), norm_val, n_bits, len, seed};
    }

    // 3. Normalize to unit vector + pad
    std::vector<double> x_hat(padded_len, 0.0);
    double inv_norm = 1.0 / static_cast<double>(norm_val);
    for (int i = 0; i < len; ++i) {
        x_hat[i] = data[i] * inv_norm;
    }

    // 4. Rotate (WHT + random sign flip)
    rotate(x_hat.data(), padded_len, seed);

    // 5. Scale: rotated unit vector entries ≈ N(0, 1/d), multiply by √d → N(0, 1)
    double scale = std::sqrt(static_cast<double>(padded_len));
    for (int i = 0; i < padded_len; ++i) {
        x_hat[i] *= scale;
    }

    // 6. Quantize with Lloyd-Max codebook
    std::vector<uint8_t> indices(padded_len);
    quantize_scalar(x_hat.data(), indices.data(), padded_len, cb);

    return CompressedBlock{std::move(indices), norm_val, n_bits, len, seed};
}

std::vector<double> polar_dequantize_block(const CompressedBlock& block) {
    // Zero-vector fast-path
    if (block.norm == 0.0f) {
        return std::vector<double>(block.block_size, 0.0);
    }

    const Codebook& cb = get_codebook(block.n_bits);
    int d = static_cast<int>(block.indices.size());

    // 1. Dequantize
    std::vector<double> z(d);
    dequantize_scalar(block.indices.data(), z.data(), d, cb);

    // 2. Unscale
    double inv_scale = 1.0 / std::sqrt(static_cast<double>(d));
    for (int i = 0; i < d; ++i) {
        z[i] *= inv_scale;
    }

    // 3. Inverse rotate
    inverse_rotate(z.data(), d, block.seed);

    // 4. Trim padding + rescale by norm
    std::vector<double> result(block.block_size);
    double norm_d = static_cast<double>(block.norm);
    for (int i = 0; i < block.block_size; ++i) {
        result[i] = z[i] * norm_d;
    }

    return result;
}

CompressedTensor polar_quantize(const double* data, int total_elements,
                                const int* shape, int ndim,
                                int n_bits, int seed, int block_size) {
    if (n_bits != 2 && n_bits != 3 && n_bits != 4) {
        throw std::invalid_argument(
            "n_bits must be 2, 3, or 4, got " + std::to_string(n_bits));
    }
    if (block_size < 1) {
        throw std::invalid_argument(
            "block_size must be >= 1, got " + std::to_string(block_size));
    }

    int n_blocks = (total_elements + block_size - 1) / block_size;
    std::vector<CompressedBlock> blocks(n_blocks);

    #pragma omp parallel for schedule(dynamic, 4) if(n_blocks > 32)
    for (int i = 0; i < n_blocks; ++i) {
        int start = i * block_size;
        int end = std::min(start + block_size, total_elements);
        blocks[i] = polar_quantize_block(data + start, end - start,
                                          n_bits, seed + i);
    }

    std::vector<int> orig_shape(shape, shape + ndim);

    return CompressedTensor{
        std::move(blocks), std::move(orig_shape), n_bits, block_size};
}

std::vector<double> polar_dequantize(const CompressedTensor& ct) {
    int total = 1;
    for (int s : ct.original_shape) total *= s;

    int n_blocks = static_cast<int>(ct.blocks.size());

    // Decompress all blocks in parallel, then concatenate
    std::vector<std::vector<double>> parts(n_blocks);

    #pragma omp parallel for schedule(dynamic, 4) if(n_blocks > 32)
    for (int i = 0; i < n_blocks; ++i) {
        parts[i] = polar_dequantize_block(ct.blocks[i]);
    }

    std::vector<double> result;
    result.reserve(total);
    for (auto& p : parts) {
        result.insert(result.end(), p.begin(), p.end());
    }

    result.resize(total);
    return result;
}

double compression_ratio(int n_bits, int block_size) {
    return 16.0 / (n_bits + 32.0 / block_size);
}

}  // namespace turboquant
