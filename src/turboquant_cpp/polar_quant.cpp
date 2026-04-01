#include "polar_quant.h"
#include "codebook.h"
#include "rotation.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace turboquant {

// -----------------------------------------------------------------------
// Thread-local scratch buffers — eliminates malloc/free per block.
// With block_size=128 (padded_len=128), each block would otherwise
// allocate 2 vectors (~1KB).  Over 10,000+ blocks per tensor, this
// saves millions of allocations.
// -----------------------------------------------------------------------
static thread_local std::vector<float> tl_float_buf;
static thread_local std::vector<uint8_t> tl_index_buf;

static float* get_float_scratch(int n) {
    if (static_cast<int>(tl_float_buf.size()) < n) {
        tl_float_buf.resize(n);
    }
    return tl_float_buf.data();
}

static uint8_t* get_index_scratch(int n) {
    if (static_cast<int>(tl_index_buf.size()) < n) {
        tl_index_buf.resize(n);
    }
    return tl_index_buf.data();
}

// -----------------------------------------------------------------------
// Block-level compress
// -----------------------------------------------------------------------

CompressedBlock polar_quantize_block(const float* data, int len,
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

    // 1. Compute L2 norm (AVX2 vectorized)
    float norm_sq = 0.0f;
#ifdef __AVX2__
    {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < len; i += 8) {
            __m256 v = _mm256_loadu_ps(&data[i]);
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        // Horizontal sum
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc);
        for (int j = 0; j < 8; ++j) norm_sq += tmp[j];
        for (; i < len; ++i) norm_sq += data[i] * data[i];
    }
#else
    for (int i = 0; i < len; ++i) {
        norm_sq += data[i] * data[i];
    }
#endif
    float norm_val = std::sqrt(norm_sq);

    // 2. Zero-vector fast-path
    if (norm_val == 0.0f) {
        float* scratch = get_float_scratch(padded_len);
        std::memset(scratch, 0, padded_len * sizeof(float));
        std::vector<uint8_t> indices(padded_len);
        quantize_scalar(scratch, indices.data(), padded_len, cb);
        return CompressedBlock{
            std::move(indices), norm_val, n_bits, len, seed};
    }

    // 3. Normalize + pad using scratch buffer (no allocation)
    float* x_hat = get_float_scratch(padded_len);
    float inv_norm = 1.0f / norm_val;

#ifdef __AVX2__
    {
        __m256 inv_v = _mm256_set1_ps(inv_norm);
        int i = 0;
        for (; i + 7 < len; i += 8) {
            __m256 v = _mm256_loadu_ps(&data[i]);
            _mm256_storeu_ps(&x_hat[i], _mm256_mul_ps(v, inv_v));
        }
        for (; i < len; ++i) x_hat[i] = data[i] * inv_norm;
    }
#else
    for (int i = 0; i < len; ++i) {
        x_hat[i] = data[i] * inv_norm;
    }
#endif
    // Zero-pad
    for (int i = len; i < padded_len; ++i) {
        x_hat[i] = 0.0f;
    }

    // 4. Rotate (WHT + random sign flip) — in-place, AVX2 accelerated
    rotate(x_hat, padded_len, seed);

    // 5. Scale: rotated unit vector entries ~ N(0, 1/d), * sqrt(d) -> N(0, 1)
    float scale = std::sqrt(static_cast<float>(padded_len));
#ifdef __AVX2__
    {
        __m256 scale_v = _mm256_set1_ps(scale);
        int i = 0;
        for (; i + 7 < padded_len; i += 8) {
            __m256 v = _mm256_loadu_ps(&x_hat[i]);
            _mm256_storeu_ps(&x_hat[i], _mm256_mul_ps(v, scale_v));
        }
        for (; i < padded_len; ++i) x_hat[i] *= scale;
    }
#else
    for (int i = 0; i < padded_len; ++i) {
        x_hat[i] *= scale;
    }
#endif

    // 6. Quantize with Lloyd-Max codebook
    std::vector<uint8_t> indices(padded_len);
    quantize_scalar(x_hat, indices.data(), padded_len, cb);

    return CompressedBlock{std::move(indices), norm_val, n_bits, len, seed};
}

// -----------------------------------------------------------------------
// Block-level decompress
// -----------------------------------------------------------------------

std::vector<float> polar_dequantize_block(const CompressedBlock& block) {
    // Zero-vector fast-path
    if (block.norm == 0.0f) {
        return std::vector<float>(block.block_size, 0.0f);
    }

    const Codebook& cb = get_codebook(block.n_bits);
    int d = static_cast<int>(block.indices.size());

    // 1. Dequantize into scratch buffer
    float* z = get_float_scratch(d);
    dequantize_scalar(block.indices.data(), z, d, cb);

    // 2. Unscale
    float inv_scale = 1.0f / std::sqrt(static_cast<float>(d));
#ifdef __AVX2__
    {
        __m256 inv_v = _mm256_set1_ps(inv_scale);
        int i = 0;
        for (; i + 7 < d; i += 8) {
            __m256 v = _mm256_loadu_ps(&z[i]);
            _mm256_storeu_ps(&z[i], _mm256_mul_ps(v, inv_v));
        }
        for (; i < d; ++i) z[i] *= inv_scale;
    }
#else
    for (int i = 0; i < d; ++i) {
        z[i] *= inv_scale;
    }
#endif

    // 3. Inverse rotate (in-place, AVX2 accelerated)
    inverse_rotate(z, d, block.seed);

    // 4. Trim padding + rescale by norm
    std::vector<float> result(block.block_size);
    float norm_f = block.norm;
#ifdef __AVX2__
    {
        __m256 norm_v = _mm256_set1_ps(norm_f);
        int i = 0;
        for (; i + 7 < block.block_size; i += 8) {
            __m256 v = _mm256_loadu_ps(&z[i]);
            _mm256_storeu_ps(&result[i], _mm256_mul_ps(v, norm_v));
        }
        for (; i < block.block_size; ++i) result[i] = z[i] * norm_f;
    }
#else
    for (int i = 0; i < block.block_size; ++i) {
        result[i] = z[i] * norm_f;
    }
#endif

    return result;
}

// -----------------------------------------------------------------------
// Tensor-level compress (OpenMP parallel across blocks)
// -----------------------------------------------------------------------

CompressedTensor polar_quantize(const float* data, int total_elements,
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

// -----------------------------------------------------------------------
// Tensor-level decompress (OpenMP parallel + pre-allocated output)
// -----------------------------------------------------------------------

std::vector<float> polar_dequantize(const CompressedTensor& ct) {
    int total = 1;
    for (int s : ct.original_shape) total *= s;

    int n_blocks = static_cast<int>(ct.blocks.size());

    // Pre-allocate output and write directly (avoids concat copies)
    std::vector<float> result(total, 0.0f);

    #pragma omp parallel for schedule(dynamic, 4) if(n_blocks > 32)
    for (int i = 0; i < n_blocks; ++i) {
        auto part = polar_dequantize_block(ct.blocks[i]);
        int start = i * ct.block_size;
        int copy_len = static_cast<int>(part.size());
        if (start + copy_len > total) copy_len = total - start;
        std::memcpy(&result[start], part.data(), copy_len * sizeof(float));
    }

    return result;
}

double compression_ratio(int n_bits, int block_size) {
    return 16.0 / (n_bits + 32.0 / block_size);
}

}  // namespace turboquant
