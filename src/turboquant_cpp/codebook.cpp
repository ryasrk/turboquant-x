#include "codebook.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef __AVX2__
#include <immintrin.h>
#endif

// Standard normal PDF and CDF (no external dependency)
namespace {

constexpr double INV_SQRT_2PI = 0.3989422804014327;

double norm_pdf(double x) {
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}

double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

}  // namespace

namespace turboquant {

Codebook lloyd_max_codebook(int n_bits, int n_iterations, double tol) {
    if (n_bits < 1 || n_bits > 8) {
        throw std::invalid_argument(
            "n_bits must be in [1, 8], got " + std::to_string(n_bits));
    }

    int n_levels = 1 << n_bits;
    // Internal computation in double for precision
    std::vector<double> centroids(n_levels);
    for (int i = 0; i < n_levels; ++i) {
        centroids[i] = -3.0 + 6.0 * i / (n_levels - 1);
    }

    std::vector<double> boundaries(n_levels - 1);

    for (int iter = 0; iter < n_iterations; ++iter) {
        for (int i = 0; i < n_levels - 1; ++i) {
            boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
        }

        std::vector<double> b(n_levels + 1);
        b[0] = -1e30;
        for (int i = 0; i < n_levels - 1; ++i) {
            b[i + 1] = boundaries[i];
        }
        b[n_levels] = 1e30;

        std::vector<double> new_centroids(n_levels);
        double max_change = 0.0;

        for (int i = 0; i < n_levels; ++i) {
            double lo = b[i], hi = b[i + 1];
            double cdf_diff = norm_cdf(hi) - norm_cdf(lo);
            if (cdf_diff < 1e-15) {
                new_centroids[i] = (lo + hi) / 2.0;
            } else {
                double pdf_diff = norm_pdf(lo) - norm_pdf(hi);
                new_centroids[i] = pdf_diff / cdf_diff;
            }
            max_change = std::max(max_change,
                                  std::abs(new_centroids[i] - centroids[i]));
        }

        centroids = new_centroids;
        if (max_change < tol) break;
    }

    for (int i = 0; i < n_levels - 1; ++i) {
        boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
    }

    // Convert to float for storage
    std::vector<float> f_centroids(n_levels);
    std::vector<float> f_boundaries(n_levels - 1);
    for (int i = 0; i < n_levels; ++i) {
        f_centroids[i] = static_cast<float>(centroids[i]);
    }
    for (int i = 0; i < n_levels - 1; ++i) {
        f_boundaries[i] = static_cast<float>(boundaries[i]);
    }

    return Codebook{n_bits, std::move(f_centroids), std::move(f_boundaries)};
}

// Precomputed codebooks (float precision — matches Python within float32 ULP)
static const Codebook TURBO2{
    2,
    {-1.5104176f, -0.45278004f, 0.45278004f, 1.5104176f},
    {-0.9815988f, 0.0f, 0.9815988f},
};

static const Codebook TURBO3{
    3,
    {-2.1519457f, -1.3439093f, -0.7560053f, -0.24509418f,
      0.24509418f,  0.7560053f,  1.3439093f,  2.1519457f},
    {-1.7479275f, -1.0499573f, -0.5005497f, 0.0f,
      0.5005497f,  1.0499573f,  1.7479275f},
};

static const Codebook TURBO4{
    4,
    {-2.7329730f, -2.0694504f, -1.6184883f, -1.2566477f,
     -0.9427008f, -0.6570370f, -0.3882236f, -0.12845491f,
      0.12845491f,  0.3882236f,  0.6570370f,  0.9427008f,
      1.2566477f,  1.6184883f,  2.0694504f,  2.7329730f},
    {-2.4012117f, -1.8439694f, -1.4375680f, -1.0996742f,
     -0.7998689f, -0.5226303f, -0.25833923f, 0.0f,
      0.25833923f,  0.5226303f,  0.7998689f,  1.0996742f,
      1.4375680f,  1.8439694f,  2.4012117f},
};

const Codebook& get_codebook(int n_bits) {
    switch (n_bits) {
        case 2: return TURBO2;
        case 3: return TURBO3;
        case 4: return TURBO4;
        default:
            throw std::invalid_argument(
                "No precomputed codebook for " + std::to_string(n_bits) +
                "-bit. Supported: 2, 3, 4.");
    }
}

// -----------------------------------------------------------------------
// Branchless scalar quantization — AVX2 accelerated for TURBO4 (15 bounds)
// Instead of binary search O(n log k), we sum comparison results O(n * k)
// which is branchless and SIMD-friendly.
// For k=15, branchless beats binary search due to branch mispredictions.
// -----------------------------------------------------------------------

#ifdef __AVX2__
static void quantize_scalar_avx2_15(const float* x, uint8_t* indices, int n,
                                     const float* bounds) {
    // Broadcast all 15 boundaries into AVX registers
    __m256 b0  = _mm256_set1_ps(bounds[0]);
    __m256 b1  = _mm256_set1_ps(bounds[1]);
    __m256 b2  = _mm256_set1_ps(bounds[2]);
    __m256 b3  = _mm256_set1_ps(bounds[3]);
    __m256 b4  = _mm256_set1_ps(bounds[4]);
    __m256 b5  = _mm256_set1_ps(bounds[5]);
    __m256 b6  = _mm256_set1_ps(bounds[6]);
    __m256 b7  = _mm256_set1_ps(bounds[7]);
    __m256 b8  = _mm256_set1_ps(bounds[8]);
    __m256 b9  = _mm256_set1_ps(bounds[9]);
    __m256 b10 = _mm256_set1_ps(bounds[10]);
    __m256 b11 = _mm256_set1_ps(bounds[11]);
    __m256 b12 = _mm256_set1_ps(bounds[12]);
    __m256 b13 = _mm256_set1_ps(bounds[13]);
    __m256 b14 = _mm256_set1_ps(bounds[14]);

    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 val = _mm256_loadu_ps(&x[i]);

        // _CMP_GE_OQ: val >= bound_k → all bits set (0xFFFFFFFF = -1 as int32)
        // Subtract each comparison from accumulator (subtracting -1 = adding 1)
        __m256i sum = _mm256_setzero_si256();
        sum = _mm256_sub_epi32(sum, _mm256_castps_si256(_mm256_cmp_ps(val, b0,  _CMP_GE_OQ)));
        sum = _mm256_sub_epi32(sum, _mm256_castps_si256(_mm256_cmp_ps(val, b1,  _CMP_GE_OQ)));
        sum = _mm256_sub_epi32(sum, _mm256_castps_si256(_mm256_cmp_ps(val, b2,  _CMP_GE_OQ)));
        sum = _mm256_sub_epi32(sum, _mm256_castps_si256(_mm256_cmp_ps(val, b3,  _CMP_GE_OQ)));
        sum = _mm256_sub_epi32(sum, _mm256_castps_si256(_mm256_cmp_ps(val, b4,  _CMP_GE_OQ)));
        sum = _mm256_sub_epi32(sum, _mm256_castps_si256(_mm256_cmp_ps(val, b5,  _CMP_GE_OQ)));
        sum = _mm256_sub_epi32(sum, _mm256_castps_si256(_mm256_cmp_ps(val, b6,  _CMP_GE_OQ)));
        sum = _mm256_sub_epi32(sum, _mm256_castps_si256(_mm256_cmp_ps(val, b7,  _CMP_GE_OQ)));
        sum = _mm256_sub_epi32(sum, _mm256_castps_si256(_mm256_cmp_ps(val, b8,  _CMP_GE_OQ)));
        sum = _mm256_sub_epi32(sum, _mm256_castps_si256(_mm256_cmp_ps(val, b9,  _CMP_GE_OQ)));
        sum = _mm256_sub_epi32(sum, _mm256_castps_si256(_mm256_cmp_ps(val, b10, _CMP_GE_OQ)));
        sum = _mm256_sub_epi32(sum, _mm256_castps_si256(_mm256_cmp_ps(val, b11, _CMP_GE_OQ)));
        sum = _mm256_sub_epi32(sum, _mm256_castps_si256(_mm256_cmp_ps(val, b12, _CMP_GE_OQ)));
        sum = _mm256_sub_epi32(sum, _mm256_castps_si256(_mm256_cmp_ps(val, b13, _CMP_GE_OQ)));
        sum = _mm256_sub_epi32(sum, _mm256_castps_si256(_mm256_cmp_ps(val, b14, _CMP_GE_OQ)));

        // Extract 32-bit indices and pack to uint8
        alignas(32) int32_t idx32[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(idx32), sum);
        for (int j = 0; j < 8; ++j) {
            indices[i + j] = static_cast<uint8_t>(idx32[j]);
        }
    }

    // Scalar remainder
    for (; i < n; ++i) {
        float val = x[i];
        uint8_t idx = 0;
        idx += (val >= bounds[0]);  idx += (val >= bounds[1]);
        idx += (val >= bounds[2]);  idx += (val >= bounds[3]);
        idx += (val >= bounds[4]);  idx += (val >= bounds[5]);
        idx += (val >= bounds[6]);  idx += (val >= bounds[7]);
        idx += (val >= bounds[8]);  idx += (val >= bounds[9]);
        idx += (val >= bounds[10]); idx += (val >= bounds[11]);
        idx += (val >= bounds[12]); idx += (val >= bounds[13]);
        idx += (val >= bounds[14]);
        indices[i] = idx;
    }
}
#endif

void quantize_scalar(const float* x, uint8_t* indices, int n,
                     const Codebook& cb) {
    int n_bounds = static_cast<int>(cb.boundaries.size());

#ifdef __AVX2__
    if (n_bounds == 15) {
        quantize_scalar_avx2_15(x, indices, n, cb.boundaries.data());
        return;
    }
#endif

    // Branchless for any small codebook (2/3/4-bit)
    if (n_bounds <= 15) {
        const float* bounds = cb.boundaries.data();
        for (int i = 0; i < n; ++i) {
            float val = x[i];
            uint8_t idx = 0;
            for (int b = 0; b < n_bounds; ++b) {
                idx += (val >= bounds[b]);
            }
            indices[i] = idx;
        }
    } else {
        // Binary search fallback for large codebooks
        for (int i = 0; i < n; ++i) {
            auto it = std::lower_bound(cb.boundaries.begin(),
                                       cb.boundaries.end(), x[i]);
            indices[i] = static_cast<uint8_t>(it - cb.boundaries.begin());
        }
    }
}

void dequantize_scalar(const uint8_t* indices, float* out, int n,
                       const Codebook& cb) {
    const float* centroids = cb.centroids.data();
    for (int i = 0; i < n; ++i) {
        out[i] = centroids[indices[i]];
    }
}

}  // namespace turboquant
