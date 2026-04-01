#include "rotation.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace turboquant {

void validate_power_of_two(int d) {
    if (d <= 0 || (d & (d - 1)) != 0) {
        throw std::invalid_argument(
            "d must be a positive power of 2, got " + std::to_string(d));
    }
}

int next_power_of_two(int n) {
    if (n <= 0) {
        throw std::invalid_argument(
            "n must be positive, got " + std::to_string(n));
    }
    if (n == 1) return 1;
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

std::vector<float> random_sign_diagonal(int d, int seed) {
    validate_power_of_two(d);
    std::mt19937_64 rng(static_cast<uint64_t>(seed));
    std::uniform_int_distribution<int> dist(0, 1);
    std::vector<float> signs(d);
    for (int i = 0; i < d; ++i) {
        signs[i] = dist(rng) == 0 ? -1.0f : 1.0f;
    }
    return signs;
}

// -----------------------------------------------------------------------
// Fast Walsh-Hadamard Transform — O(d log d) butterfly with AVX2 SIMD
// Processes 8 floats per cycle in the butterfly inner loop.
// Float32 gives 2x SIMD throughput over double and halves memory bandwidth.
// -----------------------------------------------------------------------

static void fwht_inplace(float* data, int d) {
    for (int half_size = 1; half_size < d; half_size <<= 1) {
        for (int i = 0; i < d; i += half_size << 1) {
#ifdef __AVX2__
            int j = i;
            int end = i + half_size;
            // AVX2: process 8 floats per iteration
            for (; j + 7 < end; j += 8) {
                __m256 a = _mm256_loadu_ps(&data[j]);
                __m256 b = _mm256_loadu_ps(&data[j + half_size]);
                _mm256_storeu_ps(&data[j],             _mm256_add_ps(a, b));
                _mm256_storeu_ps(&data[j + half_size], _mm256_sub_ps(a, b));
            }
            // Scalar remainder
            for (; j < end; ++j) {
                float a = data[j];
                float b = data[j + half_size];
                data[j]             = a + b;
                data[j + half_size] = a - b;
            }
#else
            for (int j = i; j < i + half_size; ++j) {
                float a = data[j];
                float b = data[j + half_size];
                data[j]             = a + b;
                data[j + half_size] = a - b;
            }
#endif
        }
    }
    // Normalize by 1/sqrt(d)
    float norm = 1.0f / std::sqrt(static_cast<float>(d));
#ifdef __AVX2__
    __m256 norm_v = _mm256_set1_ps(norm);
    int i = 0;
    for (; i + 7 < d; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        _mm256_storeu_ps(&data[i], _mm256_mul_ps(v, norm_v));
    }
    for (; i < d; ++i) data[i] *= norm;
#else
    for (int i = 0; i < d; ++i) data[i] *= norm;
#endif
}

// Kept for backward compat — builds full matrix via FWHT on identity columns
std::vector<float> hadamard_matrix(int d) {
    validate_power_of_two(d);
    std::vector<float> H(static_cast<size_t>(d) * d, 0.0f);
    for (int col = 0; col < d; ++col) {
        H[col * d + col] = 1.0f;
        fwht_inplace(&H[col * d], d);
    }
    return H;
}

// AVX2 element-wise multiply helper
static inline void vec_mul_inplace(float* data, const float* signs, int d) {
#ifdef __AVX2__
    int i = 0;
    for (; i + 7 < d; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        __m256 s = _mm256_loadu_ps(&signs[i]);
        _mm256_storeu_ps(&data[i], _mm256_mul_ps(v, s));
    }
    for (; i < d; ++i) data[i] *= signs[i];
#else
    for (int i = 0; i < d; ++i) data[i] *= signs[i];
#endif
}

void rotate(float* data, int d, int seed) {
    validate_power_of_two(d);
    auto signs = random_sign_diagonal(d, seed);

    // Step 1: sign flip (AVX2 vectorized)
    vec_mul_inplace(data, signs.data(), d);

    // Step 2: Fast WHT — O(d log d) with AVX2 SIMD
    fwht_inplace(data, d);
}

void inverse_rotate(float* data, int d, int seed) {
    validate_power_of_two(d);
    auto signs = random_sign_diagonal(d, seed);

    // Inverse WHT = WHT (self-inverse after normalization)
    fwht_inplace(data, d);

    // Apply signs (AVX2 vectorized)
    vec_mul_inplace(data, signs.data(), d);
}

void rotate_batch(float* data, int n_rows, int d, int seed) {
    #pragma omp parallel for schedule(static) if(n_rows > 16)
    for (int r = 0; r < n_rows; ++r) {
        rotate(data + r * d, d, seed);
    }
}

void inverse_rotate_batch(float* data, int n_rows, int d, int seed) {
    #pragma omp parallel for schedule(static) if(n_rows > 16)
    for (int r = 0; r < n_rows; ++r) {
        inverse_rotate(data + r * d, d, seed);
    }
}

}  // namespace turboquant
