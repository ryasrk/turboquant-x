#include "rotation.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

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

std::vector<double> random_sign_diagonal(int d, int seed) {
    validate_power_of_two(d);
    std::mt19937_64 rng(static_cast<uint64_t>(seed));
    std::uniform_int_distribution<int> dist(0, 1);
    std::vector<double> signs(d);
    for (int i = 0; i < d; ++i) {
        signs[i] = dist(rng) == 0 ? -1.0 : 1.0;
    }
    return signs;
}

// -----------------------------------------------------------------------
// Fast Walsh-Hadamard Transform — O(d log d) butterfly algorithm
// In-place, normalized by 1/sqrt(d) at the end.
// This replaces the O(d²) matrix-vector multiply.
// -----------------------------------------------------------------------

static void fwht_inplace(double* data, int d) {
    // Butterfly: for each level, pairs at distance half_size
    for (int half_size = 1; half_size < d; half_size <<= 1) {
        for (int i = 0; i < d; i += half_size << 1) {
            for (int j = i; j < i + half_size; ++j) {
                double a = data[j];
                double b = data[j + half_size];
                data[j]             = a + b;
                data[j + half_size] = a - b;
            }
        }
    }
    // Normalize
    double norm = 1.0 / std::sqrt(static_cast<double>(d));
    for (int i = 0; i < d; ++i) {
        data[i] *= norm;
    }
}

// Kept for backward compat but not used in hot path anymore
std::vector<double> hadamard_matrix(int d) {
    validate_power_of_two(d);
    std::vector<double> H(static_cast<size_t>(d) * d, 0.0);
    // Build using butterfly on identity columns
    for (int col = 0; col < d; ++col) {
        H[col * d + col] = 1.0;
        fwht_inplace(&H[col * d], d);
    }
    return H;
}

void rotate(double* data, int d, int seed) {
    validate_power_of_two(d);

    auto signs = random_sign_diagonal(d, seed);

    // Step 1: sign flip
    for (int i = 0; i < d; ++i) {
        data[i] *= signs[i];
    }

    // Step 2: Fast WHT — O(d log d) instead of O(d²)
    fwht_inplace(data, d);
}

void inverse_rotate(double* data, int d, int seed) {
    validate_power_of_two(d);

    auto signs = random_sign_diagonal(d, seed);

    // Inverse WHT = WHT (self-inverse after normalization)
    fwht_inplace(data, d);

    // Apply signs
    for (int i = 0; i < d; ++i) {
        data[i] *= signs[i];
    }
}

void rotate_batch(double* data, int n_rows, int d, int seed) {
    #pragma omp parallel for schedule(static) if(n_rows > 16)
    for (int r = 0; r < n_rows; ++r) {
        rotate(data + r * d, d, seed);
    }
}

void inverse_rotate_batch(double* data, int n_rows, int d, int seed) {
    #pragma omp parallel for schedule(static) if(n_rows > 16)
    for (int r = 0; r < n_rows; ++r) {
        inverse_rotate(data + r * d, d, seed);
    }
}

}  // namespace turboquant
