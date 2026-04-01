#include "codebook.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Standard normal PDF and CDF (no external dependency needed)
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
    std::vector<double> centroids(n_levels);

    // Initialize uniformly in [-3, 3]
    for (int i = 0; i < n_levels; ++i) {
        centroids[i] = -3.0 + 6.0 * i / (n_levels - 1);
    }

    std::vector<double> boundaries(n_levels - 1);

    for (int iter = 0; iter < n_iterations; ++iter) {
        // Boundaries = midpoints
        for (int i = 0; i < n_levels - 1; ++i) {
            boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
        }

        // Extended boundaries with ±∞ sentinels
        std::vector<double> b(n_levels + 1);
        b[0] = -1e30;  // Approximate -∞
        for (int i = 0; i < n_levels - 1; ++i) {
            b[i + 1] = boundaries[i];
        }
        b[n_levels] = 1e30;  // Approximate +∞

        std::vector<double> new_centroids(n_levels);
        double max_change = 0.0;

        for (int i = 0; i < n_levels; ++i) {
            double lo = b[i], hi = b[i + 1];
            double cdf_diff = norm_cdf(hi) - norm_cdf(lo);
            if (cdf_diff < 1e-15) {
                new_centroids[i] = (lo + hi) / 2.0;
            } else {
                // Conditional expectation E[X | lo < X < hi] for N(0,1)
                double pdf_diff = norm_pdf(lo) - norm_pdf(hi);
                new_centroids[i] = pdf_diff / cdf_diff;
            }
            max_change = std::max(max_change,
                                  std::abs(new_centroids[i] - centroids[i]));
        }

        centroids = new_centroids;
        if (max_change < tol) break;
    }

    // Final boundaries
    for (int i = 0; i < n_levels - 1; ++i) {
        boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
    }

    return Codebook{n_bits, std::move(centroids), std::move(boundaries)};
}

// Precomputed codebooks (exact values from Python)
static const Codebook TURBO2{
    2,
    {-1.5104176085002023, -0.4527800346370284,
      0.4527800346370285,  1.5104176085002028},
    {-0.9815988215686153, 0.0, 0.9815988215686157},
};

static const Codebook TURBO3{
    3,
    {-2.1519457045434560, -1.3439092785115518,
     -0.7560052812106792, -0.2450941789459803,
      0.2450941789459804,  0.7560052812106796,
      1.3439092785115505,  2.1519457045434582},
    {-1.7479274915275038, -1.0499572798611156,
     -0.5005497300783297,  0.0,
      0.5005497300783299,  1.0499572798611150,
      1.7479274915275043},
};

static const Codebook TURBO4{
    4,
    {-2.7329730276314383, -2.0694504218818746,
     -1.6184883483594745, -1.2566476802931554,
     -0.9427007640381189, -0.6570370151078325,
     -0.3882235722082744, -0.1284549100577299,
      0.1284549100577301,  0.3882235722082740,
      0.6570370151078336,  0.9427007640381218,
      1.2566476802931565,  1.6184883483594787,
      2.0694504218818760,  2.7329730276314304},
    {-2.4012117247566565, -1.8439693851206744,
     -1.4375680143263150, -1.0996742221656373,
     -0.7998688895729758, -0.5226302936580535,
     -0.2583392411330021,  0.0,
      0.2583392411330021,  0.5226302936580538,
      0.7998688895729778,  1.0996742221656390,
      1.4375680143263176,  1.8439693851206773,
      2.4012117247566529},
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

void quantize_scalar(const double* x, uint8_t* indices, int n,
                     const Codebook& cb) {
    // Binary search on boundaries (equivalent to np.searchsorted)
    int n_bounds = static_cast<int>(cb.boundaries.size());
    for (int i = 0; i < n; ++i) {
        // std::lower_bound gives first boundary >= x[i]
        auto it = std::lower_bound(cb.boundaries.begin(), cb.boundaries.end(),
                                   x[i]);
        indices[i] = static_cast<uint8_t>(it - cb.boundaries.begin());
    }
}

void dequantize_scalar(const uint8_t* indices, double* out, int n,
                       const Codebook& cb) {
    for (int i = 0; i < n; ++i) {
        out[i] = cb.centroids[indices[i]];
    }
}

}  // namespace turboquant
