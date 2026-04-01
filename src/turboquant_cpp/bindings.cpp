// pybind11 bindings for turboquant C++ backend.
//
// TWO paths:
//   - float64 (legacy): converts double↔float at boundary for Python compat
//   - float32 (fast):   zero-copy from numpy → C++ and back (no conversion)
//
// The compressor calls the float32 path by converting numpy arrays once
// in Python (fast SIMD) instead of per-element in C++.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "rotation.h"
#include "codebook.h"
#include "polar_quant.h"

#include <cstring>

namespace py = pybind11;
using namespace turboquant;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

static std::vector<py::ssize_t> get_shape(const py::buffer_info& buf) {
    std::vector<py::ssize_t> shape(buf.ndim);
    for (py::ssize_t i = 0; i < buf.ndim; ++i) shape[i] = buf.shape[i];
    return shape;
}

// Convert double numpy → float vector (for legacy path)
static std::vector<float> to_float_vec(const py::array_t<double>& arr) {
    auto buf = arr.request();
    const double* src = static_cast<const double*>(buf.ptr);
    std::vector<float> out(buf.size);
    for (py::ssize_t i = 0; i < buf.size; ++i) {
        out[i] = static_cast<float>(src[i]);
    }
    return out;
}

// Convert float vector → double numpy with shape (for legacy path)
static py::array_t<double> to_double_array(const float* data, int size,
                                            const std::vector<py::ssize_t>& shape) {
    auto result = py::array_t<double>(size);
    auto rbuf = result.request();
    double* dst = static_cast<double*>(rbuf.ptr);
    for (int i = 0; i < size; ++i) {
        dst[i] = static_cast<double>(data[i]);
    }
    result.resize(shape);
    return result;
}

// Build blocks list from CompressedTensor (shared by both paths)
static py::list blocks_to_pylist(const CompressedTensor& ct) {
    py::list blocks_list;
    for (const auto& b : ct.blocks) {
        py::dict bd;
        bd["indices"] = py::array_t<uint8_t>(b.indices.size(), b.indices.data());
        bd["norm"] = b.norm;
        bd["n_bits"] = b.n_bits;
        bd["block_size"] = b.block_size;
        bd["seed"] = b.seed;
        blocks_list.append(bd);
    }
    return blocks_list;
}

// Parse CompressedTensor from Python dict (shared by both paths)
static CompressedTensor parse_ct_dict(py::dict ct_dict) {
    CompressedTensor ct;
    auto blocks_list = ct_dict["blocks"].cast<py::list>();
    ct.blocks.reserve(blocks_list.size());
    for (auto& item : blocks_list) {
        auto bd = item.cast<py::dict>();
        CompressedBlock block;
        auto indices_arr = bd["indices"].cast<py::array_t<uint8_t>>();
        auto ibuf = indices_arr.request();
        block.indices.assign(
            static_cast<uint8_t*>(ibuf.ptr),
            static_cast<uint8_t*>(ibuf.ptr) + ibuf.size);
        block.norm = bd["norm"].cast<float>();
        block.n_bits = bd["n_bits"].cast<int>();
        block.block_size = bd["block_size"].cast<int>();
        block.seed = bd["seed"].cast<int>();
        ct.blocks.push_back(std::move(block));
    }
    auto shape_tuple = ct_dict["original_shape"].cast<py::tuple>();
    for (auto& s : shape_tuple) {
        ct.original_shape.push_back(s.cast<int>());
    }
    ct.n_bits = ct_dict["n_bits"].cast<int>();
    ct.block_size = ct_dict["block_size"].cast<int>();
    return ct;
}

// -----------------------------------------------------------------------
// Legacy float64 wrappers (for tests and Python fallback)
// -----------------------------------------------------------------------

static py::array_t<double> py_rotate(py::array_t<double> x, int seed) {
    auto buf = x.request();
    if (buf.ndim < 1) throw std::invalid_argument("x must be at least 1-D");
    int d = static_cast<int>(buf.shape[buf.ndim - 1]);
    auto fdata = to_float_vec(x);
    int n_rows = static_cast<int>(buf.size / d);
    rotate_batch(fdata.data(), n_rows, d, seed);
    return to_double_array(fdata.data(), static_cast<int>(buf.size), get_shape(buf));
}

static py::array_t<double> py_inverse_rotate(py::array_t<double> y, int seed) {
    auto buf = y.request();
    if (buf.ndim < 1) throw std::invalid_argument("y must be at least 1-D");
    int d = static_cast<int>(buf.shape[buf.ndim - 1]);
    auto fdata = to_float_vec(y);
    int n_rows = static_cast<int>(buf.size / d);
    inverse_rotate_batch(fdata.data(), n_rows, d, seed);
    return to_double_array(fdata.data(), static_cast<int>(buf.size), get_shape(buf));
}

static py::array_t<uint8_t> py_quantize_scalar(py::array_t<double> x, int n_bits) {
    auto buf = x.request();
    const Codebook& cb = get_codebook(n_bits);
    auto fdata = to_float_vec(x);
    auto result = py::array_t<uint8_t>(buf.size);
    auto rbuf = result.request();
    quantize_scalar(fdata.data(), static_cast<uint8_t*>(rbuf.ptr),
                    static_cast<int>(buf.size), cb);
    result.resize(get_shape(buf));
    return result;
}

static py::array_t<double> py_dequantize_scalar(py::array_t<uint8_t> indices,
                                                  int n_bits) {
    auto buf = indices.request();
    const Codebook& cb = get_codebook(n_bits);
    std::vector<float> fout(buf.size);
    dequantize_scalar(static_cast<const uint8_t*>(buf.ptr),
                      fout.data(), static_cast<int>(buf.size), cb);
    return to_double_array(fout.data(), static_cast<int>(buf.size), get_shape(buf));
}

static py::dict py_polar_quantize_block(py::array_t<double> x,
                                         int n_bits, int seed) {
    auto buf = x.request();
    if (buf.ndim != 1) throw std::invalid_argument("x must be 1-D");
    auto fdata = to_float_vec(x);
    auto block = polar_quantize_block(fdata.data(),
                                       static_cast<int>(buf.size), n_bits, seed);
    py::dict result;
    result["indices"] = py::array_t<uint8_t>(
        block.indices.size(), block.indices.data());
    result["norm"] = block.norm;
    result["n_bits"] = block.n_bits;
    result["block_size"] = block.block_size;
    result["seed"] = block.seed;
    return result;
}

static py::array_t<double> py_polar_dequantize_block(py::dict block_dict) {
    CompressedBlock block;
    auto indices_arr = block_dict["indices"].cast<py::array_t<uint8_t>>();
    auto ibuf = indices_arr.request();
    block.indices.assign(
        static_cast<uint8_t*>(ibuf.ptr),
        static_cast<uint8_t*>(ibuf.ptr) + ibuf.size);
    block.norm = block_dict["norm"].cast<float>();
    block.n_bits = block_dict["n_bits"].cast<int>();
    block.block_size = block_dict["block_size"].cast<int>();
    block.seed = block_dict["seed"].cast<int>();
    auto result_vec = polar_dequantize_block(block);
    std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(result_vec.size())};
    return to_double_array(result_vec.data(),
                           static_cast<int>(result_vec.size()), shape);
}

static py::dict py_polar_quantize(py::array_t<double> x,
                                   int n_bits, int seed, int block_size) {
    auto buf = x.request();
    auto fdata = to_float_vec(x);
    std::vector<int> shape;
    for (int i = 0; i < buf.ndim; ++i) {
        shape.push_back(static_cast<int>(buf.shape[i]));
    }
    auto ct = polar_quantize(fdata.data(), static_cast<int>(buf.size),
                              shape.data(), static_cast<int>(shape.size()),
                              n_bits, seed, block_size);
    py::dict result;
    result["blocks"] = blocks_to_pylist(ct);
    result["original_shape"] = py::tuple(py::cast(ct.original_shape));
    result["n_bits"] = ct.n_bits;
    result["block_size"] = ct.block_size;
    return result;
}

static py::array_t<double> py_polar_dequantize(py::dict ct_dict) {
    auto ct = parse_ct_dict(ct_dict);
    auto result_vec = polar_dequantize(ct);
    std::vector<py::ssize_t> py_shape;
    for (int s : ct.original_shape) py_shape.push_back(s);
    return to_double_array(result_vec.data(),
                           static_cast<int>(result_vec.size()), py_shape);
}

// -----------------------------------------------------------------------
// FAST float32 wrappers (zero-copy — used by compressor)
// -----------------------------------------------------------------------

static py::dict py_polar_quantize_f32(py::array_t<float> x,
                                       int n_bits, int seed, int block_size) {
    auto buf = x.request();
    // ZERO COPY: x.data() is already float32 — pass directly to C++
    std::vector<int> shape;
    for (int i = 0; i < buf.ndim; ++i) {
        shape.push_back(static_cast<int>(buf.shape[i]));
    }
    auto ct = polar_quantize(
        static_cast<const float*>(buf.ptr),
        static_cast<int>(buf.size),
        shape.data(), static_cast<int>(shape.size()),
        n_bits, seed, block_size);

    py::dict result;
    result["blocks"] = blocks_to_pylist(ct);
    result["original_shape"] = py::tuple(py::cast(ct.original_shape));
    result["n_bits"] = ct.n_bits;
    result["block_size"] = ct.block_size;
    return result;
}

static py::array_t<float> py_polar_dequantize_f32(py::dict ct_dict) {
    auto ct = parse_ct_dict(ct_dict);
    auto result_vec = polar_dequantize(ct);

    // ZERO COPY: return float32 directly to Python
    std::vector<py::ssize_t> py_shape;
    for (int s : ct.original_shape) py_shape.push_back(s);

    auto result = py::array_t<float>(result_vec.size(), result_vec.data());
    result.resize(py_shape);
    return result;
}

// -----------------------------------------------------------------------
// Module definition
// -----------------------------------------------------------------------

PYBIND11_MODULE(_turboquant_cpp, m) {
    m.doc() = "TurboQuant C++ backend — float32 + AVX2 + thread-local pools";

    // Rotation (legacy float64)
    m.def("rotate", &py_rotate, py::arg("x"), py::arg("seed"));
    m.def("inverse_rotate", &py_inverse_rotate, py::arg("y"), py::arg("seed"));

    // Codebook (legacy float64)
    m.def("quantize_scalar", &py_quantize_scalar, py::arg("x"), py::arg("n_bits"));
    m.def("dequantize_scalar", &py_dequantize_scalar,
          py::arg("indices"), py::arg("n_bits"));

    // Block-level (legacy float64)
    m.def("polar_quantize_block", &py_polar_quantize_block,
          py::arg("x"), py::arg("n_bits"), py::arg("seed"));
    m.def("polar_dequantize_block", &py_polar_dequantize_block,
          py::arg("block"));

    // Tensor-level (legacy float64)
    m.def("polar_quantize", &py_polar_quantize,
          py::arg("x"), py::arg("n_bits"), py::arg("seed"), py::arg("block_size"));
    m.def("polar_dequantize", &py_polar_dequantize,
          py::arg("compressed_tensor"));

    // FAST float32 tensor-level (zero-copy)
    m.def("polar_quantize_f32", &py_polar_quantize_f32,
          "Compress tensor (float32 zero-copy path)",
          py::arg("x"), py::arg("n_bits"), py::arg("seed"), py::arg("block_size"));
    m.def("polar_dequantize_f32", &py_polar_dequantize_f32,
          "Decompress tensor (returns float32 directly)",
          py::arg("compressed_tensor"));

    // Utility
    m.def("compression_ratio", &compression_ratio,
          py::arg("n_bits"), py::arg("block_size"));
}
