// pybind11 bindings for turboquant C++ backend.
//
// Exposes: rotate, inverse_rotate, quantize_scalar, dequantize_scalar,
//          polar_quantize_block, polar_dequantize_block,
//          polar_quantize, polar_dequantize, compression_ratio

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "rotation.h"
#include "codebook.h"
#include "polar_quant.h"

namespace py = pybind11;
using namespace turboquant;

// Helper: extract shape from buffer_info
static std::vector<py::ssize_t> get_shape(const py::buffer_info& buf) {
    std::vector<py::ssize_t> shape(buf.ndim);
    for (py::ssize_t i = 0; i < buf.ndim; ++i) shape[i] = buf.shape[i];
    return shape;
}

// Wrap rotate to accept/return numpy arrays
static py::array_t<double> py_rotate(py::array_t<double> x, int seed) {
    auto buf = x.request();
    if (buf.ndim < 1) throw std::invalid_argument("x must be at least 1-D");

    int d = static_cast<int>(buf.shape[buf.ndim - 1]);

    // Copy data
    auto result = py::array_t<double>(buf.size);
    auto rbuf = result.request();
    std::memcpy(rbuf.ptr, buf.ptr, buf.size * sizeof(double));

    int n_rows = static_cast<int>(buf.size / d);
    rotate_batch(static_cast<double*>(rbuf.ptr), n_rows, d, seed);

    // Reshape to match input
    result.resize(get_shape(buf));
    return result;
}

static py::array_t<double> py_inverse_rotate(py::array_t<double> y, int seed) {
    auto buf = y.request();
    if (buf.ndim < 1) throw std::invalid_argument("y must be at least 1-D");

    int d = static_cast<int>(buf.shape[buf.ndim - 1]);

    auto result = py::array_t<double>(buf.size);
    auto rbuf = result.request();
    std::memcpy(rbuf.ptr, buf.ptr, buf.size * sizeof(double));

    int n_rows = static_cast<int>(buf.size / d);
    inverse_rotate_batch(static_cast<double*>(rbuf.ptr), n_rows, d, seed);

    result.resize(get_shape(buf));
    return result;
}

static py::array_t<uint8_t> py_quantize_scalar(py::array_t<double> x, int n_bits) {
    auto buf = x.request();
    const Codebook& cb = get_codebook(n_bits);

    auto result = py::array_t<uint8_t>(buf.size);
    auto rbuf = result.request();

    quantize_scalar(static_cast<const double*>(buf.ptr),
                    static_cast<uint8_t*>(rbuf.ptr),
                    static_cast<int>(buf.size), cb);

    result.resize(get_shape(buf));
    return result;
}

static py::array_t<double> py_dequantize_scalar(py::array_t<uint8_t> indices,
                                                  int n_bits) {
    auto buf = indices.request();
    const Codebook& cb = get_codebook(n_bits);

    auto result = py::array_t<double>(buf.size);
    auto rbuf = result.request();

    dequantize_scalar(static_cast<const uint8_t*>(buf.ptr),
                      static_cast<double*>(rbuf.ptr),
                      static_cast<int>(buf.size), cb);

    result.resize(get_shape(buf));
    return result;
}

// Wrap block-level compress/decompress
static py::dict py_polar_quantize_block(py::array_t<double> x,
                                         int n_bits, int seed) {
    auto buf = x.request();
    if (buf.ndim != 1) throw std::invalid_argument("x must be 1-D");

    auto block = polar_quantize_block(
        static_cast<const double*>(buf.ptr),
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
    return py::array_t<double>(result_vec.size(), result_vec.data());
}

// Wrap tensor-level compress/decompress
static py::dict py_polar_quantize(py::array_t<double> x,
                                   int n_bits, int seed, int block_size) {
    auto buf = x.request();
    std::vector<int> shape;
    for (int i = 0; i < buf.ndim; ++i) {
        shape.push_back(static_cast<int>(buf.shape[i]));
    }

    auto ct = polar_quantize(
        static_cast<const double*>(buf.ptr),
        static_cast<int>(buf.size),
        shape.data(), static_cast<int>(shape.size()),
        n_bits, seed, block_size);

    // Convert blocks to list of dicts
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

    py::dict result;
    result["blocks"] = blocks_list;
    result["original_shape"] = py::tuple(py::cast(ct.original_shape));
    result["n_bits"] = ct.n_bits;
    result["block_size"] = ct.block_size;
    return result;
}

static py::array_t<double> py_polar_dequantize(py::dict ct_dict) {
    CompressedTensor ct;

    auto blocks_list = ct_dict["blocks"].cast<py::list>();
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

    auto result_vec = polar_dequantize(ct);

    // Reshape to original shape
    std::vector<py::ssize_t> py_shape;
    for (int s : ct.original_shape) py_shape.push_back(s);

    auto result = py::array_t<double>(result_vec.size(), result_vec.data());
    result.resize(py_shape);
    return result;
}

PYBIND11_MODULE(_turboquant_cpp, m) {
    m.doc() = "TurboQuant C++ backend — PolarQuant compression pipeline";

    // Rotation
    m.def("rotate", &py_rotate,
          "Apply WHT rotation: y = H @ diag(signs) @ x",
          py::arg("x"), py::arg("seed"));
    m.def("inverse_rotate", &py_inverse_rotate,
          "Inverse WHT rotation: x = diag(signs) @ H^T @ y",
          py::arg("y"), py::arg("seed"));

    // Codebook
    m.def("quantize_scalar", &py_quantize_scalar,
          "Quantize values to codebook indices",
          py::arg("x"), py::arg("n_bits"));
    m.def("dequantize_scalar", &py_dequantize_scalar,
          "Dequantize indices back to values",
          py::arg("indices"), py::arg("n_bits"));

    // Block-level PolarQuant
    m.def("polar_quantize_block", &py_polar_quantize_block,
          "Compress a single block using PolarQuant",
          py::arg("x"), py::arg("n_bits"), py::arg("seed"));
    m.def("polar_dequantize_block", &py_polar_dequantize_block,
          "Decompress a single PolarQuant block",
          py::arg("block"));

    // Tensor-level PolarQuant
    m.def("polar_quantize", &py_polar_quantize,
          "Compress a full tensor with blocking",
          py::arg("x"), py::arg("n_bits"),
          py::arg("seed") = 42, py::arg("block_size") = 128);
    m.def("polar_dequantize", &py_polar_dequantize,
          "Decompress a CompressedTensor",
          py::arg("compressed"));

    // Utility
    m.def("compression_ratio", &compression_ratio,
          "Compression ratio vs float16",
          py::arg("n_bits"), py::arg("block_size"));
}
