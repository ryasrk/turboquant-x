"""TurboQuant C++ backend — PolarQuant compression pipeline.

Auto-imports the compiled C++ extension if available, otherwise falls back
to the pure-Python implementation.

Usage:
    from src.turboquant_cpp import (
        rotate, inverse_rotate,
        quantize_scalar, dequantize_scalar,
        polar_quantize_block, polar_dequantize_block,
        polar_quantize, polar_dequantize,
        polar_quantize_f32, polar_dequantize_f32,
        compression_ratio,
    )
"""

try:
    from src.turboquant_cpp._turboquant_cpp import (  # type: ignore[import]
        rotate,
        inverse_rotate,
        quantize_scalar,
        dequantize_scalar,
        polar_quantize_block,
        polar_dequantize_block,
        polar_quantize,
        polar_dequantize,
        polar_quantize_f32,
        polar_dequantize_f32,
        compression_ratio,
    )
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
