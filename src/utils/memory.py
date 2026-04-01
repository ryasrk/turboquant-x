"""GPU and system memory monitoring for TurboQuant inference.

Provides real-time memory tracking for GPU (NVIDIA) and system RAM.
Uses pynvml when available, falls back to nvidia-smi subprocess.
Gracefully handles environments without GPU.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryInfo:
    """Memory usage snapshot.

    All values in bytes.
    """

    total: int
    used: int
    free: int

    @property
    def used_percent(self) -> float:
        """Percentage of memory used (0-100)."""
        if self.total == 0:
            return 0.0
        return (self.used / self.total) * 100.0

    @property
    def total_gb(self) -> float:
        return self.total / (1024**3)

    @property
    def used_gb(self) -> float:
        return self.used / (1024**3)

    @property
    def free_gb(self) -> float:
        return self.free / (1024**3)


def get_gpu_memory(device_index: int = 0) -> MemoryInfo | None:
    """Get GPU memory info.

    Tries pynvml first, then nvidia-smi subprocess.
    Returns None if no GPU is available.
    """
    # Try pynvml
    info = _get_gpu_memory_pynvml(device_index)
    if info is not None:
        return info

    # Fallback to nvidia-smi
    info = _get_gpu_memory_nvidia_smi(device_index)
    if info is not None:
        return info

    logger.debug("No GPU detected")
    return None


def _get_gpu_memory_pynvml(device_index: int) -> MemoryInfo | None:
    """Get GPU memory via pynvml."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return MemoryInfo(total=info.total, used=info.used, free=info.free)
    except Exception:
        return None


def _get_gpu_memory_nvidia_smi(device_index: int) -> MemoryInfo | None:
    """Get GPU memory via nvidia-smi subprocess."""
    if not shutil.which("nvidia-smi"):
        return None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        parts = result.stdout.strip().split(",")
        if len(parts) != 3:
            return None

        # nvidia-smi reports in MiB
        total_mib, used_mib, free_mib = (int(p.strip()) for p in parts)
        mib_to_bytes = 1024 * 1024

        return MemoryInfo(
            total=total_mib * mib_to_bytes,
            used=used_mib * mib_to_bytes,
            free=free_mib * mib_to_bytes,
        )
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return None


def get_ram_usage() -> MemoryInfo:
    """Get system RAM usage.

    Uses /proc/meminfo on Linux, fallback to psutil if available,
    or os.sysconf as last resort.
    """
    # Try /proc/meminfo (Linux)
    info = _get_ram_from_proc()
    if info is not None:
        return info

    # Fallback to psutil
    try:
        import psutil

        vm = psutil.virtual_memory()
        return MemoryInfo(total=vm.total, used=vm.used, free=vm.available)
    except ImportError:
        pass

    # Last resort: os.sysconf (limited info)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        total = page_size * total_pages
        free = page_size * avail_pages
        return MemoryInfo(total=total, used=total - free, free=free)
    except (ValueError, OSError):
        return MemoryInfo(total=0, used=0, free=0)


def _get_ram_from_proc() -> MemoryInfo | None:
    """Parse /proc/meminfo for RAM stats."""
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()

        info: dict[str, int] = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                # Values in kB
                info[key] = int(parts[1]) * 1024

        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", info.get("MemFree", 0))
        used = total - available

        return MemoryInfo(total=total, used=used, free=available)
    except (FileNotFoundError, ValueError, KeyError):
        return None


def check_model_fits(
    model_size_gb: float,
    kv_cache_gb: float,
    gpu_vram_gb: float | None = None,
    overhead_gb: float = 0.5,
) -> tuple[bool, str]:
    """Check if a model + KV cache fits in available GPU VRAM.

    Args:
        model_size_gb: Model weight size in GB.
        kv_cache_gb: KV cache size in GB.
        gpu_vram_gb: Override GPU VRAM size. If None, auto-detect.
        overhead_gb: CUDA runtime overhead.

    Returns:
        Tuple of (fits: bool, message: str).
    """
    if gpu_vram_gb is None:
        gpu_info = get_gpu_memory()
        if gpu_info is None:
            return False, "No GPU detected"
        gpu_vram_gb = gpu_info.total_gb

    required = model_size_gb + kv_cache_gb + overhead_gb
    fits = required <= gpu_vram_gb

    message = (
        f"Required: {required:.2f} GB "
        f"(model={model_size_gb:.1f} + cache={kv_cache_gb:.2f} + overhead={overhead_gb:.1f}) | "
        f"Available: {gpu_vram_gb:.1f} GB | "
        f"{'FITS' if fits else 'DOES NOT FIT'}"
    )

    return fits, message


class MemoryMonitor:
    """Periodic memory monitoring with peak tracking.

    Usage:
        monitor = MemoryMonitor()
        monitor.start()
        # ... do inference ...
        monitor.stop()
        print(monitor.peak_gpu_used_gb)
    """

    def __init__(self, interval_s: float = 1.0, device_index: int = 0) -> None:
        self._interval = interval_s
        self._device_index = device_index
        self._samples: list[dict[str, MemoryInfo | None]] = []
        self._running = False
        self._thread: Any = None
        self._peak_gpu_bytes: int = 0
        self._peak_ram_bytes: int = 0
        self._start_time: float = 0.0

    @property
    def peak_gpu_used_gb(self) -> float:
        return self._peak_gpu_bytes / (1024**3)

    @property
    def peak_ram_used_gb(self) -> float:
        return self._peak_ram_bytes / (1024**3)

    @property
    def samples(self) -> list[dict[str, MemoryInfo | None]]:
        return list(self._samples)

    def start(self) -> None:
        """Start periodic monitoring in a background thread."""
        import threading

        self._running = True
        self._start_time = time.monotonic()
        self._samples.clear()
        self._peak_gpu_bytes = 0
        self._peak_ram_bytes = 0

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring and wait for thread to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)
            self._thread = None

    def snapshot(self) -> dict[str, MemoryInfo | None]:
        """Take a single memory snapshot."""
        gpu = get_gpu_memory(self._device_index)
        ram = get_ram_usage()
        return {"gpu": gpu, "ram": ram}

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            sample = self.snapshot()
            self._samples.append(sample)

            # Track peaks
            if sample["gpu"] is not None:
                self._peak_gpu_bytes = max(self._peak_gpu_bytes, sample["gpu"].used)
            if sample["ram"] is not None:
                self._peak_ram_bytes = max(self._peak_ram_bytes, sample["ram"].used)

            time.sleep(self._interval)

    def summary(self) -> dict[str, float]:
        """Get monitoring summary."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        return {
            "elapsed_s": elapsed,
            "n_samples": len(self._samples),
            "peak_gpu_used_gb": self.peak_gpu_used_gb,
            "peak_ram_used_gb": self.peak_ram_used_gb,
        }
