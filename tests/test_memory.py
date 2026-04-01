"""Tests for memory monitoring (GPU tests mocked, RAM tests can use real system)."""

import platform
import sys
import time
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from src.utils.memory import (
    MemoryInfo,
    MemoryMonitor,
    _get_gpu_memory_nvidia_smi,
    _get_gpu_memory_pynvml,
    _get_ram_from_proc,
    check_model_fits,
    get_gpu_memory,
    get_ram_usage,
)


class TestMemoryInfo:
    def test_basic_construction(self):
        info = MemoryInfo(total=8_000_000_000, used=3_000_000_000, free=5_000_000_000)
        assert info.total == 8_000_000_000
        assert info.used == 3_000_000_000
        assert info.free == 5_000_000_000

    def test_used_percent(self):
        info = MemoryInfo(total=10_000, used=2_500, free=7_500)
        assert info.used_percent == pytest.approx(25.0)

    def test_gb_properties(self):
        gib = 1024**3
        info = MemoryInfo(total=16 * gib, used=4 * gib, free=12 * gib)
        assert info.total_gb == pytest.approx(16.0)
        assert info.used_gb == pytest.approx(4.0)
        assert info.free_gb == pytest.approx(12.0)

    def test_zero_total_returns_zero_percent(self):
        info = MemoryInfo(total=0, used=0, free=0)
        assert info.used_percent == 0.0

    def test_frozen_immutable(self):
        info = MemoryInfo(total=100, used=50, free=50)
        with pytest.raises(FrozenInstanceError):
            info.total = 200  # type: ignore[misc]


class TestGetGpuMemory:
    def test_pynvml_returns_memory_info(self):
        mock_info = MagicMock()
        mock_info.total = 24_000_000_000
        mock_info.used = 6_000_000_000
        mock_info.free = 18_000_000_000

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_info

        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            result = _get_gpu_memory_pynvml(0)

        assert result is not None
        assert result.total == 24_000_000_000
        assert result.used == 6_000_000_000
        assert result.free == 18_000_000_000

    def test_pynvml_failure_falls_back_to_nvidia_smi(self):
        nvidia_smi_output = "24564, 6120, 18444\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = nvidia_smi_output

        with (
            patch("src.utils.memory._get_gpu_memory_pynvml", return_value=None),
            patch("src.utils.memory._get_gpu_memory_nvidia_smi") as mock_smi,
        ):
            mib = 1024 * 1024
            mock_smi.return_value = MemoryInfo(
                total=24564 * mib, used=6120 * mib, free=18444 * mib
            )
            result = get_gpu_memory(0)

        assert result is not None
        assert result.total == 24564 * mib

    def test_no_gpu_returns_none(self):
        with (
            patch("src.utils.memory._get_gpu_memory_pynvml", return_value=None),
            patch("src.utils.memory._get_gpu_memory_nvidia_smi", return_value=None),
        ):
            result = get_gpu_memory(0)

        assert result is None

    def test_nvidia_smi_parse_valid_output(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "24564, 6120, 18444\n"

        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = _get_gpu_memory_nvidia_smi(0)

        assert result is not None
        mib = 1024 * 1024
        assert result.total == 24564 * mib
        assert result.used == 6120 * mib
        assert result.free == 18444 * mib

    def test_nvidia_smi_bad_output_returns_none(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "garbage data\n"

        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = _get_gpu_memory_nvidia_smi(0)

        assert result is None

    def test_nvidia_smi_not_found_returns_none(self):
        with patch("shutil.which", return_value=None):
            result = _get_gpu_memory_nvidia_smi(0)
        assert result is None

    def test_nvidia_smi_nonzero_return_code(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = _get_gpu_memory_nvidia_smi(0)

        assert result is None


class TestGetRamUsage:
    def test_returns_memory_info_with_positive_values(self):
        info = get_ram_usage()
        assert isinstance(info, MemoryInfo)
        assert info.total >= 0
        assert info.used >= 0
        assert info.free >= 0

    def test_total_greater_than_zero(self):
        info = get_ram_usage()
        assert info.total > 0

    def test_used_greater_than_zero(self):
        info = get_ram_usage()
        assert info.used > 0


class TestCheckModelFits:
    def test_fits_with_explicit_vram(self):
        fits, msg = check_model_fits(
            model_size_gb=7.0, kv_cache_gb=1.0, gpu_vram_gb=24.0
        )
        assert fits is True
        assert "FITS" in msg
        assert "DOES NOT FIT" not in msg

    def test_does_not_fit_with_small_vram(self):
        fits, msg = check_model_fits(
            model_size_gb=14.0, kv_cache_gb=2.0, gpu_vram_gb=8.0
        )
        assert fits is False
        assert "DOES NOT FIT" in msg

    def test_returns_descriptive_message(self):
        fits, msg = check_model_fits(
            model_size_gb=7.0, kv_cache_gb=1.5, gpu_vram_gb=24.0, overhead_gb=0.5
        )
        assert "Required:" in msg
        assert "Available:" in msg
        assert "model=7.0" in msg
        assert "cache=1.50" in msg
        assert "overhead=0.5" in msg

    def test_no_gpu_auto_detect_returns_false(self):
        with (
            patch("src.utils.memory._get_gpu_memory_pynvml", return_value=None),
            patch("src.utils.memory._get_gpu_memory_nvidia_smi", return_value=None),
        ):
            fits, msg = check_model_fits(model_size_gb=7.0, kv_cache_gb=1.0)

        assert fits is False
        assert "No GPU detected" in msg

    def test_exact_boundary_fits(self):
        fits, _ = check_model_fits(
            model_size_gb=7.0, kv_cache_gb=1.0, gpu_vram_gb=8.5, overhead_gb=0.5
        )
        assert fits is True

    def test_just_over_boundary_does_not_fit(self):
        fits, _ = check_model_fits(
            model_size_gb=7.0, kv_cache_gb=1.0, gpu_vram_gb=8.49, overhead_gb=0.5
        )
        assert fits is False


class TestMemoryMonitor:
    def test_start_stop_lifecycle(self):
        monitor = MemoryMonitor(interval_s=0.05)
        monitor.start()
        time.sleep(0.15)
        monitor.stop()
        assert len(monitor.samples) >= 1

    def test_snapshot_returns_dict_with_expected_keys(self):
        monitor = MemoryMonitor()
        snap = monitor.snapshot()
        assert "gpu" in snap
        assert "ram" in snap
        assert isinstance(snap["ram"], MemoryInfo)

    def test_summary_returns_expected_keys(self):
        monitor = MemoryMonitor(interval_s=0.05)
        monitor.start()
        time.sleep(0.1)
        monitor.stop()

        summary = monitor.summary()
        assert "elapsed_s" in summary
        assert "n_samples" in summary
        assert "peak_gpu_used_gb" in summary
        assert "peak_ram_used_gb" in summary
        assert summary["elapsed_s"] > 0
        assert summary["n_samples"] >= 1

    def test_peak_tracking_with_mocked_gpu(self):
        increasing_values = iter(
            [
                MemoryInfo(total=24_000_000_000, used=2_000_000_000, free=22_000_000_000),
                MemoryInfo(total=24_000_000_000, used=8_000_000_000, free=16_000_000_000),
                MemoryInfo(total=24_000_000_000, used=5_000_000_000, free=19_000_000_000),
            ]
        )

        def mock_get_gpu_memory(device_index=0):
            try:
                return next(increasing_values)
            except StopIteration:
                return MemoryInfo(
                    total=24_000_000_000, used=5_000_000_000, free=19_000_000_000
                )

        with patch("src.utils.memory.get_gpu_memory", side_effect=mock_get_gpu_memory):
            monitor = MemoryMonitor(interval_s=0.05)
            monitor.start()
            time.sleep(0.2)
            monitor.stop()

        # Peak should be 8 GB
        assert monitor.peak_gpu_used_gb == pytest.approx(
            8_000_000_000 / (1024**3), abs=0.01
        )

    def test_peak_ram_tracked(self):
        monitor = MemoryMonitor(interval_s=0.05)
        monitor.start()
        time.sleep(0.1)
        monitor.stop()
        assert monitor.peak_ram_used_gb > 0

    def test_samples_returns_copy(self):
        monitor = MemoryMonitor(interval_s=0.05)
        monitor.start()
        time.sleep(0.1)
        monitor.stop()
        s1 = monitor.samples
        s2 = monitor.samples
        assert s1 is not s2


class TestRamFromProc:
    @pytest.mark.skipif(
        platform.system() != "Linux", reason="Only runs on Linux with /proc/meminfo"
    )
    def test_parses_real_proc_meminfo(self):
        result = _get_ram_from_proc()
        assert result is not None
        assert result.total > 0
        assert result.used > 0
        assert result.free > 0

    def test_returns_none_when_file_missing(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = _get_ram_from_proc()
        assert result is None

    def test_parses_synthetic_meminfo(self):
        synthetic = (
            "MemTotal:       16384000 kB\n"
            "MemFree:         2048000 kB\n"
            "MemAvailable:    8192000 kB\n"
            "Buffers:          512000 kB\n"
        )
        from unittest.mock import mock_open

        with patch("builtins.open", mock_open(read_data=synthetic)):
            result = _get_ram_from_proc()

        assert result is not None
        assert result.total == 16384000 * 1024
        assert result.free == 8192000 * 1024
        assert result.used == (16384000 - 8192000) * 1024


# ---------------------------------------------------------------------------
# Missing coverage tests
# ---------------------------------------------------------------------------


class TestPynvmlException:
    """Test _get_gpu_memory_pynvml exception handling (lines 82-83)."""

    def test_returns_none_on_pynvml_error(self):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = RuntimeError("CUDA error")
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _get_gpu_memory_pynvml(0)
        assert result is None

    def test_returns_info_on_pynvml_success(self):
        mock_info = MagicMock()
        mock_info.total = 8_000_000_000
        mock_info.used = 2_000_000_000
        mock_info.free = 6_000_000_000

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_info
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _get_gpu_memory_pynvml(0)
        assert result is not None
        assert result.total == 8_000_000_000


class TestNvidiaSmiParse:
    """Test nvidia-smi CSV parsing (lines 119-120)."""

    def test_parses_valid_csv(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "8192, 2048, 6144\n"

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", return_value=mock_result):
                result = _get_gpu_memory_nvidia_smi(0)
        assert result is not None
        assert result.total == 8192 * 1024 * 1024

    def test_returns_none_on_bad_csv(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "bad,output\n"

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", return_value=mock_result):
                result = _get_gpu_memory_nvidia_smi(0)
        assert result is None

    def test_returns_none_on_timeout(self):
        import subprocess

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
                result = _get_gpu_memory_nvidia_smi(0)
        assert result is None

    def test_returns_none_on_value_error(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "not_a_number, 2048, 6144\n"

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", return_value=mock_result):
                result = _get_gpu_memory_nvidia_smi(0)
        assert result is None


class TestGetRamUsageFallbacks:
    """Test get_ram_usage fallback chain (lines 135-152)."""

    def test_fallback_to_psutil(self):
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = MagicMock(
            total=16_000_000_000, used=4_000_000_000, available=12_000_000_000,
        )
        with patch("src.utils.memory._get_ram_from_proc", return_value=None):
            with patch.dict("sys.modules", {"psutil": mock_psutil}):
                result = get_ram_usage()
        assert result.total == 16_000_000_000

    def test_fallback_to_sysconf(self):
        with patch("src.utils.memory._get_ram_from_proc", return_value=None):
            with patch.dict("sys.modules", {"psutil": None}):
                with patch("os.sysconf", side_effect=lambda x: {
                    "SC_PAGE_SIZE": 4096,
                    "SC_PHYS_PAGES": 1000,
                    "SC_AVPHYS_PAGES": 500,
                }[x]):
                    result = get_ram_usage()
        assert result.total == 4096 * 1000
        assert result.free == 4096 * 500

    def test_fallback_to_zero(self):
        with patch("src.utils.memory._get_ram_from_proc", return_value=None):
            with patch.dict("sys.modules", {"psutil": None}):
                with patch("os.sysconf", side_effect=OSError):
                    result = get_ram_usage()
        assert result.total == 0


class TestCheckModelFitsAutoDetect:
    """Test check_model_fits with gpu_vram_gb=None (line 199)."""

    def test_auto_detect_no_gpu(self):
        with patch("src.utils.memory.get_gpu_memory", return_value=None):
            fits, msg = check_model_fits(4.0, 1.0)
        assert fits is False
        assert "No GPU" in msg

    def test_auto_detect_with_gpu(self):
        mock_gpu = MemoryInfo(total=24 * 1024**3, used=1 * 1024**3, free=23 * 1024**3)
        with patch("src.utils.memory.get_gpu_memory", return_value=mock_gpu):
            fits, msg = check_model_fits(4.0, 1.0)
        assert fits is True
