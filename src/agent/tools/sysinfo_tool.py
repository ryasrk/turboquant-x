"""System information tool: hardware and process monitoring."""
from __future__ import annotations

import os
import platform
import shutil
from typing import Any

from src.agent.base import Tool


class SystemInfoTool(Tool):
    """Get system information: CPU, RAM, GPU, disk, and OS details."""

    @property
    def name(self) -> str:
        return "system_info"

    @property
    def description(self) -> str:
        return (
            "Get system hardware and OS information including CPU, RAM, "
            "GPU memory, disk space, and Python version."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        lines: list[str] = []

        # OS
        lines.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
        lines.append(f"Python: {platform.python_version()}")
        lines.append(f"Hostname: {platform.node()}")

        # CPU
        try:
            cpu_count = os.cpu_count() or 0
            with open("/proc/cpuinfo") as f:
                cpu_info = f.read()
            model_name = ""
            for line in cpu_info.split("\n"):
                if line.startswith("model name"):
                    model_name = line.split(":")[-1].strip()
                    break
            lines.append(f"CPU: {model_name} ({cpu_count} cores)")
        except Exception:
            lines.append(f"CPU: {os.cpu_count()} cores")

        # RAM
        try:
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            total = avail = 0
            for line in meminfo.split("\n"):
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / 1024 / 1024  # GB
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 1024 / 1024  # GB
            used = total - avail
            lines.append(f"RAM: {used:.1f}GB used / {total:.1f}GB total ({used/total*100:.0f}%)")
        except Exception:
            lines.append("RAM: unavailable")

        # GPU
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                total_gb = mem.total / 1024**3
                used_gb = mem.used / 1024**3
                lines.append(
                    f"GPU {i}: {name} — {used_gb:.1f}GB / {total_gb:.1f}GB "
                    f"({used_gb/total_gb*100:.0f}%)"
                )
            pynvml.nvmlShutdown()
        except Exception:
            lines.append("GPU: not available or no NVIDIA driver")

        # Disk
        try:
            usage = shutil.disk_usage("/")
            total_gb = usage.total / 1024**3
            used_gb = usage.used / 1024**3
            free_gb = usage.free / 1024**3
            lines.append(f"Disk: {used_gb:.1f}GB used / {total_gb:.1f}GB total ({free_gb:.1f}GB free)")
        except Exception:
            lines.append("Disk: unavailable")

        return "\n".join(lines)
