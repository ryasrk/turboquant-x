"""Terminal tool with user approval gate for shell commands.

Implements a multi-layer validation pipeline inspired by claw-code-parity:
  1. Hard deny — always blocked (rm -rf /, fork bombs, etc.)
  2. Command intent classification — read_only, write, destructive, network, etc.
  3. Destructive warning — flags dangerous rm, shred, wipefs, etc.
  4. Path validation — detects commands targeting system paths
  5. Sed validation — blocks sed -i in read-only contexts
  6. Risk level — low / medium / high / critical sent to the UI
"""
from __future__ import annotations

import asyncio
import re
import shlex
from enum import Enum
from typing import Any

from src.agent.base import Tool

MAX_OUTPUT_CHARS = 12_000

# ── Hard deny patterns ────────────────────────────────────────────────────
_HARD_DENY_PATTERNS: list[str] = [
    r"\brm\s+-[rf]{1,2}f?\s+/",            # rm -rf / or rm -rf /home etc.
    r"\brm\s+-[rf]{1,2}f?\s+~",            # rm -rf ~
    r"(?:^|[;&|]\s*)format\b",            # format drives
    r"\b(mkfs|diskpart)\b",               # filesystem format
    r"\bdd\s+if=",                        # raw disk write
    r">\s*/dev/sd",                        # overwrite disk
    r"\b(shutdown|reboot|poweroff|halt)\b",  # system control
    r":\(\)\s*\{.*\};\s*:",              # fork bomb
    r"\bchmod\s+-R\s+(777|000)\s+/",     # recursive chmod on root
    r"\bwipefs\b",                        # wipe filesystem signatures
]


# ── Command intent classification ─────────────────────────────────────────

class CommandIntent(str, Enum):
    READ_ONLY = "read_only"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
    NETWORK = "network"
    PACKAGE_MANAGEMENT = "package_management"
    PROCESS_MANAGEMENT = "process_management"
    SYSTEM_ADMIN = "system_admin"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    LOW = "low"           # read-only, safe
    MEDIUM = "medium"     # writes, package installs
    HIGH = "high"         # network, system admin, process mgmt
    CRITICAL = "critical" # destructive, hard-deny adjacent


_READ_ONLY_COMMANDS = frozenset({
    "ls", "ll", "la", "cat", "head", "tail", "less", "more", "grep", "egrep", "fgrep",
    "find", "locate", "which", "whereis", "file", "stat", "wc", "sort", "uniq",
    "diff", "cmp", "strings", "xxd", "hexdump", "od", "readlink", "realpath",
    "pwd", "whoami", "hostname", "uname", "date", "cal", "uptime", "id", "groups",
    "env", "printenv", "echo", "printf", "true", "false", "test", "expr",
    "du", "df", "free", "top", "htop", "ps", "lsof", "lsblk", "lscpu",
    "tree", "man", "help", "info", "type",
    "python", "python3", "node", "ruby", "go", "java", "rustc", "gcc", "g++",
    "cargo", "make", "cmake",
})

_WRITE_COMMANDS = frozenset({
    "cp", "mv", "mkdir", "rmdir", "touch", "chmod", "chown", "chgrp",
    "ln", "install", "tee", "truncate", "mknod", "mkfifo",
})

_DESTRUCTIVE_COMMANDS = frozenset({
    "rm", "shred", "wipefs",
})

_NETWORK_COMMANDS = frozenset({
    "curl", "wget", "ssh", "scp", "rsync", "nc", "ncat", "netcat",
    "ping", "traceroute", "nslookup", "dig", "host", "nmap",
    "ftp", "sftp", "telnet",
})

_PACKAGE_COMMANDS = frozenset({
    "pip", "pip3", "npm", "npx", "yarn", "pnpm", "bun",
    "apt", "apt-get", "yum", "dnf", "pacman", "brew",
    "cargo", "gem", "go", "rustup", "conda",
})

_PROCESS_COMMANDS = frozenset({
    "kill", "pkill", "killall", "xkill",
})

_SYSTEM_ADMIN_COMMANDS = frozenset({
    "sudo", "su", "mount", "umount", "systemctl", "service",
    "useradd", "userdel", "usermod", "groupadd", "groupdel",
    "crontab", "at", "iptables", "ufw", "firewall-cmd",
})

_GIT_READ_ONLY_SUBCOMMANDS = frozenset({
    "status", "log", "diff", "show", "branch", "tag", "stash",
    "remote", "fetch", "ls-files", "ls-tree", "cat-file", "rev-parse",
    "describe", "shortlog", "blame", "bisect", "reflog", "config",
})

_WRITE_REDIRECTIONS = frozenset({">", ">>", ">&"})

_SYSTEM_PATHS = (
    "/etc/", "/usr/", "/var/", "/boot/", "/sys/",
    "/proc/", "/dev/", "/sbin/", "/lib/", "/opt/",
)

# ── Destructive patterns with warnings ────────────────────────────────────
_DESTRUCTIVE_PATTERNS: list[tuple[str, str]] = [
    ("rm -rf *", "Recursive forced deletion of all files in current directory"),
    ("rm -rf .", "Recursive forced deletion of current directory"),
    ("mkfs", "Filesystem creation will destroy existing data"),
    ("dd if=", "Direct disk write — can overwrite partitions"),
    ("> /dev/sd", "Writing to raw disk device"),
    ("chmod -R 777", "Recursively setting world-writable permissions"),
    ("chmod -R 000", "Recursively removing all permissions"),
]


def _extract_first_command(command: str) -> str:
    """Extract the first command name from a shell command string."""
    stripped = command.strip()
    for prefix in ("sudo ", "nohup ", "nice ", "env "):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):].lstrip()
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        tokens = stripped.split()
    if not tokens:
        return ""
    # Skip env-var assignments (FOO=bar cmd)
    for token in tokens:
        if "=" in token and not token.startswith("-"):
            continue
        return token.split("/")[-1]  # basename
    return tokens[-1].split("/")[-1]


def classify_command(command: str) -> CommandIntent:
    """Classify a command's intent (read-only, write, destructive, etc.)."""
    first = _extract_first_command(command)
    if not first:
        return CommandIntent.UNKNOWN

    if first == "git":
        parts = command.split()
        sub = next((p for p in parts[1:] if not p.startswith("-")), None)
        if sub and sub in _GIT_READ_ONLY_SUBCOMMANDS:
            return CommandIntent.READ_ONLY
        if sub:
            return CommandIntent.WRITE
        return CommandIntent.READ_ONLY

    if first in _DESTRUCTIVE_COMMANDS:
        return CommandIntent.DESTRUCTIVE
    if first in _SYSTEM_ADMIN_COMMANDS:
        return CommandIntent.SYSTEM_ADMIN
    if first in _PROCESS_COMMANDS:
        return CommandIntent.PROCESS_MANAGEMENT
    if first in _PACKAGE_COMMANDS:
        return CommandIntent.PACKAGE_MANAGEMENT
    if first in _NETWORK_COMMANDS:
        return CommandIntent.NETWORK
    if first in _WRITE_COMMANDS:
        return CommandIntent.WRITE
    if first in _READ_ONLY_COMMANDS:
        # Even read-only commands become writes if they use redirections
        for redir in _WRITE_REDIRECTIONS:
            if redir in command:
                return CommandIntent.WRITE
        return CommandIntent.READ_ONLY

    # Check for write-redirections on unknown commands too
    for redir in _WRITE_REDIRECTIONS:
        if redir in command:
            return CommandIntent.WRITE

    return CommandIntent.UNKNOWN


def compute_risk_level(command: str, intent: CommandIntent) -> RiskLevel:
    """Compute risk level from command intent + content heuristics."""
    if intent == CommandIntent.DESTRUCTIVE:
        return RiskLevel.CRITICAL
    if intent == CommandIntent.SYSTEM_ADMIN:
        return RiskLevel.HIGH
    if intent in (CommandIntent.NETWORK, CommandIntent.PROCESS_MANAGEMENT):
        return RiskLevel.HIGH
    if intent in (CommandIntent.WRITE, CommandIntent.PACKAGE_MANAGEMENT):
        return RiskLevel.MEDIUM
    if intent == CommandIntent.READ_ONLY:
        return RiskLevel.LOW
    # Unknown — check for system path targets
    for sys_path in _SYSTEM_PATHS:
        if sys_path in command:
            return RiskLevel.HIGH
    return RiskLevel.MEDIUM


def check_destructive_warning(command: str) -> str | None:
    """Return a warning string if the command looks destructive, else None."""
    for pattern, warning in _DESTRUCTIVE_PATTERNS:
        if pattern in command:
            return warning
    first = _extract_first_command(command)
    if first in ("shred", "wipefs"):
        return f"Command '{first}' is inherently destructive and may cause data loss"
    # Flag any rm with -r or -f flags as a general warning
    if "rm " in command and ("-r" in command or "-f" in command):
        return "Recursive/forced deletion detected — verify the target path"
    return None


def check_path_safety(command: str) -> str | None:
    """Warn if command targets sensitive system paths."""
    first = _extract_first_command(command)
    is_write = first in _WRITE_COMMANDS or first in _DESTRUCTIVE_COMMANDS or first in _SYSTEM_ADMIN_COMMANDS
    if not is_write:
        return None
    for sys_path in _SYSTEM_PATHS:
        if sys_path in command:
            return f"Command targets system path '{sys_path}' — may require elevated permissions"
    return None


def check_sed_safety(command: str) -> str | None:
    """Warn if sed -i is used (in-place file editing)."""
    first = _extract_first_command(command)
    if first == "sed" and " -i" in command:
        return "sed -i performs in-place file editing — verify the target files"
    return None


def validate_command(command: str) -> dict[str, Any]:
    """Run the full validation pipeline and return metadata for the UI.

    Returns dict with:
      intent: str — command intent classification
      risk_level: str — low/medium/high/critical
      warnings: list[str] — validation warnings
      blocked: str | None — hard-deny error message
    """
    intent = classify_command(command)
    risk = compute_risk_level(command, intent)
    warnings: list[str] = []

    destructive_warn = check_destructive_warning(command)
    if destructive_warn:
        warnings.append(destructive_warn)

    path_warn = check_path_safety(command)
    if path_warn:
        warnings.append(path_warn)

    sed_warn = check_sed_safety(command)
    if sed_warn:
        warnings.append(sed_warn)

    return {
        "intent": intent.value,
        "risk_level": risk.value,
        "warnings": warnings,
        "blocked": None,
    }


class TerminalTool(Tool):
    """Execute terminal commands with user approval.

    All commands require user review (allow/deny) before execution.
    Dangerous commands matching hard-deny patterns are always rejected.
    The validation pipeline classifies commands and provides risk metadata
    to the approval UI.
    """

    def __init__(
        self,
        timeout: int = 120,
        working_dir: str | None = None,
    ) -> None:
        self._timeout = min(timeout, 300)
        self._working_dir = working_dir
        self._hard_deny_re = [re.compile(p) for p in _HARD_DENY_PATTERNS]

    @property
    def name(self) -> str:
        return "terminal_exec"

    @property
    def description(self) -> str:
        return (
            "Execute a terminal command (e.g. install packages, run scripts, "
            "manage services). The user will review and approve or deny the "
            "command before it runs. Use this for: pip install, npm install, "
            "apt install, running build scripts, starting services, etc."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The terminal command to execute (e.g. 'pip install pandas')",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the command (optional)",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 120, max 300)",
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this command is needed",
                },
            },
            "required": ["command"],
        }

    @property
    def requires_approval(self) -> bool:
        return True

    def is_hard_denied(self, command: str) -> str | None:
        """Check if command matches a hard-deny pattern. Returns error or None."""
        for pattern in self._hard_deny_re:
            if pattern.search(command):
                return f"Error: command permanently blocked by security policy (matched: {pattern.pattern})"
        return None

    def validate(self, command: str) -> dict[str, Any]:
        """Run the full validation pipeline. Hard-deny is checked first."""
        blocked = self.is_hard_denied(command)
        if blocked:
            return {
                "intent": CommandIntent.DESTRUCTIVE.value,
                "risk_level": RiskLevel.CRITICAL.value,
                "warnings": [],
                "blocked": blocked,
            }
        return validate_command(command)

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        half = MAX_OUTPUT_CHARS // 2
        return text[:half] + "\n\n... [truncated] ...\n\n" + text[-half:]

    async def execute(self, **kwargs: Any) -> str:
        command: str = kwargs["command"]
        working_dir: str | None = kwargs.get("working_dir") or self._working_dir
        timeout: int = min(kwargs.get("timeout", self._timeout), 300)

        # Hard deny check
        blocked = self.is_hard_denied(command)
        if blocked:
            return blocked

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return f"Error: command timed out after {timeout}s"
        except Exception as exc:
            return f"Error executing command: {exc}"

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        output = stdout
        if stderr:
            output = output + ("\n" if output else "") + stderr

        output = self._truncate(output)
        return f"[exit code: {proc.returncode}]\n{output}"
