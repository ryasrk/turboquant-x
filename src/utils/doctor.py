"""Config doctor — validates, diagnoses, and fixes TurboQuant-X configuration.

Usage:
    python -m src.main --doctor          # Check config health
    python -m src.main --doctor fix      # Auto-fix issues, preserve values
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path("config")
DEFAULT_YAML = CONFIG_DIR / "default.yaml"
CLOUD_YAML = CONFIG_DIR / "cloud.yaml"
ENV_FILE = CONFIG_DIR / ".env"
ENV_EXAMPLE = CONFIG_DIR / ".env.example"
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")

# ANSI colors
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_DIM = "\033[2m"

_OK = f"{_GREEN}✓{_RESET}"
_FAIL = f"{_RED}✗{_RESET}"
_WARN = f"{_YELLOW}!{_RESET}"
_INFO = f"{_CYAN}ℹ{_RESET}"


# ── Cloud YAML template ─────────────────────────────────────────────

_CLOUD_YAML_TEMPLATE = """\
# Cloud LLM Provider Configuration
# API keys should go in .env (gitignored), not here.
# This file defines provider settings (models, base URLs, parameters).

default_provider: "openai"

providers:
  openai:
    # api_key loaded from .env: TURBOQUANT_CLOUD_OPENAI_API_KEY
    model: "gpt-4o"
    max_tokens: 4096
    temperature: 0.7
    top_p: 0.95
    timeout: 120

  anthropic:
    # api_key loaded from .env: TURBOQUANT_CLOUD_ANTHROPIC_API_KEY
    model: "claude-sonnet-4-20250514"
    max_tokens: 4096
    temperature: 0.7
    top_p: 0.95
    timeout: 120

  moonshot:
    # api_key loaded from .env: TURBOQUANT_CLOUD_MOONSHOT_API_KEY
    model: "moonshot-v1-128k"
    max_tokens: 4096
    temperature: 0.7
    top_p: 0.95
    timeout: 120

  zhipu:
    # api_key loaded from .env: TURBOQUANT_CLOUD_ZHIPU_API_KEY
    model: "glm-4.5-flash"
    max_tokens: 4096
    temperature: 0.7
    top_p: 0.95
    timeout: 120

  deepseek:
    # api_key loaded from .env: TURBOQUANT_CLOUD_DEEPSEEK_API_KEY
    model: "deepseek-chat"
    max_tokens: 4096
    temperature: 0.7
    top_p: 0.95
    timeout: 120

  groq:
    # api_key loaded from .env: TURBOQUANT_CLOUD_GROQ_API_KEY
    model: "llama-3.3-70b-versatile"
    max_tokens: 4096
    temperature: 0.7
    top_p: 0.95
    timeout: 60
"""

_ENV_TEMPLATE = """\
# TurboQuant-X Environment Variables
# Secrets go here — this file is gitignored.

# ── Cloud LLM API Keys ───────────────────────────────────────────────
TURBOQUANT_CLOUD_OPENAI_API_KEY=
TURBOQUANT_CLOUD_ANTHROPIC_API_KEY=
TURBOQUANT_CLOUD_MOONSHOT_API_KEY=
TURBOQUANT_CLOUD_ZHIPU_API_KEY=
TURBOQUANT_CLOUD_DEEPSEEK_API_KEY=
TURBOQUANT_CLOUD_GROQ_API_KEY=

# ── Cloud Provider Override ──────────────────────────────────────────
# TURBOQUANT_CLOUD_PROVIDER=openai
# TURBOQUANT_CLOUD_MODEL=gpt-4o

# ── Server Overrides ─────────────────────────────────────────────────
# TURBOQUANT_HOST=0.0.0.0
# TURBOQUANT_PORT=8000
# TURBOQUANT_INFERENCE_MODE=cloud

# ── Model Overrides ──────────────────────────────────────────────────
# TURBOQUANT_MODEL_PATH=./models/my-model.gguf
# TURBOQUANT_N_CTX=32768
"""

# Known provider → env var mapping
_PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "TURBOQUANT_CLOUD_OPENAI_API_KEY",
    "anthropic": "TURBOQUANT_CLOUD_ANTHROPIC_API_KEY",
    "moonshot": "TURBOQUANT_CLOUD_MOONSHOT_API_KEY",
    "zhipu": "TURBOQUANT_CLOUD_ZHIPU_API_KEY",
    "deepseek": "TURBOQUANT_CLOUD_DEEPSEEK_API_KEY",
    "groq": "TURBOQUANT_CLOUD_GROQ_API_KEY",
    "together": "TURBOQUANT_CLOUD_TOGETHER_API_KEY",
    "openrouter": "TURBOQUANT_CLOUD_OPENROUTER_API_KEY",
    "siliconflow": "TURBOQUANT_CLOUD_SILICONFLOW_API_KEY",
    "custom": "TURBOQUANT_CLOUD_CUSTOM_API_KEY",
}

VALID_INFERENCE_MODES = {"standard", "turboquant", "zero-quant", "ultra-quant", "cloud"}


# ── Check results ────────────────────────────────────────────────────

class CheckResult:
    """Result of a single config check."""

    def __init__(self, name: str, ok: bool, message: str, fixable: bool = False):
        self.name = name
        self.ok = ok
        self.message = message
        self.fixable = fixable

    def __repr__(self) -> str:
        status = "OK" if self.ok else "FAIL"
        return f"CheckResult({self.name}: {status})"


class DoctorReport:
    """Collected results from all config checks."""

    def __init__(self) -> None:
        self.checks: list[CheckResult] = []

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def fixable_issues(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.ok and c.fixable]

    @property
    def unfixable_issues(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.ok and not c.fixable]

    def print_report(self) -> None:
        """Print a formatted health report to stdout."""
        print(f"\n{_BOLD}TurboQuant-X Config Doctor{_RESET}")
        print("=" * 50)

        for check in self.checks:
            if check.ok:
                print(f"  {_OK} {check.name}: {check.message}")
            elif check.fixable:
                print(f"  {_WARN} {check.name}: {check.message} {_DIM}(fixable){_RESET}")
            else:
                print(f"  {_FAIL} {check.name}: {check.message}")

        print()
        passed = sum(1 for c in self.checks if c.ok)
        total = len(self.checks)
        fixable = len(self.fixable_issues)

        if self.all_ok:
            print(f"  {_GREEN}{_BOLD}All {total} checks passed!{_RESET}")
        else:
            failed = total - passed
            print(f"  {passed}/{total} passed, {_RED}{failed} issues{_RESET}", end="")
            if fixable:
                print(f" ({_YELLOW}{fixable} fixable{_RESET} with --doctor fix)")
            else:
                print()
        print()


# ── Checks ───────────────────────────────────────────────────────────

def _check_config_dir() -> CheckResult:
    """Check config/ directory exists."""
    if CONFIG_DIR.is_dir():
        return CheckResult("config/", True, "Directory exists")
    return CheckResult("config/", False, "Directory missing", fixable=True)


def _check_default_yaml() -> CheckResult:
    """Check config/default.yaml exists and is valid YAML."""
    if not DEFAULT_YAML.exists():
        return CheckResult("default.yaml", False, "File missing", fixable=False)
    try:
        with open(DEFAULT_YAML) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return CheckResult("default.yaml", False, "Not a valid YAML dict", fixable=False)

        # Check inference_mode is valid
        mode = data.get("inference_mode", "standard")
        if mode not in VALID_INFERENCE_MODES:
            return CheckResult(
                "default.yaml",
                False,
                f"Invalid inference_mode: '{mode}' (valid: {', '.join(VALID_INFERENCE_MODES)})",
                fixable=False,
            )

        return CheckResult("default.yaml", True, f"Valid (mode: {mode})")
    except yaml.YAMLError as e:
        return CheckResult("default.yaml", False, f"YAML parse error: {e}", fixable=False)


def _check_cloud_yaml() -> CheckResult:
    """Check config/cloud.yaml exists and has valid structure."""
    if not CLOUD_YAML.exists():
        return CheckResult("cloud.yaml", False, "File missing", fixable=True)
    try:
        with open(CLOUD_YAML) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return CheckResult("cloud.yaml", False, "Not a valid YAML dict", fixable=True)

        providers = data.get("providers", {})
        if not providers:
            return CheckResult("cloud.yaml", False, "No providers defined", fixable=True)

        names = list(providers.keys())
        default = data.get("default_provider", "")
        if default and default not in names:
            return CheckResult(
                "cloud.yaml",
                False,
                f"default_provider '{default}' not in providers: {names}",
                fixable=False,
            )

        return CheckResult("cloud.yaml", True, f"Valid ({len(names)} providers: {', '.join(names)})")
    except yaml.YAMLError as e:
        return CheckResult("cloud.yaml", False, f"YAML parse error: {e}", fixable=True)


def _check_env_file() -> CheckResult:
    """Check .env file exists."""
    if ENV_FILE.exists():
        return CheckResult(".env", True, "File exists")
    return CheckResult(".env", False, "File missing — API keys not loaded", fixable=True)


def _check_cloud_api_keys() -> list[CheckResult]:
    """Check which cloud API keys are set (from .env + environment)."""
    results = []

    # Load .env if it exists (without modifying os.environ)
    env_vars: dict[str, str] = {}
    if ENV_FILE.exists():
        env_vars = _parse_env_file(ENV_FILE)

    # Also check actual environment (takes precedence)
    for provider, env_key in _PROVIDER_ENV_KEYS.items():
        value = os.environ.get(env_key, "") or env_vars.get(env_key, "")
        if value and value.strip():
            # Mask the key for display
            masked = value[:8] + "..." + value[-4:] if len(value) > 16 else "***"
            results.append(
                CheckResult(f"API key: {provider}", True, f"Set ({masked})")
            )
        else:
            results.append(
                CheckResult(f"API key: {provider}", True, f"Not set {_DIM}(optional){_RESET}")
            )

    # Check if at least one key is set when cloud mode is configured
    try:
        with open(DEFAULT_YAML) as f:
            cfg = yaml.safe_load(f) or {}
        if cfg.get("inference_mode") == "cloud":
            any_key = any(
                (os.environ.get(v, "") or env_vars.get(v, "")).strip()
                for v in _PROVIDER_ENV_KEYS.values()
            )
            if not any_key:
                results.append(
                    CheckResult(
                        "Cloud mode keys",
                        False,
                        "inference_mode=cloud but no API keys set! Add keys to config/.env",
                        fixable=True,
                    )
                )
    except Exception:
        pass

    return results


def _check_models_dir() -> CheckResult:
    """Check models/ directory and GGUF files."""
    if not MODELS_DIR.is_dir():
        return CheckResult("models/", False, "Directory missing", fixable=True)
    gguf_files = list(MODELS_DIR.glob("*.gguf"))
    if not gguf_files:
        return CheckResult("models/", True, f"Directory exists (no .gguf files — OK for cloud mode)")
    sizes = [f"{f.name} ({f.stat().st_size / (1024**3):.1f}GB)" for f in gguf_files[:5]]
    return CheckResult("models/", True, f"{len(gguf_files)} model(s): {', '.join(sizes)}")


def _check_logs_dir() -> CheckResult:
    """Check logs/ directory."""
    if LOGS_DIR.is_dir():
        return CheckResult("logs/", True, "Directory exists")
    return CheckResult("logs/", False, "Directory missing", fixable=True)


def _check_gitignore() -> CheckResult:
    """Check .gitignore protects secrets."""
    gitignore = Path(".gitignore")
    if not gitignore.exists():
        return CheckResult(".gitignore", False, "File missing — secrets may leak!", fixable=True)

    content = gitignore.read_text()
    issues = []
    if ".env" not in content:
        issues.append(".env not gitignored")
    if "config/.env" not in content and ".env" not in content:
        issues.append("config/.env not gitignored")

    if issues:
        return CheckResult(
            ".gitignore",
            False,
            f"Missing entries: {', '.join(issues)} — secrets may leak!",
            fixable=True,
        )
    return CheckResult(".gitignore", True, "Secrets protected")


def _check_cloud_yaml_sync() -> CheckResult:
    """Check if cloud section in default.yaml is in sync with cloud.yaml."""
    if not CLOUD_YAML.exists() or not DEFAULT_YAML.exists():
        return CheckResult("Config sync", True, "Skipped (missing files)")

    try:
        with open(DEFAULT_YAML) as f:
            default = yaml.safe_load(f) or {}
        with open(CLOUD_YAML) as f:
            cloud = yaml.safe_load(f) or {}

        cloud_in_default = default.get("cloud", {}).get("providers", {})
        cloud_in_file = cloud.get("providers", {})

        if cloud_in_default and cloud_in_file:
            # Both exist — just informational
            return CheckResult(
                "Config sync",
                True,
                "cloud.yaml is the source of truth for cloud providers",
            )
        return CheckResult("Config sync", True, "OK")
    except Exception:
        return CheckResult("Config sync", True, "Skipped")


def _validate_api_key(provider: str, api_key: str) -> CheckResult:
    """Make a lightweight test call to validate an API key."""
    try:
        from src.engine.cloud.provider import CloudConfig
        from src.engine.cloud.registry import create_provider

        config = CloudConfig(provider=provider, api_key=api_key, max_tokens=5)
        prov = create_provider(config)
        models = prov.list_models()
        if models:
            return CheckResult(
                f"Validate: {provider}",
                True,
                f"API key valid ({len(models)} models available)",
            )
        # list_models might return empty for some providers — try a tiny chat
        return CheckResult(f"Validate: {provider}", True, "Connected (models endpoint empty)")
    except Exception as e:
        err = str(e)
        if "401" in err or "Unauthorized" in err:
            return CheckResult(f"Validate: {provider}", False, "Invalid API key (401 Unauthorized)")
        if "403" in err or "Forbidden" in err:
            return CheckResult(f"Validate: {provider}", False, "API key forbidden (403)")
        return CheckResult(f"Validate: {provider}", False, f"Connection failed: {err[:80]}")


# ── Fix actions ──────────────────────────────────────────────────────

def _fix_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  {_INFO} Created {CONFIG_DIR}/")


def _fix_cloud_yaml() -> None:
    """Generate cloud.yaml, preserving existing values."""
    existing: dict[str, Any] = {}
    if CLOUD_YAML.exists():
        try:
            with open(CLOUD_YAML) as f:
                existing = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            pass

    if existing and existing.get("providers"):
        # File exists with content — just report
        print(f"  {_INFO} cloud.yaml exists with values — keeping as-is")
        return

    # Write fresh template
    CLOUD_YAML.write_text(_CLOUD_YAML_TEMPLATE)
    print(f"  {_INFO} Created {CLOUD_YAML} with provider templates")


def _fix_env_file() -> None:
    """Generate .env, preserving existing API keys."""
    existing_vars: dict[str, str] = {}
    if ENV_FILE.exists():
        existing_vars = _parse_env_file(ENV_FILE)

    if not existing_vars:
        ENV_FILE.write_text(_ENV_TEMPLATE)
        print(f"  {_INFO} Created {ENV_FILE} — add your API keys there")
        return

    # Regenerate template but keep existing values
    lines = []
    for line in _ENV_TEMPLATE.splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in existing_vars and existing_vars[key]:
                lines.append(f"{key}={existing_vars[key]}")
                continue
        lines.append(line)

    ENV_FILE.write_text("\n".join(lines) + "\n")
    preserved = sum(1 for v in existing_vars.values() if v)
    print(f"  {_INFO} Regenerated {ENV_FILE} (preserved {preserved} existing values)")


def _fix_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  {_INFO} Created {MODELS_DIR}/")


def _fix_logs_dir() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  {_INFO} Created {LOGS_DIR}/")


def _fix_gitignore() -> None:
    gitignore = Path(".gitignore")
    entries_to_add = []

    content = gitignore.read_text() if gitignore.exists() else ""

    if ".env" not in content:
        entries_to_add.append("# Secrets")
        entries_to_add.append(".env")
        entries_to_add.append("config/.env")

    if entries_to_add:
        with open(gitignore, "a") as f:
            f.write("\n" + "\n".join(entries_to_add) + "\n")
        print(f"  {_INFO} Added .env entries to .gitignore")


# ── .env parser ──────────────────────────────────────────────────────

def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict. Ignores comments and blank lines.

    Handles both plain ``KEY=value`` and bash-style ``export KEY=value``.
    """
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip optional 'export ' prefix (bash-compatible .env files)
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value
    return result


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """Load .env file and set values in os.environ (if not already set).

    Returns the parsed vars dict.
    """
    path = path or ENV_FILE
    env_vars = _parse_env_file(path)
    for key, value in env_vars.items():
        if value and key not in os.environ:
            os.environ[key] = value
    return env_vars


def load_cloud_yaml(path: Path | None = None) -> dict[str, Any]:
    """Load cloud.yaml and return the config dict.

    Merges into the standard config format expected by build_cloud_configs.
    """
    path = path or CLOUD_YAML
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return {"cloud": data}
    except yaml.YAMLError:
        return {}


# ── Main entry points ────────────────────────────────────────────────

def run_doctor(fix: bool = False, validate_keys: bool = False) -> DoctorReport:
    """Run all config checks and optionally fix issues.

    Args:
        fix: If True, auto-fix fixable issues.
        validate_keys: If True, test API keys with live calls.

    Returns:
        DoctorReport with all check results.
    """
    report = DoctorReport()

    # Structural checks
    report.add(_check_config_dir())
    report.add(_check_default_yaml())
    report.add(_check_cloud_yaml())
    report.add(_check_env_file())
    report.add(_check_models_dir())
    report.add(_check_logs_dir())
    report.add(_check_gitignore())
    report.add(_check_cloud_yaml_sync())

    # API key checks
    for result in _check_cloud_api_keys():
        report.add(result)

    # Live validation (optional, slow)
    if validate_keys:
        env_vars = _parse_env_file(ENV_FILE) if ENV_FILE.exists() else {}
        for provider, env_key in _PROVIDER_ENV_KEYS.items():
            key = os.environ.get(env_key, "") or env_vars.get(env_key, "")
            if key and key.strip():
                report.add(_validate_api_key(provider, key.strip()))

    # Print report
    report.print_report()

    # Fix if requested
    if fix and not report.all_ok:
        print(f"{_BOLD}Fixing issues...{_RESET}")
        for check in report.fixable_issues:
            name = check.name
            if name == "config/":
                _fix_config_dir()
            elif name == "cloud.yaml":
                _fix_cloud_yaml()
            elif name == ".env":
                _fix_env_file()
            elif name == "models/":
                _fix_models_dir()
            elif name == "logs/":
                _fix_logs_dir()
            elif name == ".gitignore":
                _fix_gitignore()
            elif name == "Cloud mode keys":
                _fix_env_file()

        print(f"\n{_GREEN}Fixes applied.{_RESET} Run {_BOLD}--doctor{_RESET} again to verify.\n")

    return report
