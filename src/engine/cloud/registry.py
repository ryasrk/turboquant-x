"""Cloud provider registry — maps provider names to implementations."""

from __future__ import annotations

import logging
from typing import Any

from src.engine.cloud.provider import CloudConfig, CloudProvider

logger = logging.getLogger(__name__)

# Supported providers and their display names
SUPPORTED_PROVIDERS: dict[str, str] = {
    "openai": "OpenAI (GPT-4o, GPT-4, o1, o3)",
    "anthropic": "Anthropic (Claude Opus, Sonnet, Haiku)",
    "moonshot": "Moonshot AI / Kimi",
    "zhipu": "Zhipu AI / GLM-4",
    "deepseek": "DeepSeek",
    "groq": "Groq (LLaMA, Mixtral)",
    "together": "Together AI",
    "openrouter": "OpenRouter (multi-provider gateway)",
    "siliconflow": "SiliconFlow",
    "custom": "Custom OpenAI-compatible endpoint",
}


class ProviderRegistry:
    """Registry of cloud LLM provider implementations."""

    _providers: dict[str, CloudProvider] = {}

    @classmethod
    def register(cls, name: str, provider: CloudProvider) -> None:
        """Register a provider instance."""
        cls._providers[name] = provider
        logger.info("Registered cloud provider: %s (%s)", name, provider.__class__.__name__)

    @classmethod
    def get(cls, name: str) -> CloudProvider | None:
        """Get a registered provider by name."""
        return cls._providers.get(name)

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names."""
        return list(cls._providers.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers."""
        cls._providers.clear()


def create_provider(config: CloudConfig) -> CloudProvider:
    """Create a cloud provider instance from config.

    Uses Anthropic's native API for Claude; all others use the
    OpenAI-compatible protocol.

    Args:
        config: Cloud provider configuration.

    Returns:
        Configured CloudProvider instance.

    Raises:
        ValueError: If provider config is invalid.
    """
    if not config.api_key:
        raise ValueError(
            f"API key required for cloud provider '{config.provider}'. "
            "Set it in config or via environment variable."
        )

    if config.provider == "anthropic":
        from src.engine.cloud.anthropic_provider import AnthropicProvider
        return AnthropicProvider(config)

    # All other providers use OpenAI-compatible protocol
    from src.engine.cloud.openai_compat import OpenAICompatibleProvider
    return OpenAICompatibleProvider(config)


def build_cloud_configs(config: dict[str, Any]) -> dict[str, CloudConfig]:
    """Build CloudConfig instances from the cloud section of the config.

    Config format (YAML):
        cloud:
          default_provider: openai
          providers:
            openai:
              api_key: sk-...
              model: gpt-4o
            anthropic:
              api_key: sk-ant-...
              model: claude-sonnet-4-20250514
            moonshot:
              api_key: sk-...
              model: moonshot-v1-128k
            zhipu:
              api_key: ...
              model: glm-4-flash
            custom:
              api_key: ...
              base_url: https://your-endpoint.com/v1
              model: your-model

    Environment variable overrides:
        TURBOQUANT_CLOUD_OPENAI_API_KEY
        TURBOQUANT_CLOUD_ANTHROPIC_API_KEY
        TURBOQUANT_CLOUD_MOONSHOT_API_KEY
        TURBOQUANT_CLOUD_ZHIPU_API_KEY
        etc.

    Returns:
        Dict mapping provider name to CloudConfig.
    """
    import os

    cloud = config.get("cloud", {})
    providers_cfg = cloud.get("providers", {})
    configs: dict[str, CloudConfig] = {}

    for name, pcfg in providers_cfg.items():
        if not isinstance(pcfg, dict):
            continue

        # Environment variable override for API key
        env_key = f"TURBOQUANT_CLOUD_{name.upper()}_API_KEY"
        api_key = os.environ.get(env_key, pcfg.get("api_key", ""))

        if not api_key:
            logger.debug("Skipping cloud provider '%s': no API key", name)
            continue

        configs[name] = CloudConfig(
            provider=name,
            api_key=api_key,
            base_url=pcfg.get("base_url"),
            model=pcfg.get("model", ""),
            max_tokens=pcfg.get("max_tokens", 2048),
            temperature=pcfg.get("temperature", 0.7),
            top_p=pcfg.get("top_p", 0.95),
            extra_headers=pcfg.get("extra_headers", {}),
            timeout=pcfg.get("timeout", 120.0),
        )

    return configs
