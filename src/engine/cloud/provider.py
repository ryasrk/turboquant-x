"""Abstract base for cloud LLM providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CloudConfig:
    """Configuration for a cloud LLM provider.

    Attributes:
        provider: Provider name (openai, anthropic, moonshot, zhipu, custom).
        api_key: API key for authentication.
        base_url: API base URL (provider-specific default if omitted).
        model: Model identifier (e.g. gpt-4o, claude-sonnet-4-20250514).
        max_tokens: Default max tokens for generation.
        temperature: Default sampling temperature.
        top_p: Default nucleus sampling parameter.
        extra_headers: Additional HTTP headers (e.g. for custom endpoints).
        timeout: Request timeout in seconds.
    """

    provider: str
    api_key: str
    base_url: str | None = None
    model: str = ""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 120.0


@dataclass(frozen=True)
class CloudMessage:
    """A chat message for cloud providers."""

    role: str
    content: str


@dataclass(frozen=True)
class CloudResponse:
    """Response from a cloud LLM provider."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] | None = None


class CloudProvider(ABC):
    """Abstract base class for cloud LLM providers.

    All providers implement a common interface:
    - chat(): synchronous chat completion
    - chat_stream(): async streaming chat completion
    - list_models(): list available models
    """

    def __init__(self, config: CloudConfig) -> None:
        self._config = config

    @property
    def config(self) -> CloudConfig:
        return self._config

    @property
    def provider_name(self) -> str:
        return self._config.provider

    @property
    def model(self) -> str:
        return self._config.model

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> CloudResponse:
        """Synchronous chat completion.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.
            top_p: Override default top_p.

        Returns:
            CloudResponse with generated content and usage stats.
        """

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Async streaming chat completion.

        Yields content chunks as strings.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.
            top_p: Override default top_p.
        """
        yield ""  # pragma: no cover

    @abstractmethod
    def list_models(self) -> list[str]:
        """List available models for this provider.

        Returns:
            List of model identifier strings.
        """
