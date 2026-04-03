"""Tests for cloud LLM engine and providers (all mocked — no API keys required)."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.engine.cloud.provider import CloudConfig, CloudResponse
from src.engine.cloud.registry import (
    ProviderRegistry,
    create_provider,
    build_cloud_configs,
)
from src.engine.cloud.openai_compat import OpenAICompatibleProvider
from src.engine.cloud.anthropic_provider import AnthropicProvider
from src.engine.cloud_engine import CloudEngine


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def openai_config():
    return CloudConfig(
        provider="openai",
        api_key="sk-test-key-openai",
        model="gpt-4o",
        max_tokens=1024,
    )


@pytest.fixture
def anthropic_config():
    return CloudConfig(
        provider="anthropic",
        api_key="sk-ant-test-key",
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
    )


@pytest.fixture
def moonshot_config():
    return CloudConfig(
        provider="moonshot",
        api_key="sk-test-key-moonshot",
        model="moonshot-v1-128k",
    )


@pytest.fixture
def zhipu_config():
    return CloudConfig(
        provider="zhipu",
        api_key="sk-test-key-zhipu",
        model="glm-4-flash",
    )


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the provider registry before each test."""
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


# ======================================================================
# CloudConfig tests
# ======================================================================


class TestCloudConfig:
    def test_default_values(self):
        cfg = CloudConfig(provider="openai", api_key="sk-test")
        assert cfg.provider == "openai"
        assert cfg.api_key == "sk-test"
        assert cfg.base_url is None
        assert cfg.model == ""
        assert cfg.max_tokens == 2048
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.95
        assert cfg.extra_headers == {}
        assert cfg.timeout == 120.0

    def test_frozen(self):
        cfg = CloudConfig(provider="openai", api_key="sk-test")
        with pytest.raises(AttributeError):
            cfg.api_key = "modified"

    def test_custom_values(self):
        cfg = CloudConfig(
            provider="custom",
            api_key="key",
            base_url="https://custom.example.com/v1",
            model="my-model",
            max_tokens=4096,
            temperature=0.5,
            top_p=0.9,
            extra_headers={"X-Custom": "value"},
            timeout=60.0,
        )
        assert cfg.base_url == "https://custom.example.com/v1"
        assert cfg.extra_headers == {"X-Custom": "value"}


# ======================================================================
# ProviderRegistry tests
# ======================================================================


class TestProviderRegistry:
    def test_register_and_get(self, openai_config):
        provider = OpenAICompatibleProvider(openai_config)
        ProviderRegistry.register("openai", provider)
        assert ProviderRegistry.get("openai") is provider

    def test_get_nonexistent(self):
        assert ProviderRegistry.get("nonexistent") is None

    def test_list_providers(self, openai_config, anthropic_config):
        ProviderRegistry.register("openai", OpenAICompatibleProvider(openai_config))
        ProviderRegistry.register("anthropic", AnthropicProvider(anthropic_config))
        names = ProviderRegistry.list_providers()
        assert "openai" in names
        assert "anthropic" in names

    def test_clear(self, openai_config):
        ProviderRegistry.register("openai", OpenAICompatibleProvider(openai_config))
        ProviderRegistry.clear()
        assert ProviderRegistry.list_providers() == []


# ======================================================================
# create_provider tests
# ======================================================================


class TestCreateProvider:
    def test_creates_openai_provider(self, openai_config):
        provider = create_provider(openai_config)
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_creates_anthropic_provider(self, anthropic_config):
        provider = create_provider(anthropic_config)
        assert isinstance(provider, AnthropicProvider)

    def test_creates_moonshot_provider(self, moonshot_config):
        provider = create_provider(moonshot_config)
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_creates_zhipu_provider(self, zhipu_config):
        provider = create_provider(zhipu_config)
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_rejects_empty_api_key(self):
        cfg = CloudConfig(provider="openai", api_key="")
        with pytest.raises(ValueError, match="API key required"):
            create_provider(cfg)


# ======================================================================
# build_cloud_configs tests
# ======================================================================


class TestBuildCloudConfigs:
    def test_builds_from_yaml_dict(self):
        config = {
            "cloud": {
                "default_provider": "openai",
                "providers": {
                    "openai": {
                        "api_key": "sk-test-openai",
                        "model": "gpt-4o",
                        "max_tokens": 4096,
                    },
                    "anthropic": {
                        "api_key": "sk-ant-test",
                        "model": "claude-sonnet-4-20250514",
                    },
                },
            }
        }
        configs = build_cloud_configs(config)
        assert "openai" in configs
        assert "anthropic" in configs
        assert configs["openai"].model == "gpt-4o"
        assert configs["openai"].max_tokens == 4096
        assert configs["anthropic"].model == "claude-sonnet-4-20250514"

    def test_skips_providers_without_api_key(self):
        config = {
            "cloud": {
                "providers": {
                    "openai": {"model": "gpt-4o"},  # no api_key
                },
            }
        }
        configs = build_cloud_configs(config)
        assert "openai" not in configs

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("TURBOQUANT_CLOUD_OPENAI_API_KEY", "sk-env-key")
        config = {
            "cloud": {
                "providers": {
                    "openai": {"model": "gpt-4o"},  # no yaml key
                },
            }
        }
        configs = build_cloud_configs(config)
        assert "openai" in configs
        assert configs["openai"].api_key == "sk-env-key"

    def test_empty_config(self):
        configs = build_cloud_configs({})
        assert configs == {}


# ======================================================================
# OpenAICompatibleProvider tests
# ======================================================================


class TestOpenAICompatibleProvider:
    def test_default_base_url(self, openai_config):
        provider = OpenAICompatibleProvider(openai_config)
        assert "api.openai.com" in provider._base_url

    def test_moonshot_default_url(self, moonshot_config):
        provider = OpenAICompatibleProvider(moonshot_config)
        assert "moonshot.cn" in provider._base_url

    def test_zhipu_default_url(self, zhipu_config):
        provider = OpenAICompatibleProvider(zhipu_config)
        assert "bigmodel.cn" in provider._base_url

    def test_custom_base_url(self):
        cfg = CloudConfig(
            provider="custom",
            api_key="key",
            base_url="https://my-endpoint.com/v1",
            model="my-model",
        )
        provider = OpenAICompatibleProvider(cfg)
        assert provider._base_url == "https://my-endpoint.com/v1"

    def test_rejects_unknown_provider_without_base_url(self):
        cfg = CloudConfig(provider="unknown_xyz", api_key="key")
        with pytest.raises(ValueError, match="No base_url"):
            OpenAICompatibleProvider(cfg)

    @patch("src.engine.cloud.openai_compat.httpx.Client")
    def test_chat_success(self, mock_client_cls, openai_config):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
            },
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        provider = OpenAICompatibleProvider(openai_config)
        response = provider.chat(
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert isinstance(response, CloudResponse)
        assert response.content == "Hello!"
        assert response.model == "gpt-4o"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 3
        assert response.total_tokens == 13

    @patch("src.engine.cloud.openai_compat.httpx.Client")
    def test_list_models(self, mock_client_cls, openai_config):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"id": "gpt-4o"},
                {"id": "gpt-4"},
                {"id": "gpt-3.5-turbo"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        provider = OpenAICompatibleProvider(openai_config)
        models = provider.list_models()

        assert "gpt-4o" in models
        assert "gpt-4" in models
        assert len(models) == 3


# ======================================================================
# AnthropicProvider tests
# ======================================================================


class TestAnthropicProvider:
    def test_default_base_url(self, anthropic_config):
        provider = AnthropicProvider(anthropic_config)
        assert "api.anthropic.com" in provider._base_url

    def test_convert_messages_extracts_system(self, anthropic_config):
        provider = AnthropicProvider(anthropic_config)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        system, converted = provider._convert_messages(messages)
        assert system == "You are helpful"
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_convert_messages_no_system(self, anthropic_config):
        provider = AnthropicProvider(anthropic_config)
        messages = [{"role": "user", "content": "Hi"}]
        system, converted = provider._convert_messages(messages)
        assert system is None
        assert len(converted) == 1

    @patch("src.engine.cloud.anthropic_provider.httpx.Client")
    def test_chat_success(self, mock_client_cls, anthropic_config):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": "Hello from Claude!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 5,
            },
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        provider = AnthropicProvider(anthropic_config)
        response = provider.chat(
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert isinstance(response, CloudResponse)
        assert response.content == "Hello from Claude!"
        assert response.model == "claude-sonnet-4-20250514"
        assert response.prompt_tokens == 12
        assert response.completion_tokens == 5

    def test_list_models_returns_known_models(self, anthropic_config):
        provider = AnthropicProvider(anthropic_config)
        models = provider.list_models()
        assert len(models) > 0
        assert any("claude" in m for m in models)

    def test_temperature_clamp(self, anthropic_config):
        provider = AnthropicProvider(anthropic_config)
        payload = provider._build_payload(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            temperature=2.0,  # Should be clamped to 1.0
            top_p=0.9,
        )
        assert payload["temperature"] == 1.0


# ======================================================================
# CloudEngine tests
# ======================================================================


class TestCloudEngine:
    def test_init(self, openai_config):
        engine = CloudEngine(openai_config)
        assert not engine.is_loaded
        assert engine.provider_name == "openai"
        assert engine.model_name == "gpt-4o"

    @patch("src.engine.cloud_engine.create_provider")
    def test_load_model(self, mock_create, openai_config):
        mock_provider = MagicMock()
        mock_create.return_value = mock_provider

        engine = CloudEngine(openai_config)
        engine.load_model()

        assert engine.is_loaded
        mock_create.assert_called_once_with(openai_config)

    @patch("src.engine.cloud_engine.create_provider")
    def test_load_model_twice_raises(self, mock_create, openai_config):
        mock_create.return_value = MagicMock()
        engine = CloudEngine(openai_config)
        engine.load_model()

        with pytest.raises(RuntimeError, match="already loaded"):
            engine.load_model()

    @patch("src.engine.cloud_engine.create_provider")
    def test_unload(self, mock_create, openai_config):
        mock_create.return_value = MagicMock()
        engine = CloudEngine(openai_config)
        engine.load_model()
        engine.unload()

        assert not engine.is_loaded

    def test_chat_without_loading_raises(self, openai_config):
        engine = CloudEngine(openai_config)
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.chat(messages=[{"role": "user", "content": "Hi"}])

    @patch("src.engine.cloud_engine.create_provider")
    def test_chat_returns_message_and_stats(self, mock_create, openai_config):
        mock_provider = MagicMock()
        mock_provider.chat.return_value = CloudResponse(
            content="Test response",
            model="gpt-4o",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        mock_create.return_value = mock_provider

        engine = CloudEngine(openai_config)
        engine.load_model()

        msg, stats = engine.chat(
            messages=[{"role": "user", "content": "Hello"}]
        )

        assert msg["role"] == "assistant"
        assert msg["content"] == "Test response"
        assert stats.prompt_tokens == 10
        assert stats.completion_tokens == 5
        assert stats.total_tokens == 15
        assert stats.generation_time_s > 0

    @patch("src.engine.cloud_engine.create_provider")
    def test_generate(self, mock_create, openai_config):
        mock_provider = MagicMock()
        mock_provider.chat.return_value = CloudResponse(
            content="Generated text",
            model="gpt-4o",
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
        )
        mock_create.return_value = mock_provider

        engine = CloudEngine(openai_config)
        engine.load_model()

        text, stats = engine.generate(prompt="Complete this:")
        assert text == "Generated text"
        assert stats.total_tokens == 8

    @patch("src.engine.cloud_engine.create_provider")
    def test_get_stats(self, mock_create, openai_config):
        mock_create.return_value = MagicMock()
        engine = CloudEngine(openai_config)
        engine.load_model()

        stats = engine.get_stats()
        assert stats["mode"] == "cloud"
        assert stats["provider"] == "openai"
        assert stats["is_loaded"] is True
        assert stats["model_name"] == "gpt-4o"

    @patch("src.engine.cloud_engine.create_provider")
    def test_list_models(self, mock_create, openai_config):
        mock_provider = MagicMock()
        mock_provider.list_models.return_value = ["gpt-4o", "gpt-4"]
        mock_create.return_value = mock_provider

        engine = CloudEngine(openai_config)
        engine.load_model()

        models = engine.list_models()
        assert "gpt-4o" in models

    @patch("src.engine.cloud_engine.create_provider")
    def test_switch_model(self, mock_create, openai_config):
        mock_provider = MagicMock()
        mock_create.return_value = mock_provider

        engine = CloudEngine(openai_config)
        engine.load_model()
        engine.switch_model("gpt-4-turbo")

        assert engine.model_name == "gpt-4-turbo"
        # Provider should have been recreated
        assert mock_create.call_count == 2

    def test_model_config_duck_type(self, openai_config):
        engine = CloudEngine(openai_config)
        mc = engine.model_config
        assert mc.model_name == "gpt-4o"
        assert "cloud://" in mc.model_path
        assert mc.n_ctx == 0


# ======================================================================
# main.py cloud config integration tests
# ======================================================================


class TestMainCloudConfig:
    def test_build_cloud_config_with_env(self, monkeypatch):
        monkeypatch.setenv("TURBOQUANT_CLOUD_OPENAI_API_KEY", "sk-test-from-env")
        from src.main import build_cloud_config

        config = {
            "cloud": {
                "default_provider": "openai",
                "providers": {
                    "openai": {"model": "gpt-4o"},
                },
            }
        }
        cloud_cfg = build_cloud_config(config)
        assert cloud_cfg is not None
        assert cloud_cfg.provider == "openai"
        assert cloud_cfg.api_key == "sk-test-from-env"
        assert cloud_cfg.model == "gpt-4o"

    def test_build_cloud_config_no_providers(self):
        from src.main import build_cloud_config

        cloud_cfg = build_cloud_config({})
        assert cloud_cfg is None

    def test_build_cloud_config_selects_default_provider(self, monkeypatch):
        monkeypatch.setenv("TURBOQUANT_CLOUD_OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("TURBOQUANT_CLOUD_ANTHROPIC_API_KEY", "sk-anthropic")
        from src.main import build_cloud_config

        config = {
            "cloud": {
                "default_provider": "anthropic",
                "providers": {
                    "openai": {"model": "gpt-4o"},
                    "anthropic": {"model": "claude-sonnet-4-20250514"},
                },
            }
        }
        cloud_cfg = build_cloud_config(config)
        assert cloud_cfg is not None
        assert cloud_cfg.provider == "anthropic"

    def test_build_cloud_config_model_override(self, monkeypatch):
        monkeypatch.setenv("TURBOQUANT_CLOUD_OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("TURBOQUANT_CLOUD_MODEL", "gpt-4-turbo")
        from src.main import build_cloud_config

        config = {
            "cloud": {
                "default_provider": "openai",
                "providers": {
                    "openai": {"model": "gpt-4o"},
                },
            }
        }
        cloud_cfg = build_cloud_config(config)
        assert cloud_cfg.model == "gpt-4-turbo"

    def test_build_cloud_config_provider_override(self, monkeypatch):
        monkeypatch.setenv("TURBOQUANT_CLOUD_OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("TURBOQUANT_CLOUD_ANTHROPIC_API_KEY", "sk-anthropic")
        monkeypatch.setenv("TURBOQUANT_CLOUD_PROVIDER", "anthropic")
        from src.main import build_cloud_config

        config = {
            "cloud": {
                "default_provider": "openai",
                "providers": {
                    "openai": {"model": "gpt-4o"},
                    "anthropic": {"model": "claude-sonnet-4-20250514"},
                },
            }
        }
        cloud_cfg = build_cloud_config(config)
        # Env var should override YAML default
        assert cloud_cfg.provider == "anthropic"
