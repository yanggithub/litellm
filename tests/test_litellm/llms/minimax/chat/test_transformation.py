"""
Test MiniMax OpenAI-compatible API support
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory to the system path

import litellm
from litellm import completion
from litellm.litellm_core_utils.get_model_cost_map import (
    GetModelCostMap,
    _expand_model_aliases,
)
from litellm.llms.minimax.chat.transformation import MinimaxChatConfig


CODEX_MINIMAX_M27_MODEL = "minimax/codex-minimax-m2.7"


def test_minimax_chat_config():
    """Test that MinimaxChatConfig is properly configured"""
    config = MinimaxChatConfig()

    # Test get_api_base default
    api_base = config.get_api_base()
    assert api_base == "https://api.minimax.io/v1"

    # Test get_api_base with custom value
    custom_base = config.get_api_base(api_base="https://api.minimaxi.com/v1")
    assert custom_base == "https://api.minimaxi.com/v1"

    # Test get_complete_url
    complete_url = config.get_complete_url(
        api_base="https://api.minimax.io/v1",
        api_key=None,
        model="MiniMax-M2.1",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert complete_url == "https://api.minimax.io/v1/chat/completions"


def test_minimax_chat_config_url_variations():
    """Test URL handling with different base URL formats"""
    config = MinimaxChatConfig()

    # Test with /v1 ending
    url1 = config.get_complete_url(
        api_base="https://api.minimax.io/v1",
        api_key=None,
        model="MiniMax-M2.1",
        optional_params={},
        litellm_params={},
    )
    assert url1 == "https://api.minimax.io/v1/chat/completions"

    # Test with trailing slash
    url2 = config.get_complete_url(
        api_base="https://api.minimax.io/",
        api_key=None,
        model="MiniMax-M2.1",
        optional_params={},
        litellm_params={},
    )
    assert url2 == "https://api.minimax.io/v1/chat/completions"

    # Test without trailing slash
    url3 = config.get_complete_url(
        api_base="https://api.minimax.io",
        api_key=None,
        model="MiniMax-M2.1",
        optional_params={},
        litellm_params={},
    )
    assert url3 == "https://api.minimax.io/v1/chat/completions"

    # Test with full path already
    url4 = config.get_complete_url(
        api_base="https://api.minimax.io/v1/chat/completions",
        api_key=None,
        model="MiniMax-M2.1",
        optional_params={},
        litellm_params={},
    )
    assert url4 == "https://api.minimax.io/v1/chat/completions"


def test_minimax_provider_routing():
    """Test that minimax provider is properly routed"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    # Test with minimax/ prefix
    model, provider, api_key, api_base = get_llm_provider(
        model="minimax/MiniMax-M2.1", api_base="https://api.minimax.io/v1"
    )
    assert provider == "minimax"
    assert model == "MiniMax-M2.1"


def test_codex_minimax_m27_model_info(monkeypatch):
    """Codex MiniMax M2.7 should be registered as a native MiniMax chat model."""
    local_model_cost = _expand_model_aliases(
        GetModelCostMap.load_local_model_cost_map()
    )
    assert CODEX_MINIMAX_M27_MODEL in local_model_cost
    monkeypatch.setattr(litellm, "model_cost", local_model_cost)

    model_info = litellm.get_model_info(CODEX_MINIMAX_M27_MODEL)

    assert model_info["litellm_provider"] == "minimax"
    assert model_info["mode"] == "chat"
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_system_messages"] is True
    assert model_info["supports_reasoning"] is True


def test_codex_minimax_m27_model_info_matches_root_model_cost_map():
    """The package backup and published root map should stay in sync."""
    repo_root = Path(__file__).resolve().parents[5]
    root_model_cost = json.loads(
        (repo_root / "model_prices_and_context_window.json").read_text(encoding="utf-8")
    )
    local_model_cost = GetModelCostMap.load_local_model_cost_map()

    assert (
        root_model_cost[CODEX_MINIMAX_M27_MODEL]
        == local_model_cost[CODEX_MINIMAX_M27_MODEL]
    )


def test_codex_minimax_m27_supported_params_are_narrow():
    """Codex MiniMax M2.7 should not advertise unsupported Codex/Responses params."""
    config = MinimaxChatConfig()

    supported_params = config.get_supported_openai_params(model="codex-minimax-m2.7")

    assert "messages" not in supported_params
    assert "tools" in supported_params
    assert "tool_choice" in supported_params
    assert "stream" in supported_params
    assert "temperature" in supported_params
    assert "top_p" in supported_params
    assert "max_tokens" in supported_params
    assert "parallel_tool_calls" not in supported_params
    assert "web_search_options" not in supported_params
    assert "stream_options" in supported_params


def test_codex_minimax_m27_map_openai_params_drops_unsupported_params():
    """MiniMax Codex adapter should remove params MiniMax rejects."""
    config = MinimaxChatConfig()

    optional_params = config.map_openai_params(
        non_default_params={
            "stream": True,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 1024,
            "parallel_tool_calls": True,
            "web_search_options": {"search_context_size": "low"},
            "stream_options": {"include_usage": True},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "description": "run a shell command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
        optional_params={},
        model="codex-minimax-m2.7",
        drop_params=True,
    )

    assert optional_params["stream"] is True
    assert optional_params["temperature"] == 0.2
    assert optional_params["top_p"] == 0.9
    assert optional_params["max_tokens"] == 1024
    assert optional_params["tools"][0]["type"] == "function"
    assert "parallel_tool_calls" not in optional_params
    assert "web_search_options" not in optional_params
    assert optional_params["stream_options"] == {"include_usage": True}


def test_codex_minimax_m27_rejects_non_function_chat_tools():
    """MiniMax Codex should reject bad chat tool shapes before HTTP dispatch."""
    import litellm

    config = MinimaxChatConfig()

    with pytest.raises(litellm.BadRequestError) as exc_info:
        config.map_openai_params(
            non_default_params={
                "tools": [
                    {
                        "type": "namespace",
                        "namespace": "tools",
                    }
                ]
            },
            optional_params={},
            model="codex-minimax-m2.7",
            drop_params=True,
        )

    assert "Unsupported tool type 'namespace'" in str(exc_info.value)
    assert "function tools only" in str(exc_info.value)


def test_codex_minimax_m27_rejects_malformed_function_chat_tools():
    """MiniMax Codex should reject function tools without nested function names."""
    import litellm

    config = MinimaxChatConfig()

    with pytest.raises(litellm.BadRequestError) as exc_info:
        config.map_openai_params(
            non_default_params={
                "tools": [
                    {
                        "type": "function",
                        "name": "shell",
                        "parameters": {"type": "object"},
                    }
                ]
            },
            optional_params={},
            model="codex-minimax-m2.7",
            drop_params=True,
        )

    assert "Unsupported function tool" in str(exc_info.value)
    assert "function.name" in str(exc_info.value)


def test_codex_minimax_m27_rejects_unsupported_tool_choice_shape():
    """MiniMax Codex should reject Responses-only forced tool_choice forms."""
    import litellm

    config = MinimaxChatConfig()

    with pytest.raises(litellm.BadRequestError) as exc_info:
        config.map_openai_params(
            non_default_params={
                "tool_choice": {
                    "type": "namespace",
                    "namespace": "tools",
                }
            },
            optional_params={},
            model="codex-minimax-m2.7",
            drop_params=True,
        )

    assert "Unsupported tool_choice" in str(exc_info.value)
    assert "function tools only" in str(exc_info.value)


def test_codex_minimax_m27_allows_function_tool_choice():
    """MiniMax Codex should keep standard OpenAI function tool_choice."""
    config = MinimaxChatConfig()

    optional_params = config.map_openai_params(
        non_default_params={
            "tool_choice": {
                "type": "function",
                "function": {"name": "shell"},
            }
        },
        optional_params={},
        model="codex-minimax-m2.7",
        drop_params=True,
    )

    assert optional_params["tool_choice"] == {
        "type": "function",
        "function": {"name": "shell"},
    }


@pytest.mark.parametrize("tool_choice_str", ["auto", "none", "required"])
def test_codex_minimax_m27_allows_standard_string_tool_choices(tool_choice_str):
    """MiniMax Codex should accept standard OpenAI string tool_choice values."""
    config = MinimaxChatConfig()

    optional_params = config.map_openai_params(
        non_default_params={"tool_choice": tool_choice_str},
        optional_params={},
        model="codex-minimax-m2.7",
        drop_params=True,
    )

    assert optional_params["tool_choice"] == tool_choice_str


def test_non_codex_minimax_models_unaffected_by_codex_restrictions():
    """Standard MiniMax models should keep full OpenAI param support."""
    config = MinimaxChatConfig()

    codex_params = config.get_supported_openai_params(model="codex-minimax-m2.7")
    standard_params = config.get_supported_openai_params(model="MiniMax-M2.1")

    # Standard models should have strictly more supported params than Codex
    assert len(standard_params) > len(codex_params)
    # Standard models should still include params that Codex restricts
    assert "stream_options" in standard_params


def test_codex_minimax_m27_stream_options_not_stripped():
    """MiniMax Codex should preserve stream_options for Responses usage tracking."""
    config = MinimaxChatConfig()

    optional_params = config.map_openai_params(
        non_default_params={
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        optional_params={},
        model="codex-minimax-m2.7",
        drop_params=True,
    )

    assert optional_params.get("stream_options") == {"include_usage": True}


def test_minimax_provider_config_manager():
    """Test that ProviderConfigManager returns MinimaxChatConfig"""
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_chat_config(
        model="MiniMax-M2.1", provider=LlmProviders.MINIMAX
    )

    assert config is not None
    assert isinstance(config, MinimaxChatConfig)


@pytest.mark.skip(reason="Requires actual MiniMax API key")
def test_minimax_chat_completion_basic():
    """Test basic chat completion with MiniMax OpenAI-compatible API"""
    response = completion(
        model="minimax/MiniMax-M2.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        api_key=os.getenv("MINIMAX_API_KEY"),
        api_base="https://api.minimax.io/v1",
    )

    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0


@pytest.mark.skip(reason="Requires actual MiniMax API key")
def test_minimax_chat_completion_with_reasoning_split():
    """Test completion with reasoning_split parameter (MiniMax M2.1 feature)"""
    response = completion(
        model="minimax/MiniMax-M2.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Solve this problem: 2+2=?"},
        ],
        api_key=os.getenv("MINIMAX_API_KEY"),
        api_base="https://api.minimax.io/v1",
        extra_body={"reasoning_split": True},
    )

    assert response is not None
    # Check if reasoning_details is present in response
    if hasattr(response.choices[0].message, "reasoning_details"):
        assert response.choices[0].message.reasoning_details is not None


@pytest.mark.skip(reason="Requires actual MiniMax API key")
def test_minimax_chat_completion_with_tools():
    """Test completion with tool calling (function calling)"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = completion(
        model="minimax/MiniMax-M2.1",
        messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
        tools=tools,
        api_key=os.getenv("MINIMAX_API_KEY"),
        api_base="https://api.minimax.io/v1",
    )

    assert response is not None
    assert hasattr(response, "choices")


@pytest.mark.skip(reason="Requires actual MiniMax API key")
def test_minimax_chat_completion_streaming():
    """Test streaming completion"""
    response = completion(
        model="minimax/MiniMax-M2.1",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True,
        api_key=os.getenv("MINIMAX_API_KEY"),
        api_base="https://api.minimax.io/v1",
    )

    chunks = []
    for chunk in response:
        chunks.append(chunk)

    assert len(chunks) > 0


if __name__ == "__main__":
    # Run basic tests that don't require API key
    print("Testing MiniMax Chat Config...")
    test_minimax_chat_config()
    print("✓ Config test passed")

    print("\nTesting MiniMax Chat Config URL Variations...")
    test_minimax_chat_config_url_variations()
    print("✓ URL variations test passed")

    print("\nTesting MiniMax Provider Routing...")
    test_minimax_provider_routing()
    print("✓ Routing test passed")

    print("\nTesting MiniMax Provider Config Manager...")
    test_minimax_provider_config_manager()
    print("✓ Provider config manager test passed")

    print("\n✅ All basic tests passed!")
