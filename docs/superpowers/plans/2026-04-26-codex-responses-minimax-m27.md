# Codex Responses MiniMax M2.7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LiteLLM Proxy support Codex `/v1/responses` for `chatgpt/gpt-5.5` planning and `minimax/codex-minimax-m2.7` implementation without sending unsupported tools or params to MiniMax.

**Architecture:** Keep `/v1/responses` on the existing proxy route. Let GPT models continue through native ChatGPT Responses config, and let MiniMax continue through the existing Responses-to-chat bridge with MiniMax-specific request validation and param filtering before the outbound chat-completion call.

**Tech Stack:** Python 3.12, LiteLLM provider configs, OpenAI Responses API bridge, pytest, JSON model metadata.

**Python Environment:** Use `uv` and the project virtual environment (`.venv`) to manage Python dependencies. Do NOT use bare `python` or `pytest` commands; always prefix Python commands with `uv run`. If any task creates or modifies a dependency file such as `pyproject.toml`, `uv.lock`, or dependency-group configuration, run `uv sync` immediately after that dependency-file change to install/update the virtual environment before running tests. Note: uv is already installed; use it directly without upgrading.

---

## File Structure

- Modify `model_prices_and_context_window.json`
  - Add only `minimax/codex-minimax-m2.7` with conservative MiniMax chat metadata.

- Modify `litellm/llms/minimax/chat/transformation.py`
  - Add Codex MiniMax model detection.
  - Restrict supported OpenAI params for `codex-minimax-m2.7`.
  - Filter unsupported params in `map_openai_params`.
  - Validate MiniMax tool shapes before the HTTP request.
  - Raise `litellm.BadRequestError` for unsupported tools or unsupported `tool_choice` forms.

- Modify `litellm/responses/litellm_completion_transformation/transformation.py`
  - Add an optional `custom_llm_provider` argument to `transform_responses_api_tools_to_chat_completion_tools`.
  - For MiniMax, reject non-function Responses tools before they become chat-completion tools.
  - Keep generic behavior unchanged for all other providers.

- Modify `tests/test_litellm/llms/minimax/chat/test_transformation.py`
  - Add unit coverage for provider routing, metadata, param filtering, tool validation, and error messages.

- Modify `tests/test_litellm/responses/litellm_completion_transformation/test_litellm_completion_responses.py`
  - Add unit coverage for MiniMax-specific Responses tool conversion behavior.

- Modify `tests/test_litellm_proxy_responses_config.py`
  - Add a small regression test that `chatgpt/gpt-5.5` still uses `ChatGPTResponsesAPIConfig` and MiniMax does not register native Responses config.

- Existing streaming coverage lives in `tests/test_litellm/responses/litellm_completion_transformation/test_tool_call_streaming_transformation.py`.
  - Add a MiniMax-labeled streaming regression only if existing generic tests do not already assert text/tool-call Responses events independent of provider.

## Task 1: Add MiniMax M2.7 Model Metadata

**Files:**
- Modify: `model_prices_and_context_window.json`
- Test: `tests/test_litellm/llms/minimax/chat/test_transformation.py`

- [ ] **Step 1: Add the failing metadata test**

Append this test near the existing provider routing/config tests in `tests/test_litellm/llms/minimax/chat/test_transformation.py`:

```python
def test_codex_minimax_m27_model_info():
    """Codex MiniMax M2.7 should be registered as a native MiniMax chat model."""
    import litellm

    model_info = litellm.get_model_info("minimax/codex-minimax-m2.7")

    assert model_info["litellm_provider"] == "minimax"
    assert model_info["mode"] == "chat"
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_system_messages"] is True
    assert model_info["supports_reasoning"] is True
```

- [ ] **Step 2: Run the failing metadata test**

Run:

```bash
uv run pytest tests/test_litellm/llms/minimax/chat/test_transformation.py::test_codex_minimax_m27_model_info -q
```

Expected: FAIL because `minimax/codex-minimax-m2.7` is not in `model_prices_and_context_window.json`.

- [ ] **Step 3: Add the model metadata**

Insert this JSON entry near the existing native MiniMax chat entries:

```json
"minimax/codex-minimax-m2.7": {
    "input_cost_per_token": 3e-07,
    "output_cost_per_token": 1.2e-06,
    "cache_read_input_token_cost": 3e-08,
    "cache_creation_input_token_cost": 3.75e-07,
    "litellm_provider": "minimax",
    "mode": "chat",
    "supports_function_calling": true,
    "supports_tool_choice": true,
    "supports_prompt_caching": true,
    "supports_reasoning": true,
    "supports_system_messages": true,
    "max_input_tokens": 1000000,
    "max_output_tokens": 8192
}
```

Place a comma before or after the entry according to the surrounding JSON object.

- [ ] **Step 4: Run the metadata test**

Run:

```bash
uv run pytest tests/test_litellm/llms/minimax/chat/test_transformation.py::test_codex_minimax_m27_model_info -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add model_prices_and_context_window.json tests/test_litellm/llms/minimax/chat/test_transformation.py
git commit -m "feat: add codex minimax m27 metadata"
```

## Task 2: Restrict MiniMax Codex Chat Params

**Files:**
- Modify: `litellm/llms/minimax/chat/transformation.py`
- Test: `tests/test_litellm/llms/minimax/chat/test_transformation.py`

- [ ] **Step 1: Add failing tests for supported params and filtering**

Append these tests to `tests/test_litellm/llms/minimax/chat/test_transformation.py`:

```python
def test_codex_minimax_m27_supported_params_are_narrow():
    """Codex MiniMax M2.7 should not advertise unsupported Codex/Responses params."""
    config = MinimaxChatConfig()

    supported_params = config.get_supported_openai_params(
        model="codex-minimax-m2.7"
    )

    assert "messages" not in supported_params
    assert "tools" in supported_params
    assert "tool_choice" in supported_params
    assert "stream" in supported_params
    assert "temperature" in supported_params
    assert "top_p" in supported_params
    assert "max_tokens" in supported_params
    assert "parallel_tool_calls" not in supported_params
    assert "web_search_options" not in supported_params
    assert "stream_options" not in supported_params


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
    assert "stream_options" not in optional_params
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
uv run pytest tests/test_litellm/llms/minimax/chat/test_transformation.py::test_codex_minimax_m27_supported_params_are_narrow tests/test_litellm/llms/minimax/chat/test_transformation.py::test_codex_minimax_m27_map_openai_params_drops_unsupported_params -q
```

Expected: FAIL because `MinimaxChatConfig` still inherits OpenAI's broad params.

- [ ] **Step 3: Implement model detection and narrow param list**

In `litellm/llms/minimax/chat/transformation.py`, update the imports:

```python
from typing import Any, Dict, List, Optional, Tuple
```

Add this constant and helper above `class MinimaxChatConfig`:

```python
CODEX_MINIMAX_M27_MODEL = "codex-minimax-m2.7"


def _is_codex_minimax_m27_model(model: str) -> bool:
    normalized_model = model.split("/", 1)[-1].lower()
    return normalized_model == CODEX_MINIMAX_M27_MODEL
```

Then replace `get_supported_openai_params` with:

```python
    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported OpenAI parameters for MiniMax.

        Codex MiniMax M2.7 is intentionally narrow because it is used through
        LiteLLM's Responses-to-chat bridge and MiniMax rejects several
        OpenAI/Codex request fields.
        """
        if _is_codex_minimax_m27_model(model):
            supported_params = [
                "frequency_penalty",
                "max_tokens",
                "presence_penalty",
                "seed",
                "stop",
                "stream",
                "temperature",
                "top_p",
                "tools",
                "tool_choice",
                "extra_headers",
                "reasoning_effort",
                "thinking",
                "reasoning_split",
            ]
            return supported_params

        base_params = super().get_supported_openai_params(model=model)
        additional_params = ["reasoning_split"]

        try:
            if litellm.supports_reasoning(model=model, custom_llm_provider="minimax"):
                additional_params.append("thinking")
        except Exception:
            pass

        return base_params + additional_params
```

- [ ] **Step 4: Implement filtering through `map_openai_params`**

Still in `MinimaxChatConfig`, add this method below `get_supported_openai_params`:

```python
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        optional_params = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

        if _is_codex_minimax_m27_model(model):
            unsupported_params = {
                "parallel_tool_calls",
                "web_search_options",
                "stream_options",
                "context_management",
                "metadata",
                "service_tier",
            }
            for param in unsupported_params:
                optional_params.pop(param, None)

        return optional_params
```

- [ ] **Step 5: Run the tests**

Run:

```bash
uv run pytest tests/test_litellm/llms/minimax/chat/test_transformation.py::test_codex_minimax_m27_supported_params_are_narrow tests/test_litellm/llms/minimax/chat/test_transformation.py::test_codex_minimax_m27_map_openai_params_drops_unsupported_params -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add litellm/llms/minimax/chat/transformation.py tests/test_litellm/llms/minimax/chat/test_transformation.py
git commit -m "fix: restrict codex minimax chat params"
```

## Task 3: Reject Unsupported MiniMax Tools In Responses Conversion

**Files:**
- Modify: `litellm/responses/litellm_completion_transformation/transformation.py`
- Test: `tests/test_litellm/responses/litellm_completion_transformation/test_litellm_completion_responses.py`

- [ ] **Step 1: Add failing MiniMax Responses tool tests**

Append these tests near existing `transform_responses_api_tools_to_chat_completion_tools` tests in `tests/test_litellm/responses/litellm_completion_transformation/test_litellm_completion_responses.py`:

```python
def test_minimax_responses_function_tools_are_preserved():
    """MiniMax Codex should preserve function tools required for agent turns."""
    tools, web_search_options = (
        LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=[
                {
                    "type": "function",
                    "name": "shell",
                    "description": "run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {"type": "string"},
                        },
                        "required": ["cmd"],
                    },
                    "strict": True,
                }
            ],
            custom_llm_provider="minimax",
            model="codex-minimax-m2.7",
        )
    )

    assert web_search_options is None
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "run shell commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"},
                    },
                    "required": ["cmd"],
                },
                "strict": True,
            },
        }
    ]


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "mcp", "server_label": "filesystem"},
        {"type": "namespace", "namespace": "tools"},
        {"type": "web_search_preview", "search_context_size": "low"},
        {"type": "web_search", "search_context_size": "low"},
        {"type": "image_generation"},
    ],
)
def test_minimax_responses_unsupported_tools_raise_bad_request(tool):
    """MiniMax Codex should fail fast instead of forwarding unsupported tools."""
    import litellm

    with pytest.raises(litellm.BadRequestError) as exc_info:
        LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=[tool],
            custom_llm_provider="minimax",
            model="codex-minimax-m2.7",
        )

    error_message = str(exc_info.value)
    assert "codex-minimax-m2.7" in error_message
    assert "minimax" in error_message.lower()
    assert "function tools only" in error_message
    assert "GPT" in error_message
```

If `pytest` is not already imported in this test file, add:

```python
import pytest
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
uv run pytest tests/test_litellm/responses/litellm_completion_transformation/test_litellm_completion_responses.py::test_minimax_responses_function_tools_are_preserved tests/test_litellm/responses/litellm_completion_transformation/test_litellm_completion_responses.py::test_minimax_responses_unsupported_tools_raise_bad_request -q
```

Expected: FAIL because the converter does not accept `custom_llm_provider` and still passes unsupported non-function tools.

- [ ] **Step 3: Add MiniMax validation helpers**

In `litellm/responses/litellm_completion_transformation/transformation.py`, add this helper inside `LiteLLMCompletionResponsesConfig` above `transform_responses_api_tools_to_chat_completion_tools`:

```python
    @staticmethod
    def _is_codex_minimax_m27_responses_bridge(
        custom_llm_provider: Optional[str],
        model: Optional[str],
    ) -> bool:
        if custom_llm_provider != "minimax" or model is None:
            return False
        return model.split("/", 1)[-1].lower() == "codex-minimax-m2.7"

    @staticmethod
    def _raise_unsupported_minimax_responses_tool(
        tool_type: str,
        model: Optional[str],
    ) -> None:
        import litellm

        raise litellm.BadRequestError(
            message=(
                f"Unsupported tool type '{tool_type}' for minimax/{model}. "
                "MiniMax Codex V1 supports function tools only. "
                "Use GPT models for richer Responses features such as MCP namespace tools, "
                "web search, or image generation."
            ),
            model=model or "codex-minimax-m2.7",
            llm_provider="minimax",
        )
```

- [ ] **Step 4: Update the converter signature and validation**

Change the function signature to:

```python
    def transform_responses_api_tools_to_chat_completion_tools(
        tools: Optional[List[Union[FunctionToolParam, OpenAIMcpServerTool]]],
        custom_llm_provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Tuple[
        List[Union[ChatCompletionToolParam, OpenAIMcpServerTool]],
        Optional[OpenAIWebSearchOptions],
    ]:
```

Then add this near the top of the loop, before existing type branches:

```python
            tool_type = str(tool.get("type") or "")
            if LiteLLMCompletionResponsesConfig._is_codex_minimax_m27_responses_bridge(
                custom_llm_provider=custom_llm_provider,
                model=model,
            ) and tool_type != "function":
                LiteLLMCompletionResponsesConfig._raise_unsupported_minimax_responses_tool(
                    tool_type=tool_type,
                    model=model,
                )
```

- [ ] **Step 5: Pass provider and model from request conversion**

In `transform_responses_api_request_to_chat_completion_request`, update the existing call to:

```python
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            responses_api_request.get("tools") or [],  # type: ignore
            custom_llm_provider=custom_llm_provider,
            model=model,
        )
```

- [ ] **Step 6: Run the MiniMax Responses tool tests**

Run:

```bash
uv run pytest tests/test_litellm/responses/litellm_completion_transformation/test_litellm_completion_responses.py::test_minimax_responses_function_tools_are_preserved tests/test_litellm/responses/litellm_completion_transformation/test_litellm_completion_responses.py::test_minimax_responses_unsupported_tools_raise_bad_request -q
```

Expected: PASS.

- [ ] **Step 7: Run nearby generic Responses tool tests**

Run:

```bash
uv run pytest tests/test_litellm/responses/litellm_completion_transformation/test_litellm_completion_responses.py -q
```

Expected: PASS. This verifies existing non-MiniMax behavior still passes through `mcp` and web-search tools as before.

- [ ] **Step 8: Commit**

Run:

```bash
git add litellm/responses/litellm_completion_transformation/transformation.py tests/test_litellm/responses/litellm_completion_transformation/test_litellm_completion_responses.py
git commit -m "fix: reject unsupported minimax responses tools"
```

## Task 4: Reject Unsupported MiniMax Chat Tool Shapes And Tool Choice

**Files:**
- Modify: `litellm/llms/minimax/chat/transformation.py`
- Test: `tests/test_litellm/llms/minimax/chat/test_transformation.py`

- [ ] **Step 1: Add failing tests for chat-level validation**

Append these tests to `tests/test_litellm/llms/minimax/chat/test_transformation.py`:

```python
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
```

- [ ] **Step 2: Run the failing validation tests**

Run:

```bash
uv run pytest tests/test_litellm/llms/minimax/chat/test_transformation.py::test_codex_minimax_m27_rejects_non_function_chat_tools tests/test_litellm/llms/minimax/chat/test_transformation.py::test_codex_minimax_m27_rejects_unsupported_tool_choice_shape tests/test_litellm/llms/minimax/chat/test_transformation.py::test_codex_minimax_m27_allows_function_tool_choice -q
```

Expected: FAIL because chat-level MiniMax validation is not implemented.

- [ ] **Step 3: Add validation helpers to `MinimaxChatConfig`**

In `litellm/llms/minimax/chat/transformation.py`, add these methods inside `MinimaxChatConfig`:

```python
    @staticmethod
    def _raise_codex_minimax_bad_request(
        message: str,
        model: str,
    ) -> None:
        raise litellm.BadRequestError(
            message=message,
            model=model,
            llm_provider="minimax",
        )

    def _validate_codex_minimax_tools(
        self,
        tools: Optional[List[Dict[str, Any]]],
        model: str,
    ) -> None:
        if not tools:
            return

        for tool in tools:
            tool_type = str(tool.get("type") or "")
            if tool_type != "function":
                self._raise_codex_minimax_bad_request(
                    message=(
                        f"Unsupported tool type '{tool_type}' for minimax/{model}. "
                        "MiniMax Codex V1 supports function tools only. "
                        "Use GPT models for richer Responses features such as MCP namespace tools, "
                        "web search, or image generation."
                    ),
                    model=model,
                )

    def _validate_codex_minimax_tool_choice(
        self,
        tool_choice: Any,
        model: str,
    ) -> None:
        if tool_choice is None:
            return
        if isinstance(tool_choice, str):
            if tool_choice in {"auto", "none"}:
                return
            self._raise_codex_minimax_bad_request(
                message=(
                    f"Unsupported tool_choice '{tool_choice}' for minimax/{model}. "
                    "MiniMax Codex V1 supports function tools only. "
                    "Use GPT models for richer Responses features."
                ),
                model=model,
            )
        if isinstance(tool_choice, dict):
            if (
                tool_choice.get("type") == "function"
                and isinstance(tool_choice.get("function"), dict)
                and tool_choice["function"].get("name")
            ):
                return
            self._raise_codex_minimax_bad_request(
                message=(
                    f"Unsupported tool_choice for minimax/{model}. "
                    "MiniMax Codex V1 supports function tools only. "
                    "Use GPT models for richer Responses features."
                ),
                model=model,
            )
        self._raise_codex_minimax_bad_request(
            message=(
                f"Unsupported tool_choice for minimax/{model}. "
                "MiniMax Codex V1 supports function tools only. "
                "Use GPT models for richer Responses features."
            ),
            model=model,
        )
```

- [ ] **Step 4: Call validation from `map_openai_params`**

At the end of the `_is_codex_minimax_m27_model(model)` block in `map_openai_params`, add:

```python
            self._validate_codex_minimax_tools(
                tools=optional_params.get("tools"),
                model=model,
            )
            self._validate_codex_minimax_tool_choice(
                tool_choice=optional_params.get("tool_choice"),
                model=model,
            )
```

- [ ] **Step 5: Run the validation tests**

Run:

```bash
uv run pytest tests/test_litellm/llms/minimax/chat/test_transformation.py::test_codex_minimax_m27_rejects_non_function_chat_tools tests/test_litellm/llms/minimax/chat/test_transformation.py::test_codex_minimax_m27_rejects_unsupported_tool_choice_shape tests/test_litellm/llms/minimax/chat/test_transformation.py::test_codex_minimax_m27_allows_function_tool_choice -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add litellm/llms/minimax/chat/transformation.py tests/test_litellm/llms/minimax/chat/test_transformation.py
git commit -m "fix: validate codex minimax tool inputs"
```

## Task 5: Verify GPT Responses Boundary

**Files:**
- Modify: `tests/test_litellm_proxy_responses_config.py`

- [ ] **Step 1: Add GPT and MiniMax Responses config regression tests**

Append these tests to `tests/test_litellm_proxy_responses_config.py`:

```python
def test_chatgpt_gpt_55_uses_native_responses_config():
    """GPT planning model should keep native ChatGPT Responses handling."""
    from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig

    config = ProviderConfigManager.get_provider_responses_api_config(
        model="gpt-5.5",
        provider=LlmProviders.CHATGPT,
    )

    assert isinstance(config, ChatGPTResponsesAPIConfig)


def test_minimax_codex_m27_does_not_register_native_responses_config():
    """MiniMax M2.7 should use the Responses-to-chat bridge, not native Responses."""
    config = ProviderConfigManager.get_provider_responses_api_config(
        model="codex-minimax-m2.7",
        provider=LlmProviders.MINIMAX,
    )

    assert config is None
```

- [ ] **Step 2: Run the regression tests**

Run:

```bash
uv run pytest tests/test_litellm_proxy_responses_config.py::test_chatgpt_gpt_55_uses_native_responses_config tests/test_litellm_proxy_responses_config.py::test_minimax_codex_m27_does_not_register_native_responses_config -q
```

Expected: PASS. If this fails, inspect `ProviderConfigManager._get_python_responses_api_config` and keep MiniMax out of native Responses config for V1.

- [ ] **Step 3: Commit**

Run:

```bash
git add tests/test_litellm_proxy_responses_config.py
git commit -m "test: cover codex responses provider boundary"
```

## Task 6: Verify Streaming Responses Events

**Files:**
- Inspect: `tests/test_litellm/responses/litellm_completion_transformation/test_tool_call_streaming_transformation.py`
- Modify: `tests/test_litellm/responses/litellm_completion_transformation/test_tool_call_streaming_transformation.py` only if the existing tests do not cover function-call Responses events.

- [ ] **Step 1: Inspect existing streaming tests**

Run:

```bash
rg -n "function_call_arguments|output_item.added|response.completed|LiteLLMCompletionStreamingIterator" tests/test_litellm/responses/litellm_completion_transformation/test_tool_call_streaming_transformation.py
```

Expected: Output includes assertions for `response.output_item.added`, `response.function_call_arguments.delta`, and `response.completed`.

- [ ] **Step 2: If coverage exists, run the existing streaming test file**

Run:

```bash
uv run pytest tests/test_litellm/responses/litellm_completion_transformation/test_tool_call_streaming_transformation.py -q
```

Expected: PASS.

- [ ] **Step 3: If coverage is missing, add this test**

Add this test to `tests/test_litellm/responses/litellm_completion_transformation/test_tool_call_streaming_transformation.py`:

```python
@pytest.mark.asyncio
async def test_minimax_labeled_tool_call_stream_emits_responses_events():
    """Provider label should not prevent chat tool-call chunks from becoming Responses events."""
    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )
    from litellm.types.llms.openai import ResponsesAPIStreamEvents

    class FakeStream:
        logging_obj = None

        def __aiter__(self):
            self._chunks = iter(
                [
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "codex-minimax-m2.7",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_123",
                                            "type": "function",
                                            "function": {
                                                "name": "shell",
                                                "arguments": "{\"cmd\":\"ls\"}",
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "codex-minimax-m2.7",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    },
                ]
            )
            return self

        async def __anext__(self):
            try:
                from litellm.types.utils import ModelResponseStream

                return ModelResponseStream(**next(self._chunks))
            except StopIteration:
                raise StopAsyncIteration

    iterator = LiteLLMCompletionStreamingIterator(
        model="codex-minimax-m2.7",
        litellm_custom_stream_wrapper=FakeStream(),
        request_input="list files",
        responses_api_request={"stream": True},
        custom_llm_provider="minimax",
    )

    events = []
    async for event in iterator:
        events.append(event.type)

    assert ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED in events
    assert ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA in events
    assert ResponsesAPIStreamEvents.RESPONSE_COMPLETED in events
```

- [ ] **Step 4: Run the streaming tests**

Run:

```bash
uv run pytest tests/test_litellm/responses/litellm_completion_transformation/test_tool_call_streaming_transformation.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit if the test file changed**

If no file changed, skip this commit. If the test file changed, run:

```bash
git add tests/test_litellm/responses/litellm_completion_transformation/test_tool_call_streaming_transformation.py
git commit -m "test: cover minimax responses stream events"
```

## Task 7: Final Verification And Formatting

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run pytest tests/test_litellm/llms/minimax/chat/test_transformation.py tests/test_litellm/responses/litellm_completion_transformation/test_litellm_completion_responses.py tests/test_litellm_proxy_responses_config.py tests/test_litellm/responses/litellm_completion_transformation/test_tool_call_streaming_transformation.py -q
```

Expected: PASS.

- [ ] **Step 2: Format Python files**

Run:

```bash
uv run black litellm/llms/minimax/chat/transformation.py litellm/responses/litellm_completion_transformation/transformation.py tests/test_litellm/llms/minimax/chat/test_transformation.py tests/test_litellm/responses/litellm_completion_transformation/test_litellm_completion_responses.py tests/test_litellm_proxy_responses_config.py tests/test_litellm/responses/litellm_completion_transformation/test_tool_call_streaming_transformation.py
```

Expected: black reports formatted files or leaves them unchanged.

- [ ] **Step 3: Run focused tests again**

Run:

```bash
uv run pytest tests/test_litellm/llms/minimax/chat/test_transformation.py tests/test_litellm/responses/litellm_completion_transformation/test_litellm_completion_responses.py tests/test_litellm_proxy_responses_config.py tests/test_litellm/responses/litellm_completion_transformation/test_tool_call_streaming_transformation.py -q
```

Expected: PASS.

- [ ] **Step 4: Inspect the diff**

Run:

```bash
git diff --stat
git diff -- model_prices_and_context_window.json litellm/llms/minimax/chat/transformation.py litellm/responses/litellm_completion_transformation/transformation.py tests/test_litellm/llms/minimax/chat/test_transformation.py tests/test_litellm/responses/litellm_completion_transformation/test_litellm_completion_responses.py tests/test_litellm_proxy_responses_config.py tests/test_litellm/responses/litellm_completion_transformation/test_tool_call_streaming_transformation.py
```

Expected: Diff only contains the model metadata, MiniMax adapter validation/filtering, Responses tool conversion validation, and tests.

- [ ] **Step 5: Commit final formatting or verification changes**

If `black` changed files or Task 7 introduced cleanup changes, run:

```bash
git add model_prices_and_context_window.json litellm/llms/minimax/chat/transformation.py litellm/responses/litellm_completion_transformation/transformation.py tests/test_litellm/llms/minimax/chat/test_transformation.py tests/test_litellm/responses/litellm_completion_transformation/test_litellm_completion_responses.py tests/test_litellm_proxy_responses_config.py tests/test_litellm/responses/litellm_completion_transformation/test_tool_call_streaming_transformation.py
git commit -m "chore: format codex minimax responses support"
```

If there are no changes after verification, do not create an empty commit.

## Self-Review Notes

Spec coverage:

- Single LiteLLM proxy entrypoint: covered by keeping work out of proxy route and testing provider config boundary.
- GPT planning model: covered by Task 5.
- MiniMax M2.7 model metadata: covered by Task 1.
- Strip unsupported params: covered by Task 2.
- Preserve function tools: covered by Task 3.
- Raise for unsupported tools/features: covered by Tasks 3 and 4.
- Preserve streaming behavior: covered by Task 6.
- Avoid broader provider matrix: covered by model detection for only `codex-minimax-m2.7`.

Plan hygiene:

- No implementation task depends on a separate custom proxy.
- Each task has a failing test first unless it is verification-only.
- Commits are scoped by behavior.
