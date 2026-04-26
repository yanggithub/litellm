# Codex Responses Support For MiniMax M2.7

## Goal

Support Codex using LiteLLM Proxy as a single `model_provider` for:

- `chatgpt/gpt-5.5` for GPT planning.
- `minimax/codex-minimax-m2.7` for MiniMax implementation.

Codex sends OpenAI Responses API requests to `/v1/responses` and expects Responses-style streaming events back. LiteLLM should keep the proxy entrypoint unchanged and fix provider adaptation inside LiteLLM, not by adding a separate custom proxy.

## V1 Scope

V1 is intentionally narrow:

- Support Codex `/v1/responses` workflow.
- Preserve existing GPT/ChatGPT native Responses handling.
- Add only the native LiteLLM MiniMax model alias `minimax/codex-minimax-m2.7`.
- Support MiniMax through its OpenAI-compatible chat completions endpoint.
- Do not cover MiniMax through OpenRouter, Bedrock, Novita, Vertex, or other provider aliases.
- Do not attempt to emulate unsupported Responses features on MiniMax.

## Current Code Path

`/v1/responses` is handled by `litellm/proxy/response_api_endpoints/endpoints.py` and routed as `aresponses`.

`litellm/responses/main.py` checks `ProviderConfigManager.get_provider_responses_api_config`. GPT planning models such as `chatgpt/gpt-5.5` resolve to `ChatGPTResponsesAPIConfig`, so they use the native Responses path.

MiniMax currently has no Responses provider config. Requests for `custom_llm_provider="minimax"` fall back to `LiteLLMCompletionTransformationHandler`, which converts Responses requests to chat completion requests and then converts chat completion responses or streams back to Responses API objects/events.

The generic bridge is close to the desired architecture, but it currently passes through unsupported Codex/Responses params and non-function tools that MiniMax rejects.

## Architecture

Keep the existing proxy route and generic Responses-to-chat bridge.

Add MiniMax-specific behavior at the provider/adapter boundary:

1. Register model metadata for `minimax/codex-minimax-m2.7`.
2. Use MiniMax chat-completion transformations to filter unsupported OpenAI/Codex params before the MiniMax HTTP request.
3. Reuse the existing Responses-to-chat request conversion.
4. Reuse the existing chat-stream-to-Responses streaming iterator.

This keeps GPT planning and MiniMax implementation on the same LiteLLM proxy surface while avoiding proxy-route special cases.

## Model Metadata

Add `minimax/codex-minimax-m2.7` to `model_prices_and_context_window.json` with conservative metadata:

- `litellm_provider`: `minimax`
- `mode`: `chat`
- `supports_function_calling`: `true`
- `supports_tool_choice`: `true` only if MiniMax M2.7 accepts the supported `tool_choice` values after normalization
- `supports_system_messages`: `true`
- `supports_reasoning`: `true` only if the existing MiniMax M2.x reasoning handling applies to M2.7
- Token and pricing fields copied conservatively from the closest supported MiniMax M2.x entry until exact M2.7 pricing is available

This metadata should advertise only the capabilities V1 actually preserves.

## Request Adaptation

For `custom_llm_provider="minimax"` and model `codex-minimax-m2.7`, LiteLLM should strip unsupported request fields before sending to MiniMax:

- Drop `parallel_tool_calls`.
- Drop `web_search_options`.
- Drop Responses-only context fields that only belong to `/v1/responses`.
- Drop `stream_options` if MiniMax rejects it; otherwise keep it only when verified to work.
- Preserve `messages`, `stream`, `max_tokens`, `temperature`, `top_p`, supported reasoning/thinking params, and supported function tool params.

`tool_choice` should be normalized to MiniMax-supported values. V1 should pass simple supported values such as `auto` and `none`, and reject unsupported forced namespace/tool forms with a clear LiteLLM error.

## Tool Handling

MiniMax V1 must preserve Codex coding-agent function tools. Dropping function tools would make the implementation model unusable for normal agent turns.

Supported:

- Responses `type: "function"` tools converted to OpenAI chat-completion function tools:
  - `{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}`
  - Preserve valid JSON schema parameters.
  - Ensure missing parameters default to an object schema when required by provider logic.

Unsupported:

- `mcp`
- `namespace`
- `web_search`
- `web_search_preview`
- `image_generation`
- Any non-function tool type

When unsupported tools are present in a request routed to MiniMax, LiteLLM should raise a targeted `BadRequestError` before making the MiniMax API call. The error should state that MiniMax Codex V1 supports function tools only and that GPT planning models should be used for richer Responses features.

This is preferable to silently dropping unsupported tools because silent drops can remove required Codex agent capabilities.

## GPT Feature Boundary

GPT models remain the correct route for Responses features that MiniMax V1 does not support. This includes MCP namespace tools, image generation, web search, and other native Responses features.

LiteLLM should make that boundary visible:

- GPT planning models continue to use native Responses support.
- MiniMax implementation model supports function-tool agent work only.
- Unsupported feature requests routed to MiniMax fail fast with a message pointing the caller to GPT-capable models for those features.

## Streaming

MiniMax streaming should continue through LiteLLM chat completion streaming. `LiteLLMCompletionStreamingIterator` already converts chat-completion streaming chunks back into Responses events.

V1 should verify that MiniMax-style streamed chunks produce the events Codex needs:

- `response.created`
- `response.in_progress`
- `response.output_item.added`
- `response.output_text.delta` for assistant text
- `response.function_call_arguments.delta` for tool argument streaming
- function-call done events
- `response.completed`

If MiniMax sends complete tool arguments in one chunk, the existing iterator behavior that splits large argument deltas can be reused.

## Error Handling

Use `litellm.BadRequestError` for unsupported MiniMax V1 request features.

The error message should include:

- model name
- provider name
- unsupported feature or tool type
- short guidance that MiniMax Codex V1 supports function tools only, while GPT models should be used for richer Responses features

The adapter should reject before the provider HTTP call so MiniMax never sees invalid tool types such as `namespace`.

## Tests

Add focused unit tests rather than broad integration coverage:

- `chatgpt/gpt-5.5` still resolves to the ChatGPT Responses provider config.
- `minimax/codex-minimax-m2.7` resolves as a MiniMax chat model.
- A Codex Responses request for MiniMax strips unsupported params before chat completion dispatch.
- Function tools survive and are shaped as MiniMax-compatible chat-completion function tools.
- Unsupported tools raise `BadRequestError` and are not sent to MiniMax.
- Streaming chat chunks with text convert to Responses streaming events.
- Streaming chat chunks with function tool calls convert to Responses function-call events.

## Non-Goals

- No custom proxy separate from LiteLLM.
- No support for MiniMax aliases on OpenRouter, Bedrock, Novita, Vertex, or other providers.
- No native MiniMax Responses API config unless MiniMax exposes a real Responses endpoint.
- No silent dropping of unsupported tools.
- No full MCP emulation for MiniMax.
- No image generation or web search support on MiniMax V1.

## Open Implementation Notes

The likely implementation points are:

- `model_prices_and_context_window.json`
- `litellm/llms/minimax/chat/transformation.py`
- `litellm/responses/litellm_completion_transformation/transformation.py`
- Tests under the existing Responses and MiniMax test areas

Keep the changes adapter-level. Avoid adding provider-specific logic to the FastAPI proxy route unless a test proves the routing layer needs a small metadata fix.
