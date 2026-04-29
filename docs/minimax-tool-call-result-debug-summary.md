# MiniMax Tool Call Result Debug Summary

Date: 2026-04-29

## Scope

This note summarizes the current MiniMax failure observed when Codex calls the local LiteLLM proxy with model alias `minimax-m2.7`. It is documentation only; no implementation fix is included for this issue.

## Current Failure

LiteLLM request id:

- `88e5b00a-8657-4f20-b10d-c0a635029405`

Spend-log mapping:

- Start time: `2026-04-29T15:08:07.691+00:00`
- Model: `minimax-m2.7`
- Status: `failure`
- Session id: `91cc5ffc-9b5f-452c-a0ef-a7446184ed29`
- MiniMax provider request id: `064151d456bca7100dd9a7c5f4f9173d`

Provider error:

```text
invalid params, tool call result does not follow tool call (2013)
```

This is a different failure from the earlier MiniMax error:

```text
invalid params, invalid chat setting (2013)
```

The earlier error was tied to MiniMax receiving multiple separate `system` messages. The current failure occurs after that path is cleared.

## Captured Request Shape

Raw request debug was enabled temporarily for the MiniMax alias only. The captured provider-bound request had three retries with the same shape:

- Top-level keys: `client_metadata`, `messages`, `model`, `stream`, `tools`
- Provider model: `codex-minimax-m2.7`
- Stream: `true`
- Tool count: `9`
- Message count: `7`

Message sequence, with content redacted:

| Index | Role | Content type | Tool information |
| --- | --- | --- | --- |
| 0 | `system` | `str` | merged system prompt |
| 1 | `user` | `list` | text block |
| 2 | `user` | `list` | text block |
| 3 | `assistant` | `NoneType` | `tool_calls`: `call_function_vqcmhdrvqwgy_1`, `call_function_vqcmhdrvqwgy_2`; both `exec_command` |
| 4 | `assistant` | `list` | text block |
| 5 | `tool` | `str` | `tool_call_id`: `call_function_vqcmhdrvqwgy_1` |
| 6 | `tool` | `str` | `tool_call_id`: `call_function_vqcmhdrvqwgy_2` |

The suspicious ordering is:

```text
assistant(tool_calls=[call_1, call_2])
assistant(text)
tool(tool_call_id=call_1)
tool(tool_call_id=call_2)
```

MiniMax reports that the tool call result does not follow the tool call. The extra assistant text message between the assistant tool-call message and the tool result messages is the most likely request-shape problem.

## Relevant Code Path

The failure is in the Responses API to Chat Completions bridge path before the MiniMax provider call:

- `litellm/responses/litellm_completion_transformation/transformation.py`
- `transform_responses_api_input_to_messages`
- `_transform_response_input_param_to_chat_completion_message`
- `_ensure_tool_results_have_corresponding_tool_calls`

MiniMax then receives the transformed chat-completion request through:

- `litellm/llms/minimax/chat/transformation.py`

## Notes

- The request has valid tool ids: the `tool` messages refer to ids present in the assistant `tool_calls` message.
- The rejection appears to be about ordering, not missing ids.
- A synthetic orphan tool result produces a different MiniMax error: `tool result's tool id(...) not found`.
- The captured real request is therefore best described as a tool-call adjacency/order issue.
- Temporary `litellm_request_debug: true` was removed again after capture to avoid ongoing raw prompt logging.

## Suggested Next Investigation

Add a failing unit test around the Responses-to-chat transformation that produces:

```text
assistant(tool_calls=[...])
assistant(text)
tool(...)
```

Then decide whether MiniMax should:

- drop or move the intervening assistant text message,
- merge it into the assistant tool-call message content when safe, or
- preserve it elsewhere while ensuring each `tool` result immediately follows the assistant `tool_calls` message required by MiniMax.
