# Code Review: Codex Responses MiniMax M2.7 Integration

**Date:** 2026-04-28  
**Commits reviewed:** `40371dceeb2455edeb7f106b6e4bc00710ccae34` to `5118e4956e` (HEAD)  
**Reviewer:** Claude Opus 4.6  
**Spec:** `docs/superpowers/specs/2026-04-26-codex-responses-minimax-m27-design.md`  
**Plan:** `docs/superpowers/plans/2026-04-26-codex-responses-minimax-m27.md`

---

## Summary

The implementation adds LiteLLM support for Codex `/v1/responses` with `minimax/codex-minimax-m2.7` as the implementation model. The architecture correctly keeps GPT models on native Responses handling and routes MiniMax through the existing Responses-to-chat bridge with MiniMax-specific validation.

**Overall quality:** Good. The layered validation (Responses layer + chat layer), the comprehensive test coverage, and the clear error messages are well-designed. Two bugs need fixing before production use with Codex CLI.

---

## Findings

| # | Severity | Category | Finding |
|---|----------|----------|---------|
| 1 | HIGH | Bug | `tool_choice="required"` incorrectly rejected |
| 2 | MEDIUM | Bug | `stream_options` stripping breaks Responses usage tracking |
| 3 | MEDIUM | Design | Model detection logic duplicated across layers |
| 4 | LOW | Design | Inline `import litellm` violates project coding convention |
| 5 | MEDIUM | Test gap | No test for `tool_choice="required"/"auto"/"none"` strings |
| 6 | LOW | Test gap | No regression test ensuring non-Codex MiniMax models unaffected |
| 7 | LOW | Style | Validation uses sequential `if` instead of `elif` |
| 8 | LOW | Test gap | No MiniMax-labeled streaming Responses event test |
| 9 | NIT | UX | Error message references "GPT" which may be confusing outside Codex |

---

## Detailed Findings

### 1. BUG (HIGH): `tool_choice="required"` incorrectly rejected

**File:** `litellm/llms/minimax/chat/transformation.py`, lines 218-228

**Problem:**

```python
def _validate_codex_minimax_tool_choice(self, tool_choice, model):
    ...
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none"}:
            return
        self._raise_codex_minimax_bad_request(...)  # <-- "required" hits this
```

The string tool_choice validator only allows `"auto"` and `"none"`. But `"required"` is a standard OpenAI `tool_choice` value that forces the model to call at least one tool. Codex CLI uses `tool_choice="required"` to ensure the implementation model produces tool calls during agent turns.

**Impact:** Any Codex agent turn that sets `tool_choice="required"` will fail with `BadRequestError` before reaching MiniMax, making the implementation model unable to reliably produce tool calls when the agent demands it.

**Fix:**

```python
if tool_choice in {"auto", "none", "required"}:
    return
```

---

### 2. BUG (MEDIUM): `stream_options` stripping breaks Responses usage tracking

**Files:**
- `litellm/llms/minimax/chat/transformation.py`, line 149 (strips `stream_options`)
- `litellm/responses/litellm_completion_transformation/transformation.py`, lines 232-236 (injects it)
- `litellm/utils.py`, line 4023 (bypasses supported_params check)

**Problem:**

The Responses-to-chat bridge **always** injects `stream_options={"include_usage": True}` for streaming requests:

```python
# In transform_responses_api_request_to_chat_completion_request:
if stream is True:
    stream_options = {"include_usage": True}
    litellm_completion_request["stream_options"] = stream_options
```

This is required because Responses API `response.completed` events include usage data (input_tokens, output_tokens, total_tokens).

However, the MiniMax chat adapter strips it:

```python
unsupported_params = {
    ...
    "stream_options",  # <-- stripped
    ...
}
for param in unsupported_params:
    optional_params.pop(param, None)
```

I verified that `stream_options` bypasses the `get_supported_openai_params` check in `litellm/utils.py:4023` (it's in a special allowlist alongside `user` and `stream`), so it enters `map_openai_params` and then gets explicitly popped by the MiniMax override.

**Impact:** The `response.completed` streaming event emitted by `LiteLLMCompletionStreamingIterator` may contain zero-value usage data, which would break Codex's cost tracking and token counting.

**Fix options:**
1. **Remove `stream_options` from the strip list** if MiniMax's `api.minimax.io/v1` endpoint accepts or ignores `stream_options` (most OpenAI-compatible endpoints do).
2. **Keep stripping but handle usage differently** — if MiniMax truly rejects `stream_options`, the streaming iterator may need to compute usage from prompt/completion token counts provided elsewhere.

**Recommendation:** Option 1. MiniMax's endpoint at `api.minimax.io/v1` claims OpenAI compatibility, and `stream_options` is an OpenAI standard parameter. Test by sending a request with `stream_options` to confirm.

---

### 3. DESIGN (MEDIUM): Model detection logic duplicated across layers

**Files:**
- `litellm/llms/minimax/chat/transformation.py` — `_is_codex_minimax_m27_model()`
- `litellm/responses/litellm_completion_transformation/transformation.py` — `_is_codex_minimax_m27_responses_bridge()`

Both implement the same core check:
```python
model.split("/", 1)[-1].lower() == "codex-minimax-m2.7"
```

The Responses version adds `custom_llm_provider == "minimax"` as an additional guard.

**Issue:** The magic string `"codex-minimax-m2.7"` appears in both files independently. If a future `codex-minimax-m2.8` or `codex-minimax-m3.0` is added, both files must be updated.

**Suggestion:** Extract the constant and detection function to `litellm/llms/minimax/common_utils.py`:

```python
# litellm/llms/minimax/common_utils.py
CODEX_MINIMAX_MODELS = {"codex-minimax-m2.7"}

def is_codex_minimax_model(model: str) -> bool:
    normalized = model.split("/", 1)[-1].lower()
    return normalized in CODEX_MINIMAX_MODELS
```

Then both files import from this shared location. Adding new Codex MiniMax models becomes a one-line change.

---

### 4. DESIGN (LOW): Inline `import litellm` violates project convention

**File:** `litellm/responses/litellm_completion_transformation/transformation.py`, lines 1368, 1386

```python
@staticmethod
def _raise_unsupported_minimax_responses_tool(tool_type, model):
    import litellm  # <-- inline import
    raise litellm.BadRequestError(...)
```

The project's `CLAUDE.md` states: *"Avoid imports within methods — place all imports at the top of the file (module-level)."*

**Note:** This may be intentional to avoid circular imports. The file already does `from litellm.caching import InMemoryCache` and `from litellm.types.llms.openai import ...` at the top, but not a bare `import litellm`.

**Fix:** Add `import litellm` at the top of the file (after the existing `from litellm...` imports), then remove the inline imports. If this causes a circular import, add a comment explaining why the inline import is necessary.

---

### 5. IMPROVEMENT (MEDIUM): Missing tests for string `tool_choice` values

**File:** `tests/test_litellm/llms/minimax/chat/test_transformation.py`

Tests exist for:
- `tool_choice={"type": "function", "function": {"name": "shell"}}` — allowed (tested)
- `tool_choice={"type": "namespace", ...}` — rejected (tested)

Missing tests for:
- `tool_choice="required"` — should be allowed (currently broken, see Bug #1)
- `tool_choice="auto"` — should be allowed
- `tool_choice="none"` — should be allowed

**Suggested test:**

```python
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
```

---

### 6. IMPROVEMENT (LOW): No regression test for non-Codex MiniMax models

**File:** `tests/test_litellm/llms/minimax/chat/test_transformation.py`

The tests verify Codex MiniMax M2.7 has restricted params, but nothing asserts that `MiniMax-M2.1` or other standard MiniMax models are NOT affected by the new restrictions.

**Suggested test:**

```python
def test_non_codex_minimax_models_unaffected_by_codex_restrictions():
    """Standard MiniMax models should keep full OpenAI param support."""
    config = MinimaxChatConfig()
    codex_params = config.get_supported_openai_params(model="codex-minimax-m2.7")
    standard_params = config.get_supported_openai_params(model="MiniMax-M2.1")
    # Standard models should have strictly more supported params
    assert len(standard_params) > len(codex_params)
```

---

### 7. STYLE (LOW): Validation uses `if/if` instead of `elif`

**File:** `litellm/llms/minimax/chat/transformation.py`, lines 218-251

```python
if isinstance(tool_choice, str):
    ...  # returns or raises
if isinstance(tool_choice, dict):  # <-- should be elif
    ...  # returns or raises
self._raise_codex_minimax_bad_request(...)  # fallback
```

Since the first `if` block always returns or raises for strings, the second `if` is unreachable when `tool_choice` is a string. Using `elif` makes the mutual exclusivity explicit and avoids the unnecessary `isinstance` check.

---

### 8. TEST GAP (LOW): No MiniMax-labeled streaming test

**File:** `tests/test_litellm/responses/litellm_completion_transformation/test_tool_call_streaming_transformation.py`

The plan called for a MiniMax-labeled streaming regression test, but no changes were made to this file. The existing generic streaming tests likely cover the path since `LiteLLMCompletionStreamingIterator` is provider-agnostic, but an explicit test with `model="codex-minimax-m2.7"` and `custom_llm_provider="minimax"` would provide confidence.

---

### 9. NIT: Error message "GPT" may be confusing

**File:** `litellm/responses/litellm_completion_transformation/transformation.py`, line 1373

```python
"Use GPT models for richer Responses features such as MCP namespace tools, web search, or image generation."
```

This makes sense in the Codex context where GPT-5.5 is the planning model, but if other users hit this error outside Codex, "GPT models" is less clear than "models with native Responses API support". Minor issue — acceptable for V1.

---

## What's Done Well

1. **Layered validation** — Tools are validated at both the Responses conversion layer (early, before chat conversion) AND at the chat param mapping layer (defense-in-depth). This prevents bad requests from reaching MiniMax regardless of entry path.

2. **Clear error messages** — All `BadRequestError` messages include the model name, provider, unsupported feature, and guidance for the correct model to use.

3. **Comprehensive test coverage** — Tests cover model metadata, param filtering, tool rejection (parametrized), tool_choice validation, and provider boundary (GPT vs MiniMax routing).

4. **Conservative metadata** — The model entry in `model_prices_and_context_window.json` only advertises capabilities that V1 actually preserves.

5. **No proxy route changes** — All logic stays at the adapter/transformation level, avoiding provider-specific code in the FastAPI endpoints.

---

## Recommended Fix Priority

1. **Fix Bug #1** (tool_choice="required") — Blocking for Codex usage
2. **Investigate Bug #2** (stream_options) — Test against MiniMax API, remove from strip list if accepted
3. **Add tests** (#5, #6) — Prevents regression and catches Bug #1
4. **Extract model constant** (#3) — Preparedness for future model variants
5. **Fix inline imports** (#4) — Code hygiene
