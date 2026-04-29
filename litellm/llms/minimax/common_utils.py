"""
Shared utilities for the MiniMax provider.
"""

from typing import Set

# Model identifiers for Codex MiniMax models that need restricted
# OpenAI parameter handling through LiteLLM's Responses-to-chat bridge.
CODEX_MINIMAX_MODELS: Set[str] = {"codex-minimax-m2.7"}


def is_codex_minimax_model(model: str) -> bool:
    """
    Check whether `model` is a Codex MiniMax model that requires
    restricted parameter handling.

    Accepts both bare names ("codex-minimax-m2.7") and provider-prefixed
    names ("minimax/codex-minimax-m2.7").
    """
    normalized = model.split("/", 1)[-1].lower()
    return normalized in CODEX_MINIMAX_MODELS
