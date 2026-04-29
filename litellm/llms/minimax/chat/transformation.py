"""
MiniMax OpenAI transformation config - extends OpenAI chat config for MiniMax's OpenAI-compatible API
"""

from typing import Any, Dict, List, Optional, Tuple, cast

import litellm
from litellm.llms.minimax.common_utils import is_codex_minimax_model
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam


class MinimaxChatConfig(OpenAIGPTConfig):
    """
    MiniMax OpenAI configuration that extends OpenAIGPTConfig.
    MiniMax provides an OpenAI-compatible API at:
    - International: https://api.minimax.io/v1
    - China: https://api.minimaxi.com/v1

    Supported models:
    - MiniMax-M2.1
    - MiniMax-M2.1-lightning
    - MiniMax-M2
    """

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """
        Get MiniMax API key from environment or parameters.
        """
        return api_key or get_secret_str("MINIMAX_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(
        api_base: Optional[str] = None,
    ) -> str:
        """
        Get MiniMax API base URL.
        Defaults to international endpoint: https://api.minimax.io/v1
        For China, set to: https://api.minimaxi.com/v1
        """
        return (
            api_base
            or get_secret_str("MINIMAX_API_BASE")
            or "https://api.minimax.io/v1"
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for MiniMax OpenAI API.
        Override to ensure we use MiniMax's endpoint.
        """
        # Get the base URL (either provided or default MiniMax endpoint)
        base_url = self.get_api_base(api_base=api_base)

        # Ensure it ends with /chat/completions
        if base_url.endswith("/chat/completions"):
            return base_url
        elif base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        elif base_url.endswith("/"):
            return f"{base_url}v1/chat/completions"
        else:
            return f"{base_url}/v1/chat/completions"

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: List[AllMessageValues],
        tools: Optional[List[ChatCompletionToolParam]] = None,
    ) -> Tuple[List[AllMessageValues], Optional[List[ChatCompletionToolParam]]]:
        """
        Override to preserve cache_control for MiniMax.
        MiniMax supports cache_control - don't strip it.
        """
        # MiniMax supports cache_control, so return messages and tools unchanged
        return messages, tools

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        if is_codex_minimax_model(model):
            messages = self._merge_codex_minimax_system_messages(messages)

        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    @staticmethod
    def _merge_codex_minimax_system_messages(
        messages: List[AllMessageValues],
    ) -> List[AllMessageValues]:
        system_message_indices = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "system"
        ]
        if len(system_message_indices) <= 1:
            return messages

        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            convert_content_list_to_str,
        )

        merged_system_content = "\n\n".join(
            convert_content_list_to_str(messages[index])
            for index in system_message_indices
        )
        first_system_index = system_message_indices[0]
        merged_system_message = cast(
            AllMessageValues, dict(messages[first_system_index])
        )
        merged_system_message["content"] = merged_system_content

        merged_messages: List[AllMessageValues] = []
        for index, message in enumerate(messages):
            if index == first_system_index:
                merged_messages.append(merged_system_message)
            elif message.get("role") != "system":
                merged_messages.append(message)

        return merged_messages

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported OpenAI parameters for MiniMax.

        Codex MiniMax M2.7 is intentionally narrow because it is used through
        LiteLLM's Responses-to-chat bridge and MiniMax rejects several
        OpenAI/Codex request fields.
        """
        if is_codex_minimax_model(model):
            supported_params = [
                "frequency_penalty",
                "max_tokens",
                "presence_penalty",
                "seed",
                "stop",
                "stream",
                "stream_options",
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

        if is_codex_minimax_model(model):
            unsupported_params = {
                "parallel_tool_calls",
                "web_search_options",
                "context_management",
                "metadata",
                "service_tier",
            }
            for param in unsupported_params:
                optional_params.pop(param, None)

            self._validate_codex_minimax_tools(
                tools=optional_params.get("tools"),
                model=model,
            )
            self._validate_codex_minimax_tool_choice(
                tool_choice=optional_params.get("tool_choice"),
                model=model,
            )

        return optional_params

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
            function = tool.get("function")
            if not isinstance(function, dict) or not function.get("name"):
                self._raise_codex_minimax_bad_request(
                    message=(
                        f"Unsupported function tool for minimax/{model}. "
                        "MiniMax Codex V1 requires OpenAI chat function tools with function.name."
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
            if tool_choice in {"auto", "none", "required"}:
                return
            self._raise_codex_minimax_bad_request(
                message=(
                    f"Unsupported tool_choice '{tool_choice}' for minimax/{model}. "
                    "MiniMax Codex V1 supports function tools only. "
                    "Use GPT models for richer Responses features."
                ),
                model=model,
            )
        elif isinstance(tool_choice, dict):
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
        else:
            self._raise_codex_minimax_bad_request(
                message=(
                    f"Unsupported tool_choice for minimax/{model}. "
                    "MiniMax Codex V1 supports function tools only. "
                    "Use GPT models for richer Responses features."
                ),
                model=model,
            )
