"""
Test MiniMax tool-call adjacency enforcement.

MiniMax requires that tool result messages immediately follow the assistant
message that issued the tool calls. An intervening assistant(text) message
between assistant(tool_calls) and tool(result) messages causes:

    "invalid params, tool call result does not follow tool call (2013)"

These tests verify that MinimaxChatConfig.transform_request reorders messages
so that tool results are always adjacent to their corresponding tool_calls.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../"))

from litellm.llms.minimax.chat.transformation import MinimaxChatConfig


CODEX_MODEL = "codex-minimax-m2.7"
STANDARD_MODEL = "MiniMax-M2.1"


class TestMinimaxToolCallAdjacency:
    """Tests for ensuring tool results immediately follow assistant tool_calls."""

    def test_assistant_text_between_tool_calls_and_results_is_reordered(self):
        """
        Reproduces the exact failure from production:

        Input ordering:
            assistant(tool_calls=[call_1, call_2])
            assistant(text="here is some commentary")
            tool(tool_call_id=call_1)
            tool(tool_call_id=call_2)

        Expected output ordering:
            assistant(text="here is some commentary")
            assistant(tool_calls=[call_1, call_2])
            tool(tool_call_id=call_1)
            tool(tool_call_id=call_2)

        Or alternatively, the text can be merged into the tool_calls message.
        The key invariant: tool results MUST immediately follow the assistant
        tool_calls message with no intervening messages.
        """
        config = MinimaxChatConfig()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Run two commands for me."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_function_abc_1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"command": "echo hello"}',
                        },
                    },
                    {
                        "id": "call_function_abc_2",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"command": "echo world"}',
                        },
                    },
                ],
            },
            # This assistant text message incorrectly sits between tool_calls and results
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Running both commands now."}],
            },
            {
                "role": "tool",
                "content": "hello\n",
                "tool_call_id": "call_function_abc_1",
            },
            {
                "role": "tool",
                "content": "world\n",
                "tool_call_id": "call_function_abc_2",
            },
        ]

        request = config.transform_request(
            model=CODEX_MODEL,
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        result_messages = request["messages"]

        # Find where the tool_calls assistant message is
        tool_calls_idx = None
        for i, msg in enumerate(result_messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls_idx = i
                break

        assert tool_calls_idx is not None, "assistant tool_calls message not found"

        # Verify that the messages immediately following the assistant(tool_calls)
        # are the tool result messages — no intervening assistant(text)
        next_idx = tool_calls_idx + 1
        assert next_idx < len(result_messages), "no messages after tool_calls"
        assert result_messages[next_idx].get("role") == "tool", (
            f"Expected tool message immediately after assistant(tool_calls), "
            f"got role={result_messages[next_idx].get('role')}"
        )

        # Verify both tool results follow consecutively
        assert next_idx + 1 < len(result_messages)
        assert result_messages[next_idx + 1].get("role") == "tool", (
            f"Expected second tool message, "
            f"got role={result_messages[next_idx + 1].get('role')}"
        )

        # Verify the assistant text message still exists somewhere in the output
        # The text content should be preserved (either as a separate message or merged)
        has_text = any(
            "Running both commands" in str(msg.get("content", ""))
            for msg in result_messages
        )
        assert has_text, "assistant text content was lost during reordering"

    def test_tool_results_already_adjacent_unchanged(self):
        """
        When tool results already immediately follow assistant(tool_calls),
        messages should remain unchanged (no unnecessary reordering).
        """
        config = MinimaxChatConfig()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "SF"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "content": "Sunny, 72F",
                "tool_call_id": "call_weather_1",
            },
        ]

        request = config.transform_request(
            model=CODEX_MODEL,
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        result_messages = request["messages"]

        # Messages should be in the same order since they're already correct
        assert result_messages[0]["role"] == "system"
        assert result_messages[1]["role"] == "user"
        assert result_messages[2]["role"] == "assistant"
        assert result_messages[2].get("tool_calls") is not None
        assert result_messages[3]["role"] == "tool"
        assert result_messages[3]["tool_call_id"] == "call_weather_1"

    def test_multiple_intervening_messages_between_tool_calls_and_results(self):
        """
        Even with multiple messages between tool_calls and results,
        tool results must end up immediately after their tool_calls message.
        """
        config = MinimaxChatConfig()

        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Do some work."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_xyz_1",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": '{"cmd": "ls"}',
                        },
                    },
                ],
            },
            # Intervening assistant text
            {
                "role": "assistant",
                "content": "Let me check the directory listing.",
            },
            # Tool result
            {
                "role": "tool",
                "content": "file1.txt\nfile2.txt",
                "tool_call_id": "call_xyz_1",
            },
        ]

        request = config.transform_request(
            model=CODEX_MODEL,
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        result_messages = request["messages"]

        # Find assistant with tool_calls
        tool_calls_idx = None
        for i, msg in enumerate(result_messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls_idx = i
                break

        assert tool_calls_idx is not None

        # Next message must be the tool result
        next_msg = result_messages[tool_calls_idx + 1]
        assert next_msg.get("role") == "tool", (
            f"Expected tool message after assistant(tool_calls), "
            f"got role={next_msg.get('role')}"
        )
        assert next_msg.get("tool_call_id") == "call_xyz_1"

    def test_standard_minimax_model_also_enforces_adjacency(self):
        """
        Tool-call adjacency should be enforced for all MiniMax models,
        not just codex models.
        """
        config = MinimaxChatConfig()

        messages = [
            {"role": "user", "content": "Get weather."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_std_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "Checking weather for you...",
            },
            {
                "role": "tool",
                "content": '{"temp": 65}',
                "tool_call_id": "call_std_1",
            },
        ]

        request = config.transform_request(
            model=STANDARD_MODEL,
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        result_messages = request["messages"]

        # Find the tool_calls message
        tool_calls_idx = None
        for i, msg in enumerate(result_messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls_idx = i
                break

        assert tool_calls_idx is not None
        assert result_messages[tool_calls_idx + 1].get("role") == "tool"
