# Custom LiteLLM Build

This branch contains a custom LiteLLM Proxy build for a Codex workflow with:

- `chatgpt/gpt-5.5` as the planning model through native ChatGPT Responses handling.
- `minimax/codex-minimax-m2.7` as the implementation model through LiteLLM's Responses-to-chat bridge.

The MiniMax adapter is intentionally narrow. It preserves function tools for agent turns and rejects Responses features that MiniMax does not support.

## Build

Install the local development environment:

```bash
make install-proxy-dev
```

Or use `uv` directly if the environment is already bootstrapped:

```bash
uv sync --group proxy-dev --extra proxy
```

Build a local Docker image from this checkout:

```bash
docker build -t litellm-codex-minimax:local .
```

## Configure

Set the MiniMax API key before starting the proxy:

```bash
export MINIMAX_API_KEY="YOUR_MINIMAX_API_KEY"
```

The `chatgpt/*` provider uses OpenAI OAuth through LiteLLM's ChatGPT authenticator, not an `OPENAI_API_KEY`. On first use it may print a device-code login prompt and write tokens to `~/.config/litellm/chatgpt/auth.json` by default. You can override that location:

```bash
export CHATGPT_TOKEN_DIR="$HOME/.config/litellm/chatgpt"
```

### OpenAI OAuth for `chatgpt/gpt-5.5`

Start the proxy normally:

```bash
uv run litellm --config custom_config.yaml --port 4000
```

Trigger the ChatGPT provider once, for example with a health check:

```bash
curl http://localhost:4000/health
```

If no valid token cache exists, the proxy logs will print a device-code flow:

```text
Sign in with ChatGPT using device code:
1) Visit https://auth.openai.com/codex/device
2) Enter code: <device-code>
Device codes are a common phishing target. Never share this code.
```

Open the URL, enter the code, and complete the OpenAI login in your browser. The proxy stores OAuth tokens in:

```text
~/.config/litellm/chatgpt/auth.json
```

Do not commit or share this file. It contains OAuth credentials. When the access token expires, LiteLLM refreshes it with the stored refresh token. If refresh fails, delete the token cache or restart the flow and sign in again.

For a custom token cache location, set both the environment variable and the Docker mount consistently:

```bash
export CHATGPT_TOKEN_DIR="$PWD/.local-chatgpt-token-cache"
mkdir -p "$CHATGPT_TOKEN_DIR"
```

This checkout includes a recommended `custom_config.yaml` at the repo root:

```yaml
model_list:
  - model_name: gpt-5.5
    litellm_params:
      model: chatgpt/gpt-5.5

  - model_name: codex-planner
    litellm_params:
      model: chatgpt/gpt-5.5

  - model_name: codex-implementer
    litellm_params:
      model: minimax/codex-minimax-m2.7
      api_key: os.environ/MINIMAX_API_KEY

  - model_name: codex-minimax-m2.7
    litellm_params:
      model: minimax/codex-minimax-m2.7
      api_key: os.environ/MINIMAX_API_KEY

  - model_name: minimax/codex-minimax-m2.7
    litellm_params:
      model: minimax/codex-minimax-m2.7
      api_key: os.environ/MINIMAX_API_KEY

litellm_settings:
  drop_params: true
```

Use `codex-planner` for planning turns that need native Responses features. The `gpt-5.5` entry is a compatibility alias for clients that still send the raw model name. Use `codex-implementer` for implementation turns that only require chat-style text plus function tools. The `codex-minimax-m2.7` and `minimax/codex-minimax-m2.7` entries are compatibility aliases for clients that send the raw MiniMax model name.

MiniMax does not support web search. If a Codex sub-agent uses `codex-implementer`, disable web search in that agent config:

```toml
model = "codex-implementer"
model_provider = "litellm-custom"
web_search = "disabled"
```

## Run

Run the proxy from the checkout:

```bash
uv run litellm --config custom_config.yaml --port 4000
```

Wait for the health endpoint:

```bash
curl http://localhost:4000/health
```

Run the Docker image:

```bash
docker run --rm -p 4000:4000 \
  -e MINIMAX_API_KEY="$MINIMAX_API_KEY" \
  -e CHATGPT_TOKEN_DIR=/root/.config/litellm/chatgpt \
  -v "$PWD/custom_config.yaml:/app/custom_config.yaml:ro" \
  litellm-codex-minimax:local \
  --config /app/custom_config.yaml --port 4000
```

If you want Docker to reuse an existing ChatGPT token cache, mount it too:

```bash
docker run --rm -p 4000:4000 \
  -e MINIMAX_API_KEY="$MINIMAX_API_KEY" \
  -e CHATGPT_TOKEN_DIR=/root/.config/litellm/chatgpt \
  -v "$PWD/custom_config.yaml:/app/custom_config.yaml:ro" \
  -v "$HOME/.config/litellm/chatgpt:/root/.config/litellm/chatgpt" \
  litellm-codex-minimax:local \
  --config /app/custom_config.yaml --port 4000
```

For first-time OAuth in Docker, run the container in an interactive terminal and watch the logs for the device code. Complete login in your browser, then keep the mounted token directory for future runs.

## Supported Parameters

For `minimax/codex-minimax-m2.7`, the chat adapter supports:

- `frequency_penalty`
- `max_tokens`
- `presence_penalty`
- `seed`
- `stop`
- `stream`
- `temperature`
- `top_p`
- `tools`
- `tool_choice`
- `extra_headers`
- `reasoning_effort`
- `thinking`
- `reasoning_split`

Through `/v1/responses`, use the Responses names where applicable:

- `input`
- `instructions`
- `max_output_tokens`
- `reasoning`
- `stream`
- `temperature`
- `top_p`
- `tools`
- `tool_choice`

MiniMax tool support is function-only:

```json
{
  "type": "function",
  "name": "shell",
  "description": "Run a shell command",
  "parameters": {
    "type": "object",
    "properties": {
      "cmd": { "type": "string" }
    },
    "required": ["cmd"]
  }
}
```

Supported `tool_choice` values are `auto`, `none`, or a named function choice:

```json
{
  "type": "function",
  "function": { "name": "shell" }
}
```

The MiniMax adapter drops unsupported request fields before dispatch when `drop_params` is enabled, including:

- `parallel_tool_calls`
- `web_search_options`
- `stream_options`
- `context_management`
- `metadata`
- `service_tier`

The adapter rejects unsupported tool shapes instead of forwarding them to MiniMax. Use `chatgpt/gpt-5.5` for richer Responses features such as MCP tools, namespace tools, web search, image generation, or other non-function tool types.

## Smoke Test

Start the proxy, then send a Responses request to the ChatGPT planning alias. ChatGPT expects list-shaped Responses input:

```bash
curl http://localhost:4000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "codex-planner",
    "input": [
      {
        "role": "user",
        "content": [
          { "type": "input_text", "text": "Reply with the single word ok." }
        ]
      }
    ],
    "max_output_tokens": 16,
    "temperature": 0.2
  }'
```

Then send a Responses request to the MiniMax implementation alias:

```bash
curl http://localhost:4000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "codex-implementer",
    "input": "Say hello in one sentence.",
    "max_output_tokens": 128,
    "temperature": 0.2
  }'
```

For tool-heavy planning or non-function Responses tools, route the request to `codex-planner` instead.
