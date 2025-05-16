# Databricks Claude 3.7 Integration Guide

This document provides a guide for replacing the direct Anthropic Claude 3.7 integration with Databricks Claude 3.7 in the SWE-bench agent.

## Overview

The SWE-bench agent currently uses Claude 3.7 Sonnet directly through the Anthropic API. This guide outlines the steps to modify the agent to use Claude 3.7 through Databricks' Mosaic AI Model Serving platform instead.

## Why Use Databricks Claude 3.7?

- **Simplified Management**: Databricks provides a managed environment for AI models
- **Integrated Security**: Leverage Databricks' security features
- **Cost Management**: Potential cost benefits through Databricks' pricing model
- **Unified Platform**: Integrate with other Databricks services

## Prerequisites

Before implementing the changes, ensure you have:

1. A Databricks workspace with access to Claude 3.7 Sonnet
2. Databricks Personal Access Token with appropriate permissions
3. The endpoint name for Claude 3.7 in your Databricks workspace

## Implementation Steps

### 1. Add Dependencies

Add the Databricks SDK to the project dependencies in `pyproject.toml`:

```toml
[project]
# ... existing dependencies
dependencies = [
    # ... existing dependencies
    "databricks-sdk>=0.7.0",
]
```

### 2. Create a New Client Class

Create a new `DatabricksClaudeClient` class in `utils/llm_client.py` that implements the `LLMClient` interface:

```python
class DatabricksClaudeClient(LLMClient):
    """Use Claude models via Databricks API."""

    def __init__(
        self,
        model_name="claude-3-7-sonnet-20250219",
        max_retries=2,
        use_caching=True,
        use_low_qos_server: bool = False,
        thinking_tokens: int = 0,
        endpoint_name: str = "claude-3-7-sonnet",
    ):
        """Initialize the Databricks Claude client.
        
        Args:
            model_name: The model name to use
            max_retries: Maximum number of retries for API calls
            use_caching: Whether to use prompt caching
            use_low_qos_server: Whether to use a low QoS server
            thinking_tokens: Number of tokens to allocate for thinking
            endpoint_name: The name of the Databricks serving endpoint
        """
        # Get Databricks credentials from environment variables
        self.databricks_host = os.getenv("DATABRICKS_HOST")
        self.databricks_token = os.getenv("DATABRICKS_TOKEN")
        
        if not self.databricks_host or not self.databricks_token:
            raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN environment variables must be set")
        
        # Initialize the Databricks client
        from databricks.sdk import WorkspaceClient
        self.client = WorkspaceClient(
            host=self.databricks_host,
            token=self.databricks_token
        )
        
        self.model_name = model_name
        self.max_retries = max_retries
        self.use_caching = use_caching
        self.thinking_tokens = thinking_tokens
        self.endpoint_name = endpoint_name
        
    def generate(
        self,
        messages: List[Union[TextPrompt, ToolUsePrompt, ToolResultPrompt]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Tuple[List[Union[TextPrompt, ToolUsePrompt]], Dict[str, Any]]:
        """Generate a response from the model."""
        # Convert Augment messages to Anthropic format
        anthropic_messages = []
        for message in messages:
            if isinstance(message, TextPrompt):
                anthropic_messages.append(
                    {"role": message.role, "content": message.text}
                )
            elif isinstance(message, ToolUsePrompt):
                anthropic_messages.append(
                    {
                        "role": message.role,
                        "content": "",
                        "tool_calls": [
                            {
                                "id": message.tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": message.name,
                                    "arguments": json.dumps(message.arguments),
                                },
                            }
                        ],
                    }
                )
            elif isinstance(message, ToolResultPrompt):
                anthropic_messages.append(
                    {
                        "role": "tool",
                        "content": message.content,
                        "tool_call_id": message.tool_call_id,
                    }
                )
            else:
                raise ValueError(f"Unknown message type: {type(message)}")

        # Prepare tool parameters if tools are provided
        tool_params = None
        tool_choice_param = None
        if tools:
            tool_params = tools
            if tool_choice:
                tool_choice_param = tool_choice

        # Extra headers for prompt caching if enabled
        extra_headers = {}
        if self.use_caching:
            extra_headers = {"anthropic-beta": "prompt-caching-2024-07-31"}

        # Extra body parameters for thinking tokens if specified
        extra_body = {}
        if self.thinking_tokens > 0:
            extra_body = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": self.thinking_tokens,
                }
            }

        response = None
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        if tool_params:
            payload["tools"] = tool_params
            
        if tool_choice_param:
            payload["tool_choice"] = tool_choice_param
            
        # Add extra body parameters if any
        payload.update(extra_body)

        # Make the API request with retries
        for retry in range(self.max_retries):
            try:
                # Call Databricks serving endpoint
                response = self.client.serving_endpoints_data_plane.query(
                    endpoint_name=self.endpoint_name,
                    inputs=[payload]
                )
                break
            except Exception as e:
                if retry == self.max_retries - 1:
                    print(f"Failed Databricks request after {retry + 1} retries")
                    raise e
                else:
                    print(f"Retrying LLM request: {retry + 1}/{self.max_retries}")
                    # Sleep 4-6 seconds with jitter to avoid thundering herd
                    time.sleep(5 * random.uniform(0.8, 1.2))

        # Convert response back to Augment format
        augment_messages = []
        assert response is not None
        
        # Extract the actual response content from Databricks response
        # The exact structure will depend on how Databricks formats the response
        content = response.predictions[0]
        
        # Parse the content based on its type
        if isinstance(content, dict) and "content" in content:
            # Handle text response
            augment_messages.append(TextPrompt(role="assistant", text=content["content"]))
        elif isinstance(content, dict) and "tool_calls" in content:
            # Handle tool use response
            for tool_call in content["tool_calls"]:
                augment_messages.append(
                    ToolUsePrompt(
                        role="assistant",
                        name=tool_call["function"]["name"],
                        arguments=json.loads(tool_call["function"]["arguments"]),
                        tool_call_id=tool_call["id"],
                    )
                )
        else:
            # Fallback for unexpected response format
            warning_msg = "\n".join(
                ["!" * 80, "WARNING: Unexpected response format", "!" * 80]
            )
            print(warning_msg)
            augment_messages.append(TextPrompt(role="assistant", text=str(content)))

        # Create metadata from response
        # This would need to be adjusted based on actual Databricks response format
        message_metadata = {
            "raw_response": response,
            "input_tokens": response.metadata.get("input_tokens", 0),
            "output_tokens": response.metadata.get("output_tokens", 0),
        }

        return augment_messages, message_metadata
```

### 3. Update the `get_client` Function

Update the `get_client` function in `utils/llm_client.py` to support the new client type:

```python
def get_client(client_name: str, **kwargs) -> LLMClient:
    """Get a client for a given client name."""
    if client_name == "anthropic-direct":
        return AnthropicDirectClient(**kwargs)
    elif client_name == "openai-direct":
        return OpenAIDirectClient(**kwargs)
    elif client_name == "databricks-claude":
        return DatabricksClaudeClient(**kwargs)
    else:
        raise ValueError(f"Unknown client name: {client_name}")
```

### 4. Update the Client Initialization in `cli.py`

Update the client initialization in `cli.py`:

```python
# Initialize LLM client
client = get_client(
    "databricks-claude",
    model_name="claude-3-7-sonnet-20250219",
    use_caching=True,
    endpoint_name="claude-3-7-sonnet",  # Adjust based on your Databricks endpoint name
)
```

## Configuration

### Environment Variables

Set the following environment variables:

```bash
# Databricks credentials
export DATABRICKS_HOST="https://your-databricks-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your-databricks-personal-access-token"

# Existing environment variables
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Keep for backward compatibility
export OPENAI_API_KEY="your-openai-api-key"  # For ensembler
```

### Databricks Endpoint Configuration

1. In your Databricks workspace, ensure you have a serving endpoint for Claude 3.7 Sonnet
2. Note the endpoint name (e.g., "claude-3-7-sonnet")
3. Use this name in the `endpoint_name` parameter when initializing the client

## Testing

After implementation, test the integration:

1. Run a simple query to verify the connection works
2. Test with tool use to ensure tool calls are properly handled
3. Verify token counting and metadata are correctly reported

## Troubleshooting

Common issues and solutions:

1. **Authentication Errors**: Verify DATABRICKS_HOST and DATABRICKS_TOKEN are correctly set
2. **Endpoint Not Found**: Confirm the endpoint name matches your Databricks configuration
3. **Response Format Issues**: Adjust the response parsing logic based on actual Databricks response format

## Fallback Strategy

If issues arise with the Databricks integration, you can easily revert to the direct Anthropic API by changing the client name back to "anthropic-direct" in `cli.py`.

## References

- [Databricks SDK for Python Documentation](https://docs.databricks.com/dev-tools/sdk/python.html)
- [Databricks Model Serving Documentation](https://docs.databricks.com/machine-learning/model-serving/index.html)
- [Claude 3.7 Documentation](https://docs.anthropic.com/claude/docs/claude-3-7-models)
