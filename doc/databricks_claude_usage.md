# Using Databricks Claude 3.7 with SWE-bench Agent

This document provides instructions for using the SWE-bench agent with Claude 3.7 through Databricks' Mosaic AI Model Serving platform.

## Prerequisites

Before using the Databricks Claude integration, ensure you have:

1. A Databricks workspace with access to Claude 3.7 Sonnet
2. A Databricks Personal Access Token with appropriate permissions
3. A serving endpoint for Claude 3.7 in your Databricks workspace

## Setup

### 1. Install Dependencies

First, install the required dependencies:

```bash
pip install -e .
```

This will install all dependencies, including the Databricks SDK.

### 2. Set Environment Variables

Set the following environment variables:

```bash
# Databricks credentials
export DATABRICKS_HOST="https://your-databricks-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your-databricks-personal-access-token"
```

### 3. Run the Agent with Databricks Claude

Run the agent with the `--client-type` argument set to `databricks-claude`:

```bash
python cli.py --client-type databricks-claude
```

By default, the agent will use the endpoint named `claude-3-7-sonnet`. If your endpoint has a different name, specify it with the `--endpoint-name` argument:

```bash
python cli.py --client-type databricks-claude --endpoint-name your-endpoint-name
```

## Command-Line Arguments

The following command-line arguments are available for configuring the Databricks Claude integration:

- `--client-type`: Type of LLM client to use. Options are `anthropic-direct` (default), `databricks-claude`, or `openai-direct`.
- `--endpoint-name`: Name of the Databricks serving endpoint (only used with `databricks-claude` client). Default is `claude-3-7-sonnet`.

## Databricks Endpoint Configuration

To use Claude 3.7 with Databricks, you need to set up a serving endpoint in your Databricks workspace:

1. In your Databricks workspace, navigate to "Machine Learning" > "Serving" > "Model Serving"
2. Click "Create Serving Endpoint"
3. Select "Foundation Model" as the endpoint type
4. Choose "Claude 3.7 Sonnet" as the model
5. Configure the endpoint settings as needed
6. Click "Create"

Once the endpoint is created, note the endpoint name to use with the `--endpoint-name` argument.

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   - Ensure `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables are correctly set
   - Verify that your token has the necessary permissions

2. **Endpoint Not Found**:
   - Confirm the endpoint name matches your Databricks configuration
   - Check if the endpoint is active in your Databricks workspace

3. **Response Format Issues**:
   - If you encounter unexpected response format errors, check the Databricks endpoint configuration
   - Ensure the endpoint is configured to return responses in the expected format

### Fallback to Direct Anthropic API

If you encounter issues with the Databricks integration, you can easily revert to using the direct Anthropic API:

```bash
python cli.py --client-type anthropic-direct
```

## Advanced Configuration

### Thinking Mode

The Databricks Claude client supports Claude's thinking mode. To enable it, you can modify the `client_kwargs` in `cli.py` to include the `thinking_tokens` parameter:

```python
client_kwargs = {
    "model_name": "claude-3-7-sonnet-20250219",
    "use_caching": True,
    "thinking_tokens": 8192,  # Enable thinking mode with 8192 tokens
}
```

### Custom Endpoint Configuration

If you need to customize the endpoint configuration, you can modify the `DatabricksClaudeClient` class in `utils/llm_client.py`.

## References

- [Databricks SDK for Python Documentation](https://docs.databricks.com/dev-tools/sdk/python.html)
- [Databricks Model Serving Documentation](https://docs.databricks.com/machine-learning/model-serving/index.html)
- [Claude 3.7 Documentation](https://docs.anthropic.com/claude/docs/claude-3-7-models)
- [Databricks Claude Integration Guide](databricks_claude_integration_guide.md)
