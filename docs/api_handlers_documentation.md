# API Handlers

The API handlers provide functions for interacting with different LLM providers through their APIs. These functions handle:

- API authentication
- Request formatting
- Error handling and retries
- Rate limiting
- Response parsing
- Cost estimation

## Important Note on Binary Classification

All API functions in this framework are specifically configured for binary classification tasks that return 0 or 1:

- The system prompt is set to: `"You are a helpful assistant that responds with only 0 or 1."`
- Response parsing is designed to extract the first 0 or 1 from the model's output
- All prompt templates should end with instructions to answer with only 0 or 1

If you need different output formats (multi-class, text generation, etc.), you'll need to:
1. Modify the system prompt in the API handler functions
2. Update the response parsing logic in the functions
3. Adjust your prompt templates accordingly

For example, to modify for multi-class classification, you would need to update both the system prompt and the response parsing logic in the API handler functions.

## Core Functions

### Text Cleaning

```
clean_text(text)
```

Cleans and normalizes text before sending to APIs:
- Removes non-ASCII characters
- Normalizes whitespace
- Trims leading/trailing spaces

**Parameters:**
- `text`: Text string to clean

**Returns:**
- Cleaned text string

### OpenAI API Calls

```
call_gpt4o(prompt, api_key, model_name="gpt-4o")
call_gpt4o_mini(prompt, api_key, model_name="gpt-4o-mini")
```

Makes calls to OpenAI models and extracts binary (0/1) responses.

**Parameters:**
- `prompt`: Full text prompt to send to the model
- `api_key`: OpenAI API key
- `model_name`: Specific model version (default provides the latest version)

**Returns:**
- `0` or `1` for successful classification
- `NULL`/`None` if error occurs

### Anthropic API Calls

```
call_claude_opus(prompt, api_key)
call_claude_sonnet(prompt, api_key)
call_claude_haiku(prompt, api_key)
```

Makes calls to Anthropic's Claude models and extracts binary (0/1) responses. The implementations use these specific model versions:
- Claude Opus: `claude-3-opus-20240229`
- Claude Sonnet: `claude-3-5-sonnet-20241022`
- Claude Haiku: `claude-3-5-haiku-20241022`

**Parameters:**
- `prompt`: Full text prompt to send to the model
- `api_key`: Anthropic API key

**Returns:**
- `0` or `1` for successful classification
- `NULL`/`None` if error occurs

### Other API Calls

```
call_deepseek_chat(prompt, api_key)
```

Makes calls to other LLM providers.

**Parameters:**
- `prompt`: Full text prompt to send to the model
- `api_key`: Provider's API key

**Returns:**
- `0` or `1` for successful classification
- `NULL`/`None` if error occurs

### Cost Estimation

```
calculate_api_cost(prompt, model, n_requests=1)
```

Estimates the cost of API calls based on prompt length and model pricing.

**Parameters:**
- `prompt`: Example prompt to estimate token count
- `model`: Model identifier (e.g., "gpt4o", "claude_opus", "claude_sonnet")
- `n_requests`: Number of requests to estimate cost for (default: 1)

**Returns:**
- Estimated cost in USD

## Model Cost Comparison

| Model          | Input Cost (per 1M tokens) | Output Cost (per 1M tokens) |
|----------------|----------------------------|----------------------------|
| GPT-4o         | $2.50                      | $10.00                     |
| GPT-4o-mini    | $0.15                      | $0.60                      |
| Claude Opus    | $1.50                      | $7.50                      |
| Claude Sonnet  | $0.80                      | $4.00                      |
| Claude Haiku   | $0.25                      | $1.25                      |
| Deepseek Chat  | $0.27                      | $1.10                      |

## Implementation Details

### Request Format

Each API requires specific request formatting:

- **OpenAI**: Uses the chat completions API with system and user messages
- **Anthropic**: Uses the messages API with system prompt and user message
- **Others**: Vary by provider

### Rate Limiting

Each function implements basic rate limiting to avoid API throttling:

- Sleep between requests (varies by provider and model size)
- Exponential backoff for retries on rate limit errors

### Error Handling

Functions handle common errors:
- Rate limiting (429 errors)
- Authentication errors
- Server errors
- Parsing errors

### Response Parsing

All functions extract a binary (0/1) response from the model's text output:
- Uses regex pattern matching to find the first 0 or 1 in the response
- Returns NULL/None if no valid response found

## Testing

The framework includes test scripts for both Python and R implementations:

### Python Test
```bash
# Run from project root
python python/tests/test_api_handlers.py
```

### R Test
```bash
# Run from project root 
Rscript r/tests/test_api_handlers.R
# To test API calls, add the --test-api flag
Rscript r/tests/test_api_handlers.R --test-api
```

## Usage Examples

### R Example

```r
# Load the API handlers
source("r/src/api_handlers_r.R")

# Set API keys
api_keys <- list(
  openai = "your-openai-key",
  anthropic = "your-anthropic-key"
)

# Clean text
text <- "Product description with irregular spacing."
cleaned_text <- clean_text(text)

# Call different models
prompt <- paste("Classify if this product benefits society. Answer 0 or 1:", cleaned_text)

# OpenAI models
gpt4o_result <- call_gpt4o(prompt, api_keys$openai)
gpt4o_mini_result <- call_gpt4o_mini(prompt, api_keys$openai)

# Anthropic models
claude_opus_result <- call_claude_opus(prompt, api_keys$anthropic)
claude_sonnet_result <- call_claude_sonnet(prompt, api_keys$anthropic)
claude_haiku_result <- call_claude_haiku(prompt, api_keys$anthropic)

# Estimate costs
gpt4o_cost <- calculate_api_cost(prompt, "gpt4o")
claude_opus_cost <- calculate_api_cost(prompt, "claude_opus")
```

### Python Example

```python
# Import the API handlers
from python.src.api_handlers_python import (
    clean_text, call_gpt4o, call_gpt4o_mini, 
    call_claude_opus, call_claude_sonnet, call_claude_haiku,
    calculate_api_cost
)

# Set API keys
api_keys = {
    "openai": "your-openai-key",
    "anthropic": "your-anthropic-key"
}

# Clean text
original_text = "Product description with irregular spacing."
cleaned_text = clean_text(original_text)

# Call different models
prompt = f"Classify if this product benefits society. Answer 0 or 1: {cleaned_text}"

# OpenAI models
gpt4o_result = call_gpt4o(prompt, api_keys["openai"])
gpt4o_mini_result = call_gpt4o_mini(prompt, api_keys["openai"])

# Anthropic models
claude_opus_result = call_claude_opus(prompt, api_keys["anthropic"])
claude_sonnet_result = call_claude_sonnet(prompt, api_keys["anthropic"])
claude_haiku_result = call_claude_haiku(prompt, api_keys["anthropic"])

# Estimate costs
gpt4o_cost = calculate_api_cost(prompt, "gpt4o")
claude_opus_cost = calculate_api_cost(prompt, "claude_opus")
```

## Best Practices

1. **Error Handling**: Always check for NULL/None responses
2. **Rate Limiting**: Adjust sleep times based on your API usage limits
3. **Cost Management**: Use cost estimation before running large batches
4. **Prompt Construction**: Construct prompts carefully to ensure proper binary responses
5. **API Keys**: Store API keys securely, never hardcode them in scripts
6. **Model Selection**: Choose models based on tradeoffs between accuracy, speed, and cost

## API Versions

This implementation is compatible with:
- OpenAI API v1 (as of March 2025)
- Anthropic API 2023-06-01
- Deepseek API (as of March 2025)

API specifications may change over time, so check the respective provider documentation for updates.
