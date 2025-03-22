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

Makes calls to Anthropic's Claude models and extracts binary (0/1) responses.

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

## Usage Examples

### R Example

```r
source("api_handlers.R")

# Set API keys
api_keys <- list(
  openai = "your-openai-key",
  anthropic = "your-anthropic-key"
)

# Clean text
text <- "Product description with irregular spacing."
clean_text <- clean_text(text)

# Call different models
prompt <- paste("Classify if this product benefits society. Answer 0 or 1:", clean_text)

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
from api_handlers import clean_text, call_gpt4o, call_claude_sonnet, calculate_api_cost

# Set API keys
api_keys = {
    "openai": "your-openai-key",
    "anthropic": "your-anthropic-key"
}

# Clean text
text = "Product description with irregular spacing."
clean_text = clean_text(text)

# Call different models
prompt = f"Classify if this product benefits society. Answer 0 or 1: {clean_text}"

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
