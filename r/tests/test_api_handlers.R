# Load the API handlers
print("Loading API handlers...")
source("src/api_handlers_r.R")
print("API handlers loaded successfully")


# Set your API keys
api_keys <- list(
  openai = "YOUR_OPENAI_API_KEY",
  anthropic = "YOUR_ANTHROPIC_API_KEY"
)

# Test text cleaning
print("Testing text cleaning...")
original_text <- "Product   description with    irregular spacing."
cleaned_text <- clean_text(original_text)
print(paste("Original:", original_text))
print(paste("Cleaned:", cleaned_text))
print("Text cleaning test complete")

# Test cost calculation for all models
print("\nTesting cost calculation...")
test_prompt <- "Please classify if this product benefits society beyond its consumers. Answer only with 0 or 1: Solar-powered water purifier for communities."
models <- c("gpt4o", "gpt4o_mini", "claude_opus", "claude_sonnet", "claude_haiku")
for (model in models) {
  cost <- calculate_api_cost(test_prompt, model)
  print(paste("Estimated", model, "cost: $", sprintf("%.6f", cost)))
}
print("Cost calculation test complete")

# Check if run with --test-api flag
args <- commandArgs(trailingOnly = TRUE)
test_api <- "--test-api" %in% args

if (!test_api) {
  print("\nSkipping API calls. Run with --test-api flag to test API calls.")
  quit()
}

# Test OpenAI API calls
print("\n=== Testing OpenAI API Calls ===")

print("\nTesting GPT-4o API call...")
tryCatch({
  result <- call_gpt4o(test_prompt, api_keys$openai)
  print(paste("GPT-4o result:", result))
}, error = function(e) {
  print(paste("GPT-4o API error:", e$message))
})

print("\nTesting GPT-4o-mini API call...")
tryCatch({
  result <- call_gpt4o_mini(test_prompt, api_keys$openai)
  print(paste("GPT-4o-mini result:", result))
}, error = function(e) {
  print(paste("GPT-4o-mini API error:", e$message))
})

# Test Anthropic API calls
print("\n=== Testing Anthropic API Calls ===")

print("\nTesting Claude Opus API call...")
tryCatch({
  result <- call_claude_opus(test_prompt, api_keys$anthropic)
  print(paste("Claude Opus result:", result))
}, error = function(e) {
  print(paste("Claude Opus API error:", e$message))
})

print("\nTesting Claude Sonnet API call...")
tryCatch({
  result <- call_claude_sonnet(test_prompt, api_keys$anthropic)
  print(paste("Claude Sonnet result:", result))
}, error = function(e) {
  print(paste("Claude Sonnet API error:", e$message))
})

print("\nTesting Claude Haiku API call...")
tryCatch({
  result <- call_claude_haiku(test_prompt, api_keys$anthropic)
  print(paste("Claude Haiku result:", result))
}, error = function(e) {
  print(paste("Claude Haiku API error:", e$message))
})

print("\nAll tests completed")