### API Handler Functions for LLM Annotation Framework
### This file contains functions for calling different LLM providers through their APIs

library(stringr)
library(httr)
library(jsonlite)

#' Clean and normalize text for API submission
#'
#' @param text Text string to clean
#' @return Cleaned text string
clean_text <- function(text) {
  # Convert to ASCII, removing non-ASCII characters
  text <- iconv(text, to = "ASCII", sub = "")
  
  # Normalize whitespace (replace multiple spaces with single space)
  text <- gsub("\\s+", " ", text)
  
  # Trim whitespace
  text <- trimws(text)
  
  return(text)
}

#' Call the GPT-4o model via OpenAI API
#'
#' @param prompt Full text prompt to send to the model
#' @param api_key OpenAI API key
#' @param model_name Optional model name (default: "gpt-4o")
#' @return 0 or 1 for successful classification, NA if error occurs
call_gpt4o <- function(prompt, api_key, model_name = "gpt-4o") {
  # Basic rate limiting
  Sys.sleep(0.5)
  
  # Clean the prompt text
  clean_prompt <- clean_text(prompt)
  
  # Make API request
  response <- tryCatch({
    POST(
      url = "https://api.openai.com/v1/chat/completions",
      add_headers(
        Authorization = paste("Bearer", api_key),
        "Content-Type" = "application/json"
      ),
      body = toJSON(list(
        model = model_name,
        messages = list(
          list(
            role = "system",
            content = "You are a helpful assistant that responds with only 0 or 1."
          ),
          list(
            role = "user",
            content = clean_prompt
          )
        ),
        temperature = 0,
        max_tokens = 10
      ), auto_unbox = TRUE),
      encode = "raw"
    )
  }, error = function(e) {
    print(paste("API call error:", e$message))
    return(NULL)
  })
  
  # Handle rate limiting
  if (!is.null(response) && response$status_code == 429) {
    wait_time <- 60
    print(sprintf("Rate limit hit. Waiting %s seconds...", wait_time))
    Sys.sleep(wait_time)
    return(call_gpt4o(prompt, api_key))
  }
  
  # Check for other errors
  if (is.null(response) || response$status_code != 200) {
    print(paste("Error:", if(!is.null(response)) response$status_code else "NULL response"))
    return(NA)
  }
  
  # Parse response
  result <- tryCatch({
    content(response)$choices[[1]]$message$content
  }, error = function(e) {
    print(sprintf("Error parsing response: %s", e$message))
    return(NA)
  })
  
  # Extract binary response (0 or 1)
  clean_result <- tryCatch({
    as.numeric(str_extract(result, "[0-1]"))
  }, error = function(e) {
    print("Could not extract numeric response")
    return(NA)
  })
  
  return(clean_result)
}

#' Call the GPT-4o-mini model via OpenAI API
#'
#' @param prompt Full text prompt to send to the model
#' @param api_key OpenAI API key
#' @param model_name Optional model name (default: "gpt-4o-mini")
#' @return 0 or 1 for successful classification, NA if error occurs
call_gpt4o_mini <- function(prompt, api_key, model_name = "gpt-4o-mini") {
  # Basic rate limiting
  Sys.sleep(0.4)
  
  # Clean the prompt text
  clean_prompt <- clean_text(prompt)
  
  # Make API request
  response <- tryCatch({
    POST(
      url = "https://api.openai.com/v1/chat/completions",
      add_headers(
        Authorization = paste("Bearer", api_key),
        "Content-Type" = "application/json"
      ),
      body = toJSON(list(
        model = model_name,
        messages = list(
          list(
            role = "system",
            content = "You are a helpful assistant that responds with only 0 or 1."
          ),
          list(
            role = "user",
            content = clean_prompt
          )
        ),
        temperature = 0,
        max_tokens = 10
      ), auto_unbox = TRUE),
      encode = "raw"
    )
  }, error = function(e) {
    print(paste("API call error:", e$message))
    return(NULL)
  })
  
  # Handle rate limiting
  if (!is.null(response) && response$status_code == 429) {
    wait_time <- 60
    print(sprintf("Rate limit hit. Waiting %s seconds...", wait_time))
    Sys.sleep(wait_time)
    return(call_gpt4o_mini(prompt, api_key))
  }
  
  # Check for other errors
  if (is.null(response) || response$status_code != 200) {
    print(paste("Error:", if(!is.null(response)) response$status_code else "NULL response"))
    return(NA)
  }
  
  # Parse response
  result <- tryCatch({
    content(response)$choices[[1]]$message$content
  }, error = function(e) {
    print(sprintf("Error parsing response: %s", e$message))
    return(NA)
  })
  
  # Extract binary response (0 or 1)
  clean_result <- tryCatch({
    as.numeric(str_extract(result, "[0-1]"))
  }, error = function(e) {
    print("Could not extract numeric response")
    return(NA)
  })
  
  return(clean_result)
}

# O1 model removed

#' Call Claude-3-Opus model via Anthropic API
#'
#' @param prompt Full text prompt to send to the model
#' @param api_key Anthropic API key
#' @return 0 or 1 for successful classification, NA if error occurs
call_claude_opus <- function(prompt, api_key) {
  # Basic rate limiting
  Sys.sleep(0.5)  # Slightly longer rate limit for larger model
  
  # Clean the prompt text
  clean_prompt <- clean_text(prompt)
  
  # Make API request
  response <- tryCatch({
    POST(
      url = "https://api.anthropic.com/v1/messages",
      add_headers(
        "x-api-key" = api_key, 
        "anthropic-version" = "2023-06-01",
        "Content-Type" = "application/json"
      ),
      body = toJSON(list(
        model = "claude-3-opus-20240229",
        messages = list(
          list(
            role = "user",
            content = clean_prompt
          )
        ),
        temperature = 0,
        max_tokens = 1024,
        system = "You are a helpful assistant that responds with only 0 or 1."
      ), auto_unbox = TRUE),
      encode = "raw"
    )
  }, error = function(e) {
    print(paste("API call error:", e$message))
    return(NULL)
  })
  
  # Handle rate limiting and server overload
  if (!is.null(response) && response$status_code %in% c(429, 529)) {
    wait_time <- 60
    print(sprintf("Rate limit/overload hit. Waiting %s seconds...", wait_time))
    Sys.sleep(wait_time)
    return(call_claude_opus(prompt, api_key))
  }
  
  # Check for other errors
  if (is.null(response) || response$status_code != 200) {
    print(paste("Error:", if(!is.null(response)) response$status_code else "NULL response"))
    return(NA)
  }
  
  # Parse response
  result <- tryCatch({
    content(response)$content[[1]]$text
  }, error = function(e) {
    print(sprintf("Error parsing response: %s", e$message))
    return(NA)
  })
  
  # Extract binary response (0 or 1)
  clean_result <- tryCatch({
    as.numeric(str_extract(result, "[0-1]"))
  }, error = function(e) {
    print("Could not extract numeric response")
    return(NA)
  })
  
  return(clean_result)
}

#' Call Claude-3-Sonnet model via Anthropic API
#'
#' @param prompt Full text prompt to send to the model
#' @param api_key Anthropic API key
#' @return 0 or 1 for successful classification, NA if error occurs
call_claude_sonnet <- function(prompt, api_key) {
  # Basic rate limiting
  Sys.sleep(0.25)
  
  # Clean the prompt text
  clean_prompt <- clean_text(prompt)
  
  # Make API request
  response <- tryCatch({
    POST(
      url = "https://api.anthropic.com/v1/messages",
      add_headers(
        "x-api-key" = api_key, 
        "anthropic-version" = "2023-06-01",
        "Content-Type" = "application/json"
      ),
      body = toJSON(list(
        model = "claude-3-5-sonnet-20241022",
        messages = list(
          list(
            role = "user",
            content = clean_prompt
          )
        ),
        temperature = 0,
        max_tokens = 1024,
        system = "You are a helpful assistant that responds with only 0 or 1."
      ), auto_unbox = TRUE),
      encode = "raw"
    )
  }, error = function(e) {
    print(paste("API call error:", e$message))
    return(NULL)
  })
  
  # Handle rate limiting and server overload
  if (!is.null(response) && response$status_code %in% c(429, 529)) {
    wait_time <- 60
    print(sprintf("Rate limit/overload hit. Waiting %s seconds...", wait_time))
    Sys.sleep(wait_time)
    return(call_claude_sonnet(prompt, api_key))
  }
  
  # Check for other errors
  if (is.null(response) || response$status_code != 200) {
    print(paste("Error:", if(!is.null(response)) response$status_code else "NULL response"))
    return(NA)
  }
  
  # Parse response
  result <- tryCatch({
    content(response)$content[[1]]$text
  }, error = function(e) {
    print(sprintf("Error parsing response: %s", e$message))
    return(NA)
  })
  
  # Extract binary response (0 or 1)
  clean_result <- tryCatch({
    as.numeric(str_extract(result, "[0-1]"))
  }, error = function(e) {
    print("Could not extract numeric response")
    return(NA)
  })
  
  return(clean_result)
}

#' Call Claude-3-Haiku model via Anthropic API
#'
#' @param prompt Full text prompt to send to the model
#' @param api_key Anthropic API key
#' @return 0 or 1 for successful classification, NA if error occurs
call_claude_haiku <- function(prompt, api_key) {
  # Basic rate limiting
  Sys.sleep(0.15)  # Slightly faster rate limit for smaller model
  
  # Clean the prompt text
  clean_prompt <- clean_text(prompt)
  
  # Make API request
  response <- tryCatch({
    POST(
      url = "https://api.anthropic.com/v1/messages",
      add_headers(
        "x-api-key" = api_key, 
        "anthropic-version" = "2023-06-01",
        "Content-Type" = "application/json"
      ),
      body = toJSON(list(
        model = "claude-3-5-haiku-20241022",
        messages = list(
          list(
            role = "user",
            content = clean_prompt
          )
        ),
        temperature = 0,
        max_tokens = 1024,
        system = "You are a helpful assistant that responds with only 0 or 1."
      ), auto_unbox = TRUE),
      encode = "raw"
    )
  }, error = function(e) {
    print(paste("API call error:", e$message))
    return(NULL)
  })
  
  # Handle rate limiting and server overload
  if (!is.null(response) && response$status_code %in% c(429, 529)) {
    wait_time <- 60
    print(sprintf("Rate limit/overload hit. Waiting %s seconds...", wait_time))
    Sys.sleep(wait_time)
    return(call_claude_haiku(prompt, api_key))
  }
  
  # Check for other errors
  if (is.null(response) || response$status_code != 200) {
    print(paste("Error:", if(!is.null(response)) response$status_code else "NULL response"))
    return(NA)
  }
  
  # Parse response
  result <- tryCatch({
    content(response)$content[[1]]$text
  }, error = function(e) {
    print(sprintf("Error parsing response: %s", e$message))
    return(NA)
  })
  
  # Extract binary response (0 or 1)
  clean_result <- tryCatch({
    as.numeric(str_extract(result, "[0-1]"))
  }, error = function(e) {
    print("Could not extract numeric response")
    return(NA)
  })
  
  return(clean_result)
}

#' Call Deepseek Chat model
#'
#' @param prompt Full text prompt to send to the model
#' @param api_key Deepseek API key
#' @return 0 or 1 for successful classification, NA if error occurs
call_deepseek_chat <- function(prompt, api_key) {
  # Basic rate limiting
  Sys.sleep(0.25)
  
  # Clean the prompt text
  clean_prompt <- clean_text(prompt)
  
  # Make API request
  response <- tryCatch({
    POST(
      url = "https://api.deepseek.com/chat/completions",
      add_headers(
        Authorization = paste("Bearer", api_key),
        "Content-Type" = "application/json"
      ),
      body = toJSON(list(
        model = "deepseek-chat",
        messages = list(
          list(
            role = "system",
            content = "You are a helpful assistant that responds with only 0 or 1."
          ),
          list(
            role = "user",
            content = clean_prompt
          )
        ),
        temperature = 0,
        max_tokens = 10,
        stream = FALSE
      ), auto_unbox = TRUE),
      encode = "raw"
    )
  }, error = function(e) {
    print(paste("API call error:", e$message))
    return(NULL)
  })
  
  # Handle rate limiting
  if (!is.null(response) && response$status_code == 429) {
    wait_time <- 60
    print(sprintf("Rate limit hit. Waiting %s seconds...", wait_time))
    Sys.sleep(wait_time)
    return(call_deepseek_chat(prompt, api_key))
  }
  
  # Check for other errors
  if (is.null(response) || response$status_code != 200) {
    print(paste("Error:", if(!is.null(response)) response$status_code else "NULL response"))
    return(NA)
  }
  
  # Parse response
  result <- tryCatch({
    content(response)$choices[[1]]$message$content
  }, error = function(e) {
    print(sprintf("Error parsing response: %s", e$message))
    return(NA)
  })
  
  # Extract binary response (0 or 1)
  clean_result <- tryCatch({
    as.numeric(str_extract(result, "[0-1]"))
  }, error = function(e) {
    print("Could not extract numeric response")
    return(NA)
  })
  
  return(clean_result)
}

#' Calculate the estimated cost of an API call
#'
#' @param prompt Example prompt to estimate token count
#' @param model Model identifier (e.g., "gpt4o", "claude", etc.)
#' @param n_requests Number of requests to estimate for (default: 1)
#' @return Estimated cost in USD
calculate_api_cost <- function(prompt, model, n_requests = 1) {
  # Calculate tokens in prompt (approximate)
  tokens <- nchar(prompt) / 4  # rough approximation
  
  # Cost per 1K tokens (as of 2024)
  costs <- list(
    gpt4o = list(
      input = 0.0025,    # $2.50 per 1M tokens = $0.0025 per 1K tokens
      output = 0.01      # $10.00 per 1M tokens = $0.01 per 1K tokens
    ),
    gpt4o_mini = list(
      input = 0.00015,   # $0.150 per 1M tokens = $0.00015 per 1K tokens
      output = 0.0006    # $0.600 per 1M tokens = $0.0006 per 1K tokens
    ),
    claude_opus = list(
      input = 0.0015,    # $1.50 per 1M tokens = $0.0015 per 1K tokens
      output = 0.0075    # $7.50 per 1M tokens = $0.0075 per 1K tokens
    ),
    claude_sonnet = list(
      input = 0.0008,    # $0.80 per 1M tokens = $0.0008 per 1K tokens
      output = 0.004     # $4.00 per 1M tokens = $0.004 per 1K tokens
    ),
    claude_haiku = list(
      input = 0.00025,   # $0.25 per 1M tokens = $0.00025 per 1K tokens
      output = 0.00125   # $1.25 per 1M tokens = $0.00125 per 1K tokens
    ),
    deepseek_chat = list(
      input = 0.00027,   # $0.27 per 1M tokens = $0.00027 per 1K tokens (cache miss)
      output = 0.0011    # $1.10 per 1M tokens = $0.0011 per 1K tokens
    )
  )
  
  # Get costs for the specified model
  model_costs <- costs[[model]]
  if (is.null(model_costs)) {
    warning(paste("Cost data not available for model", model))
    return(0)
  }
  
  # Estimate total cost (assuming short output for binary classification)
  total_cost <- (tokens * model_costs$input + 10 * model_costs$output) / 1000 * n_requests
  return(total_cost)
}
