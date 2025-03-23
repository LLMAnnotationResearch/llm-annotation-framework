# Load necessary libraries
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(stringr)
library(stats)

# Source the API handlers
source("r/src/api_handlers_r.R")

#' Evaluate an LLM model for classification performance and regression impact
#'
#' @param data A dataframe containing the text to classify and ground truth labels
#' @param model_name Name of the model to use (gpt4o, gpt4o_mini, claude_opus, claude_sonnet, claude_haiku)
#' @param api_key API key for the relevant service
#' @param prompt_template Template text for the prompt (text will be appended to this)
#' @param text_column Name of the column containing the text to classify
#' @param human_label_column Name of the column containing the ground truth labels
#' @param regression_formula Formula for regression (e.g., "success ~ LLM_LABEL + control1 + control2")
#' @param regression_family Family for GLM regression (e.g., "binomial", "gaussian")
#' @param max_retries Maximum number of retries for API calls
#' @param retry_delay Delay between retries in seconds
#' @param verbose Whether to print progress information
#'
#' @return A list containing:
#'   - results: Dataframe with original data and LLM predictions
#'   - metrics: List of evaluation metrics
#'   - cost: Total API cost
#'   - regression: List of regression results (if applicable)
#'   - n_valid: Number of valid predictions
#'   - n_total: Total number of examples
#'
#' @export
evaluate_model <- function(data, 
                           model_name, 
                           api_key, 
                           prompt_template,
                           text_column = "text",
                           human_label_column = "HUMAN",
                           regression_formula = NULL,
                           regression_family = "gaussian",
                           max_retries = 3,
                           retry_delay = 5,
                           verbose = TRUE) {
  
  # Validate model name
  supported_models <- c("gpt4o", "gpt4o_mini", "claude_opus", "claude_sonnet", "claude_haiku")
  if (!model_name %in% supported_models) {
    stop(paste("Unsupported model:", model_name, 
               "Available models:", paste(supported_models, collapse = ", ")))
  }
  
  # Clean text first
  if (verbose) {
    cat("Cleaning text data...\n")
  }
  
  data <- data %>%
    mutate(clean_text = sapply(!!sym(text_column), clean_text))
  
  # Process all examples
  if (verbose) {
    cat(sprintf("Processing %d examples with %s...\n", nrow(data), model_name))
  }
  
  results <- data %>%
    mutate(
      LLM_LABEL = map_dbl(clean_text, function(text) {
        full_prompt <- paste(prompt_template, text)
        
        # Try with retries
        for (attempt in 1:max_retries) {
          result <- tryCatch({
            if (verbose) cat(".")  # Progress indicator
            
            # Call the appropriate model function
            model_result <- switch(model_name,
                                  "gpt4o" = call_gpt4o(full_prompt, api_key),
                                  "gpt4o_mini" = call_gpt4o_mini(full_prompt, api_key),
                                  "claude_opus" = call_claude_opus(full_prompt, api_key),
                                  "claude_sonnet" = call_claude_sonnet(full_prompt, api_key),
                                  "claude_haiku" = call_claude_haiku(full_prompt, api_key))
            
            if (is.null(model_result)) {
              NA_real_
            } else {
              model_result
            }
          }, error = function(e) {
            if (attempt < max_retries) {
              if (verbose) {
                cat(sprintf("\nError: %s. Retrying in %d seconds...\n", e$message, retry_delay))
              }
              Sys.sleep(retry_delay)
              return(NULL)  # Continue to next attempt
            } else {
              if (verbose) {
                cat(sprintf("\nFailed after %d attempts: %s\n", max_retries, e$message))
              }
              return(NA_real_)
            }
          })
          
          if (!is.null(result)) {
            return(result)  # Break out of retry loop if successful
          }
        }
        return(NA_real_)  # Only reached if all retries failed and error handler didn't return
      })
    )
  
  if (verbose) cat("\n")  # End progress indicators
  
  # Calculate metrics
  valid_preds <- !is.na(results$LLM_LABEL)
  n_valid <- sum(valid_preds)
  n_total <- nrow(data)
  
  if (n_valid > 0 && human_label_column %in% colnames(results)) {
    valid_data <- results[valid_preds, ]
    
    metrics <- list(
      accuracy = sum(valid_data[[human_label_column]] == valid_data$LLM_LABEL) / n_valid,
      false_positive_rate = sum(valid_data[[human_label_column]] == 0 & valid_data$LLM_LABEL == 1) / n_total,
      false_negative_rate = sum(valid_data[[human_label_column]] == 1 & valid_data$LLM_LABEL == 0) / n_total
    )
  } else {
    metrics <- list(
      accuracy = NA_real_,
      false_positive_rate = NA_real_,
      false_negative_rate = NA_real_
    )
  }
  
  # Calculate costs
  total_cost <- sum(sapply(data$clean_text, function(text) {
    calculate_api_cost(paste(prompt_template, text), model_name)
  }))
  
  # Run regression if formula is provided and we have enough data
  reg_results <- list(
    coefficient = NA_real_,
    std_error = NA_real_,
    p_value = NA_real_,
    converged = FALSE
  )
  
  if (!is.null(regression_formula) && n_valid >= 10) {
    if (verbose) {
      cat("Running regression analysis...\n")
    }
    
    # Prepare regression data
    reg_data <- results[valid_preds, ]
    
    # Create the formula by replacing "LLM_LABEL" with the actual column name
    formula_str <- regression_formula
    
    # Print regression data info for debugging
    if (verbose) {
      cat(sprintf("\nRegression data shape: %d rows, %d columns\n", nrow(reg_data), ncol(reg_data)))
      cat(sprintf("Regression formula: %s\n", formula_str))
      cat(sprintf("Regression columns available: %s\n", paste(colnames(reg_data), collapse = ", ")))
    }
    
    # Run the regression
    tryCatch({
      if (regression_family == "binomial") {
        reg_model <- glm(as.formula(formula_str), data = reg_data, family = binomial())
      } else if (regression_family == "gaussian") {
        reg_model <- lm(as.formula(formula_str), data = reg_data)
      } else {
        stop(paste("Unsupported regression family:", regression_family))
      }
      
      # Extract results if the model converged
      converged <- !is.null(reg_model$converged) || is.null(reg_model$converged)  # lm doesn't have converged attribute
      
      # Check if LLM_LABEL is in the model summary
      model_summary <- summary(reg_model)
      coef_matrix <- coef(model_summary)
      
      if ("LLM_LABEL" %in% rownames(coef_matrix)) {
        reg_results <- list(
          coefficient = coef_matrix["LLM_LABEL", "Estimate"],
          std_error = coef_matrix["LLM_LABEL", "Std. Error"],
          p_value = coef_matrix["LLM_LABEL", ifelse(regression_family == "binomial", "Pr(>|z|)", "Pr(>|t|)")],
          converged = converged,
          full_results = if (verbose) capture.output(print(model_summary)) else NULL
        )
      } else {
        warning("LLM_LABEL not found in regression results")
      }
    }, error = function(e) {
      warning(paste("Regression error:", e$message))
      if (verbose) {
        print(e)
      }
    })
  }
  
  # Return all results
  return(list(
    results = results,
    metrics = metrics,
    cost = total_cost,
    regression = reg_results,
    n_valid = n_valid,
    n_total = n_total
  ))
}

#' Run a sensitivity analysis across multiple prompt variations
#'
#' @param prompts_df Dataframe containing prompt variations with columns 'prompt' and 'strategy'
#' @param data Dataframe containing the text to classify and ground truth labels
#' @param api_key API key for the relevant service
#' @param model_name Name of the model to use
#' @param text_column Name of the column containing the text to classify
#' @param human_label_column Name of the column containing the ground truth labels
#' @param regression_formula Formula for regression
#' @param regression_family Family for GLM regression
#' @param save_path Path to save intermediate results and plots
#' @param figure_name Name for the output figure (defaults to model_name)
#'
#' @return A list containing:
#'   - results: Dataframe with results for each prompt
#'   - plot: ggplot object
#'
#' @export
run_sensitivity_analysis <- function(prompts_df,
                                    data,
                                    api_key,
                                    model_name = "gpt4o",
                                    text_column = "text",
                                    human_label_column = "HUMAN",
                                    regression_formula = NULL,
                                    regression_family = "gaussian",
                                    save_path = "./",
                                    figure_name = NULL) {
  
  if (is.null(figure_name)) {
    figure_name <- paste0("coefficient_distribution_", model_name)
  }
  
  # Create save path if it doesn't exist
  if (!dir.exists(save_path)) {
    dir.create(save_path, recursive = TRUE)
  }
  
  # Initialize results dataframe
  results_df <- data.frame(
    prompt_id = numeric(),
    strategy = character(),
    accuracy = numeric(),
    reg_coef = numeric(),
    reg_se = numeric(),
    p_value = numeric(),
    converged = logical(),
    n_valid = numeric(),
    n_total = numeric(),
    cost = numeric(),
    stringsAsFactors = FALSE
  )
  
  # For each prompt variation
  for (i in 1:nrow(prompts_df)) {
    cat(sprintf("\nProcessing prompt %d/%d (%s)...\n", 
                i, nrow(prompts_df), prompts_df$strategy[i]))
    
    # Run evaluation
    results <- evaluate_model(
      data,
      model_name,
      api_key,
      prompts_df$prompt[i],
      text_column = text_column,
      human_label_column = human_label_column,
      regression_formula = regression_formula,
      regression_family = regression_family
    )
    
    # Add results to dataframe
    results_df <- rbind(results_df, data.frame(
      prompt_id = i,
      strategy = prompts_df$strategy[i],
      accuracy = results$metrics$accuracy,
      reg_coef = results$regression$coefficient,
      reg_se = results$regression$std_error,
      p_value = results$regression$p_value,
      converged = results$regression$converged,
      n_valid = results$n_valid,
      n_total = results$n_total,
      cost = results$cost
    ))
    
    # Save intermediate results
    results_file <- file.path(save_path, paste0("sensitivity_analysis_", model_name, ".rds"))
    saveRDS(results_df, results_file)
    
    cat(sprintf("Results saved to %s\n", results_file))
  }
  
  # Filter out non-converged models for the histogram
  plot_df <- results_df %>%
    filter(converged == TRUE)
  
  if (nrow(plot_df) > 0) {
    # Calculate statistics for reference lines
    mean_coef <- mean(plot_df$reg_coef, na.rm = TRUE)
    sd_coef <- sd(plot_df$reg_coef, na.rm = TRUE)
    
    # Create plot
    p <- ggplot(plot_df, aes(x = reg_coef)) +
      geom_histogram(binwidth = (max(plot_df$reg_coef, na.rm = TRUE) - 
                                min(plot_df$reg_coef, na.rm = TRUE)) / 20, 
                    fill = "steelblue", color = "white", alpha = 0.7) +
      geom_density(alpha = 0.3, fill = "lightblue") +
      geom_vline(xintercept = mean_coef, 
                color = "red", linetype = "dashed", size = 1) +
      geom_vline(xintercept = c(mean_coef - sd_coef, mean_coef + sd_coef), 
                color = "darkred", linetype = "dotted", size = 0.8) +
      annotate("text", x = mean_coef, y = Inf, 
              label = sprintf("Mean: %.2f", mean_coef),
              vjust = 2, hjust = -0.2) +
      annotate("text", x = mean_coef - sd_coef, y = Inf, 
              label = "-1 SD", vjust = 2, hjust = 1.2) +
      annotate("text", x = mean_coef + sd_coef, y = Inf, 
              label = "+1 SD", vjust = 2, hjust = -0.2) +
      labs(title = paste("Distribution of Regression Coefficients (", model_name, ")", sep = ""),
           subtitle = sprintf("Mean: %.2f, SD: %.2f", mean_coef, sd_coef),
           x = "Coefficient Value",
           y = "Count") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5),
            plot.subtitle = element_text(hjust = 0.5))
    
    # Save plot
    plot_file <- file.path(save_path, paste0(figure_name, ".png"))
    ggsave(plot_file, p, width = 8, height = 6)
    
    cat(sprintf("Plot saved to %s\n", plot_file))
  } else {
    warning("No converged models to plot")
    p <- NULL
  }
  
  return(list(
    results = results_df,
    plot = p
  ))
}

#' Load prompt variations from a text file
#'
#' The file should have sections starting with "## Strategy Name",
#' followed by the prompt text.
#'
#' @param file_path Path to the text file with prompt variations
#' @return Dataframe with columns 'strategy' and 'prompt'
#'
#' @export
load_prompt_variations <- function(file_path) {
  # Read the file content
  content <- readLines(file_path, warn = FALSE)
  content <- paste(content, collapse = "\n")
  
  # Split by sections
  sections <- strsplit(content, "##\\s+")[[1]]
  
  # Skip the first element if it's empty
  if (sections[1] == "") {
    sections <- sections[-1]
  }
  
  # Process each section
  strategies <- character()
  prompts <- character()
  
  for (section in sections) {
    # Split the section into lines
    lines <- strsplit(section, "\n")[[1]]
    
    # First line is the strategy name
    strategy <- lines[1]
    
    # Rest is the prompt
    prompt <- paste(lines[-1], collapse = "\n")
    
    # Trim whitespace
    strategy <- trimws(strategy)
    prompt <- trimws(prompt)
    
    # Add to vectors
    strategies <- c(strategies, strategy)
    prompts <- c(prompts, prompt)
  }
  
  # Create dataframe
  data.frame(
    strategy = strategies,
    prompt = prompts,
    stringsAsFactors = FALSE
  )
}

# Example usage
if (FALSE) {
  # Load data
  data <- read.csv("data/test_sample.csv")
  
  # Load prompts
  prompts <- read.csv("data/prompt_variations.csv")
  
  # Set API keys
  api_keys <- list(
    openai = "your-openai-api-key",
    anthropic = "your-anthropic-api-key"
  )
  
  # Run model evaluation
  results <- evaluate_model(
    data = data,
    model_name = "gpt4o_mini",
    api_key = api_keys$openai,
    prompt_template = prompts$prompt[1],
    text_column = "text",
    human_label_column = "HUMAN",
    regression_formula = "success ~ LLM_LABEL"
  )
  
  # Run sensitivity analysis
  sensitivity <- run_sensitivity_analysis(
    prompts_df = prompts,
    data = data,
    api_key = api_keys$openai,
    model_name = "gpt4o_mini",
    regression_formula = "success ~ LLM_LABEL"
  )
}
