# Basic sensitivity analysis test script for the LLM Annotation Framework.
# This script follows the sensitivity analysis example in the README.

# Load necessary libraries
library(dplyr)
library(ggplot2)

# Source the evaluation functions
print("Loading evaluation functions...")
source("R/src/evaluation_functions_r.R")
print("Successfully loaded evaluation functions")

run_sensitivity_test <- function() {
  print("Running basic sensitivity analysis test for LLM Annotation Framework...")
  
  # Set your API keys - replace these with actual keys
  api_keys <- list(
    openai = "[INSERT_YOUR_OPENAI_KEY_HERE]",
    anthropic = "[INSERT_YOUR_ANTHROPIC_KEY_HERE]"
  )
  
  # Load your dataset
  tryCatch({
    data_path <- file.path("data", "test_sample.csv")
    data <- read.csv(data_path)
    print(paste("Successfully loaded", nrow(data), "records from", data_path))
  }, error = function(e) {
    print(paste("Error loading data:", e$message))
    return()
  })
  
  # Load prompt variations
  tryCatch({
    prompts_path <- file.path("data", "prompt_variations.csv")
    prompts_df <- read.csv(prompts_path)
    print(paste("Successfully loaded", nrow(prompts_df), "prompt variations from", prompts_path))
    
    # Print a sample prompt
    if (nrow(prompts_df) > 0) {
      print("\nSample prompt strategy:")
      print(paste("Strategy:", prompts_df$strategy[1]))
      prompt_preview <- substr(prompts_df$prompt[1], 1, 100)
      print(paste("Prompt:", prompt_preview, "..."))  # Show first 100 chars
    }
  }, error = function(e) {
    print(paste("Error loading prompt variations:", e$message))
    return()
  })
  
  # Use a small subset of prompts and data for testing
  prompt_limit <- min(3, nrow(prompts_df))
  limited_prompts <- prompts_df[1:prompt_limit, ]
  
  test_size <- min(10, nrow(data))  # Use at least 10 samples for regression
  test_data <- data[1:test_size, ]
  
  print(paste("\nRunning sensitivity analysis with", prompt_limit, "prompts and", test_size, "samples..."))
  
  # Create a folder for results
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  output_dir <- file.path("results", paste0("sensitivity_", timestamp))
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Determine which model to use - using claude_opus
  model_name <- "claude_opus"
  api_key <- api_keys$anthropic
  
  # Run sensitivity analysis
  tryCatch({
    results <- run_sensitivity_analysis(
      prompts_df = limited_prompts,
      data = test_data,
      api_key = api_key,
      model_name = model_name,
      text_column = "text",
      human_label_column = "HUMAN",
      regression_formula = "success ~ LLM_LABEL + lgoal + year",
      regression_family = "gaussian",
      save_path = output_dir,
      figure_name = paste0("sensitivity_", model_name)
    )
    
    # View results summary
    summary_df <- results$results
    print("\nSensitivity Analysis Results:")
    print(summary_df[, c("strategy", "accuracy", "reg_coef", "p_value", "cost")])
    
    # Calculate the mean and standard deviation of the coefficient
    mean_coef <- mean(summary_df$reg_coef, na.rm = TRUE)
    sd_coef <- sd(summary_df$reg_coef, na.rm = TRUE)
    print(paste("\nMean coefficient:", round(mean_coef, 4), 
                "(SD:", round(sd_coef, 4), ")"))
    
    print("\nSensitivity analysis completed successfully!")
    print(paste("Results saved to:", output_dir))
    
  }, error = function(e) {
    print(paste("Error during sensitivity analysis:", e$message))
  })
}

# Run the test
run_sensitivity_test()
