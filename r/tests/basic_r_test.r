# Basic test script for the LLM Annotation Framework that follows the README example.
# Run this script to verify that the basic functionality works as described.

# Load necessary libraries
library(dplyr)

# Source the evaluation functions
print("Loading evaluation functions...")
source("R/src/evaluation_functions_r.R")
print("Successfully loaded evaluation functions")

run_basic_test <- function() {
  print("Running basic R test for LLM Annotation Framework...")
  
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
    print(paste("Columns:", paste(colnames(data), collapse = ", ")))
  }, error = function(e) {
    print(paste("Error loading data:", e$message))
    return()
  })
  
  # Define your prompt template
  prompt_template <- "
  Please classify if this product benefits society beyond its consumers.
  Answer only with 0 or 1. Here is the text:
  "
  
  # Use a small subset for testing
  test_size <- 5
  test_data <- data[1:test_size, ]
  print(paste("\nTesting with", test_size, "samples..."))
  
  # Determine which model to use - using claude_opus instead of claude_haiku
  model_name <- "claude_opus"  # Options: gpt4o, gpt4o_mini, claude_opus, claude_sonnet, claude_haiku
  api_key <- api_keys$anthropic  # Use appropriate key based on the model
  
  # Run the model on your data
  tryCatch({
    results <- evaluate_model(
      data = test_data,
      model_name = model_name,
      api_key = api_key,
      prompt_template = prompt_template,
      text_column = "text",
      human_label_column = "HUMAN",
      regression_formula = "success ~ LLM_LABEL + lgoal + year",
      regression_family = "gaussian"
    )
    
    # Access the annotated data
    annotated_data <- results$results
    print(paste("Annotated", nrow(annotated_data), "items"))
    
    # View model performance metrics
    metrics <- results$metrics
    print(paste("Accuracy:", format(metrics$accuracy, digits = 2)))
    print(paste("Cost: $", format(results$cost, digits = 4)))
    
    # Check regression results
    regression <- results$regression
    print(paste("LLM label coefficient:", format(regression$coefficient, digits = 4)))
    print(paste("p-value:", format(regression$p_value, digits = 4)))
    
    # Print model responses
    print("Model responses:")
    for (i in 1:nrow(annotated_data)) {
      print(paste("Sample", i, ":", annotated_data$LLM_LABEL[i]))
    }
    
    print("Basic test completed successfully!")
    
  }, error = function(e) {
    print(paste("Error during model evaluation:", e$message))
  })
}

# Run the test
run_basic_test()
