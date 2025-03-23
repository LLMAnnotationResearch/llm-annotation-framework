# Test script for evaluation functions in R
# Load necessary libraries
library(dplyr)
library(ggplot2)

# Source the evaluation functions
source("R/src/evaluation_functions_r.R")
print("Successfully imported evaluation functions")

# Run the test function with default values
test_evaluation_functions <- function() {
  # Setup API keys - replace these with your actual keys
  api_keys <- list(
    openai = "[INSERT OPENAI KEY HERE]",
    anthropic = "[INSERT ANTHROPIC KEY HERE]"
  )
  
  # Step 1: Load the test data
  print("Loading test data...")
  
  # Try different potential data paths
  potential_paths <- c(
    file.path("data", "test_sample.csv"),  # If running from main directory
    file.path("..", "data", "test_sample.csv")  # If running from R/tests
  )
  
  data_path <- NULL
  for (path in potential_paths) {
    if (file.exists(path)) {
      data_path <- path
      print(paste("Found data file at:", path))
      break
    }
  }
  
  if (is.null(data_path)) {
    stop("Could not find data file. Please check the file path.")
  }
  
  # Try different encodings
  encodings <- c('utf-8', 'latin1', 'windows-1252')
  data <- NULL
  
  for (encoding in encodings) {
    tryCatch({
      print(paste("Trying to load with", encoding, "encoding..."))
      data <- read.csv(data_path, encoding = encoding, stringsAsFactors = FALSE)
      print(paste("Successfully loaded with", encoding, "encoding"))
      break
    }, error = function(e) {
      print(paste("Failed with", encoding, "encoding, trying next..."))
    })
  }
  
  if (is.null(data)) {
    stop("Could not load the file with any of the attempted encodings")
  }
  
  print(paste("Loaded", nrow(data), "rows from", data_path))
  print(paste("Columns:", paste(colnames(data), collapse = ", ")))
  
  # Step 2: Load prompt variations
  print("Loading prompt variations...")
  
  # Try different potential data paths
  potential_paths <- c(
    file.path("data", "prompt_variations.csv"),  # If running from main directory
    file.path("..", "data", "prompt_variations.csv")  # If running from R/tests
  )
  
  prompts_path <- NULL
  for (path in potential_paths) {
    if (file.exists(path)) {
      prompts_path <- path
      print(paste("Found prompts file at:", path))
      break
    }
  }
  
  if (is.null(prompts_path)) {
    stop("Could not find prompts file. Please check the file path.")
  }
  
  # Try different encodings
  prompts_df <- NULL
  
  for (encoding in encodings) {
    tryCatch({
      print(paste("Trying to load prompts with", encoding, "encoding..."))
      prompts_df <- read.csv(prompts_path, encoding = encoding, stringsAsFactors = FALSE)
      print(paste("Successfully loaded with", encoding, "encoding"))
      break
    }, error = function(e) {
      print(paste("Failed with", encoding, "encoding, trying next..."))
    })
  }
  
  if (is.null(prompts_df)) {
    stop("Could not load the prompts file with any of the attempted encodings")
  }
  
  print(paste("Loaded", nrow(prompts_df), "prompt variations from", prompts_path))
  
  # Show a sample prompt
  if (nrow(prompts_df) > 0) {
    print(paste("Sample prompt (strategy:", prompts_df$strategy[1], "):"))
    print(prompts_df$prompt[1])
  }
  
  # Step 3: Test a single model evaluation with fixed parameters
  print("TESTING SINGLE MODEL EVALUATION")
  
  # Use gpt4o_mini as default model
  model_name <- "gpt4o_mini"
  api_key <- api_keys$openai
  
  # Test with at least 10 samples for regression to work
  test_size <- 10
  test_data <- data[1:test_size, ]
  print(paste("Testing", model_name, "with", test_size, "samples..."))
  
  # Use a simple regression formula with success as the dependent variable
  reg_formula <- "success ~ LLM_LABEL"
  print(paste("Using regression formula:", reg_formula))
  
  # Run evaluation
  results <- evaluate_model(
    data = test_data,
    model_name = model_name,
    api_key = api_key,
    prompt_template = prompts_df$prompt[1],
    text_column = "text",
    human_label_column = "HUMAN",
    regression_formula = reg_formula,
    regression_family = "gaussian",
    verbose = TRUE
  )
  
  # Print results
  print("Evaluation Results:")
  print(paste("Accuracy:", format(results$metrics$accuracy, digits = 2)))
  print(paste("Cost: $", format(results$cost, digits = 6)))
  print(paste("Valid predictions:", results$n_valid, "/", results$n_total))
  print(paste("Regression coefficient:", format(results$regression$coefficient, digits = 4)))
  print(paste("Regression p-value:", format(results$regression$p_value, digits = 4)))
  
  # Step 4: Test sensitivity analysis with fixed parameters
  print("TESTING SENSITIVITY ANALYSIS")
  
  # Run sensitivity analysis with 2 prompts as an example
  prompt_limit <- min(2, nrow(prompts_df))
  limited_prompts <- prompts_df[1:prompt_limit, ]
  
  # Create a timestamp for output files
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  output_dir <- file.path("output", timestamp)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Using the same test data (10 samples)
  print(paste("Running sensitivity analysis with", prompt_limit, "prompts and", test_size, "samples..."))
  
  # Run sensitivity analysis
  results <- run_sensitivity_analysis(
    prompts_df = limited_prompts,
    data = test_data,
    api_key = api_key,
    model_name = model_name,
    text_column = "text",
    human_label_column = "HUMAN",
    regression_formula = reg_formula,
    regression_family = "gaussian",
    save_path = output_dir,
    figure_name = paste0("sensitivity_", model_name)
  )
  
  # Print results summary
  print("Sensitivity Analysis Results:")
  print(paste("Results saved to:", output_dir))
  
  # Show the first few rows of results
  print("Results summary:")
  print(head(results$results))
  
  print("Test script completed successfully!")
}

# Run the test function
test_evaluation_functions()