# LLM Annotation Framework for Social Science Research

A comprehensive framework for using large language models (LLMs) to annotate text data for social science research, with emphasis on methodological rigor and sensitivity analysis.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Components](#components)
- [Basic Usage](#basic-usage)
- [Conducting Sensitivity Analysis](#conducting-sensitivity-analysis)
- [Example: Analyzing Crowdfunding Success](#example-analyzing-crowdfunding-success)
- [Tips for Effective Use](#tips-for-effective-use)

## Overview

The LLM Annotation Framework provides tools for:

1. **Text Annotation**: Using LLMs to classify text data with binary labels
2. **Evaluation**: Assessing model performance against human annotations
3. **Sensitivity Analysis**: Testing how different prompt formulations affect results
4. **Downstream Analysis**: Incorporating LLM classifications into regression models

This framework is designed to help social scientists leverage LLMs for data annotation while maintaining methodological rigor through robust sensitivity analysis.

## Installation

### Prerequisites

- Python 3.6+ or R 4.0+
- API keys for OpenAI and/or Anthropic

### Python Installation

```bash
git clone https://github.com/LLMAnnotationResearch/llm-annotation-framework.git
cd llm-annotation-framework

# Install required packages
pip install pandas numpy matplotlib statsmodels tqdm seaborn openai anthropic
```

### R Installation

```r
# Install required packages
install.packages(c("dplyr", "tidyr", "ggplot2", "readr", "httr", "jsonlite"))

# Clone the repository using your preferred method
```

## Components

The framework consists of several components:

### 1. API Handlers

Functions for making API calls to various LLM providers:

- **Python**: `src/api_handlers_python.py`
- **R**: `src/api_handlers_r.R`

Supported models:
- OpenAI: GPT-4o, GPT-4o-mini
- Anthropic: Claude Opus, Claude Sonnet, Claude Haiku

### 2. Evaluation Functions

Tools for evaluating model performance and running sensitivity analyses:

- **Python**: `src/evaluation_functions_python.py`
- **R**: `src/evaluation_functions_r.R`

### 3. Prompt Creation Tools

Templates for generating effective prompt variations:

- **Template**: `data/prompt_creation_template.txt`
- **Examples**: `data/prompt_creation_example.txt`

### 4. Test Scripts

Scripts to test API and evaluation functions:

- **API Tests**:
  - Python: `python/tests/test_api_handlers.py`
  - R: `R/tests/test_api_handlers.R`
- **Evaluation Tests**:
  - Python: `python/tests/test_evaluation_functions.py`
  - R: `R/tests/test_evaluation_functions.R`

## Basic Usage

### Python Example

```python
import pandas as pd
from src.evaluation_functions_python import evaluate_model

# Load your dataset
data = pd.read_csv("data/test_sample.csv")

# Set your API keys
api_keys = {
    "openai": "your-openai-api-key-here",
    "anthropic": "your-anthropic-api-key-here"
}

# Define your prompt template
prompt_template = """
Please classify if this product benefits society beyond its consumers.
Answer only with 0 or 1. Here is the text:
"""

# Run the model on your data
results = evaluate_model(
    data=data,
    model_name="gpt4o_mini",  # Options: gpt4o, gpt4o_mini, claude_opus, claude_sonnet, claude_haiku
    api_key=api_keys["openai"],  # Use appropriate key based on the model
    prompt_template=prompt_template,
    text_column="text",  # Column containing text to classify
    human_label_column="HUMAN",  # Optional ground truth column
    regression_formula="success ~ LLM_LABEL + lgoal + year",  # Optional regression
    regression_family="gaussian"  # "gaussian" for OLS, "binomial" for logistic
)

# Access the annotated data
annotated_data = results['results']
print(f"Annotated {len(annotated_data)} items")

# View model performance metrics
metrics = results['metrics']
print(f"Accuracy: {metrics['accuracy']:.2f}")
print(f"Cost: ${results['cost']:.4f}")

# Check regression results
regression = results['regression']
print(f"LLM label coefficient: {regression['coefficient']:.4f}")
print(f"p-value: {regression['p_value']:.4f}")
```

### R Example

```r
library(dplyr)
source("R/src/evaluation_functions_r.R")

# Load your dataset
data <- read.csv("data/test_sample.csv")

# Set your API keys
api_keys <- list(
  openai = "your-openai-api-key-here",
  anthropic = "your-anthropic-api-key-here"
)

# Define your prompt template
prompt_template <- "
Please classify if this product benefits society beyond its consumers.
Answer only with 0 or 1. Here is the text:
"

# Run the model on your data
results <- evaluate_model(
  data = data,
  model_name = "claude_haiku",
  api_key = api_keys$anthropic,  # Use appropriate key based on the model
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
print(paste("Accuracy:", round(metrics$accuracy, 2)))
print(paste("Cost: $", round(results$cost, 4)))

# Check regression results
regression <- results$regression
print(paste("LLM label coefficient:", round(regression$coefficient, 4)))
print(paste("p-value:", round(regression$p_value, 4)))
```

## Conducting Sensitivity Analysis

Sensitivity analysis tests how your results vary across different prompt formulations. This is crucial for ensuring the robustness of your findings.

### Creating Prompt Variations

The framework includes templates for generating prompt variations:

1. **Using the Prompt Creation Template**: The file `data/prompt_creation_template.txt` provides a structured approach for generating diverse prompt variations.
2. **Example Prompts**: See `data/prompt_creation_example.txt` for examples of generated prompts.

These templates help create prompt variations using different strategies like zero-shot, few-shot, chain-of-thought, and role-based approaches. We've had good success generating these prompt variations using Claude 3.5 Sonnet in a chat window.

### Python Example

```python
from src.evaluation_functions_python import run_sensitivity_analysis
import pandas as pd

# Load your dataset
data = pd.read_csv("data/test_sample.csv")

# Load prompt variations
prompts_df = pd.read_csv("data/prompt_variations.csv")

# Set your API keys
api_keys = {
    "openai": "your-openai-api-key-here",
    "anthropic": "your-anthropic-api-key-here"
}

# Run sensitivity analysis
sensitivity_results = run_sensitivity_analysis(
    prompts_df=prompts_df,
    data=data,
    api_key=api_keys["openai"],  # Use appropriate key based on the model
    model_name="gpt4o_mini",
    text_column="text",
    human_label_column="HUMAN",
    regression_formula="success ~ LLM_LABEL + lgoal + year",
    regression_family="gaussian",
    save_path="results/sensitivity_analysis/"
)

# View results summary
summary_df = sensitivity_results['results']
print(summary_df[['strategy', 'accuracy', 'reg_coef', 'p_value', 'cost']])

# The function automatically saves:
# 1. A CSV with all results
# 2. A plot showing the distribution of regression coefficients
```

## Example: Analyzing Crowdfunding Success

This example demonstrates a complete workflow for using LLM annotations to analyze factors affecting crowdfunding success.

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.evaluation_functions_python import run_sensitivity_analysis

# Set your API keys
api_keys = {
    "openai": "your-openai-api-key-here",
    "anthropic": "your-anthropic-api-key-here"
}

# Step 1: Load crowdfunding data
data = pd.read_csv("data/test_sample.csv")
print(f"Loaded {len(data)} crowdfunding campaigns")

# Step 2: Load prompt variations to test different formulations
prompts = pd.read_csv("data/prompt_variations.csv")
print(f"Loaded {len(prompts)} prompt variations")

# Step 3: Run sensitivity analysis
results = run_sensitivity_analysis(
    prompts_df=prompts,
    data=data,
    api_key=api_keys["openai"],  # Use appropriate key based on the model
    model_name="gpt4o_mini",
    text_column="text",
    human_label_column="HUMAN",
    regression_formula="success ~ LLM_LABEL + lgoal + subcategory + year",
    regression_family="gaussian",
    save_path="results/crowdfunding_analysis/"
)

# Step 4: Analyze the results
summary = results['results']
print("\nSensitivity Analysis Results:")
print(summary[['strategy', 'accuracy', 'reg_coef', 'p_value']].head())

# Calculate the mean and standard deviation of the coefficient
mean_coef = summary['reg_coef'].mean()
sd_coef = summary['reg_coef'].std()
print(f"\nMean coefficient: {mean_coef:.4f} (SD: {sd_coef:.4f})")

# Step 5: Create a visualization of how prompt variations affect results
plt.figure(figsize=(10, 6))
plt.errorbar(
    x=range(len(summary)),
    y=summary['reg_coef'],
    yerr=summary['reg_se'],
    fmt='o',
    capsize=5
)
plt.axhline(y=0, color='r', linestyle='--')
plt.axhline(y=mean_coef, color='blue', linestyle='-', alpha=0.5)
plt.fill_between(
    range(-1, len(summary)+1),
    mean_coef - sd_coef,
    mean_coef + sd_coef,
    color='blue',
    alpha=0.1
)
plt.xticks(range(len(summary)), summary['strategy'], rotation=45, ha='right')
plt.ylabel('Coefficient')
plt.xlabel('Prompt Strategy')
plt.title('Effect of Societal Benefit on Crowdfunding Success')
plt.tight_layout()
plt.savefig("results/coefficient_plot.png")
```

## Flexible Regression Analysis

The framework supports flexible regression analysis, allowing you to specify:

1. A custom regression formula (e.g., `y ~ x1 + x2 + x3`)
2. The specific coefficient of interest to extract from the results
3. The type of regression (Gaussian/OLS or Binomial/Logistic)

This enables various research designs where LLM annotations might be:
- Independent variables in your model
- Dependent variables being predicted
- Control variables in a larger model

### Python Example:

```python
# When LLM annotation is the independent variable of interest
results = evaluate_model(
    data=data,
    model_name="gpt4o",
    api_key=api_keys["openai"],
    prompt_template=prompt_template,
    regression_formula="outcome ~ LLM_LABEL + control1 + control2",
    coefficient_of_interest="LLM_LABEL"  # Default
)

# When LLM annotation is the dependent variable
results = evaluate_model(
    data=data,
    model_name="gpt4o",
    api_key=api_keys["openai"],
    prompt_template=prompt_template,
    regression_formula="LLM_LABEL ~ predictor1 + predictor2",
    coefficient_of_interest="predictor1"  # We're interested in this predictor
)

# When interested in interaction effects
results = evaluate_model(
    data=data,
    model_name="gpt4o",
    api_key=api_keys["openai"],
    prompt_template=prompt_template,
    regression_formula="outcome ~ LLM_LABEL * treatment + control",
    coefficient_of_interest="LLM_LABEL:treatment"  # Interaction term
)
```

### R Example:

```r
# When LLM annotation is the independent variable of interest
results <- evaluate_model(
    data = data,
    model_name = "claude_opus",
    api_key = api_keys$anthropic,
    prompt_template = prompt_template,
    regression_formula = "outcome ~ LLM_LABEL + control1 + control2",
    coefficient_of_interest = "LLM_LABEL"  # Default
)

# When LLM annotation is the dependent variable
results <- evaluate_model(
    data = data,
    model_name = "claude_opus",
    api_key = api_keys$anthropic,
    prompt_template = prompt_template,
    regression_formula = "LLM_LABEL ~ predictor1 + predictor2",
    coefficient_of_interest = "predictor1"  # We're interested in this predictor
)
```

The framework will automatically extract and report the coefficient, standard error, and p-value for your specified coefficient of interest.
