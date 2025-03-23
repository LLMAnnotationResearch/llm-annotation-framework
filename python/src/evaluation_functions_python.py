import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import glm
from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm
from tqdm import tqdm
import time
import sys
import os
import re
import json
from typing import List, Dict, Any, Callable, Union, Optional, Tuple

# Assume api_handlers_python.py is in the same directory
from src.api_handlers_python import (
    clean_text, call_gpt4o, call_gpt4o_mini, call_claude_opus, 
    call_claude_sonnet, call_claude_haiku, calculate_api_cost
)

def evaluate_model(
    data: pd.DataFrame,
    model_name: str,
    api_key: str,
    prompt_template: str,
    text_column: str = "text",
    human_label_column: str = "HUMAN",
    regression_formula: Optional[str] = None,
    regression_family: str = "gaussian",
    max_retries: int = 3,
    retry_delay: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate an LLM model for classification performance and regression impact.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the text to classify and ground truth labels
    model_name : str
        Name of the model to use (gpt4o, gpt4o_mini, claude_opus, claude_sonnet, claude_haiku)
    api_key : str
        API key for the relevant service
    prompt_template : str
        Template text for the prompt (text will be appended to this)
    text_column : str
        Name of the column containing the text to classify
    human_label_column : str
        Name of the column containing the ground truth labels
    regression_formula : str, optional
        R-style formula for regression (e.g., "success ~ LLM_LABEL + control1 + control2")
    regression_family : str
        Family for GLM regression (e.g., "binomial", "gaussian")
    max_retries : int
        Maximum number of retries for API calls
    retry_delay : int
        Delay between retries in seconds
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    Dict containing:
        - results: DataFrame with original data and LLM predictions
        - metrics: Dict of evaluation metrics
        - cost: Total API cost
        - regression: Dict of regression results (if applicable)
        - n_valid: Number of valid predictions
        - n_total: Total number of examples
    """
    # Clean text first
    if verbose:
        print("Cleaning text data...")
    
    data = data.copy()
    data['clean_text'] = data[text_column].apply(clean_text)
    
    # Choose the appropriate model function
    model_functions = {
        "gpt4o": call_gpt4o,
        "gpt4o_mini": call_gpt4o_mini,
        "claude_opus": call_claude_opus,
        "claude_sonnet": call_claude_sonnet,
        "claude_haiku": call_claude_haiku
    }
    
    if model_name not in model_functions:
        raise ValueError(f"Unsupported model: {model_name}. Available models: {list(model_functions.keys())}")
    
    model_function = model_functions[model_name]
    
    # Process all examples
    if verbose:
        print(f"Processing {len(data)} examples with {model_name}...")
        iterator = tqdm(data['clean_text'])
    else:
        iterator = data['clean_text']
    
    llm_labels = []
    
    for text in iterator:
        full_prompt = f"{prompt_template} {text}"
        
        # Try with retries
        for attempt in range(max_retries):
            try:
                result = model_function(full_prompt, api_key)
                llm_labels.append(result)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    if verbose:
                        print(f"Error: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    if verbose:
                        print(f"Failed after {max_retries} attempts: {e}")
                    llm_labels.append(np.nan)
    
    # Add results to data
    data['LLM_LABEL'] = llm_labels
    
    # Calculate metrics
    valid_preds = ~data['LLM_LABEL'].isna()
    n_valid = valid_preds.sum()
    n_total = len(data)
    
    metrics = {}
    if n_valid > 0:
        valid_data = data[valid_preds]
        
        # Calculate classification metrics
        metrics['accuracy'] = (valid_data[human_label_column] == valid_data['LLM_LABEL']).mean()
        
        # False positive and negative rates
        fps = ((valid_data[human_label_column] == 0) & (valid_data['LLM_LABEL'] == 1)).sum()
        fns = ((valid_data[human_label_column] == 1) & (valid_data['LLM_LABEL'] == 0)).sum()
        
        metrics['false_positive_rate'] = fps / n_total
        metrics['false_negative_rate'] = fns / n_total
    else:
        metrics = {
            'accuracy': float('nan'),
            'false_positive_rate': float('nan'),
            'false_negative_rate': float('nan')
        }
    
    # Calculate costs
    total_cost = sum(calculate_api_cost(f"{prompt_template} {text}", model_name) for text in data['clean_text'])
    
    # Run regression if formula is provided and we have enough data
    reg_results = {
        'coefficient': float('nan'),
        'std_error': float('nan'),
        'p_value': float('nan'),
        'converged': False
    }
    
    if regression_formula is not None and n_valid >= 10:
        if verbose:
            print("Running regression analysis...")
        
        # Parse the formula to extract the dependent variable
        match = re.match(r'(\w+)\s*~\s*.*', regression_formula)
        if not match:
            print("Invalid regression formula format. Should be like 'y ~ x1 + x2'")
        else:
            try:
                # Replace LLM_LABEL in the formula if it exists
                formula = regression_formula.replace("LLM_LABEL", "LLM_LABEL")
                
                # Prepare data for regression
                reg_data = data[valid_preds].copy()
                
                # Print regression data info for debugging
                if verbose:
                    print(f"\nRegression data shape: {reg_data.shape}")
                    print(f"Regression formula: {formula}")
                    print(f"Regression columns available: {', '.join(reg_data.columns)}")
                
                # Run the regression
                if regression_family.lower() == "binomial":
                    model = sm.formula.glm(formula=formula, data=reg_data, family=sm.families.Binomial())
                elif regression_family.lower() == "gaussian":
                    model = sm.formula.ols(formula=formula, data=reg_data)
                else:
                    raise ValueError(f"Unsupported regression family: {regression_family}")
                
                results = model.fit()
                
                # Extract coefficient for LLM_LABEL if it exists in the model
                if "LLM_LABEL" in results.params.index:
                    reg_results = {
                        'coefficient': results.params["LLM_LABEL"],
                        'std_error': results.bse["LLM_LABEL"],
                        'p_value': results.pvalues["LLM_LABEL"],
                        'converged': results.mle_retvals['converged'] if hasattr(results, 'mle_retvals') else True,
                        'full_results': results.summary().as_text() if verbose else None
                    }
                else:
                    print("Warning: LLM_LABEL not found in regression results")
            except Exception as e:
                print(f"Regression error: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
    
    # Return all results
    return {
        'results': data,
        'metrics': metrics,
        'cost': total_cost,
        'regression': reg_results,
        'n_valid': n_valid,
        'n_total': n_total
    }

def run_sensitivity_analysis(
    prompts_df: pd.DataFrame,
    data: pd.DataFrame,
    api_key: str,
    model_name: str = "gpt4o",
    text_column: str = "text",
    human_label_column: str = "HUMAN",
    regression_formula: Optional[str] = None,
    regression_family: str = "gaussian",
    save_path: str = "./",
    figure_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a sensitivity analysis across multiple prompt variations.
    
    Parameters:
    -----------
    prompts_df : pd.DataFrame
        DataFrame containing prompt variations with columns 'prompt' and 'strategy'
    data : pd.DataFrame
        DataFrame containing the text to classify and ground truth labels
    api_key : str
        API key for the relevant service
    model_name : str
        Name of the model to use
    text_column : str
        Name of the column containing the text to classify
    human_label_column : str
        Name of the column containing the ground truth labels
    regression_formula : str, optional
        R-style formula for regression
    regression_family : str
        Family for GLM regression
    save_path : str
        Path to save intermediate results and plots
    figure_name : str, optional
        Name for the output figure (defaults to model_name)
    
    Returns:
    --------
    Dict containing:
        - results_df: DataFrame with results for each prompt
        - plot: Path to the saved plot
    """
    if figure_name is None:
        figure_name = f"coefficient_distribution_{model_name}"
    
    # Create save path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize results dataframe
    results_df = pd.DataFrame(columns=[
        'prompt_id', 'strategy', 'accuracy', 'reg_coef', 'reg_se', 'p_value',
        'converged', 'n_valid', 'n_total', 'cost'
    ])
    
    # For each prompt variation
    for i, row in enumerate(prompts_df.itertuples(), 1):
        print(f"\nProcessing prompt {i}/{len(prompts_df)} ({row.strategy})...")
        
        # Run evaluation
        results = evaluate_model(
            data,
            model_name,
            api_key,
            row.prompt,
            text_column=text_column,
            human_label_column=human_label_column,
            regression_formula=regression_formula,
            regression_family=regression_family
        )
        
        # Add results to dataframe
        new_row = {
            'prompt_id': i,
            'strategy': row.strategy,
            'accuracy': results['metrics']['accuracy'],
            'reg_coef': results['regression']['coefficient'],
            'reg_se': results['regression']['std_error'],
            'p_value': results['regression']['p_value'],
            'converged': results['regression']['converged'],
            'n_valid': results['n_valid'],
            'n_total': results['n_total'],
            'cost': results['cost']
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save intermediate results
        results_file = os.path.join(save_path, f"sensitivity_analysis_{model_name}.csv")
        results_df.to_csv(results_file, index=False)
        
        print(f"Results saved to {results_file}")
    
    # Filter out non-converged models for the histogram
    plot_df = results_df[results_df['converged'] == True].copy()
    
    if len(plot_df) > 0:
        # Calculate statistics for reference lines
        mean_coef = plot_df['reg_coef'].mean()
        sd_coef = plot_df['reg_coef'].std()
        
        # Create plot
        plt.figure(figsize=(10, 7))
        sns.histplot(plot_df['reg_coef'], bins=20, kde=True, color='steelblue')
        
        # Add reference lines
        plt.axvline(mean_coef, color='red', linestyle='dashed', linewidth=1.5, 
                   label=f'Mean: {mean_coef:.2f}')
        plt.axvline(mean_coef - sd_coef, color='darkred', linestyle='dotted', linewidth=1.5,
                   label=f'-1 SD: {(mean_coef - sd_coef):.2f}')
        plt.axvline(mean_coef + sd_coef, color='darkred', linestyle='dotted', linewidth=1.5,
                   label=f'+1 SD: {(mean_coef + sd_coef):.2f}')
        
        # Labels and title
        plt.title(f"Distribution of Regression Coefficients ({model_name})")
        plt.xlabel("Coefficient Value")
        plt.ylabel("Count")
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(save_path, f"{figure_name}.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Plot saved to {plot_path}")
    else:
        print("Warning: No converged models to plot")
        plot_path = None
    
    return {
        'results': results_df,
        'plot_path': plot_path
    }

# Helper for loading prompt variations from text file
def load_prompt_variations(file_path: str) -> pd.DataFrame:
    """
    Load prompt variations from a text file with a specific format.
    
    The file should have sections starting with "## Strategy Name",
    followed by the prompt text.
    
    Parameters:
    -----------
    file_path : str
        Path to the text file with prompt variations
    
    Returns:
    --------
    pd.DataFrame with columns 'strategy' and 'prompt'
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    sections = re.split(r'##\s+(.+?)\n', content)[1:]  # Skip the first empty element
    
    strategies = []
    prompts = []
    
    for i in range(0, len(sections), 2):
        if i + 1 < len(sections):
            strategies.append(sections[i].strip())
            prompts.append(sections[i + 1].strip())
    
    return pd.DataFrame({
        'strategy': strategies,
        'prompt': prompts
    })

# Example usage code:
if __name__ == "__main__":
    print("Example usage of evaluation functions:")
    print("1. Load your data")
    print("data = pd.read_csv('path/to/your/data.csv')")
    print("\n2. Load prompt variations")
    print("prompts = load_prompt_variations('path/to/prompt_variations.txt')")
    print("\n3. Run a single model evaluation")
    print("results = evaluate_model(data, 'gpt4o', 'YOUR_API_KEY', prompts.iloc[0].prompt)")
    print("\n4. Run a full sensitivity analysis")
    print("analysis = run_sensitivity_analysis(prompts, data, 'YOUR_API_KEY', 'gpt4o')")