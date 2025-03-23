"""
Basic sensitivity analysis test script for the LLM Annotation Framework.
This script follows the sensitivity analysis example in the README.
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the evaluation functions
from src.evaluation_functions_python import run_sensitivity_analysis

def run_sensitivity_test():
    print("Running basic sensitivity analysis test for LLM Annotation Framework...")
    
    # Set your API keys - replace these with actual keys
    api_keys = {
        "openai": "[INSERT_YOUR_OPENAI_KEY_HERE]",
        "anthropic": "[INSERT_YOUR_ANTHROPIC_KEY_HERE]"
    }
    
    # Load your dataset
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        data = None
        
        data_path = os.path.join("data", "test_sample.csv")
        for encoding in encodings:
            try:
                print(f"Trying to load with {encoding} encoding...")
                data = pd.read_csv(data_path, encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                print(f"Failed with {encoding} encoding, trying next...")
        
        if data is None:
            raise ValueError("Could not load the file with any of the attempted encodings")
            
        print(f"Successfully loaded {len(data)} records from {data_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Load prompt variations
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        prompts_df = None
        
        prompts_path = os.path.join("data", "prompt_variations.csv")
        for encoding in encodings:
            try:
                print(f"Trying to load prompts with {encoding} encoding...")
                prompts_df = pd.read_csv(prompts_path, encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                print(f"Failed with {encoding} encoding, trying next...")
        
        if prompts_df is None:
            raise ValueError("Could not load the prompts file with any of the attempted encodings")
            
        print(f"Successfully loaded {len(prompts_df)} prompt variations from {prompts_path}")
        
        # Print a sample prompt
        if len(prompts_df) > 0:
            print("\nSample prompt strategy:")
            print(f"Strategy: {prompts_df.iloc[0]['strategy']}")
            print(f"Prompt: {prompts_df.iloc[0]['prompt'][:100]}...")  # Show first 100 chars
    except Exception as e:
        print(f"Error loading prompt variations: {e}")
        return
    
    # Use a small subset of prompts and data for testing
    prompt_limit = min(3, len(prompts_df))
    limited_prompts = prompts_df.head(prompt_limit)
    
    test_size = min(10, len(data))  # Use at least 10 samples for regression
    test_data = data.head(test_size).copy()
    
    print(f"\nRunning sensitivity analysis with {prompt_limit} prompts and {test_size} samples...")
    
    # Create a folder for results
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"sensitivity_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which model to use - defaulting to gpt4o_mini
    model_name = "gpt4o_mini"
    api_key = api_keys["openai"]
    
    # Run sensitivity analysis
    try:
        results = run_sensitivity_analysis(
            prompts_df=limited_prompts,
            data=test_data,
            api_key=api_key,
            model_name=model_name,
            text_column="text",
            human_label_column="HUMAN",
            regression_formula="success ~ LLM_LABEL + lgoal + year",
            regression_family="gaussian",
            save_path=output_dir,
            figure_name=f"sensitivity_{model_name}"
        )
        
        # View results summary
        summary_df = results['results']
        print("\nSensitivity Analysis Results:")
        print(summary_df[['strategy', 'accuracy', 'reg_coef', 'p_value', 'cost']])
        
        if 'plot_path' in results and results['plot_path']:
            print(f"\nPlot saved to: {results['plot_path']}")
        
        # Calculate the mean and standard deviation of the coefficient
        mean_coef = summary_df['reg_coef'].mean()
        sd_coef = summary_df['reg_coef'].std()
        print(f"\nMean coefficient: {mean_coef:.4f} (SD: {sd_coef:.4f})")
        
        print("\nSensitivity analysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during sensitivity analysis: {e}")

if __name__ == "__main__":
    run_sensitivity_test()
