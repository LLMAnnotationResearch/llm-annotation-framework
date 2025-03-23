import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to the path so we can import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the evaluation functions
from src.evaluation_functions_python import (
    evaluate_model, run_sensitivity_analysis, load_prompt_variations
)
print("Successfully imported evaluation functions")

def test_evaluation_functions():
    """
    Simplified test script for the evaluation functions in the LLM Annotation Framework.
    This demonstrates basic usage of the functions for researchers.
    """
    # Setup API keys
    api_keys = {
        "openai": "[INSERT OPENAI KEY HERE]",
        "anthropic": "[INSERT ANTHROPIC KEY HERE]"
    }
    
    # Step 1: Load the test data
    print("Loading test data...")
    try:
        # Try different potential data paths
        potential_paths = [
            os.path.join("data", "test_sample.csv"),  # If running from main directory
            os.path.join("..", "data", "test_sample.csv"),  # If running from python/tests
        ]
        
        data_path = None
        for path in potential_paths:
            if os.path.exists(path):
                data_path = path
                print(f"Found data file at: {path}")
                break
        
        if data_path is None:
            print("Could not find data file. Please check the file path.")
            return
        
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        data = None
        
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
            
        print(f"Loaded {len(data)} rows from {data_path}")
        print(f"Columns: {', '.join(data.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Step 2: Load prompt variations
    print("\nLoading prompt variations...")
    try:
        # Try different potential data paths
        potential_paths = [
            os.path.join("data", "prompt_variations.csv"),  # If running from main directory
            os.path.join("..", "data", "prompt_variations.csv"),  # If running from python/tests
        ]
        
        prompts_path = None
        for path in potential_paths:
            if os.path.exists(path):
                prompts_path = path
                print(f"Found prompts file at: {path}")
                break
        
        if prompts_path is None:
            print("Could not find prompts file. Please check the file path.")
            return
        
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        prompts_df = None
        
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
            
        print(f"Loaded {len(prompts_df)} prompt variations from {prompts_path}")
        
        # Show a sample prompt
        if len(prompts_df) > 0:
            print("\nSample prompt (strategy: {}):\n{}".format(
                prompts_df.iloc[0]['strategy'],
                prompts_df.iloc[0]['prompt']
            ))
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return
    
    # Step 3: Test a single model evaluation with fixed parameters
    print("\n" + "="*80)
    print("TESTING SINGLE MODEL EVALUATION")
    print("="*80)
    
    # Ask user which model to test
    print("\nAvailable models:")
    models = ["gpt4o", "gpt4o_mini", "claude_opus", "claude_sonnet", "claude_haiku"]
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    model_choice = input("\nWhich model would you like to test? (1-5, or 'skip' to skip API testing): ")
    if model_choice.lower() == 'skip':
        print("Skipping API testing.")
    else:
        try:
            model_idx = int(model_choice) - 1
            model_name = models[model_idx]
            
            # Determine which API key to use
            api_key = api_keys["openai"] if model_name.startswith("gpt") else api_keys["anthropic"]
            
            # Test with a small subset of data
            test_size = min(5, len(data))
            test_data = data.head(test_size).copy()
            print(f"\nTesting {model_name} with {test_size} samples...")
            
            # Use a simple regression formula with success as the dependent variable
            reg_formula = "success ~ LLM_LABEL" 
            print(f"Using regression formula: {reg_formula}")
            
            # Run evaluation
            results = evaluate_model(
                data=test_data,
                model_name=model_name,
                api_key=api_key,
                prompt_template=prompts_df.iloc[0]['prompt'],
                text_column="text",
                human_label_column="HUMAN",
                regression_formula=reg_formula,
                regression_family="gaussian",
                verbose=True
            )
            
            # Print results
            print("\nEvaluation Results:")
            print(f"Accuracy: {results['metrics'].get('accuracy', 'N/A')}")
            print(f"Cost: ${results['cost']:.6f}")
            print(f"Valid predictions: {results['n_valid']}/{results['n_total']}")
            print(f"Regression coefficient: {results['regression']['coefficient']}")
            print(f"Regression p-value: {results['regression']['p_value']}")
            
            # Print actual LLM responses
            print("\nLLM Responses:")
            for i, row in results['results'].iterrows():
                print(f"Sample {i+1}: {row.get('LLM_LABEL', 'N/A')}")
            
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
    
    # Step 4: Test sensitivity analysis with fixed parameters
    print("\n" + "="*80)
    print("TESTING SENSITIVITY ANALYSIS")
    print("="*80)
    
    run_full = input("\nWould you like to run a sensitivity analysis? (yes/no): ")
    if run_full.lower() == 'yes':
        # Ask which model to use
        model_choice = input("\nWhich model would you like to use? (1-5): ")
        try:
            model_idx = int(model_choice) - 1
            model_name = models[model_idx]
            
            # Determine which API key to use
            api_key = api_keys["openai"] if model_name.startswith("gpt") else api_keys["anthropic"]
            
            # Run sensitivity analysis with limited prompts
            prompt_limit = min(3, len(prompts_df))
            limited_prompts = prompts_df.head(prompt_limit)
            print(f"\nRunning sensitivity analysis with {prompt_limit} prompts...")
            
            # Use a simple regression formula with success as the dependent variable
            reg_formula = "success ~ LLM_LABEL"
            print(f"Using regression formula: {reg_formula}")
            
            # Create a timestamp for output files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("output", timestamp)
            os.makedirs(output_dir, exist_ok=True)
            
            # Test with a small subset of data
            test_size = min(10, len(data))
            test_data = data.head(test_size).copy()
            
            # Run sensitivity analysis
            results = run_sensitivity_analysis(
                prompts_df=limited_prompts,
                data=test_data,
                api_key=api_key,
                model_name=model_name,
                text_column="text",
                human_label_column="HUMAN",
                regression_formula=reg_formula,
                regression_family="gaussian",
                save_path=output_dir,
                figure_name=f"sensitivity_{model_name}"
            )
            
            # Print results summary
            print("\nSensitivity Analysis Results:")
            print(f"Results saved to: {output_dir}")
            if results.get('plot_path'):
                print(f"Plot saved to: {results['plot_path']}")
            
            # Show the first few rows of results
            print("\nResults summary:")
            print(results['results'].head())
            
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
    else:
        print("Skipping sensitivity analysis.")
    
    print("\nTest script completed.")

if __name__ == "__main__":
    test_evaluation_functions()