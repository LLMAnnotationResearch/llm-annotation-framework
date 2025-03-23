"""
Basic test script for the LLM Annotation Framework that follows the README example.
Run this script to verify that the basic functionality works as described.
"""

import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the evaluation functions
from src.evaluation_functions_python import evaluate_model

def run_basic_test():
    print("Running basic Python test for LLM Annotation Framework...")
    
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
        print(f"Columns: {', '.join(data.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Define your prompt template
    prompt_template = """
    Please classify if this product benefits society beyond its consumers.
    Answer only with 0 or 1. Here is the text:
    """
    
    # Use a small subset for testing
    test_size = 5
    test_data = data.head(test_size).copy()
    print(f"\nTesting with {test_size} samples...")
    
    # Determine which model to use - defaulting to gpt4o_mini
    model_name = "gpt4o_mini"  # Options: gpt4o, gpt4o_mini, claude_opus, claude_sonnet, claude_haiku
    api_key = api_keys["openai"]  # Use appropriate key based on the model
    
    # Run the model on your data
    try:
        results = evaluate_model(
            data=test_data,
            model_name=model_name,
            api_key=api_key,
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
        print(f"LLM label coefficient: {regression['coefficient']}")
        print(f"p-value: {regression['p_value']}")
        
        # Print model responses
        print("\nModel responses:")
        for i, row in annotated_data.iterrows():
            print(f"Sample {i+1}: {row.get('LLM_LABEL', 'N/A')}")
        
        print("\nBasic test completed successfully!")
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")

if __name__ == "__main__":
    run_basic_test()
