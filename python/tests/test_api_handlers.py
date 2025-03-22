
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api_handlers_python import clean_text, call_gpt4o, call_gpt4o_mini, call_claude_opus, call_claude_sonnet, call_claude_haiku, calculate_api_cost

api_keys = {
  "openai": "[ADD YOUR OPENAI KEY HERE]",
  "anthropic": "[ADD YOUR ANTHROPIC KEY HERE]"
}

def test_api_handlers():
    # Test text cleaning
    print("Testing text cleaning...")
    original_text = "Product   description with    irregular spacing."
    cleaned_text = clean_text(original_text)
    print(f"Original: '{original_text}'")
    print(f"Cleaned: '{cleaned_text}'")
    print()
    
    # Simple test prompt
    test_prompt = "Please classify if this product benefits society beyond its consumers. Answer only with 0 or 1: Solar-powered water purifier for communities."
    
    # Test cost calculation for all models
    print("Testing cost calculation...")
    models = ["gpt4o", "gpt4o_mini", "claude_opus", "claude_sonnet", "claude_haiku"]
    for model in models:
        cost = calculate_api_cost(test_prompt, model)
        print(f"Estimated {model} cost: ${cost:.6f}")
    print()
    
    # Only proceed with API calls if user confirms
    proceed = input("Do you want to test actual API calls? This will incur costs. (yes/no): ")
    if proceed.lower() != "yes":
        print("Skipping API calls.")
        return
    
    # Test OpenAI API calls
    try:
        print("\nTesting GPT-4o API call...")
        result = call_gpt4o(test_prompt, api_keys["openai"])
        print(f"GPT-4o result: {result}")
    except Exception as e:
        print(f"GPT-4o API error: {e}")
    
    try:
        print("\nTesting GPT-4o-mini API call...")
        result = call_gpt4o_mini(test_prompt, api_keys["openai"])
        print(f"GPT-4o-mini result: {result}")
    except Exception as e:
        print(f"GPT-4o-mini API error: {e}")
    
    # Test Anthropic API calls
    try:
        print("\nTesting Claude Opus API call...")
        result = call_claude_opus(test_prompt, api_keys["anthropic"])
        print(f"Claude Opus result: {result}")
    except Exception as e:
        print(f"Claude Opus API error: {e}")
        
    try:
        print("\nTesting Claude Sonnet API call...")
        result = call_claude_sonnet(test_prompt, api_keys["anthropic"])
        print(f"Claude Sonnet result: {result}")
    except Exception as e:
        print(f"Claude Sonnet API error: {e}")
        
    try:
        print("\nTesting Claude Haiku API call...")
        result = call_claude_haiku(test_prompt, api_keys["anthropic"])
        print(f"Claude Haiku result: {result}")
    except Exception as e:
        print(f"Claude Haiku API error: {e}")

if __name__ == "__main__":
    test_api_handlers()