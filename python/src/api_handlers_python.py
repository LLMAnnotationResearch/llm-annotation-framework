#!/usr/bin/env python3
"""
API Handler functions for LLM Annotation Framework.
This module contains functions for calling different LLM providers through their APIs.
"""

import time
import re
import json
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text):
    """
    Clean and normalize text for API submission.
    
    Args:
        text: Text string to clean
        
    Returns:
        Cleaned text string
    """
    # Remove non-ASCII characters
    if isinstance(text, str):
        text = text.encode('ascii', 'ignore').decode('ascii')
    else:
        # Convert to string if not already
        text = str(text).encode('ascii', 'ignore').decode('ascii')
    
    # Normalize whitespace (replace multiple spaces with single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Trim whitespace
    text = text.strip()
    
    return text

def call_gpt4o(prompt, api_key, model_name="gpt-4o"):
    """
    Call the GPT-4o model via OpenAI API.
    
    Args:
        prompt: Full text prompt to send to the model
        api_key: OpenAI API key
        model_name: Specific OpenAI model to use (default: "gpt-4o")
            Other supported models: "gpt-4-0613", "gpt-4-1106-preview", etc.
        
    Returns:
        0 or 1 for successful classification, None if error occurs
    """
    # Basic rate limiting
    time.sleep(0.5)
    
    # Clean the prompt text
    clean_prompt = clean_text(prompt)
    
    # Prepare the API request
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that responds with only 0 or 1."},
            {"role": "user", "content": clean_prompt}
        ],
        "temperature": 0,
        "max_tokens": 10
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        # Handle rate limiting
        if response.status_code == 429:
            wait_time = 60
            logger.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            return call_gpt4o(prompt, api_key)
        
        # Check for other errors
        if response.status_code != 200:
            logger.error(f"Error: {response.status_code}")
            return None
        
        # Parse response
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Extract binary response (0 or 1)
        match = re.search(r'[0-1]', content)
        if match:
            return int(match.group())
        else:
            logger.warning(f"Could not extract numeric response from: {content}")
            return None
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return None
        
def calculate_api_cost(prompt, model, n_requests=1):
    """
    Calculate the estimated cost of an API call.
    
    Args:
        prompt: Example prompt to estimate token count
        model: Model identifier (e.g., "gpt4o", "claude", etc.)
        n_requests: Number of requests to estimate for (default: 1)
        
    Returns:
        Estimated cost in USD
    """
    # Calculate tokens in prompt (approximate)
    tokens = len(prompt) / 4  # rough approximation
    
    # Cost per 1K tokens (as of 2024)
    costs = {
        "gpt4o": {
            "input": 0.0025,    # $2.50 per 1M tokens = $0.0025 per 1K tokens
            "output": 0.01      # $10.00 per 1M tokens = $0.01 per 1K tokens
        },
        "gpt4o_mini": {
            "input": 0.00015,   # $0.150 per 1M tokens = $0.00015 per 1K tokens
            "output": 0.0006    # $0.600 per 1M tokens = $0.0006 per 1K tokens
        },
        "claude_opus": {
            "input": 0.0015,    # $1.50 per 1M tokens = $0.0015 per 1K tokens
            "output": 0.0075    # $7.50 per 1M tokens = $0.0075 per 1K tokens
        },
        "claude_sonnet": {
            "input": 0.0008,    # $0.80 per 1M tokens = $0.0008 per 1K tokens
            "output": 0.004     # $4.00 per 1M tokens = $0.004 per 1K tokens
        },
        "claude_haiku": {
            "input": 0.00025,   # $0.25 per 1M tokens = $0.00025 per 1K tokens
            "output": 0.00125   # $1.25 per 1M tokens = $0.00125 per 1K tokens
        },
        "deepseek_chat": {
            "input": 0.00027,   # $0.27 per 1M tokens = $0.00027 per 1K tokens
            "output": 0.0011    # $1.10 per 1M tokens = $0.0011 per 1K tokens
        }
    }
    
    # Get costs for the specified model
    if model not in costs:
        logger.warning(f"Cost data not available for model {model}")
        return 0.0
        
    model_costs = costs[model]
    
    # Estimate total cost (assuming short output for binary classification)
    total_cost = (tokens * model_costs["input"] + 10 * model_costs["output"]) / 1000 * n_requests
    return total_cost

def call_gpt4o_mini(prompt, api_key, model_name="gpt-4o-mini"):
    """
    Call the GPT-4o-mini model via OpenAI API.
    
    Args:
        prompt: Full text prompt to send to the model
        api_key: OpenAI API key
        model_name: Specific OpenAI model to use (default: "gpt-4o-mini")
        
    Returns:
        0 or 1 for successful classification, None if error occurs
    """
    # Basic rate limiting
    time.sleep(0.4)
    
    # Clean the prompt text
    clean_prompt = clean_text(prompt)
    
    # Prepare the API request
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that responds with only 0 or 1."},
            {"role": "user", "content": clean_prompt}
        ],
        "temperature": 0,
        "max_tokens": 10
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        # Handle rate limiting
        if response.status_code == 429:
            wait_time = 60
            logger.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            return call_gpt4o_mini(prompt, api_key)
        
        # Check for other errors
        if response.status_code != 200:
            logger.error(f"Error: {response.status_code}")
            return None
        
        # Parse response
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Extract binary response (0 or 1)
        match = re.search(r'[0-1]', content)
        if match:
            return int(match.group())
        else:
            logger.warning(f"Could not extract numeric response from: {content}")
            return None
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return None

# O1 model removed

# Deepseek Reasoner model removed

def call_deepseek_chat(prompt, api_key):
    """
    Call Deepseek Chat model.
    
    Args:
        prompt: Full text prompt to send to the model
        api_key: Deepseek API key
        
    Returns:
        0 or 1 for successful classification, None if error occurs
    """
    # Basic rate limiting
    time.sleep(0.25)
    
    # Clean the prompt text
    clean_prompt = clean_text(prompt)
    
    # Prepare the API request
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that responds with only 0 or 1."},
            {"role": "user", "content": clean_prompt}
        ],
        "temperature": 0,
        "max_tokens": 10,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        # Handle rate limiting
        if response.status_code == 429:
            wait_time = 60
            logger.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            return call_deepseek_chat(prompt, api_key)
        
        # Check for other errors
        if response.status_code != 200:
            logger.error(f"Error: {response.status_code} - {response.text}")
            return None
        
        # Parse response
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Extract binary response (0 or 1)
        match = re.search(r'[0-1]', content)
        if match:
            return int(match.group())
        else:
            logger.warning(f"Could not extract numeric response from: {content}")
            return None
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return None

def call_claude_opus(prompt, api_key):
    """
    Call the Claude-3-Opus model via Anthropic API.
    
    Args:
        prompt: Full text prompt to send to the model
        api_key: Anthropic API key
        
    Returns:
        0 or 1 for successful classification, None if error occurs
    """
    # Basic rate limiting
    time.sleep(0.5)  # Slightly longer rate limit for larger model
    
    # Clean the prompt text
    clean_prompt = clean_text(prompt)
    
    # Prepare the API request
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "claude-3-opus-20240229",
        "messages": [
            {"role": "user", "content": clean_prompt}
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "system": "You are a helpful assistant that responds with only 0 or 1."
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        # Handle rate limiting and server overload
        if response.status_code in (429, 529):
            wait_time = 60
            logger.warning(f"Rate limit/overload hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            return call_claude_opus(prompt, api_key)
        
        # Check for other errors
        if response.status_code != 200:
            logger.error(f"Error: {response.status_code} - {response.text}")
            return None
        
        # Parse response
        result = response.json()
        content = result["content"][0]["text"]
        
        # Extract binary response (0 or 1)
        match = re.search(r'[0-1]', content)
        if match:
            return int(match.group())
        else:
            logger.warning(f"Could not extract numeric response from: {content}")
            return None
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return None
    
def call_claude_sonnet(prompt, api_key):
    """
    Call the Claude-3-Sonnet model via Anthropic API.
    
    Args:
        prompt: Full text prompt to send to the model
        api_key: Anthropic API key
        
    Returns:
        0 or 1 for successful classification, None if error occurs
    """
    # Basic rate limiting
    time.sleep(0.25)
    
    # Clean the prompt text
    clean_prompt = clean_text(prompt)
    
    # Prepare the API request
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {"role": "user", "content": clean_prompt}
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "system": "You are a helpful assistant that responds with only 0 or 1."
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        # Handle rate limiting and server overload
        if response.status_code in (429, 529):
            wait_time = 60
            logger.warning(f"Rate limit/overload hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            return call_claude_sonnet(prompt, api_key)
        
        # Check for other errors
        if response.status_code != 200:
            logger.error(f"Error: {response.status_code} - {response.text}")
            return None
        
        # Parse response
        result = response.json()
        content = result["content"][0]["text"]
        
        # Extract binary response (0 or 1)
        match = re.search(r'[0-1]', content)
        if match:
            return int(match.group())
        else:
            logger.warning(f"Could not extract numeric response from: {content}")
            return None
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return None
    
def call_claude_haiku(prompt, api_key):
    """
    Call the Claude-3-Haiku model via Anthropic API.
    
    Args:
        prompt: Full text prompt to send to the model
        api_key: Anthropic API key
        
    Returns:
        0 or 1 for successful classification, None if error occurs
    """
    # Basic rate limiting
    time.sleep(0.15)  # Slightly faster rate limit for smaller model
    
    # Clean the prompt text
    clean_prompt = clean_text(prompt)
    
    # Prepare the API request
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "claude-3-5-haiku-20241022",
        "messages": [
            {"role": "user", "content": clean_prompt}
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "system": "You are a helpful assistant that responds with only 0 or 1."
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        # Handle rate limiting and server overload
        if response.status_code in (429, 529):
            wait_time = 60
            logger.warning(f"Rate limit/overload hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            return call_claude_haiku(prompt, api_key)
        
        # Check for other errors
        if response.status_code != 200:
            logger.error(f"Error: {response.status_code} - {response.text}")
            return None
        
        # Parse response
        result = response.json()
        content = result["content"][0]["text"]
        
        # Extract binary response (0 or 1)
        match = re.search(r'[0-1]', content)
        if match:
            return int(match.group())
        else:
            logger.warning(f"Could not extract numeric response from: {content}")
            return None
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return None