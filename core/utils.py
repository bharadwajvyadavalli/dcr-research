"""
Utility functions for DCR project
Shared functions for configuration, data loading, and model management
"""

import os
import json
import pickle
from typing import Dict, Any
import yaml
from dotenv import load_dotenv


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml with environment variable expansion."""
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Expand environment variables
    api_key = config['openai']['api_key']
    if api_key.startswith('${') and api_key.endswith('}'):
        env_var = api_key[2:-1]  # Remove ${ and }
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"Environment variable {env_var} not found")

    config['openai']['api_key'] = api_key
    return config


def load_mmlu_data(split: str = "train") -> Dict[str, Any]:
    """Load MMLU data with embeddings from processed files."""
    filename = f"data/mmlu_{split}.json"
    embeddings_file = f"data/{split}_embeddings.pkl"

    print(f"📚 Loading {split} data from {filename}")

    # Load main data
    with open(filename, 'r') as f:
        data = json.load(f)

    # Load embeddings from pickle for faster access
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)

    print(f"✅ Loaded {len(data['questions'])} {split} questions")
    print(f"✅ Loaded embeddings shape: {embeddings.shape}")

    return {
        'questions': data['questions'],
        'embeddings': embeddings,
        'metadata': data['metadata']
    }


def load_stratified_data():
    """Load MMLU training data with embeddings."""
    train_data = load_mmlu_data("train")
    print(f"📚 Loaded {len(train_data['questions'])} training questions")
    return train_data


def save_model(model, filename: str):
    """Save model to pickle file."""
    os.makedirs('models', exist_ok=True)
    with open(f'models/{filename}', 'wb') as f:
        pickle.dump(model, f)
    print(f"💾 Saved model to models/{filename}")


def load_model(filename: str, config: Dict[str, Any] = None):
    """Load model from pickle file and re-initialize clients if needed."""
    # Handle both 'model.pkl' and 'models/model.pkl' formats
    filepath = filename if filename.startswith('models/') else f'models/{filename}'

    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    # Re-initialize OpenAI clients for neural routers if config is provided
    if config and hasattr(model, 'client'):
        from openai import OpenAI, AsyncOpenAI
        model.client = OpenAI(api_key=config['openai']['api_key'])
        model.async_client = AsyncOpenAI(api_key=config['openai']['api_key'])

    print(f"📦 Loaded model from {filepath}")
    return model


def init_openai_clients(config: Dict[str, Any]):
    """Initialize OpenAI synchronous and asynchronous clients."""
    from openai import OpenAI, AsyncOpenAI

    api_key = config['openai']['api_key']
    client = OpenAI(api_key=api_key)
    async_client = AsyncOpenAI(api_key=api_key)

    return client, async_client


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model configuration parameters."""
    return {
        'embedding_model': config['openai'].get('embedding_model', 'text-embedding-3-small'),
        'chat_model': config['openai'].get('chat_model', 'gpt-4'),
        'max_tokens': config['openai'].get('max_tokens', 200),
        'temperature': config['openai'].get('temperature', 0.1)
    }


def load_inference_config() -> Dict[str, Any]:
    """
    Load inference pipeline configuration from config.yaml

    Returns:
        Dict with mode and provider configurations
    """
    load_dotenv()  # Load environment variables

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Get mode (testing or production)
    mode = config.get('mode', 'testing')

    # Expand environment variables for API keys
    providers = config.get('llm_providers', {})
    for provider_name, provider_config in providers.items():
        api_key = provider_config.get('api_key', '')
        if api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]  # Remove ${ and }
            api_key = os.getenv(env_var)
            if not api_key:
                raise ValueError(f"Environment variable {env_var} not found for {provider_name}")
            provider_config['api_key'] = api_key

    # Get templates and max_tokens configuration
    templates = config.get('templates', {})
    
    return {
        'mode': mode,
        'providers': providers,
        'templates': templates
    }
