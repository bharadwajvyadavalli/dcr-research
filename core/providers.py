"""
LLM Provider Implementations for DCR Research
Supports OpenAI, Google Gemini, and Anthropic Claude
"""

import os
import time
from typing import Optional, Dict, Any
import openai
import google.generativeai as genai
import anthropic


class OpenAIProvider:
    """OpenAI provider implementation"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, api_key: Optional[str] = None,
                 temperature: float = 0.0, max_retries: int = 3):
        """
        Initialize OpenAI provider

        Args:
            config: Config dict with 'mode' and provider settings
            api_key: API key (overrides config)
            temperature: Temperature setting
            max_retries: Max retry attempts
        """
        # Get configuration
        if config:
            mode = config.get('mode', 'testing')
            provider_config = config['providers']['openai']
            self.model = provider_config['models'][mode]
            self.api_key = api_key or provider_config['api_key']
            temp = temperature if temperature != 0.0 else provider_config.get('temperature', 0.0)
            self.max_tokens = provider_config.get('max_tokens', 1000)
        else:
            self.model = "gpt-4o-mini"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            temp = temperature
            self.max_tokens = 1000

        model_name = self.model.replace("gpt-", "GPT-").replace("-mini", "-mini").replace("-", "")
        self.name = f"OpenAI-{model_name}"
        self.temperature = temp
        self.max_retries = max_retries
        self.request_count = 0
        self.failure_count = 0

        if not self.api_key:
            raise ValueError("OpenAI API key not found")

        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_response(self, prompt: str, model: Optional[str] = None, max_tokens: Optional[int] = None, **kwargs) -> Optional[str]:
        """Generate response from OpenAI"""
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

    def generate_with_retry(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response with retry logic"""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                self.request_count += 1
                response = self.generate_response(prompt, **kwargs)

                if response:
                    return {
                        'response': response,
                        'provider': self.name,
                        'success': True,
                        'error': None,
                        'attempts': attempt
                    }

            except Exception as e:
                last_error = str(e)

                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)

        # All retries failed
        self.failure_count += 1
        return {
            'response': None,
            'provider': self.name,
            'success': False,
            'error': last_error,
            'attempts': self.max_retries
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics"""
        return {
            'provider': self.name,
            'total_requests': self.request_count,
            'failures': self.failure_count,
            'success_rate': (self.request_count - self.failure_count) / self.request_count if self.request_count > 0 else 0
        }


class GeminiProvider:
    """Google Gemini provider implementation"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, api_key: Optional[str] = None,
                 temperature: float = 0.0, max_retries: int = 3):
        """
        Initialize Gemini provider

        Args:
            config: Config dict with 'mode' and provider settings
            api_key: API key (overrides config)
            temperature: Temperature setting
            max_retries: Max retry attempts
        """
        # Get API key
        if config:
            provider_config = config['providers']['gemini']
            self.api_key = api_key or provider_config['api_key']
        else:
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("Google API key not found")

        genai.configure(api_key=self.api_key)

        # Get model name
        if config:
            mode = config.get('mode', 'testing')
            provider_config = config['providers']['gemini']
            self.model_name = provider_config['models'][mode]
            temp = temperature if temperature != 0.0 else provider_config.get('temperature', 0.0)
            self.max_tokens = provider_config.get('max_tokens', 8192)
        else:
            self.model_name = None
            temp = temperature
            self.max_tokens = 8192

        # Auto-detect valid model
        self.model_name = self._get_valid_model(self.model_name)
        self.name = f"Google-{self.model_name}"
        self.temperature = temp
        self.max_retries = max_retries
        self.request_count = 0
        self.failure_count = 0
        self.model = genai.GenerativeModel(self.model_name)

    def _get_valid_model(self, preferred_model: Optional[str] = None) -> str:
        """Get a valid Gemini model name, auto-detecting if needed"""
        try:
            models = genai.list_models()
            gemini_models = [
                m.name for m in models
                if 'gemini' in m.name.lower() and 'generateContent' in m.supported_generation_methods
            ]

            if not gemini_models:
                raise ValueError("No Gemini models found")

            # Try preferred model
            if preferred_model:
                for model in gemini_models:
                    if model.endswith(preferred_model) or model == f"models/{preferred_model}":
                        return model

            # Prefer flash models
            flash_models = [m for m in gemini_models if 'flash' in m.lower()]
            if flash_models:
                return flash_models[0]

            return gemini_models[0]

        except Exception:
            return 'models/gemini-1.5-flash'

    def generate_response(self, prompt: str, max_tokens: Optional[int] = None, **kwargs) -> Optional[str]:
        """Generate response from Gemini"""
        tokens_to_use = max_tokens if max_tokens is not None else self.max_tokens
        generation_config = genai.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=tokens_to_use,
            **{k: v for k, v in kwargs.items() if k != 'max_output_tokens'}
        )

        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text

    def generate_with_retry(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response with retry logic"""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                self.request_count += 1
                response = self.generate_response(prompt, **kwargs)

                if response:
                    return {
                        'response': response,
                        'provider': self.name,
                        'success': True,
                        'error': None,
                        'attempts': attempt
                    }

            except Exception as e:
                last_error = str(e)

                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)

        # All retries failed
        self.failure_count += 1
        return {
            'response': None,
            'provider': self.name,
            'success': False,
            'error': last_error,
            'attempts': self.max_retries
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics"""
        return {
            'provider': self.name,
            'total_requests': self.request_count,
            'failures': self.failure_count,
            'success_rate': (self.request_count - self.failure_count) / self.request_count if self.request_count > 0 else 0
        }


class AnthropicProvider:
    """Anthropic Claude provider implementation"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, api_key: Optional[str] = None,
                 temperature: float = 0.0, max_retries: int = 3):
        """
        Initialize Anthropic provider

        Args:
            config: Config dict with 'mode' and provider settings
            api_key: API key (overrides config)
            temperature: Temperature setting
            max_retries: Max retry attempts
        """
        # Get configuration
        if config:
            mode = config.get('mode', 'testing')
            provider_config = config['providers']['anthropic']
            self.model = provider_config['models'][mode]
            self.api_key = api_key or provider_config['api_key']
            temp = temperature if temperature != 0.0 else provider_config.get('temperature', 0.0)
            self.max_tokens = provider_config.get('max_tokens', 1000)
        else:
            self.model = "claude-3-5-haiku-20241022"
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            temp = temperature
            self.max_tokens = 1000

        model_display = "Haiku" if "haiku" in self.model.lower() else "Sonnet" if "sonnet" in self.model.lower() else "Claude"
        self.name = f"Anthropic-{model_display}"
        self.temperature = temp
        self.max_retries = max_retries
        self.request_count = 0
        self.failure_count = 0

        if not self.api_key:
            raise ValueError("Anthropic API key not found")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate_response(self, prompt: str, model: Optional[str] = None, max_tokens: Optional[int] = None, **kwargs) -> Optional[str]:
        """Generate response from Claude"""
        message = self.client.messages.create(
            model=model or self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return message.content[0].text

    def generate_with_retry(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response with retry logic"""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                self.request_count += 1
                response = self.generate_response(prompt, **kwargs)

                if response:
                    return {
                        'response': response,
                        'provider': self.name,
                        'success': True,
                        'error': None,
                        'attempts': attempt
                    }

            except Exception as e:
                last_error = str(e)

                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)

        # All retries failed
        self.failure_count += 1
        return {
            'response': None,
            'provider': self.name,
            'success': False,
            'error': last_error,
            'attempts': self.max_retries
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics"""
        return {
            'provider': self.name,
            'total_requests': self.request_count,
            'failures': self.failure_count,
            'success_rate': (self.request_count - self.failure_count) / self.request_count if self.request_count > 0 else 0
        }
