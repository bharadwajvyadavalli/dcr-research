#!/usr/bin/env python3
"""
DCR Multi-Provider Pipeline with Router Comparison

Tests 3 routing strategies on all providers:
1. Baseline (always verbose template)
2. Simple Neural Router (embedding-based)
3. RoBERTa Router (transformer-based)

For N questions and 3 providers:
  Total API calls = N × 3 strategies × 3 providers = 9N calls

Usage:
    python3 src/inference/run_dcr_pipeline.py --num-questions 100
    python3 src/inference/run_dcr_pipeline.py --num-questions 1000 --provider openai
"""

import os
import sys

# Suppress HuggingFace tokenizers fork warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Fix pickle compatibility for MLPRouter
# (handles pickles created when train.py was run as __main__)
import training.train as train
sys.modules['__main__'].MLPRouter = train.MLPRouter

from core.utils import load_inference_config
from core.providers import OpenAIProvider, GeminiProvider, AnthropicProvider
from data_prep.s3 import S3Manager

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('boto3').setLevel(logging.WARNING)


# Template definitions - Updated for MMLU multiple-choice format
SYSTEM_PROMPTS = {
    'minimal': """Answer this multiple-choice question. Start your response with "Answer: X" where X is the correct letter (A, B, C, or D), then provide a brief one-sentence explanation.""",

    'standard': """Answer this multiple-choice question. Format your response as:
Answer: [Letter]

Then provide:
1. A clear, step-by-step explanation of your reasoning
2. Brief explanation of why other options are incorrect

Replace [Letter] with A, B, C, or D.""",

    'verbose': """Answer this multiple-choice question comprehensively. Start with "Answer: [Letter]" where [Letter] is A, B, C, or D, then provide:
1. Detailed explanation with examples and context
2. Analysis of each option
3. Related concepts and background information""",

    'technical': """Answer this multiple-choice question with technical precision. Begin with "Answer: [Letter]" (A, B, C, or D), then provide:
1. Technical explanation with precise terminology and methodology
2. Detailed reasoning for your answer""",

    'executive': """Answer this multiple-choice question concisely. Start with "Answer: [Letter]" where [Letter] is A, B, C, or D, then provide:
1. High-level summary of key reasoning
2. Final recommendation"""
}

# Gemini-specific prompts (more explicit formatting)
GEMINI_SYSTEM_PROMPTS = {
    'minimal': """Answer this multiple-choice question. You MUST start your response with exactly "Answer: X" where X is A, B, C, or D. Then provide a brief one-sentence explanation.""",

    'standard': """Answer this multiple-choice question. You MUST start your first line with exactly "Answer: X" where X is the correct letter (A, B, C, or D). Then provide:
1. A clear, step-by-step explanation of your reasoning
2. Brief explanation of why other options are incorrect""",

    'verbose': """Answer this multiple-choice question comprehensively. You MUST begin your response with "Answer: X" where X is the correct letter (A, B, C, or D). After stating the answer, provide:
1. Detailed explanation with examples and context
2. Analysis of each option
3. Related concepts and background information""",

    'technical': """Answer this multiple-choice question with technical precision. You MUST start with "Answer: X" where X is A, B, C, or D. Then provide:
1. Technical explanation with precise terminology and methodology
2. Detailed reasoning for your answer""",

    'executive': """Answer this multiple-choice question concisely. You MUST begin with "Answer: X" where X is the correct letter (A, B, C, or D). Then provide:
1. High-level summary of key reasoning
2. Final recommendation"""
}


def format_mmlu_question(q_data: Dict[str, Any]) -> str:
    """
    Format MMLU question with multiple-choice options

    Args:
        q_data: Question dictionary containing 'question' and 'choices' fields

    Returns:
        Formatted question string with A/B/C/D choices
    """
    question_text = q_data['question']
    choices = q_data.get('choices', [])

    # If no choices provided, return question as-is
    if not choices:
        return question_text

    # Format with choices
    formatted = f"{question_text}\n\n"
    choice_letters = ['A', 'B', 'C', 'D', 'E', 'F']

    for i, choice in enumerate(choices):
        if i < len(choice_letters):
            formatted += f"{choice_letters[i]}) {choice}\n"

    return formatted.strip()


class DCRPipeline:
    """Multi-provider DCR pipeline with router comparison"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DCR pipeline with all providers"""
        # Load config
        if config is None:
            config = load_inference_config()
        self.config = config

        # Initialize providers
        self.providers = {}
        try:
            self.providers['openai'] = OpenAIProvider(config=config)
        except Exception as e:
            logger.warning(f"⚠ OpenAI provider not available: {e}")

        try:
            self.providers['gemini'] = GeminiProvider(config=config)
        except Exception as e:
            logger.warning(f"⚠ Gemini provider not available: {e}")

        try:
            self.providers['claude'] = AnthropicProvider(config=config)
        except Exception as e:
            logger.warning(f"⚠ Claude provider not available: {e}")

        if not self.providers:
            raise RuntimeError("No providers available. Check API keys.")

        # Initialize routers (lazy load)
        self.simple_neural_router = None
        self.roberta_router = None

    def load_routers(self, strategies_needed: Optional[List[str]] = None):
        """
        Load trained DCR routers

        Args:
            strategies_needed: List of strategies to load routers for
                             If None, tries to load all routers
        """
        print("\nLoading trained DCR routers...")

        # Determine which routers to load
        load_simple_neural = (strategies_needed is None or 'simple_neural' in strategies_needed)
        load_roberta = (strategies_needed is None or 'roberta' in strategies_needed)

        # Load Simple Neural Router
        if load_simple_neural:
            try:
                # Import MLPRouter class so pickle can find it
                from train import MLPRouter

                with open('models/simple_neural_router.pkl', 'rb') as f:
                    self.simple_neural_router = pickle.load(f)
                print("  ✓ Simple Neural Router loaded")
            except Exception as e:
                logger.warning(f"  ⚠ Simple Neural Router not found: {e}")

        # Load RoBERTa Router
        if load_roberta:
            try:
                from train_transformer import TransformerTemplateClassifier
                self.roberta_router = TransformerTemplateClassifier(
                    model_name='models/transformer_roberta-base'
                )
                print("  ✓ RoBERTa Router loaded")
            except Exception as e:
                logger.warning(f"  ⚠ RoBERTa Router not found: {e}")

        # If only baseline needed, no routers required
        if strategies_needed == ['baseline']:
            print("  ℹ️  Baseline strategy only - no routers needed")

    def load_test_data(self, num_questions: int = 100) -> Dict[str, Any]:
        """Load MMLU test data"""
        print(f"\nLoading MMLU test data (first {num_questions} questions)...")

        # Load test questions
        with open('data/mmlu_test.json', 'r') as f:
            data = json.load(f)
        questions = data['questions'][:num_questions]

        # Load test embeddings (for Simple Neural Router)
        with open('data/test_embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)[:num_questions]

        print(f"  ✓ Loaded {len(questions)} questions")
        print(f"  ✓ Loaded {len(embeddings)} embeddings")

        return {
            'questions': questions,
            'embeddings': embeddings
        }

    def get_router_predictions(self, questions_data: List[Dict], embeddings) -> List[Dict[str, str]]:
        """Get template predictions from both routers"""
        predictions = []

        for i, (q_data, embedding) in enumerate(zip(questions_data, embeddings)):
            question = q_data['question']
            pred = {
                'question': question,
                'ground_truth': q_data.get('expected_template', 'standard')  # For accuracy measurement
            }

            # Baseline: always use verbose (most expensive, for cost comparison)
            pred['baseline'] = 'verbose'

            # Simple Neural Router prediction
            if self.simple_neural_router:
                try:
                    result = self.simple_neural_router.predict_from_embedding(embedding)
                    pred['simple_neural'] = result['template']
                except Exception as e:
                    logger.warning(f"Simple Neural failed on Q{i}: {e}")
                    pred['simple_neural'] = 'standard'  # fallback

            # RoBERTa Router prediction
            if self.roberta_router:
                try:
                    templates, _ = self.roberta_router.predict([question])
                    pred['roberta'] = templates[0]
                except Exception as e:
                    logger.warning(f"RoBERTa failed on Q{i}: {e}")
                    pred['roberta'] = 'standard'  # fallback

            predictions.append(pred)

        return predictions

    def generate_prediction_counts(self, predictions: List[Dict[str, str]], strategies: List[str]) -> Dict[str, Dict[str, int]]:
        """Generate counts of template predictions per strategy"""
        # Initialize counts for all templates
        template_names = ['minimal', 'standard', 'verbose', 'technical', 'executive']
        counts = {strategy: {template: 0 for template in template_names} for strategy in strategies}

        # Count predictions
        for pred in predictions:
            for strategy in strategies:
                template = pred.get(strategy)
                if template and template in counts[strategy]:
                    counts[strategy][template] += 1

        # Add total count
        result = {}
        for strategy in strategies:
            result[strategy] = counts[strategy]
            result[strategy]['total'] = sum(counts[strategy].values())

        return result

    async def test_strategy_on_provider(self, questions: List[str], predictions: List[Dict],
                                         strategy: str, provider_name: str) -> List[Dict[str, Any]]:
        """Test one strategy on provider - concurrent requests with asyncio"""
        import time
        import asyncio
        from openai import AsyncOpenAI
        import anthropic
        import google.generativeai as genai

        # Concurrency limits per provider
        max_concurrent = {
            'openai': 50,
            'gemini': 50,
            'claude': 50  # Process 50 at once, then sleep 60s between batches
        }.get(provider_name, 10)

        provider = self.providers[provider_name]
        print(f"\n{strategy.upper()} on {provider.name}")
        print(f"  Processing {len(questions)} questions (max_concurrent={max_concurrent})")

        start_time = time.time()
        completed = 0
        token_stats = {}  # Track token usage: {provider_strategy: {input, output, count}}

        # Create async client based on provider
        if provider_name == 'openai':
            client = AsyncOpenAI(api_key=provider.api_key)
        elif provider_name == 'claude':
            client = anthropic.AsyncAnthropic(api_key=provider.api_key)
        elif provider_name == 'gemini':
            genai.configure(api_key=provider.api_key)
            client = genai.GenerativeModel(provider.model_name)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_question(i, question, pred):
            """Process one question with semaphore and aggressive retry logic"""
            nonlocal completed
            async with semaphore:
                max_retries = 20  # Keep trying up to 20 times
                template = pred.get(strategy, 'standard')

                # Use Gemini-specific prompts for Gemini provider
                if provider_name == 'gemini':
                    system_prompt = GEMINI_SYSTEM_PROMPTS.get(template, GEMINI_SYSTEM_PROMPTS['standard'])
                else:
                    system_prompt = SYSTEM_PROMPTS.get(template, SYSTEM_PROMPTS['standard'])

                # Use template-specific max_tokens (fallback to 1000 for safety)
                template_max_tokens = self.config.get('templates', {}).get('max_tokens', {})
                max_tokens = template_max_tokens.get(template, 1000)

                for attempt in range(1, max_retries + 1):
                    try:
                        # Call API based on provider
                        input_tokens = 0
                        output_tokens = 0

                        if provider_name == 'openai':
                            response = await client.chat.completions.create(
                                model=provider.model,
                                messages=[{"role": "user", "content": f"{system_prompt}\n\nQuestion: {question}"}],
                                temperature=provider.temperature,
                                max_completion_tokens=max_tokens
                            )
                            response_text = response.choices[0].message.content
                            # Extract token usage from OpenAI response
                            if hasattr(response, 'usage'):
                                input_tokens = response.usage.prompt_tokens
                                output_tokens = response.usage.completion_tokens

                        elif provider_name == 'claude':
                            message = await client.messages.create(
                                model=provider.model,
                                max_tokens=max_tokens,
                                temperature=provider.temperature,
                                messages=[{"role": "user", "content": f"{system_prompt}\n\nQuestion: {question}"}]
                            )
                            response_text = message.content[0].text
                            # Extract token usage from Claude response
                            if hasattr(message, 'usage'):
                                input_tokens = message.usage.input_tokens
                                output_tokens = message.usage.output_tokens

                        elif provider_name == 'gemini':
                            generation_config = genai.GenerationConfig(
                                temperature=provider.temperature,
                                max_output_tokens=max_tokens
                            )
                            response = await client.generate_content_async(
                                f"{system_prompt}\n\nQuestion: {question}",
                                generation_config=generation_config
                            )
                            response_text = response.text
                            # Extract token usage from Gemini response
                            if hasattr(response, 'usage_metadata'):
                                input_tokens = response.usage_metadata.prompt_token_count
                                output_tokens = response.usage_metadata.candidates_token_count

                        # Track cumulative token usage
                        token_key = f"{provider_name}_{strategy}"
                        if token_key not in token_stats:
                            token_stats[token_key] = {'input': 0, 'output': 0, 'count': 0}
                        token_stats[token_key]['input'] += input_tokens
                        token_stats[token_key]['output'] += output_tokens
                        token_stats[token_key]['count'] += 1

                        completed += 1
                        if completed % 50 == 0 or completed == len(questions):
                            elapsed = time.time() - start_time
                            rate = (completed / elapsed * 60) if elapsed > 0 else 0
                            eta = ((len(questions) - completed) / rate) if rate > 0 else 0
                            print(f"  ✓ {completed}/{len(questions)} | {rate:.0f}/min | {elapsed:.0f}s elapsed | ETA: {eta:.1f}min | Tokens: {input_tokens}→{output_tokens}")

                        return {
                            'question_id': i,
                            'question': question,
                            'strategy': strategy,
                            'template': template,
                            'provider': provider_name,
                            'response': response_text,
                            'success': True,
                            'error': None,
                            'attempts': attempt,
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens
                        }

                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {str(e)}"
                        if attempt < max_retries:
                            # Exponential backoff with 60s cap: 2s, 4s, 8s, 16s, 32s, 60s, 60s...
                            wait_time = min(2 ** attempt, 60)
                            print(f"  ⚠️  Q{i} attempt {attempt} failed: {error_msg} (retrying in {wait_time}s)")
                            await asyncio.sleep(wait_time)
                        else:
                            # Final failure after all retries
                            print(f"  ❌ Q{i} failed after {max_retries} attempts: {error_msg}")
                            completed += 1
                            return {
                                'question_id': i,
                                'question': question,
                                'strategy': strategy,
                                'template': template,
                                'provider': provider_name,
                                'response': None,
                                'success': False,
                                'error': str(e),
                                'attempts': attempt
                            }

        # Fire all requests concurrently (limited by semaphore)
        # For Claude: Process in batches of 100 RPM with 60s sleep between batches
        if provider_name == 'claude':
            batch_size = 100
            all_results = []

            for batch_start in range(0, len(questions), batch_size):
                batch_end = min(batch_start + batch_size, len(questions))
                batch_questions = questions[batch_start:batch_end]
                batch_predictions = predictions[batch_start:batch_end]

                print(f"  📦 Batch {batch_start//batch_size + 1}: Processing questions {batch_start+1}-{batch_end}")

                tasks = [
                    process_question(i, q, pred)
                    for i, (q, pred) in enumerate(zip(batch_questions, batch_predictions), start=batch_start)
                ]
                batch_results = await asyncio.gather(*tasks)
                all_results.extend(batch_results)

                # Sleep 60s between batches (except after last batch)
                if batch_end < len(questions):
                    print(f"  ⏸️  Sleeping 60s before next batch...")
                    await asyncio.sleep(60)

            results = all_results
        else:
            # Other providers: process all at once
            tasks = [
                process_question(i, q, pred)
                for i, (q, pred) in enumerate(zip(questions, predictions))
            ]
            results = await asyncio.gather(*tasks)

        # Note: Async clients will be cleaned up by garbage collector
        # Explicit cleanup causes AttributeError with some library versions

        elapsed = time.time() - start_time
        succeeded = sum(1 for r in results if r['success'])
        rate = (len(questions) / elapsed * 60) if elapsed > 0 else 0

        print(f"  ✓ {succeeded}/{len(results)} succeeded | {elapsed:.1f}s | {rate:.0f}/min")

        # Print token usage summary
        if token_stats:
            token_key = f"{provider_name}_{strategy}"
            if token_key in token_stats:
                stats = token_stats[token_key]
                total_input = stats['input']
                total_output = stats['output']
                count = stats['count']
                avg_input = total_input / count if count > 0 else 0
                avg_output = total_output / count if count > 0 else 0
                print(f"\n  📊 Token Usage Summary for {provider_name.upper()} - {strategy}:")
                print(f"     Total Input:  {total_input:,} tokens ({avg_input:.0f} avg/query)")
                print(f"     Total Output: {total_output:,} tokens ({avg_output:.0f} avg/query)")
                print(f"     Total Tokens: {total_input + total_output:,} tokens")
                print(f"     Queries:      {count}")

        return results

    def run(self, num_questions: int = 100, specific_provider: Optional[str] = None,
            specific_strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Run DCR pipeline

        Args:
            num_questions: Number of test questions to use
            specific_provider: Test only this provider (openai/gemini/claude)
            specific_strategy: Test only this strategy (baseline/simple_neural/roberta)
        """
        print(f"\n{'='*70}")
        print(f"DCR MULTI-PROVIDER PIPELINE")
        print(f"{'='*70}")
        print(f"Questions: {num_questions}")
        print(f"Providers: {list(self.providers.keys())}")
        print(f"{'='*70}\n")

        # Determine strategies to load routers for
        strategies_to_load = [specific_strategy] if specific_strategy else None

        # Load routers (only load what's needed)
        self.load_routers(strategies_needed=strategies_to_load)

        # Load test data
        test_data = self.load_test_data(num_questions)
        questions_data = test_data['questions']  # Keep full question data (includes expected_template)
        # Format questions with A/B/C/D choices for MMLU
        questions = [format_mmlu_question(q) for q in questions_data]
        embeddings = test_data['embeddings']

        # Get router predictions
        print("\nGenerating router predictions...")
        predictions = self.get_router_predictions(questions_data, embeddings)

        # Determine strategies to test
        strategies = []
        if specific_strategy:
            strategies = [specific_strategy]
        else:
            strategies = ['baseline']
            if self.simple_neural_router:
                strategies.append('simple_neural')
            if self.roberta_router:
                strategies.append('roberta')

        # Save router predictions for inspection
        os.makedirs('dcr_results', exist_ok=True)
        with open('dcr_results/router_predictions.json', 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"  ✓ Saved router predictions to dcr_results/router_predictions.json")

        # Generate prediction counts summary
        prediction_counts = self.generate_prediction_counts(predictions, strategies)
        with open('dcr_results/prediction_counts.json', 'w') as f:
            json.dump(prediction_counts, f, indent=2)
        print(f"  ✓ Saved prediction counts to dcr_results/prediction_counts.json")

        # Determine providers to test
        providers_to_test = [specific_provider] if specific_provider else list(self.providers.keys())

        # Calculate total API calls
        total_calls = len(questions) * len(strategies) * len(providers_to_test)
        print(f"\nTotal API calls: {total_calls} ({len(questions)} questions × {len(strategies)} strategies × {len(providers_to_test)} providers)")

        # Run one provider + one strategy at a time, concurrent questions
        import asyncio
        all_results = {}

        for provider_name in providers_to_test:
            print(f"\n{'='*70}")
            print(f"PROVIDER: {provider_name.upper()}")
            print(f"{'='*70}")

            all_results[provider_name] = {}

            for i, strategy in enumerate(strategies):
                results = asyncio.run(
                    self.test_strategy_on_provider(
                        questions, predictions, strategy, provider_name
                    )
                )
                all_results[provider_name][strategy] = results

                # Sleep between strategies to avoid rate limit carryover
                if provider_name in ['claude', 'openai'] and i < len(strategies) - 1:
                    print(f"\n  ⏸️  Sleeping 60s before next strategy to reset rate limits...")
                    import time
                    time.sleep(60)

        # Save results
        self.save_results(all_results, num_questions)

        # Upload to S3
        self.upload_to_s3(all_results)

        return all_results

    def save_results(self, results: Dict[str, Any], num_questions: int):
        """Save DCR results to files"""
        print(f"\n{'='*70}")
        print("Saving results...")
        print(f"{'='*70}")

        # Create output directory
        os.makedirs('dcr_results', exist_ok=True)

        # Save full results
        output_file = f'dcr_results/dcr_n{num_questions}_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ Saved: {output_file}")

        # Save per-provider results
        for provider_name, provider_results in results.items():
            provider_file = f'dcr_results/{provider_name}_dcr_results.json'
            with open(provider_file, 'w') as f:
                json.dump(provider_results, f, indent=2)
            print(f"  ✓ Saved: {provider_file}")

    def upload_to_s3(self, results: Dict[str, Any]):
        """Upload results to S3"""
        bucket_name = os.getenv('S3_BUCKET_NAME')
        if not bucket_name:
            return

        print(f"\n{'='*70}")
        print("Uploading to S3...")
        print(f"{'='*70}")

        try:
            s3_manager = S3Manager(bucket_name=bucket_name)

            # Upload all files in dcr_results/
            uploaded = 0
            for root, dirs, files in os.walk('dcr_results'):
                for filename in files:
                    local_path = os.path.join(root, filename)
                    s3_key = f'dcr_research/{local_path}'
                    if s3_manager.upload_file(local_path, s3_key):
                        uploaded += 1

            print(f"  ✓ Uploaded {uploaded} files to s3://{bucket_name}/dcr_research/dcr_results/")

        except Exception as e:
            logger.warning(f"  ⚠ S3 upload failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='DCR Multi-Provider Pipeline')
    parser.add_argument('--num-questions', type=int, default=100,
                       help='Number of test questions (default: 100)')
    parser.add_argument('--provider', choices=['openai', 'gemini', 'claude'],
                       help='Test only this provider')
    parser.add_argument('--strategy', choices=['baseline', 'simple_neural', 'roberta'],
                       help='Test only this routing strategy')

    args = parser.parse_args()

    # Run pipeline
    pipeline = DCRPipeline()
    results = pipeline.run(
        num_questions=args.num_questions,
        specific_provider=args.provider,
        specific_strategy=args.strategy
    )

    print(f"\n{'='*70}")
    print("DCR PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print("\nResults saved to:")
    print(f"  • dcr_results/")
    print(f"\nNext steps:")
    print(f"  • Run analysis: python3 src/analysis/analyze_dcr_results.py")
    print(f"  • Generate paper assets")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
