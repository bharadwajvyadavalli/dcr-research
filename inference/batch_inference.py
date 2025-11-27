#!/usr/bin/env python3
"""
Batch Inference for DCR Research
Submit batch requests to OpenAI for 50% cost savings

Workflow:
1. Submit: Create 3 batch jobs (one per strategy) → saves batch IDs
2. Status: Check progress of submitted batches
3. Download: Retrieve and format results when complete

Usage:
    python3 inference/batch_inference.py submit --num-questions 1000
    python3 inference/batch_inference.py status
    python3 inference/batch_inference.py download
"""

import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Fix pickle compatibility for MLPRouter
import training.train as train
sys.modules['__main__'].MLPRouter = train.MLPRouter

from openai import OpenAI
from core.utils import load_inference_config, load_mmlu_data

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Suppress HuggingFace tokenizers warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Template definitions (same as inference.py)
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

# Max tokens per template (same as inference.py)
TEMPLATE_MAX_TOKENS = {
    'minimal': 50,
    'standard': 200,
    'verbose': 500,
    'technical': 400,
    'executive': 350
}


def format_mmlu_question(q_data: Dict[str, Any]) -> str:
    """Format MMLU question with A/B/C/D choices"""
    question = q_data['question']
    choices = q_data.get('choices', [])

    formatted = f"{question}\n\n"
    choice_letters = ['A', 'B', 'C', 'D']
    for i, choice in enumerate(choices):
        formatted += f"{choice_letters[i]}) {choice}\n"

    return formatted.strip()


def load_routers(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load MLP and RoBERTa routers"""
    routers = {}

    # Load MLP router (simple_neural)
    mlp_path = 'models/simple_neural_router.pkl'
    if os.path.exists(mlp_path):
        logger.info(f"📦 Loading MLP router from {mlp_path}")
        with open(mlp_path, 'rb') as f:
            routers['simple_neural'] = pickle.load(f)
    else:
        logger.warning(f"⚠️  MLP router not found: {mlp_path}")
        routers['simple_neural'] = None

    # Load RoBERTa router
    roberta_path = 'models/transformer_roberta-base'
    if os.path.exists(roberta_path):
        logger.info(f"📦 Loading RoBERTa router from {roberta_path}")
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        routers['roberta'] = {
            'tokenizer': AutoTokenizer.from_pretrained(roberta_path),
            'model': AutoModelForSequenceClassification.from_pretrained(roberta_path),
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        }
        routers['roberta']['model'].to(routers['roberta']['device'])
        routers['roberta']['model'].eval()
    else:
        logger.warning(f"⚠️  RoBERTa router not found: {roberta_path}")
        routers['roberta'] = None

    return routers


def get_template_prediction(question: str, strategy: str, routers: Dict, embeddings: Any = None, idx: int = 0) -> str:
    """Get template prediction based on strategy"""

    if strategy == 'baseline':
        return 'verbose'  # Always verbose for baseline

    elif strategy == 'simple_neural':
        if routers['simple_neural'] is None:
            logger.warning("MLP router not available, falling back to verbose")
            return 'verbose'

        # Use embedding if available
        if embeddings is not None:
            embedding = embeddings[idx]
            result = routers['simple_neural'].predict_from_embedding(embedding)
            return result['template']
        else:
            return 'verbose'

    elif strategy == 'roberta':
        if routers['roberta'] is None:
            logger.warning("RoBERTa router not available, falling back to verbose")
            return 'verbose'

        import torch

        # Tokenize and predict
        tokenizer = routers['roberta']['tokenizer']
        model = routers['roberta']['model']
        device = routers['roberta']['device']

        inputs = tokenizer(question, return_tensors='pt', truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

        template_map = {0: 'minimal', 1: 'standard', 2: 'verbose', 3: 'executive', 4: 'technical'}
        return template_map.get(predicted_class, 'verbose')

    return 'verbose'


def create_batch_requests(questions_data: List[Dict], strategy: str, routers: Dict, config: Dict, embeddings: Any = None) -> List[Dict]:
    """Create batch requests for a specific strategy"""

    requests = []
    mode = config.get('mode', 'testing')
    openai_config = config['providers']['openai']
    model = openai_config['models'][mode]

    logger.info(f"  Creating {len(questions_data)} requests for strategy: {strategy}")

    for idx, q_data in enumerate(questions_data):
        # Format question with choices
        formatted_question = format_mmlu_question(q_data)

        # Get template prediction
        template = get_template_prediction(
            q_data['question'],
            strategy,
            routers,
            embeddings=embeddings,
            idx=idx
        )

        # Get system prompt and max tokens
        system_prompt = SYSTEM_PROMPTS[template]
        max_tokens = TEMPLATE_MAX_TOKENS[template]

        # Create OpenAI batch request format
        request = {
            "custom_id": f"q{idx}_{strategy}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_question}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0
            }
        }

        requests.append(request)

    return requests


def submit_openai_batch(requests: List[Dict], strategy: str, num_questions: int) -> Dict[str, Any]:
    """Submit batch request to OpenAI"""

    client = OpenAI()

    # Create batch_files directory
    batch_files_dir = Path('batch_files')
    batch_files_dir.mkdir(exist_ok=True)

    # Create .jsonl file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    jsonl_filename = batch_files_dir / f"openai_{strategy}_n{num_questions}_{timestamp}.jsonl"

    logger.info(f"  Writing {len(requests)} requests to {jsonl_filename}")
    with open(jsonl_filename, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')

    # Upload file to OpenAI
    logger.info(f"  Uploading file to OpenAI...")
    with open(jsonl_filename, 'rb') as f:
        file_response = client.files.create(
            file=f,
            purpose='batch'
        )

    file_id = file_response.id
    logger.info(f"  ✓ Uploaded file: {file_id}")

    # Create batch
    logger.info(f"  Creating batch job...")
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"DCR {strategy} strategy - {num_questions} questions",
            "strategy": strategy,
            "num_questions": str(num_questions)
        }
    )

    batch_id = batch_response.id
    logger.info(f"  ✓ Created batch: {batch_id}")

    return {
        'batch_id': batch_id,
        'file_id': file_id,
        'status': batch_response.status,
        'created_at': batch_response.created_at,
        'request_count': len(requests),
        'jsonl_file': str(jsonl_filename)
    }


def submit_batches(num_questions: int):
    """Submit 3 batch jobs to OpenAI (one per strategy)"""

    logger.info("="*80)
    logger.info("📤 SUBMITTING OPENAI BATCH JOBS")
    logger.info("="*80)

    # Load configuration
    config = load_inference_config()

    # Load test data
    logger.info(f"\n📚 Loading {num_questions} questions from MMLU test set...")
    test_data = load_mmlu_data('test')
    questions_data = test_data['questions'][:num_questions]
    embeddings = test_data['embeddings'][:num_questions] if test_data.get('embeddings') is not None else None

    logger.info(f"✓ Loaded {len(questions_data)} questions")

    # Load routers
    logger.info(f"\n🤖 Loading routers...")
    routers = load_routers(config)

    # Strategies to run
    strategies = ['baseline', 'simple_neural', 'roberta']

    # Track all batch jobs
    batch_tracker = {
        'created_at': datetime.now().isoformat(),
        'num_questions': num_questions,
        'provider': 'openai',
        'strategies': {},
        'status': 'submitted'
    }

    # Submit batch for each strategy
    for strategy in strategies:
        logger.info(f"\n📋 Processing strategy: {strategy}")

        # Create requests
        requests = create_batch_requests(questions_data, strategy, routers, config, embeddings)

        # Submit to OpenAI
        batch_info = submit_openai_batch(requests, strategy, num_questions)

        # Add to tracker
        batch_tracker['strategies'][strategy] = batch_info

        logger.info(f"✓ Submitted {strategy}: {batch_info['batch_id']}")

    # Save tracker file
    batch_jobs_dir = Path('batch_jobs')
    batch_jobs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tracker_file = batch_jobs_dir / f"openai_n{num_questions}_{timestamp}.json"

    with open(tracker_file, 'w') as f:
        json.dump(batch_tracker, indent=2, fp=f)

    logger.info(f"\n📁 Saved batch tracking to: {tracker_file}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("✅ BATCH SUBMISSION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nSubmitted 3 batch jobs:")
    for strategy, info in batch_tracker['strategies'].items():
        logger.info(f"  • {strategy:15s}: {info['batch_id']}")

    logger.info(f"\n⏰ Estimated completion: 1-6 hours")
    logger.info(f"\nTo check status:")
    logger.info(f"  python3 inference/batch_inference.py status --tracker {tracker_file.name}")
    logger.info(f"\nTo download when ready:")
    logger.info(f"  python3 inference/batch_inference.py download --tracker {tracker_file.name}")
    logger.info("="*80)


def check_status(tracker_file: str):
    """Check status of submitted batch jobs"""

    logger.info("="*80)
    logger.info("📊 CHECKING BATCH STATUS")
    logger.info("="*80)

    # Load tracker file
    tracker_path = Path('batch_jobs') / tracker_file
    if not tracker_path.exists():
        logger.error(f"❌ Tracker file not found: {tracker_path}")
        return

    with open(tracker_path, 'r') as f:
        batch_tracker = json.load(f)

    # Initialize OpenAI client
    client = OpenAI()

    logger.info(f"\nBatch submitted: {batch_tracker['created_at']}")
    logger.info(f"Questions: {batch_tracker['num_questions']}")
    logger.info(f"\nStatus:\n")

    # Check each strategy
    all_complete = True
    for strategy, info in batch_tracker['strategies'].items():
        batch_id = info['batch_id']

        # Get current status from OpenAI
        batch = client.batches.retrieve(batch_id)

        # Update tracker
        info['status'] = batch.status
        if batch.status == 'completed':
            info['completed_at'] = batch.completed_at
            info['output_file_id'] = batch.output_file_id
        elif batch.status == 'failed':
            info['failed_at'] = getattr(batch, 'failed_at', None)
            info['error'] = getattr(batch, 'error', None)

        # Progress info
        if hasattr(batch.request_counts, 'completed'):
            completed = batch.request_counts.completed
            total = batch.request_counts.total
            progress = (completed / total * 100) if total > 0 else 0
            info['progress'] = {
                'completed': completed,
                'total': total,
                'percent': progress
            }

        # Display status
        status_icon = {
            'validating': '⏳',
            'in_progress': '⏳',
            'finalizing': '⏳',
            'completed': '✓',
            'failed': '✗',
            'expired': '✗',
            'cancelling': '⏳',
            'cancelled': '✗'
        }.get(batch.status, '?')

        logger.info(f"  {status_icon} {strategy:15s}: {batch.status}")

        if 'progress' in info:
            p = info['progress']
            logger.info(f"      Progress: {p['completed']}/{p['total']} ({p['percent']:.1f}%)")

        if batch.status != 'completed':
            all_complete = False

    # Save updated tracker
    with open(tracker_path, 'w') as f:
        json.dump(batch_tracker, indent=2, fp=f)

    # Summary
    logger.info(f"\n{'='*80}")
    if all_complete:
        logger.info("✅ All batches complete! Ready to download.")
        logger.info(f"\nTo download results:")
        logger.info(f"  python3 inference/batch_inference.py download --tracker {tracker_file}")
    else:
        logger.info("⏳ Batches still processing. Check again later.")
        logger.info(f"\nTo check status again:")
        logger.info(f"  python3 inference/batch_inference.py status --tracker {tracker_file}")
    logger.info("="*80)


def download_results(tracker_file: str):
    """Download and format batch results"""

    logger.info("="*80)
    logger.info("📥 DOWNLOADING BATCH RESULTS")
    logger.info("="*80)

    # Load tracker file
    tracker_path = Path('batch_jobs') / tracker_file
    if not tracker_path.exists():
        logger.error(f"❌ Tracker file not found: {tracker_path}")
        return

    with open(tracker_path, 'r') as f:
        batch_tracker = json.load(f)

    # Initialize OpenAI client
    client = OpenAI()

    # Check if all batches are complete
    all_complete = all(
        info.get('status') == 'completed'
        for info in batch_tracker['strategies'].values()
    )

    if not all_complete:
        logger.warning("⚠️  Not all batches are complete. Run 'status' command first.")
        logger.info("\nCurrent status:")
        for strategy, info in batch_tracker['strategies'].items():
            logger.info(f"  • {strategy}: {info.get('status', 'unknown')}")
        return

    # Download results for each strategy
    logger.info(f"\nDownloading results for {batch_tracker['num_questions']} questions...\n")

    # Format results in same structure as inference.py
    all_results = {
        'baseline': [],
        'simple_neural': [],
        'roberta': []
    }

    for strategy, info in batch_tracker['strategies'].items():
        logger.info(f"📥 Downloading {strategy} results...")

        output_file_id = info.get('output_file_id')
        if not output_file_id:
            logger.error(f"  ✗ No output file ID for {strategy}")
            continue

        # Download result file
        file_response = client.files.content(output_file_id)
        result_content = file_response.text

        # Parse .jsonl results
        results = []
        for line in result_content.strip().split('\n'):
            result = json.loads(line)

            # Extract custom_id to get question index
            custom_id = result['custom_id']  # e.g., "q0_baseline"
            q_idx = int(custom_id.split('_')[0][1:])  # Extract index from "q0"

            # Extract response
            if result['response']['status_code'] == 200:
                response_body = result['response']['body']
                response_text = response_body['choices'][0]['message']['content']
                usage = response_body.get('usage', {})

                # Get original question data
                test_data = load_mmlu_data('test')
                q_data = test_data['questions'][q_idx]

                # Format result
                formatted_result = {
                    'question': q_data['question'],
                    'choices': q_data['choices'],
                    'answer': q_data['answer'],
                    'category': q_data['category'],
                    'template': result['response']['body']['choices'][0].get('template', 'unknown'),
                    'response': response_text,
                    'success': True,
                    'input_tokens': usage.get('prompt_tokens', 0),
                    'output_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0)
                }

                results.append((q_idx, formatted_result))
            else:
                # Handle errors
                logger.warning(f"  ⚠️  Request {custom_id} failed: {result['response'].get('status_code')}")

        # Sort by question index and store
        results.sort(key=lambda x: x[0])
        all_results[strategy] = [r[1] for r in results]

        logger.info(f"  ✓ Downloaded {len(results)} responses")

    # Save in dcr_results format (same as inference.py)
    dcr_results_dir = Path('dcr_results')
    dcr_results_dir.mkdir(exist_ok=True)

    num_q = batch_tracker['num_questions']
    results_file = dcr_results_dir / f'openai_batch_n{num_q}_results.json'

    with open(results_file, 'w') as f:
        json.dump(all_results, indent=2, fp=f)

    logger.info(f"\n✓ Saved results to: {results_file}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("✅ DOWNLOAD COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults saved: {results_file}")
    logger.info(f"\nStrategy results:")
    for strategy, results in all_results.items():
        successful = sum(1 for r in results if r['success'])
        logger.info(f"  • {strategy:15s}: {successful}/{len(results)} successful")

    logger.info(f"\nNext steps:")
    logger.info(f"  • Merge with other providers (if available)")
    logger.info(f"  • Run analysis: python3 cli.py analyze")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Batch Inference for OpenAI (50% cost savings)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit batch jobs
  python3 inference/batch_inference.py submit --num-questions 1000

  # Check status
  python3 inference/batch_inference.py status --tracker openai_n1000_20250105_143022.json

  # Download results
  python3 inference/batch_inference.py download --tracker openai_n1000_20250105_143022.json
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit batch jobs')
    submit_parser.add_argument('--num-questions', type=int, required=True, help='Number of questions')

    # Status command
    status_parser = subparsers.add_parser('status', help='Check batch status')
    status_parser.add_argument('--tracker', type=str, required=True, help='Tracker filename (e.g., openai_n1000_20250105_143022.json)')

    # Download command
    download_parser = subparsers.add_parser('download', help='Download batch results')
    download_parser.add_argument('--tracker', type=str, required=True, help='Tracker filename')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'submit':
        submit_batches(args.num_questions)
    elif args.command == 'status':
        check_status(args.tracker)
    elif args.command == 'download':
        download_results(args.tracker)


if __name__ == '__main__':
    main()
