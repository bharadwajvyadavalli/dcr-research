#!/usr/bin/env python3
"""
Evaluate Router Accuracy on MMLU Test Data

Tests MLP and RoBERTa routers against ground truth templates
and reports accuracy metrics.

Usage:
    python3 inference/evaluate_routers.py
    python3 inference/evaluate_routers.py --num-questions 1000
"""

import os
import sys
import pickle
import argparse
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Fix pickle compatibility
import training.train as train
sys.modules['__main__'].MLPRouter = train.MLPRouter

from core.utils import load_mmlu_data

# Suppress warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def load_mlp_router():
    """Load MLP router"""
    print("📦 Loading MLP router...")
    mlp_path = 'models/simple_neural_router.pkl'

    if not os.path.exists(mlp_path):
        print(f"❌ MLP router not found: {mlp_path}")
        return None

    try:
        # Suppress sklearn version warnings
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)

        with open(mlp_path, 'rb') as f:
            router = pickle.load(f)

        print("✓ MLP router loaded")
        return router
    except Exception as e:
        print(f"⚠️  Could not load MLP router: {str(e)[:100]}")
        print(f"   Skipping MLP evaluation (RoBERTa will still run)")
        print(f"   To fix: Re-download model or retrain with: python3 cli.py train mlp")
        return None


def load_roberta_router():
    """Load RoBERTa router"""
    print("📦 Loading RoBERTa router...")
    roberta_path = 'models/transformer_roberta-base'

    if not os.path.exists(roberta_path):
        print(f"❌ RoBERTa router not found: {roberta_path}")
        return None

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(roberta_path)
    model = AutoModelForSequenceClassification.from_pretrained(roberta_path)
    device = torch.device('mps' if torch.backends.mps.is_available() else
                         'cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    print(f"✓ RoBERTa router loaded (device: {device})")

    return {
        'tokenizer': tokenizer,
        'model': model,
        'device': device
    }


def predict_mlp(router, embeddings):
    """Predict templates using MLP router"""
    print("\n🔮 Running MLP predictions...")
    start = time.time()

    predictions = router.predict_batch(embeddings)

    elapsed = time.time() - start
    print(f"✓ MLP predictions complete ({elapsed:.2f}s)")

    return predictions


def predict_roberta(router, questions):
    """Predict templates using RoBERTa router"""
    import torch
    from tqdm import tqdm

    print("\n🔮 Running RoBERTa predictions...")
    start = time.time()

    tokenizer = router['tokenizer']
    model = router['model']
    device = router['device']

    template_map = {0: 'minimal', 1: 'standard', 2: 'verbose', 3: 'executive', 4: 'technical'}
    predictions = []

    # Batch prediction
    batch_size = 32
    for i in tqdm(range(0, len(questions), batch_size), desc="RoBERTa"):
        batch = questions[i:i+batch_size]

        inputs = tokenizer(batch, return_tensors='pt', truncation=True,
                          max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_classes = torch.argmax(outputs.logits, dim=1).cpu().tolist()

        batch_predictions = [template_map.get(cls, 'verbose') for cls in predicted_classes]
        predictions.extend(batch_predictions)

    elapsed = time.time() - start
    print(f"✓ RoBERTa predictions complete ({elapsed:.2f}s)")

    return predictions


def calculate_metrics(predictions, ground_truth):
    """Calculate accuracy metrics"""
    total = len(predictions)
    correct = sum(1 for pred, gt in zip(predictions, ground_truth) if pred == gt)
    accuracy = (correct / total * 100) if total > 0 else 0

    # Per-template accuracy
    template_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for pred, gt in zip(predictions, ground_truth):
        template_stats[gt]['total'] += 1
        if pred == gt:
            template_stats[gt]['correct'] += 1

    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for pred, gt in zip(predictions, ground_truth):
        confusion[gt][pred] += 1

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'template_stats': dict(template_stats),
        'confusion': {k: dict(v) for k, v in confusion.items()}
    }


def print_results(router_name, metrics):
    """Print evaluation results"""
    print(f"\n{'='*70}")
    print(f"{router_name.upper()} ROUTER RESULTS")
    print(f"{'='*70}")

    print(f"\nOverall Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")

    print(f"\nPer-Template Accuracy:")
    for template in sorted(metrics['template_stats'].keys()):
        stats = metrics['template_stats'][template]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {template:10s}: {stats['correct']:4d}/{stats['total']:4d} ({acc:5.1f}%)")

    print(f"\nConfusion Matrix:")
    print(f"  {'Ground Truth':<12s} -> Predictions")
    print(f"  {'-'*60}")

    all_templates = sorted(set(metrics['confusion'].keys()) |
                          set(t for conf in metrics['confusion'].values() for t in conf.keys()))

    for gt in all_templates:
        if gt in metrics['confusion']:
            conf_row = metrics['confusion'][gt]
            pred_str = ', '.join([f"{pred}: {count}" for pred, count in sorted(conf_row.items())])
            print(f"  {gt:12s} -> {pred_str}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Router Accuracy on MMLU Test Data')
    parser.add_argument('--num-questions', type=int, default=None,
                       help='Number of questions to evaluate (default: all test data)')

    args = parser.parse_args()

    print("="*70)
    print("ROUTER ACCURACY EVALUATION")
    print("="*70)

    # Load test data
    print("\n📚 Loading MMLU test data...")
    test_data = load_mmlu_data('test')

    questions_data = test_data['questions']
    embeddings = test_data['embeddings']

    if args.num_questions:
        questions_data = questions_data[:args.num_questions]
        embeddings = embeddings[:args.num_questions]

    num_questions = len(questions_data)
    print(f"✓ Loaded {num_questions} questions")

    # Extract ground truth and question texts
    ground_truth = [q['expected_template'] for q in questions_data]
    question_texts = [q['question'] for q in questions_data]

    print(f"\nGround truth distribution:")
    from collections import Counter
    gt_counts = Counter(ground_truth)
    for template, count in sorted(gt_counts.items()):
        print(f"  {template:10s}: {count:4d} ({count/num_questions*100:5.1f}%)")

    # Load routers
    mlp_router = load_mlp_router()
    roberta_router = load_roberta_router()

    if not mlp_router and not roberta_router:
        print("\n❌ No routers available. Download models first:")
        print("  python3 cli.py s3 download models")
        return

    # Evaluate MLP
    if mlp_router:
        mlp_predictions = predict_mlp(mlp_router, embeddings)
        mlp_metrics = calculate_metrics(mlp_predictions, ground_truth)
        print_results("MLP", mlp_metrics)

    # Evaluate RoBERTa
    if roberta_router:
        roberta_predictions = predict_roberta(roberta_router, question_texts)
        roberta_metrics = calculate_metrics(roberta_predictions, ground_truth)
        print_results("RoBERTa", roberta_metrics)

    # Comparison
    if mlp_router and roberta_router:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"\n{'Router':<15s} {'Accuracy':<12s} {'Correct/Total':<15s}")
        print(f"{'-'*70}")
        print(f"{'MLP':<15s} {mlp_metrics['accuracy']:>6.2f}%     "
              f"{mlp_metrics['correct']:>4d}/{mlp_metrics['total']:<4d}")
        print(f"{'RoBERTa':<15s} {roberta_metrics['accuracy']:>6.2f}%     "
              f"{roberta_metrics['correct']:>4d}/{roberta_metrics['total']:<4d}")

        diff = roberta_metrics['accuracy'] - mlp_metrics['accuracy']
        print(f"\nRoBERTa vs MLP: {diff:+.2f}% {'better' if diff > 0 else 'worse'}")

        # Agreement rate
        agreement = sum(1 for mlp, rob in zip(mlp_predictions, roberta_predictions) if mlp == rob)
        agreement_rate = (agreement / num_questions * 100) if num_questions > 0 else 0
        print(f"Agreement rate: {agreement_rate:.2f}% ({agreement}/{num_questions})")

    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
