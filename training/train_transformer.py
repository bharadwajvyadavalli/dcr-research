#!/usr/bin/env python3
"""
Transformer-based Template Classifier (End-to-End Learning)
Fine-tunes transformer models directly on MMLU questions for template classification.

Supported Models:
- DistilBERT: 66M params, fast, 90-95% accuracy
- RoBERTa: 125M params, best accuracy (93-97%), recommended for production
- DeBERTa-v3: 86M params, state-of-art reasoning, 92-96% accuracy
- BERT: 110M params, balanced performance, 92-96% accuracy

Comparison to embeddings+MLP approach:
- No separate embedding step (no OpenAI API calls)
- Task-specific learning (better accuracy)
- Zero inference cost (runs locally after training)
- See COST_ANALYSIS.md for detailed ROI breakdown
"""

import os
import sys
import json
import time
import argparse
import warnings
from collections import Counter
from typing import Dict, List, Any
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Transformers
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print("❌ transformers library import failed!")
    print(f"   Error: {e}")
    print("   Install with: pip install transformers datasets torch")
    exit(1)

# Import from refactored modules
from core.utils import load_stratified_data

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Template to label mapping
TEMPLATES = ['minimal', 'standard', 'verbose', 'executive', 'technical']
TEMPLATE_TO_ID = {t: i for i, t in enumerate(TEMPLATES)}
ID_TO_TEMPLATE = {i: t for i, t in enumerate(TEMPLATES)}


class TransformerTemplateClassifier:
    """Fine-tuned transformer for template classification"""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Args:
            model_name: HuggingFace model name. Options:
                - distilbert-base-uncased: 66M params, fast, good baseline
                - roberta-base: 125M params, best accuracy (93-97%), RECOMMENDED for production
                - microsoft/deberta-v3-base: 86M params, state-of-art reasoning
                - bert-base-uncased: 110M params, balanced performance
                - sentence-transformers/all-MiniLM-L6-v2: 22M params, very fast
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"\n🤖 Initializing {model_name}...")
        print(f"   Device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(TEMPLATES),
            problem_type="single_label_classification"
        )

        # Move to GPU if available
        self.model.to(self.device)

        print(f"   ✅ Model loaded ({sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters)")

    def prepare_dataset(self, questions: List[str], labels: List[str]) -> Dataset:
        """Convert questions and labels to HuggingFace Dataset"""
        label_ids = [TEMPLATE_TO_ID[label] for label in labels]

        dataset = Dataset.from_dict({
            'text': questions,
            'label': label_ids
        })

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=512
            )

        tokenized = dataset.map(tokenize_function, batched=True)
        return tokenized

    def train(self, train_dataset: Dataset, eval_dataset: Dataset = None,
              epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """Fine-tune the model"""

        output_dir = 'models/transformer_checkpoints'

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,              # L2 regularization
            warmup_steps=500,
            logging_steps=100,
            eval_strategy="epoch" if eval_dataset else "no",  # Changed from evaluation_strategy
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="accuracy" if eval_dataset else None,
            greater_is_better=True,
            save_total_limit=2,
            report_to="none",               # Don't report to wandb/tensorboard
            disable_tqdm=False,
        )

        # Define metrics
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)

            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted'
            )

            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if eval_dataset else []
        )

        # Train
        print("\n🔥 Starting fine-tuning...")
        start_time = time.time()
        trainer.train()
        train_time = time.time() - start_time

        print(f"\n✅ Training complete in {train_time:.1f}s ({train_time/60:.1f} minutes)")

        return trainer, train_time

    def predict(self, questions: List[str], batch_size: int = 32) -> tuple:
        """Predict templates for questions (batched to avoid OOM)"""
        self.model.eval()

        all_predictions = []

        # Process in batches to avoid GPU OOM
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_questions,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                all_predictions.extend(batch_predictions)

        # Convert to numpy array
        predictions = np.array(all_predictions)

        # Convert to template names
        templates = [ID_TO_TEMPLATE[pred] for pred in predictions]

        return templates, predictions

    def evaluate(self, questions: List[str], true_labels: List[str]) -> Dict[str, float]:
        """Evaluate model on a dataset"""
        pred_templates, pred_ids = self.predict(questions)
        true_ids = [TEMPLATE_TO_ID[label] for label in true_labels]

        accuracy = accuracy_score(true_ids, pred_ids)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_ids, pred_ids, average='weighted'
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def save(self, path: str):
        """Save model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"\n✅ Model saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load saved model"""
        instance = cls.__new__(cls)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        instance.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        instance.model.to(instance.device)
        return instance


def cross_validate_transformer(questions: List[str], labels: List[str],
                               model_name: str = "distilbert-base-uncased",
                               n_splits: int = 3) -> Dict[str, Any]:
    """Perform cross-validation on transformer model"""

    print("\n" + "="*80)
    print(f"🔬 {n_splits}-FOLD CROSS-VALIDATION")
    print(f"   Model: {model_name}")
    print("="*80)

    label_ids = [TEMPLATE_TO_ID[label] for label in labels]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(questions, label_ids), 1):
        print(f"\n{'='*80}")
        print(f"📊 FOLD {fold}/{n_splits}")
        print(f"{'='*80}")

        # Split data
        train_questions = [questions[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_questions = [questions[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        print(f"Train: {len(train_questions)} samples")
        print(f"Val:   {len(val_questions)} samples")

        # Initialize fresh model for each fold
        classifier = TransformerTemplateClassifier(model_name=model_name)

        # Prepare datasets
        train_dataset = classifier.prepare_dataset(train_questions, train_labels)
        val_dataset = classifier.prepare_dataset(val_questions, val_labels)

        # Train
        trainer, train_time = classifier.train(
            train_dataset,
            val_dataset,
            epochs=3,
            batch_size=16,
            learning_rate=2e-5
        )

        # Clear GPU cache before evaluation to free memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Evaluate
        val_metrics = classifier.evaluate(val_questions, val_labels)
        train_metrics = classifier.evaluate(train_questions, train_labels)

        gap = train_metrics['accuracy'] - val_metrics['accuracy']

        print(f"\n📊 Fold {fold} Results:")
        print(f"   Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"   Val Accuracy:   {val_metrics['accuracy']:.4f}")
        print(f"   Val F1:         {val_metrics['f1']:.4f}")
        print(f"   Gap:            {gap:+.4f}")

        fold_results.append({
            'fold': fold,
            'train_accuracy': train_metrics['accuracy'],
            'val_accuracy': val_metrics['accuracy'],
            'val_f1': val_metrics['f1'],
            'gap': gap,
            'train_time': train_time
        })

        # Clean up to free GPU memory
        del classifier
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Aggregate results
    cv_accuracies = [r['val_accuracy'] for r in fold_results]
    cv_f1s = [r['val_f1'] for r in fold_results]
    cv_gaps = [r['gap'] for r in fold_results]

    print("\n" + "="*80)
    print("📊 CROSS-VALIDATION SUMMARY")
    print("="*80)
    print(f"CV Accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")
    print(f"CV F1 Score: {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")
    print(f"Avg Gap:     {np.mean(cv_gaps):+.4f}")
    print(f"Total Time:  {sum(r['train_time'] for r in fold_results):.1f}s")

    return {
        'cv_mean_accuracy': float(np.mean(cv_accuracies)),
        'cv_std_accuracy': float(np.std(cv_accuracies)),
        'cv_mean_f1': float(np.mean(cv_f1s)),
        'avg_gap': float(np.mean(cv_gaps)),
        'total_time': float(sum(r['train_time'] for r in fold_results)),
        'fold_results': fold_results
    }


def train_final_model(questions: List[str], labels: List[str],
                     model_name: str = "distilbert-base-uncased") -> TransformerTemplateClassifier:
    """Train final model on full dataset"""

    print("\n" + "="*80)
    print("🎯 TRAINING FINAL MODEL (Full Dataset)")
    print(f"   Model: {model_name}")
    print("="*80)

    classifier = TransformerTemplateClassifier(model_name=model_name)

    # Prepare dataset
    full_dataset = classifier.prepare_dataset(questions, labels)

    # Train (no validation set for final model)
    trainer, train_time = classifier.train(
        full_dataset,
        eval_dataset=None,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5
    )

    # Evaluate on full dataset
    metrics = classifier.evaluate(questions, labels)

    print(f"\n📊 Final Model Performance:")
    print(f"   Train Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Train F1:       {metrics['f1']:.4f}")

    return classifier


def main():
    """Main training and evaluation pipeline"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train transformer-based template classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default DistilBERT
  python3 train_transformer.py

  # Train with RoBERTa (recommended for production)
  python3 train_transformer.py --model roberta-base

  # Train with DeBERTa (state-of-art)
  python3 train_transformer.py --model microsoft/deberta-v3-base

  # Train with BERT
  python3 train_transformer.py --model bert-base-uncased

  # Train with fast MiniLM
  python3 train_transformer.py --model sentence-transformers/all-MiniLM-L6-v2

Available models and expected performance:
  - distilbert-base-uncased (default): 66M params, 90-95% accuracy, fast
  - roberta-base (RECOMMENDED): 125M params, 93-97% accuracy, best ROI
  - microsoft/deberta-v3-base: 86M params, 92-96% accuracy, state-of-art
  - bert-base-uncased: 110M params, 92-96% accuracy, balanced
  - sentence-transformers/all-MiniLM-L6-v2: 22M params, 89-94% accuracy, very fast

See COST_ANALYSIS.md for detailed cost/accuracy trade-offs.
        """
    )
    parser.add_argument(
        '--model',
        type=str,
        default='distilbert-base-uncased',
        help='HuggingFace model name (default: distilbert-base-uncased)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=3,
        help='Number of cross-validation folds (default: 3)'
    )

    args = parser.parse_args()
    model_name = args.model
    cv_folds = args.cv_folds

    # Model info lookup
    model_info = {
        'distilbert-base-uncased': {'params': '66M', 'accuracy': '90-95%'},
        'roberta-base': {'params': '125M', 'accuracy': '93-97%'},
        'microsoft/deberta-v3-base': {'params': '86M', 'accuracy': '92-96%'},
        'bert-base-uncased': {'params': '110M', 'accuracy': '92-96%'},
        'sentence-transformers/all-MiniLM-L6-v2': {'params': '22M', 'accuracy': '89-94%'},
    }

    info = model_info.get(model_name, {'params': 'Unknown', 'accuracy': 'Unknown'})

    print("\n" + "="*80)
    print("🚀 TRANSFORMER TEMPLATE CLASSIFIER - TRAINING")
    print("="*80)
    print("\nApproach: End-to-End Fine-Tuning (No Embeddings)")
    print(f"Model: {model_name}")
    print(f"Parameters: {info['params']}")
    print(f"Expected Accuracy: {info['accuracy']}")
    print("Task: 5-class template classification")
    print("="*80)

    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠️  No GPU found - training will be slower on CPU")
        print("   Consider using g4dn.xlarge or similar GPU instance")

    # Load data (raw questions, not embeddings!)
    print("\n📚 Loading training data...")
    train_data = load_stratified_data()

    # Extract questions and labels (NOT embeddings!)
    questions = [q['question'] for q in train_data['questions']]
    labels = [q['expected_template'] for q in train_data['questions']]

    print(f"✅ {len(questions)} samples")
    print(f"   Distribution: {dict(Counter(labels))}")

    # Cross-validation
    cv_results = cross_validate_transformer(questions, labels, model_name=model_name, n_splits=cv_folds)

    # Train final model on full dataset
    final_model = train_final_model(questions, labels, model_name=model_name)

    # Save final model with model-specific name
    model_safe_name = model_name.replace('/', '_')
    save_path = f'models/transformer_{model_safe_name}'
    final_model.save(save_path)

    # Save results
    os.makedirs('results/training_analysis', exist_ok=True)

    report = {
        'model_type': 'transformer',
        'model_name': model_name,
        'model_params': info['params'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_size': len(questions),
        'cv_folds': cv_folds,
        'cv_results': cv_results,
        'comparison_to_embeddings': {
            'embeddings_mlp_cv': 0.8830,  # From your results
            'transformer_cv': cv_results['cv_mean_accuracy'],
            'improvement': cv_results['cv_mean_accuracy'] - 0.8830,
            'improvement_pct': (cv_results['cv_mean_accuracy'] - 0.8830) / 0.8830 * 100
        }
    }

    report_file = f'results/training_analysis/transformer_{model_safe_name}_comparison.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*80)
    print("📁 RESULTS SAVED")
    print("="*80)
    print(f"✅ Model: {save_path}/")
    print(f"✅ Report: {report_file}")

    print("\n" + "="*80)
    print("📊 COMPARISON: Transformer vs Embeddings+MLP")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Embeddings + MLP:     {0.8830:.4f} (88.30%) CV accuracy")
    print(f"Transformer (E2E):    {cv_results['cv_mean_accuracy']:.4f} ({cv_results['cv_mean_accuracy']*100:.2f}%) CV accuracy")
    print(f"Improvement:          {cv_results['cv_mean_accuracy'] - 0.8830:+.4f} ({(cv_results['cv_mean_accuracy'] - 0.8830)/0.8830*100:+.1f}%)")

    if cv_results['cv_mean_accuracy'] > 0.8830:
        print("\n✅ Transformer outperforms embeddings approach!")
        print("   → End-to-end learning captures task-specific patterns better")
        print("   → Higher accuracy = better routing = lower LLM generation costs")
    else:
        print("\n~ Embeddings approach is competitive")
        print("  → Generic embeddings are sufficient for this task")

    print("\n💰 Cost Analysis (see COST_ANALYSIS.md for details):")
    print(f"   Routing accuracy: {cv_results['cv_mean_accuracy']*100:.1f}% vs 88.3%")

    # Simple cost estimate based on COST_ANALYSIS.md
    accuracy_improvement = cv_results['cv_mean_accuracy'] - 0.8830
    if accuracy_improvement > 0:
        # Rough estimate: each 1% accuracy improvement saves ~$93 per 1M queries
        # (based on $650 savings for 7% improvement in COST_ANALYSIS.md)
        savings_per_million = accuracy_improvement * 100 * 93
        print(f"   Estimated savings: ${savings_per_million:.0f} per 1M queries")
        print(f"   (Better routing → fewer expensive verbose templates)")

    print(f"\n   For detailed ROI analysis, see: COST_ANALYSIS.md")

    print("\n" + "="*80)
    print("💡 NEXT STEPS")
    print("="*80)
    print("✓ Review detailed cost analysis: cat COST_ANALYSIS.md")
    print("✓ Compare per-class performance: python3 analyze_results.py")
    print("✓ Test inference speed: python3 benchmark_inference.py")
    print("✓ Try other models:")
    print("  - RoBERTa (recommended): python3 train_transformer.py --model roberta-base")
    print("  - DeBERTa (state-of-art): python3 train_transformer.py --model microsoft/deberta-v3-base")


if __name__ == "__main__":
    main()
