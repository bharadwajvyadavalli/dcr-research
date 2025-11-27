#!/usr/bin/env python3
"""
Ablation Study: Transformer Models
Trains and compares DistilBERT and RoBERTa routers

Usage:
    python3 experiments_transformers.py train distilbert  # Train DistilBERT
    python3 experiments_transformers.py train roberta     # Train RoBERTa
    python3 experiments_transformers.py train all         # Train both
    python3 experiments_transformers.py evaluate          # Evaluate on test

Models:
    1. DistilBERT  - 66M params, fast, ~88-89% accuracy
    2. RoBERTa     - 125M params, best, ~93-97% accuracy

GPU Recommended: Training takes 1-3 hours on GPU, 8+ hours on CPU
Results saved to: models/ablation/
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset

from core.utils import load_mmlu_data


# ============================================================================
# TRANSFORMER TRAINER CLASS
# ============================================================================

class TransformerTemplateClassifier:
    """Transformer-based template classifier"""

    def __init__(self, model_name='distilbert-base-uncased', num_labels=5):
        """
        Initialize transformer classifier

        Args:
            model_name: HuggingFace model name
            num_labels: Number of template classes (default: 5)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        # Label mapping
        self.label2id = {
            'minimal': 0,
            'standard': 1,
            'verbose': 2,
            'technical': 3,
            'educational': 4
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

    def prepare_dataset(self, questions: List[str], labels: List[str]) -> Dataset:
        """Prepare dataset for training"""
        # Convert labels to IDs
        label_ids = [self.label2id[label] for label in labels]

        # Tokenize
        encodings = self.tokenizer(
            questions,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors=None
        )

        # Create dataset
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': label_ids
        }

        return Dataset.from_dict(dataset_dict)

    def train(self, questions: List[str], labels: List[str],
              output_dir='models/ablation/temp', epochs=3):
        """Train the model"""

        # Prepare dataset
        dataset = self.prepare_dataset(questions, labels)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            save_strategy='epoch',
            evaluation_strategy='no',
            load_best_model_at_end=False,
            push_to_hub=False,
            report_to='none'
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer
        )

        # Train
        trainer.train()

        return trainer

    def predict(self, questions: List[str]) -> List[str]:
        """Predict templates for questions"""
        # Tokenize
        encodings = self.tokenizer(
            questions,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # Convert to labels
        pred_labels = [self.id2label[pred.item()] for pred in predictions]
        return pred_labels

    def save(self, path: str):
        """Save model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str):
        """Load model and tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)

        classifier = cls.__new__(cls)
        classifier.model_name = path
        classifier.num_labels = 5
        classifier.tokenizer = tokenizer
        classifier.model = model
        classifier.label2id = {
            'minimal': 0, 'standard': 1, 'verbose': 2,
            'technical': 3, 'educational': 4
        }
        classifier.id2label = {v: k for k, v in classifier.label2id.items()}

        return classifier


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_distilbert():
    """Train DistilBERT model"""
    print("=" * 80)
    print("🔬 ABLATION STUDY: Training DistilBERT")
    print("=" * 80)
    print("\nModel: distilbert-base-uncased")
    print("Parameters: 66M")
    print("Expected accuracy: ~88-89%")
    print("Expected time: ~1.5 hours on GPU")
    print("=" * 80)

    # Load data
    print("\n📚 Loading training data...")
    train_data = load_mmlu_data('train')
    questions = [q['question'] for q in train_data['questions']]
    labels = [q['expected_template'] for q in train_data['questions']]

    print(f"   Questions: {len(questions)}")
    print(f"   Distribution: {dict(Counter(labels))}")

    # Train
    print("\n🔧 Training model...")
    classifier = TransformerTemplateClassifier(model_name='distilbert-base-uncased')
    start = time.time()
    classifier.train(questions, labels, output_dir='models/ablation/distilbert_temp', epochs=3)
    train_time = time.time() - start

    # Save
    save_path = 'models/ablation/transformer_distilbert'
    classifier.save(save_path)

    print(f"\n✅ Training complete in {train_time/60:.1f} minutes")
    print(f"💾 Saved to: {save_path}")

    # Save metadata
    results = {
        'model': 'distilbert-base-uncased',
        'parameters': '66M',
        'train_time': train_time,
        'train_samples': len(questions),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('models/ablation/distilbert_training.json', 'w') as f:
        json.dump(results, f, indent=2)


def train_roberta():
    """Train RoBERTa model"""
    print("=" * 80)
    print("🔬 ABLATION STUDY: Training RoBERTa")
    print("=" * 80)
    print("\nModel: roberta-base")
    print("Parameters: 125M")
    print("Expected accuracy: ~93-97%")
    print("Expected time: ~2-3 hours on GPU")
    print("=" * 80)

    # Load data
    print("\n📚 Loading training data...")
    train_data = load_mmlu_data('train')
    questions = [q['question'] for q in train_data['questions']]
    labels = [q['expected_template'] for q in train_data['questions']]

    print(f"   Questions: {len(questions)}")
    print(f"   Distribution: {dict(Counter(labels))}")

    # Train
    print("\n🔧 Training model...")
    classifier = TransformerTemplateClassifier(model_name='roberta-base')
    start = time.time()
    classifier.train(questions, labels, output_dir='models/ablation/roberta_temp', epochs=3)
    train_time = time.time() - start

    # Save
    save_path = 'models/ablation/transformer_roberta'
    classifier.save(save_path)

    print(f"\n✅ Training complete in {train_time/60:.1f} minutes")
    print(f"💾 Saved to: {save_path}")

    # Save metadata
    results = {
        'model': 'roberta-base',
        'parameters': '125M',
        'train_time': train_time,
        'train_samples': len(questions),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('models/ablation/roberta_training.json', 'w') as f:
        json.dump(results, f, indent=2)


def evaluate_all():
    """Evaluate all transformer models"""
    print("=" * 80)
    print("📊 ABLATION STUDY: Evaluating Transformers")
    print("=" * 80)

    # Load test data
    print("\n📚 Loading test data...")
    test_data = load_mmlu_data('test')
    questions = [q['question'] for q in test_data['questions']]
    labels = [q['expected_template'] for q in test_data['questions']]

    print(f"   Test questions: {len(questions)}")

    results = {}

    # Evaluate each model
    models = {
        'DistilBERT': 'models/ablation/transformer_distilbert',
        'RoBERTa': 'models/ablation/transformer_roberta'
    }

    for name, path in models.items():
        if not os.path.exists(path):
            print(f"\n⚠️  {name} not found at {path}")
            continue

        print(f"\n{'=' * 80}")
        print(f"Evaluating: {name}")
        print('=' * 80)

        # Load model
        classifier = TransformerTemplateClassifier.load(path)

        # Predict
        print("   Predicting...")
        start = time.time()
        predictions = classifier.predict(questions)
        pred_time = time.time() - start

        # Calculate accuracy
        accuracy = accuracy_score(labels, predictions)

        results[name] = {
            'accuracy': accuracy,
            'prediction_time': pred_time,
            'avg_time_per_question': pred_time / len(questions)
        }

        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Total time: {pred_time:.2f}s")
        print(f"   Avg per question: {pred_time/len(questions)*1000:.2f}ms")

    # Print summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.1%}")
        print(f"  Speed: {metrics['avg_time_per_question']*1000:.2f}ms/question")

    # Save results
    with open('models/ablation/transformer_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results saved to models/ablation/transformer_evaluation.json")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ablation Study: Transformer Models')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('model', choices=['distilbert', 'roberta', 'all'],
                             help='Model to train')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')

    args = parser.parse_args()

    if args.command == 'train':
        os.makedirs('models/ablation', exist_ok=True)

        if args.model == 'distilbert':
            train_distilbert()
        elif args.model == 'roberta':
            train_roberta()
        elif args.model == 'all':
            train_distilbert()
            print("\n" * 3)
            train_roberta()

    elif args.command == 'evaluate':
        evaluate_all()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
