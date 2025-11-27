#!/usr/bin/env python3
"""
Ablation Study: Traditional ML Models
Trains and compares Logistic Regression, Random Forest, and MLP routers

Usage:
    python3 experiments_models.py train    # Train all models
    python3 experiments_models.py evaluate # Evaluate on test set
    python3 experiments_models.py all      # Train + evaluate

Models:
    1. LogisticRouter     - Linear baseline
    2. RandomForestRouter - Tree-based only
    3. MLPRouter          - Neural network only

Results saved to: models/ablation/
"""

import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, List, Any

from core.utils import load_config, load_mmlu_data


# ============================================================================
# ROUTER CLASSES
# ============================================================================

class LogisticRouter:
    """Logistic Regression baseline - proves nonlinearity needed"""

    def __init__(self, config: Dict[str, Any]):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            C=1.0
        )
        self.trained = False

    def train(self, embeddings: np.ndarray, labels: List[str]):
        """Train logistic regression model"""
        y_encoded = self.label_encoder.fit_transform(labels)
        X_scaled = self.scaler.fit_transform(embeddings)
        self.model.fit(X_scaled, y_encoded)
        self.trained = True

    def predict_batch(self, embeddings: np.ndarray) -> List[str]:
        """Predict templates for batch"""
        if not self.trained:
            raise ValueError("Model not trained")
        X_scaled = self.scaler.transform(embeddings)
        pred_encoded = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(pred_encoded).tolist()


class RandomForestRouter:
    """Random Forest only - tree-based approach"""

    def __init__(self, config: Dict[str, Any]):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        rf_config = config['models']['simple_neural']['random_forest']
        self.model = RandomForestClassifier(
            n_estimators=rf_config['n_estimators'],
            random_state=rf_config['random_state'],
            n_jobs=-1,
            class_weight='balanced',
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt'
        )
        self.trained = False

    def train(self, embeddings: np.ndarray, labels: List[str]):
        """Train random forest model"""
        y_encoded = self.label_encoder.fit_transform(labels)
        X_scaled = self.scaler.fit_transform(embeddings)
        self.model.fit(X_scaled, y_encoded)
        self.trained = True

    def predict_batch(self, embeddings: np.ndarray) -> List[str]:
        """Predict templates for batch"""
        if not self.trained:
            raise ValueError("Model not trained")
        X_scaled = self.scaler.transform(embeddings)
        pred_encoded = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(pred_encoded).tolist()


class MLPRouter:
    """MLP only - neural network approach"""

    def __init__(self, config: Dict[str, Any]):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        nn_config = config['models']['simple_neural']['neural_network']
        self.model = MLPClassifier(
            hidden_layer_sizes=tuple(nn_config['hidden_layer_sizes']),
            max_iter=nn_config['max_iter'],
            random_state=nn_config['random_state'],
            alpha=0.01,
            early_stopping=True,
            validation_fraction=0.2,
            verbose=False
        )
        self.trained = False

    def train(self, embeddings: np.ndarray, labels: List[str]):
        """Train MLP model"""
        y_encoded = self.label_encoder.fit_transform(labels)
        X_scaled = self.scaler.fit_transform(embeddings)
        self.model.fit(X_scaled, y_encoded)
        self.trained = True

    def predict_batch(self, embeddings: np.ndarray) -> List[str]:
        """Predict templates for batch"""
        if not self.trained:
            raise ValueError("Model not trained")
        X_scaled = self.scaler.transform(embeddings)
        pred_encoded = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(pred_encoded).tolist()


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_all_models():
    """Train all three models"""
    print("=" * 80)
    print("🔬 ABLATION STUDY: Training Traditional ML Models")
    print("=" * 80)
    print("\nModels: Logistic Regression, Random Forest, MLP")
    print("=" * 80)

    # Load config and data
    config = load_config()
    print("\n📚 Loading training data...")
    train_data = load_mmlu_data('train')
    train_embeddings = train_data['embeddings']
    train_labels = [q['expected_template'] for q in train_data['questions']]

    print(f"   Questions: {len(train_labels)}")
    print(f"   Embeddings: {train_embeddings.shape}")
    print(f"   Templates: {dict(Counter(train_labels))}")

    # Create output directory
    os.makedirs('models/ablation', exist_ok=True)
    results = {}

    # 1. Logistic Regression
    print("\n" + "=" * 80)
    print("1️⃣  LOGISTIC REGRESSION")
    print("=" * 80)
    lr_router = LogisticRouter(config)
    start = time.time()
    lr_router.train(train_embeddings, train_labels)
    train_time = time.time() - start

    with open('models/ablation/logistic_regression.pkl', 'wb') as f:
        pickle.dump(lr_router, f)

    results['logistic'] = {'train_time': train_time}
    print(f"✅ Trained in {train_time:.2f}s")

    # 2. Random Forest
    print("\n" + "=" * 80)
    print("2️⃣  RANDOM FOREST")
    print("=" * 80)
    rf_router = RandomForestRouter(config)
    start = time.time()
    rf_router.train(train_embeddings, train_labels)
    train_time = time.time() - start

    with open('models/ablation/random_forest.pkl', 'wb') as f:
        pickle.dump(rf_router, f)

    results['random_forest'] = {'train_time': train_time}
    print(f"✅ Trained in {train_time:.2f}s")

    # 3. MLP
    print("\n" + "=" * 80)
    print("3️⃣  MLP (NEURAL NETWORK)")
    print("=" * 80)
    mlp_router = MLPRouter(config)
    start = time.time()
    mlp_router.train(train_embeddings, train_labels)
    train_time = time.time() - start

    with open('models/ablation/mlp.pkl', 'wb') as f:
        pickle.dump(mlp_router, f)

    results['mlp'] = {'train_time': train_time}
    print(f"✅ Trained in {train_time:.2f}s")

    # Save training results
    with open('models/ablation/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ All models trained and saved to models/ablation/")
    print("=" * 80)


def evaluate_all_models():
    """Evaluate all models on test set"""
    print("=" * 80)
    print("📊 ABLATION STUDY: Evaluating Models")
    print("=" * 80)

    # Load test data
    print("\n📚 Loading test data...")
    test_data = load_mmlu_data('test')
    test_embeddings = test_data['embeddings']
    test_labels = [q['expected_template'] for q in test_data['questions']]

    print(f"   Test questions: {len(test_labels)}")
    print(f"   Test embeddings: {test_embeddings.shape}")

    results = {}

    # Evaluate each model
    models = {
        'Logistic Regression': 'models/ablation/logistic_regression.pkl',
        'Random Forest': 'models/ablation/random_forest.pkl',
        'MLP': 'models/ablation/mlp.pkl'
    }

    for name, path in models.items():
        if not os.path.exists(path):
            print(f"\n⚠️  {name} not found at {path}")
            continue

        print(f"\n{'=' * 80}")
        print(f"Evaluating: {name}")
        print('=' * 80)

        with open(path, 'rb') as f:
            router = pickle.load(f)

        # Predict
        start = time.time()
        predictions = router.predict_batch(test_embeddings)
        pred_time = time.time() - start

        # Calculate accuracy
        accuracy = accuracy_score(test_labels, predictions)

        results[name] = {
            'accuracy': accuracy,
            'prediction_time': pred_time,
            'avg_time_per_question': pred_time / len(test_labels)
        }

        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Total time: {pred_time:.2f}s")
        print(f"   Avg per question: {pred_time/len(test_labels)*1000:.2f}ms")

    # Print summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.1%}")
        print(f"  Speed: {metrics['avg_time_per_question']*1000:.2f}ms/question")

    # Save results
    with open('models/ablation/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results saved to models/ablation/evaluation_results.json")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ablation Study: Traditional ML Models')
    parser.add_argument('action', choices=['train', 'evaluate', 'all'],
                       help='Action to perform')
    args = parser.parse_args()

    if args.action == 'train':
        train_all_models()
    elif args.action == 'evaluate':
        evaluate_all_models()
    elif args.action == 'all':
        train_all_models()
        print("\n")
        evaluate_all_models()


if __name__ == '__main__':
    main()
