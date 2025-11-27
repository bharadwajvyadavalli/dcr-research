#!/usr/bin/env python3
"""
Train MLP Router
Simple neural network classifier using OpenAI embeddings
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, List
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

from core.utils import load_config, load_mmlu_data


class MLPRouter:
    """MLP-only router using OpenAI embeddings"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize MLP router"""
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # Load neural network configuration
        nn_config = config['models']['simple_neural']['neural_network']
        self.model = MLPClassifier(
            hidden_layer_sizes=tuple(nn_config['hidden_layer_sizes']),
            max_iter=nn_config['max_iter'],
            random_state=nn_config['random_state'],
            alpha=0.01,  # L2 regularization
            early_stopping=True,
            validation_fraction=0.2,
            verbose=False
        )
        self.trained = False

    def train(self, embeddings: np.ndarray, labels: List[str]):
        """Train the MLP model"""
        y_encoded = self.label_encoder.fit_transform(labels)
        X_scaled = self.scaler.fit_transform(embeddings)
        self.model.fit(X_scaled, y_encoded)
        self.trained = True

    def predict_from_embedding(self, embedding: np.ndarray) -> Dict[str, Any]:
        """Predict template from embedding (used by inference)"""
        if not self.trained:
            return {"template": "verbose", "confidence": 0.5, "reasoning": "Not trained"}

        X = embedding.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        pred_proba = self.model.predict_proba(X_scaled)[0]
        prediction = np.argmax(pred_proba)
        confidence = np.max(pred_proba)

        template = self.label_encoder.inverse_transform([prediction])[0]
        reasoning = f"MLP prediction with {confidence:.3f} confidence"

        return {
            "template": template,
            "confidence": float(confidence),
            "reasoning": reasoning
        }

    def predict_batch(self, embeddings: np.ndarray) -> List[str]:
        """Predict templates for batch"""
        if not self.trained:
            raise ValueError("Model not trained")

        X_scaled = self.scaler.transform(embeddings)
        pred_encoded = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(pred_encoded).tolist()


def main():
    print("=" * 80)
    print("🔄 TRAINING MLP ROUTER")
    print("=" * 80)

    # Load configuration
    config = load_config()

    # Load training data
    print("\n📚 Loading MMLU training data...")
    train_data = load_mmlu_data('train')

    train_embeddings = train_data['embeddings']
    train_labels = [q['expected_template'] for q in train_data['questions']]

    print(f"   ✅ Loaded {len(train_labels)} training samples")
    print(f"   Embeddings shape: {train_embeddings.shape}")

    # Initialize and train MLP Router
    print("\n🔧 Training MLP...")
    mlp_router = MLPRouter(config)
    mlp_router.train(train_embeddings, train_labels)
    print("   ✅ MLP training completed")

    # Save model
    output_path = 'models/simple_neural_router.pkl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n💾 Saving model to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(mlp_router, f)

    # Verify
    print("\n🔍 Verifying saved model...")
    with open(output_path, 'rb') as f:
        loaded_model = pickle.load(f)

    test_embedding = train_embeddings[0]
    result = loaded_model.predict_from_embedding(test_embedding)

    print(f"   ✅ Model loaded successfully")
    print(f"   Test prediction: {result['template']} (confidence: {result['confidence']:.3f})")

    print("\n" + "=" * 80)
    print("✅ MLP ROUTER TRAINED AND SAVED")
    print("=" * 80)
    print(f"\n📦 Model: {output_path}")
    print(f"🎯 Ready for inference pipeline")
    print("=" * 80)


if __name__ == '__main__':
    main()
