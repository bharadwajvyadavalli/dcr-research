#!/usr/bin/env python3
"""
Corrected MMLU Integration with Proper Splits
"""

from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json
import random
import os
import pickle
import numpy as np
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MMLU subject to category mapping - Complete 57 subjects
MMLU_TO_CATEGORIES = {
    # Math & Logic (7 subjects)
    'abstract_algebra': 'math', 'college_mathematics': 'math', 'elementary_mathematics': 'math',
    'high_school_mathematics': 'math', 'precalculus': 'math', 'high_school_statistics': 'math',
    'formal_logic': 'logic', 'logical_fallacies': 'logic',
    
    # Science (14 subjects)
    'anatomy': 'science', 'astronomy': 'science', 'college_biology': 'science',
    'college_chemistry': 'science', 'college_physics': 'science', 'conceptual_physics': 'science',
    'high_school_biology': 'science', 'high_school_chemistry': 'science', 'high_school_physics': 'science',
    'machine_learning': 'science', 'electrical_engineering': 'science', 'college_medicine': 'science',
    'medical_genetics': 'science', 'nutrition': 'science', 'virology': 'science',
    
    # History (7 subjects)
    'high_school_us_history': 'history', 'high_school_world_history': 'history', 'high_school_european_history': 'history',
    'prehistory': 'history', 'world_religions': 'history', 'us_foreign_policy': 'history',
    'international_law': 'history',
    
    # Reading & Literature (3 subjects)
    'high_school_literature': 'reading', 'college_composition': 'reading', 'high_school_government_and_politics': 'reading',
    'global_facts': 'reading', 'miscellaneous': 'reading', 'professional_law': 'reading',
    'security_studies': 'reading', 'high_school_geography': 'reading',
    
    # Philosophy & Psychology (9 subjects)
    'moral_disputes': 'philosophy', 'moral_scenarios': 'philosophy', 'philosophy': 'philosophy',
    'professional_psychology': 'psychology', 'clinical_knowledge': 'psychology', 'human_aging': 'psychology',
    'human_sexuality': 'psychology', 'sociology': 'psychology', 'public_relations': 'psychology',
    'professional_medicine': 'psychology', 'jurisprudence': 'philosophy', 'high_school_psychology': 'psychology',
    
    # Business & Technology (11 subjects)
    'business_ethics': 'business', 'marketing': 'business', 'management': 'business',
    'professional_accounting': 'business', 'econometrics': 'business', 'high_school_macroeconomics': 'business',
    'high_school_microeconomics': 'business', 'computer_security': 'technology', 'computer_networks': 'technology',
    'operating_systems': 'technology', 'high_school_computer_science': 'technology', 'college_computer_science': 'technology',
}

# Template mapping
TEMPLATE_MAPPING = {
    'math': 'minimal', 'logic': 'standard', 'science': 'standard', 'history': 'standard',
    'reading': 'verbose', 'philosophy': 'verbose', 'psychology': 'verbose',
    'business': 'executive', 'technology': 'technical'
}

def load_mmlu_properly():
    """Load MMLU with proper train/val/test split"""
    
    print("📥 Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all")
    
    # Combine ALL data from working splits
    all_questions = []
    
    for split_name in ['dev', 'validation', 'test']:
        print(f"🔄 Processing {split_name} split...")
        for item in dataset[split_name]:
            subject = item.get('subject', '')
            if subject not in MMLU_TO_CATEGORIES:
                continue
                
            category = MMLU_TO_CATEGORIES[subject]
            template = TEMPLATE_MAPPING[category]
            
            all_questions.append({
                'question': item['question'],
                'subject': subject,
                'category': category,
                'expected_template': template,
                'choices': item.get('choices', []),
                'answer': item.get('answer', ''),
                'original_split': split_name  # Track origin
            })
    
    print(f"📝 Total questions: {len(all_questions)}")
    
    # Stratified split: 70% train, 10% val, 20% test
    train_val, test = train_test_split(
        all_questions,
        test_size=0.2,
        stratify=[q['category'] for q in all_questions],
        random_state=42
    )
    
    train, val = train_test_split(
        train_val,
        test_size=0.125,  # 0.125 * 0.8 = 0.1 overall
        stratify=[q['category'] for q in train_val],
        random_state=42
    )
    
    print(f"\n✅ Proper MMLU Splits:")
    print(f"   Train: {len(train):>6} questions (70%)")
    print(f"   Val:   {len(val):>6} questions (10%)")  
    print(f"   Test:  {len(test):>6} questions (20%)")
    
    # Show category distribution
    print(f"\n📊 Category Distribution:")
    for split_name, split_data in [('Train', train), ('Val', val), ('Test', test)]:
        cats = {}
        for q in split_data:
            cats[q['category']] = cats.get(q['category'], 0) + 1
        print(f"   {split_name}: {cats}")
    
    return train, val, test

def generate_embeddings(questions, model_name='text-embedding-3-small'):
    """Generate embeddings for questions using OpenAI API"""
    print(f"🔤 Generating embeddings with {model_name}...")
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Extract question texts
    question_texts = [q['question'] for q in questions]
    
    # Generate embeddings in batches
    batch_size = 100  # Smaller batches for API calls
    embeddings = []
    
    for i in tqdm(range(0, len(question_texts), batch_size), desc="Generating embeddings"):
        batch = question_texts[i:i+batch_size]
        
        # Make API call for this batch
        response = client.embeddings.create(
            model=model_name,
            input=batch
        )
        
        # Extract embeddings from response
        batch_embeddings = [data.embedding for data in response.data]
        embeddings.extend(batch_embeddings)
    
    embeddings = np.array(embeddings)
    print(f"✅ Generated embeddings shape: {embeddings.shape}")
    return embeddings

def save_splits(train, val, test):
    """Save splits to files with embeddings"""
    
    os.makedirs('data', exist_ok=True)
    
    # Generate embeddings for each split
    print(f"\n🔤 Generating embeddings for all splits...")
    
    # Training data
    print(f"🔤 Generating embeddings for train split...")
    train_embeddings = generate_embeddings(train)
    
    with open('data/mmlu_train.json', 'w') as f:
        json.dump({
            'metadata': {
                'total_questions': len(train),
                'split': 'training',
                'generated_at': datetime.now().isoformat(),
                'source': 'mmlu',
                'split_strategy': 'stratified_70_10_20',
                'embedding_dim': train_embeddings.shape[1]
            },
            'questions': train,
            'embeddings': train_embeddings.tolist()
        }, f, indent=2)
    
    # Save train embeddings as pickle
    with open('data/train_embeddings.pkl', 'wb') as f:
        pickle.dump(train_embeddings, f)
    
    # Validation data
    print(f"🔤 Generating embeddings for validation split...")
    val_embeddings = generate_embeddings(val)
    
    with open('data/mmlu_validation.json', 'w') as f:
        json.dump({
            'metadata': {
                'total_questions': len(val),
                'split': 'validation',
                'generated_at': datetime.now().isoformat(),
                'source': 'mmlu',
                'split_strategy': 'stratified_70_10_20',
                'embedding_dim': val_embeddings.shape[1]
            },
            'questions': val,
            'embeddings': val_embeddings.tolist()
        }, f, indent=2)
    
    # Save val embeddings as pickle
    with open('data/validation_embeddings.pkl', 'wb') as f:
        pickle.dump(val_embeddings, f)
    
    # Test data
    print(f"🔤 Generating embeddings for test split...")
    test_embeddings = generate_embeddings(test)
    
    with open('data/mmlu_test.json', 'w') as f:
        json.dump({
            'metadata': {
                'total_questions': len(test),
                'split': 'test',
                'generated_at': datetime.now().isoformat(),
                'source': 'mmlu',
                'split_strategy': 'stratified_70_10_20',
                'embedding_dim': test_embeddings.shape[1]
            },
            'questions': test,
            'embeddings': test_embeddings.tolist()
        }, f, indent=2)
    
    # Save test embeddings as pickle
    with open('data/test_embeddings.pkl', 'wb') as f:
        pickle.dump(test_embeddings, f)
    
    print(f"\n✅ Saved all splits with embeddings to data/")
    print(f"📁 Files created:")
    print(f"  - data/mmlu_train.json (with embeddings)")
    print(f"  - data/mmlu_validation.json (with embeddings)")
    print(f"  - data/mmlu_test.json (with embeddings)")
    print(f"  - data/train_embeddings.pkl")
    print(f"  - data/validation_embeddings.pkl")
    print(f"  - data/test_embeddings.pkl")

def main():
    random.seed(42)
    
    print("🚀 Data Setup: MMLU Integration with 1536D OpenAI Embeddings")
    print("=" * 60)
    print("📥 Using working splits: dev, validation, test")
    print("📊 Creating stratified 70/10/20 split")
    print("🔤 Generating 1536D embeddings using OpenAI API")
    print("=" * 60)
    
    # Load with proper splits
    train, val, test = load_mmlu_properly()
    
    # Save with embeddings
    save_splits(train, val, test)
    
    print(f"\n🎉 Data setup complete! Ready for training.")
    print(f"🎯 Next step: python train_models.py")

if __name__ == '__main__':
    main()
