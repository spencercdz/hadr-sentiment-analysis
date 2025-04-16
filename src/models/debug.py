import pandas as pd
import torch
from pathlib import Path
from pprint import pprint
import numpy as np

from .llm.untuned import UntunedLLM
from .llm.labels import get_all_labels

# Get project root directory
project_root = Path(__file__).resolve().parents[2]

def debug_test_data():
    """Debug the test data to understand format issues"""
    test_data_path = project_root / 'data' / 'raw' / 'test1.csv'
    print(f"Reading test data from: {test_data_path}")
    
    # Read the test data
    test_data = pd.read_csv(test_data_path)
    print(f"Test data shape: {test_data.shape}")
    
    # Print column info
    print("\nColumn information:")
    for col in test_data.columns:
        print(f"{col}: {test_data[col].dtype}, unique values: {test_data[col].nunique()}")
    
    # Check genre and related columns specifically
    print("\nGenre distribution:")
    print(test_data['genre'].value_counts())
    print("\nGenre unique values:")
    print(sorted(test_data['genre'].unique()))
    
    print("\nRelated distribution:")
    print(test_data['related'].value_counts())
    print("\nRelated unique values:")
    print(sorted(test_data['related'].unique()))
    
    # Print sample rows
    print("\nSample rows:")
    print(test_data.iloc[:3].to_string())
    
    # Compare with labels mapping
    labels = get_all_labels()
    print("\nLabel mappings:")
    print("Genre labels:", labels['genre'])
    print("Related labels:", labels['related'])

def debug_model_predictions():
    """Debug model predictions by running a small batch"""
    # Initialize the model with small batch size
    model_config = {
        'preprocessing': {
            'max_length': 128,
            'padding': 'max_length',
            'truncation': True
        },
        'batch_size': 3  # Small batch for debugging
    }
    
    print("Loading model...")
    model = UntunedLLM(
        model_name='spencercdz/twitter_disaster_sentiment',
        model_config=model_config
    )
    
    # Load test data
    test_data_path = project_root / 'data' / 'raw' / 'test1.csv'
    test_data = pd.read_csv(test_data_path)
    
    # Get just a few test samples
    test_texts = test_data['message'].tolist()[:3]
    
    print("\nGenerating predictions for sample:")
    for text in test_texts:
        print(f"- '{text[:50]}...'")
    
    # Get model info
    print("\nModel and tokenizer info:")
    print(f"Models: {list(model.models.keys())}")
    print(f"Default model: {type(model.models['default'])}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = model.predict(test_texts)
    
    # Print task predictions with scores
    for i, (text, pred) in enumerate(zip(test_texts, predictions)):
        print(f"\nSample {i+1}: '{text[:30]}...'")
        for task, values in pred.items():
            print(f"  Task: {task}")
            print(f"    Prediction: {values['prediction']}")
            print(f"    Scores: {values['scores']}")

if __name__ == "__main__":
    print("Debug Test Data:")
    print("="*50)
    debug_test_data()
    
    print("\n\nDebug Model Predictions:")
    print("="*50)
    debug_model_predictions()
