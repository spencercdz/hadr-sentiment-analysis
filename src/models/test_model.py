import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU

from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any
import gc

from .llm.tuned import TunedLLM
from .llm.labels import get_all_labels

# Get project root directory
project_root = Path(__file__).resolve().parents[2]

def process_batch_predictions(args):
    """Process a batch of predictions in parallel"""
    predictions, test_data, task_stats, task_labels = args
    results = []
    
    for i, pred in enumerate(predictions):
        sample_record = {'id': i}
        
        for task, label_dict in task_labels.items():
            if task in test_data.columns and task in pred:
                ground_truth_raw = test_data[task].iloc[i]
                model_pred_raw = pred[task]['prediction']
                model_scores = pred[task]['scores']
                
                # Convert predictions based on task type
                if task == 'genre':
                    # For multi-class genre task
                    model_pred = model_pred_raw  # Already a string like 'direct', 'news', etc.
                    
                    # Handle ground truth which is a string in the CSV ('direct', 'news', 'social media')
                    if pd.notna(ground_truth_raw):
                        if isinstance(ground_truth_raw, (int, float)):
                            # If it's a numeric value, convert to string using label dictionary
                            ground_truth_int = int(ground_truth_raw)
                            if ground_truth_int in label_dict:
                                ground_truth = label_dict[ground_truth_int]
                            else:
                                print(f"Warning: Unknown genre label {ground_truth_int}, using raw value")
                                ground_truth = str(ground_truth_raw)
                        else:
                            # It's already a string from the CSV, use it directly
                            ground_truth = str(ground_truth_raw).strip().lower()
                    else:
                        ground_truth = None
                
                elif task == 'related':
                    # For related task with text labels (no/yes/maybe)
                    if isinstance(model_pred_raw, str):
                        # Handle numeric strings by converting to text
                        if model_pred_raw in ['0', '1', '2']:
                            mapping = {'0': 'no', '1': 'yes', '2': 'maybe'}
                            model_pred = mapping[model_pred_raw]
                        else:
                            # Already text format or unknown
                            model_pred = model_pred_raw.lower()
                    elif isinstance(model_pred_raw, (int, float)):
                        # Convert numeric to text
                        mapping = {0: 'no', 1: 'yes', 2: 'maybe'}
                        model_pred = mapping.get(int(model_pred_raw), 'no')
                    else:
                        model_pred = 'no'  # Default fallback
                    
                    # Convert ground truth to standard format
                    if pd.notna(ground_truth_raw):
                        if isinstance(ground_truth_raw, str):
                            if ground_truth_raw in ['0', '1', '2']:
                                # Convert numeric strings to text
                                mapping = {'0': 'no', '1': 'yes', '2': 'maybe'}
                                ground_truth = mapping.get(ground_truth_raw, 'no')
                            else:
                                # Already text format
                                ground_truth = ground_truth_raw.lower()
                        else:
                            # Convert numeric to text
                            mapping = {0: 'no', 1: 'yes', 2: 'maybe'}
                            ground_truth = mapping.get(int(ground_truth_raw), 'no')
                    else:
                        ground_truth = None
                
                else:
                    # For binary tasks
                    if isinstance(model_pred_raw, str):
                        # Convert yes/no string to 1/0 integer
                        model_pred = 1 if model_pred_raw.lower() == 'yes' else 0
                    else:
                        model_pred = int(model_pred_raw)
                    
                    # Convert ground truth to int
                    if pd.notna(ground_truth_raw):
                        if isinstance(ground_truth_raw, str):
                            ground_truth = 1 if ground_truth_raw.lower() == 'yes' else 0
                        else:
                            ground_truth = int(ground_truth_raw)
                    else:
                        ground_truth = None
                
                # Store values in comparison record
                sample_record.update({
                    f'{task}_true': ground_truth,
                    f'{task}_pred': model_pred,
                    f'{task}_scores': model_scores
                })
                
                # Update task statistics
                if ground_truth is not None:
                    task_stats[task]['total'] += 1
                    pred_key = str(model_pred)
                    task_stats[task]['distribution'][pred_key] = task_stats[task]['distribution'].get(pred_key, 0) + 1
                    
                    # Check if prediction is correct - handle string/int comparison
                    is_correct = False
                    if isinstance(ground_truth, str) and isinstance(model_pred, str):
                        is_correct = ground_truth.lower() == model_pred.lower()
                    else:
                        is_correct = str(ground_truth).lower() == str(model_pred).lower()
                    
                    if is_correct:
                        task_stats[task]['correct'] += 1
        
        results.append(sample_record)
    return results

def main():
    # Initialize the model with optimized batch size
    model_config = {
        'preprocessing': {
            'max_length': 128,
            'padding': 'max_length',
            'truncation': True
        },
        'batch_size': 64  # Increased batch size for better throughput
    }
    
    print("Loading model...")
    model = TunedLLM(
        model_name='spencercdz/xlm-roberta-twitter-sentiment',
        model_config=model_config
    )
    
    # Load test data efficiently
    test_data_path = project_root / 'data' / 'raw' / 'test1.csv'
    print(f"\nLoading test data from: {test_data_path}")
    
    # Read only necessary columns
    label_mappings = get_all_labels()
    required_columns = ['message'] + [key for key in label_mappings.keys() if key != 'sentiment']
    test_data = pd.read_csv(test_data_path, usecols=required_columns)
    print("\nTest data columns:", list(test_data.columns))
    
    # Get texts to predict
    test_texts = test_data['message'].tolist()
    print(f"\nLoaded {len(test_texts)} test samples")
    print("Label mapping keys:", list(label_mappings.keys()))
    
    # Generate predictions with progress bar
    print("\nGenerating predictions...")
    predictions = []
    batch_size = model_config['batch_size']
    
    try:
        for i in tqdm(range(0, len(test_texts), batch_size), desc="Predicting"):
            batch_texts = test_texts[i:i + batch_size]
            batch_preds = model.predict(batch_texts)
            predictions.extend(batch_preds)
            
            # Force garbage collection after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    except KeyboardInterrupt:
        print("\nPrediction interrupted. Processing available results...")
    
    # Initialize task statistics
    task_stats = {task: {'correct': 0, 'total': 0, 'distribution': {}} for task in label_mappings.keys()}
    
    # Process predictions in parallel
    print("\nProcessing predictions...")
    num_workers = min(os.cpu_count() or 1, 4)  # Use up to 4 workers
    batch_size = max(len(predictions) // (num_workers * 2), 1)  # Ensure at least 2 batches per worker
    prediction_batches = [predictions[i:i + batch_size] for i in range(0, len(predictions), batch_size)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        batch_args = [(batch, test_data[i:i + batch_size], task_stats, label_mappings) 
                     for i, batch in zip(range(0, len(test_data), batch_size), prediction_batches)]
        results = list(executor.map(process_batch_predictions, batch_args))
    
    # Combine results
    comparison_data = [item for batch in results for item in batch]
    
    # Print statistics
    print("\nTest Results:")
    for task, stats in task_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total'] * 100
            print(f"\n{task}:")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Total samples: {stats['total']}")
            print("  Prediction distribution:", stats['distribution'])
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()