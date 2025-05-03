import sys
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
from datetime import datetime

from .llm.tuned import TunedLLM
from .llm.labels import get_all_labels

# Set up project paths
project_root = Path(__file__).resolve().parent.parent.parent

# Create logs directory if it doesn't exist
logs_dir = project_root / "logs"
os.makedirs(logs_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true, y_pred, labels, task_name, output_dir):
    """Plot and save confusion matrix for a task."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {task_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{task_name}_confusion_matrix.png"))
    plt.close()

def evaluate_model(model, test_data_path, output_dir=None):
    """Evaluate model performance on test data and generate detailed metrics."""
    if output_dir is None:
        output_dir = project_root / 'evaluation' / 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Evaluating model on test data: {test_data_path}")
    
    # Load test data
    test_data = pd.read_csv(test_data_path)
    test_texts = test_data['message'].tolist()
    
    # Get all label dictionaries
    all_labels = get_all_labels()
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = model.predict(test_texts)
    
    # Initialize results dictionary
    results = {
        'overall': {
            'accuracy': 0,
            'weighted_f1': 0,
            'macro_f1': 0,
            'tasks_evaluated': 0
        },
        'tasks': {}
    }
    
    # Process predictions and calculate metrics
    logger.info("Calculating metrics...")
    for task, label_dict in all_labels.items():
        if task not in test_data.columns:
            continue
            
        # Extract ground truth and predictions
        y_true = []
        y_pred = []
        
        for i, (text, pred) in enumerate(zip(test_texts, predictions)):
            if pd.isna(test_data[task].iloc[i]):
                continue
                
            # Get ground truth
            ground_truth_raw = test_data[task].iloc[i]
            
            # Get model prediction
            if task not in pred:
                continue
                
            model_pred_raw = pred[task]['prediction']
            
            # Convert to appropriate format based on task
            if task in ['genre', 'related']:
                # For multi-class tasks
                if task == 'genre':
                    # Handle genre which is a string in the CSV ('direct', 'news', 'social media')
                    if isinstance(ground_truth_raw, (int, float)):
                        # If it's a numeric value, use it directly
                        ground_truth = int(ground_truth_raw)
                    elif isinstance(ground_truth_raw, str):
                        # It's a string from the CSV, convert to numeric index using label dictionary
                        ground_truth_str = ground_truth_raw.strip().lower()
                        if ground_truth_str in label_dict:
                            # Use the numeric index from the label dictionary
                            ground_truth = label_dict[ground_truth_str]
                        else:
                            # Try to find the matching label
                            ground_truth = next((k for k, v in label_dict.items() 
                                              if v.lower() == ground_truth_str), None)
                            if ground_truth is None:
                                continue
                    else:
                        continue
                elif task == 'related':
                        # For related task with text labels (no/yes/maybe)
                        try:
                            if isinstance(ground_truth_raw, str):
                                if ground_truth_raw in ['0', '1', '2']:
                                    # Map numeric strings to text labels
                                    mapping = {'0': 'no', '1': 'yes', '2': 'maybe'}
                                    ground_truth = mapping.get(ground_truth_raw, 'no')
                                else:
                                    # Already a text value, just use it directly
                                    ground_truth = ground_truth_raw.lower()
                            else:
                                # Numeric value, convert to text
                                mapping = {0: 'no', 1: 'yes', 2: 'maybe'}
                                ground_truth = mapping.get(int(ground_truth_raw), 'no')
                                
                            # Get the label ID for this text value
                            ground_truth = next((k for k, v in label_dict.items() 
                                              if v.lower() == ground_truth), None)
                        except (ValueError, TypeError):
                            # Fallback for any conversion errors
                            ground_truth = next((k for k, v in label_dict.items() 
                                              if v.lower() == str(ground_truth_raw).lower()), None)
                else:
                    ground_truth = next((k for k, v in label_dict.items() 
                                        if v.lower() == str(ground_truth_raw).lower()), None)
                    if ground_truth is None:
                        continue
                
                # Map prediction string to label index
                if task == 'related':
                    # For related task with text labels
                    if isinstance(model_pred_raw, str):
                        if model_pred_raw in ['0', '1', '2']:
                            # Map numeric strings to text
                            mapping = {'0': 'no', '1': 'yes', '2': 'maybe'}
                            text_pred = mapping.get(model_pred_raw, 'no')
                        else:
                            # Already text
                            text_pred = model_pred_raw.lower()
                    else:
                        # Numeric value, convert to text
                        mapping = {0: 'no', 1: 'yes', 2: 'maybe'}
                        text_pred = mapping.get(int(model_pred_raw), 'no')
                        
                    # Get the label ID for this text prediction
                    model_pred = next((k for k, v in label_dict.items() 
                                    if v.lower() == text_pred), None)
                    if model_pred is None:
                        continue
                else:
                    # For genre task with string values
                    if task == 'genre':
                        if isinstance(model_pred_raw, str):
                            # Convert string prediction to label index
                            model_pred_str = model_pred_raw.strip().lower()
                            if model_pred_str in label_dict:
                                # Use the numeric index from the label dictionary
                                model_pred = label_dict[model_pred_str]
                            else:
                                # Try to find the matching label
                                model_pred = next((k for k, v in label_dict.items() 
                                                if v.lower() == model_pred_str), None)
                            if model_pred is None:
                                continue
                        else:
                            model_pred = int(model_pred_raw)
                    # For other multi-class tasks
                    else:
                        if isinstance(model_pred_raw, str):
                            model_pred = next((k for k, v in label_dict.items() 
                                             if v.lower() == model_pred_raw.lower()), None)
                            if model_pred is None:
                                continue
                        else:
                            model_pred = int(model_pred_raw)
            else:
                # For binary tasks
                if isinstance(ground_truth_raw, str):
                    ground_truth = 1 if ground_truth_raw.lower() == 'yes' else 0
                else:
                    ground_truth = int(ground_truth_raw)
                    
                if isinstance(model_pred_raw, str):
                    model_pred = 1 if model_pred_raw.lower() == 'yes' else 0
                else:
                    model_pred = int(model_pred_raw)
            
            y_true.append(ground_truth)
            y_pred.append(model_pred)
        
        # Skip if no valid predictions
        if not y_true:
            continue
            
        # Calculate metrics
        label_names = [label_dict[i] for i in range(len(label_dict))]
        
        # Get unique classes actually present in the data
        unique_classes = np.unique(np.array(y_true + y_pred))
        
        # Create a map from class values to indices
        class_to_idx = {class_val: idx for idx, class_val in enumerate(sorted(unique_classes))}
        
        # Get actual labels present in the data
        present_labels = [idx for idx in range(len(label_dict)) if label_dict[idx] in unique_classes or idx in unique_classes]
        
        # For related task with text labels, handle mapping
        if task == 'related':
            text_to_idx = {'no': 0, 'yes': 1, 'maybe': 2}
            # Map text predictions to indices if needed
            if all(isinstance(p, str) for p in y_pred):
                y_pred = [text_to_idx.get(p, 0) for p in y_pred]
            if all(isinstance(t, str) for t in y_true):
                y_true = [text_to_idx.get(t, 0) for t in y_true]
        
        # Make sure we have labels that match our data
        target_names = [label_dict[i] for i in present_labels]
        
        try:
            task_report = classification_report(
                y_true, y_pred,
                labels=present_labels,
                target_names=target_names,
                output_dict=True,
                zero_division=0
            )
            
            # Plot confusion matrix - use present_labels here too
            plot_confusion_matrix(y_true, y_pred, target_names, task, output_dir)
            
            # Store results
            accuracy = task_report['accuracy']
            weighted_f1 = task_report['weighted avg']['f1-score']
            macro_f1 = task_report['macro avg']['f1-score']
            
            results['tasks'][task] = {
                'accuracy': accuracy,
                'weighted_f1': weighted_f1,
                'macro_f1': macro_f1,
                'report': task_report
            }
            
            # Update overall metrics
            results['overall']['accuracy'] += accuracy
            results['overall']['weighted_f1'] += weighted_f1
            results['overall']['macro_f1'] += macro_f1
            results['overall']['tasks_evaluated'] += 1
            
            logger.info(f"Task: {task}")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Weighted F1: {weighted_f1:.4f}")
            logger.info(f"  Macro F1: {macro_f1:.4f}")
        except Exception as e:
            logger.warning(f"Could not generate classification report for task {task}: {str(e)}")
            logger.warning(f"Classes in y_true: {np.unique(y_true)}, Classes in y_pred: {np.unique(y_pred)}")
    
    # Calculate overall average metrics
    if results['overall']['tasks_evaluated'] > 0:
        results['overall']['accuracy'] /= results['overall']['tasks_evaluated']
        results['overall']['weighted_f1'] /= results['overall']['tasks_evaluated']
        results['overall']['macro_f1'] /= results['overall']['tasks_evaluated']
    
    logger.info("Overall Metrics:")
    logger.info(f"  Average Accuracy: {results['overall']['accuracy']:.4f}")
    logger.info(f"  Average Weighted F1: {results['overall']['weighted_f1']:.4f}")
    logger.info(f"  Average Macro F1: {results['overall']['macro_f1']:.4f}")
    
    # Save results to file
    import json
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json_results = {k: convert_numpy(v) if not isinstance(v, dict) else 
                      {k2: convert_numpy(v2) for k2, v2 in v.items()} 
                      for k, v in results.items()}
        json.dump(json_results, f, indent=2)
    
    return results

def main():
    try:
        logger.info("Starting model training...")
        
        # Check CUDA availability
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
        
        # Initialize model configuration with optimized parameters for GTX 1060 6GB
        model_config = {
            'preprocessing': {
                'max_length': 128,  # Reduced from 512 to save memory
                'padding': 'max_length',
                'truncation': True
            },
            'training': {
                'num_epochs': 20,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01,
                'learning_rate': 3e-5,
                'gradient_accumulation_steps': 8,  # Increased to compensate for smaller batch size
                'fp16': torch.cuda.is_available(),
                'logging_steps': 50,  # Less frequent logging
                'save_strategy': 'epoch',  # Save per epoch instead of steps
                'eval_strategy': 'epoch',  # Evaluate per epoch
                'save_total_limit': 1,  # Keep only the best model
                'load_best_model_at_end': True,
                'metric_for_best_model': 'eval_loss',
                'greater_is_better': False,
                'label_smoothing_factor': 0.1,   
                'early_stopping_patience': 2,  # Stop earlier if not improving
                'freeze_backbone': True  # Keep backbone frozen to reduce trainable parameters
            },
            'batch_size': 8,  # Reduced from 32 for GTX 1060 6GB
            'class_weights': True,
            'data_augmentation': {
                'enabled': True,  # Disable data augmentation to speed up training
                'synonym_replacement_prob': 0.0,
                'random_deletion_prob': 0.0,
                'random_swap_prob': 0.0,
                'random_insertion_prob': 0.0,
                'back_translation_prob': 0.0
            }
        }
        
        # Create and train the model
        logger.info("Initializing model...")
        model = TunedLLM(
            model_name='cardiffnlp/twitter-roberta-base-sentiment-latest',
            model_config=model_config
        )
        
        # Train the model
        logger.info("Starting training...")
        model.train()
        
        # Evaluate the model
        logger.info("Evaluating model...")
        test_data_path = project_root / 'data' / 'raw' / 'test1.csv'
        evaluation_results = evaluate_model(model, test_data_path)
        
        logger.info("Training and evaluation complete!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error("Training failed. Please check the error message above.")
        raise

if __name__ == "__main__":
    main()