# src/models/train.py

import sys
import os
from pathlib import Path
import pandas as pd
import torch
import logging
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import yaml

# Define the project root robustly.
# This assumes train.py is in src/models/
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.llm.tuned import TunedLLM
from src.models.llm.labels import get_all_labels

# --- Setup Logging ---
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = logs_dir / f"training_{current_time}.log"

# Configure the root logger. This will capture logs from all modules.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def plot_and_save_confusion_matrix(y_true, y_pred, labels, task_name, output_dir):
    """Plots and saves a confusion matrix."""
    try:
        # Ensure labels are in the correct order for the confusion matrix
        cm_labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
        
        # Map numeric labels back to string names for plotting if available
        display_labels = [labels.get(i, str(i)) for i in cm_labels]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=display_labels, yticklabels=display_labels)
        plt.title(f'Confusion Matrix - {task_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        confusion_matrix_path = output_dir / f"{task_name}_confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        plt.close()
        logger.info(f"Saved confusion matrix for {task_name} to {confusion_matrix_path}")
        
    except Exception as e:
        logger.error(f"Failed to create confusion matrix for {task_name}: {e}")


def evaluate_model(model: TunedLLM, test_data_path: Path, output_dir: Path):
    """Evaluates the trained multi-label model on the test dataset."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting evaluation on test data: {test_data_path}")

        # --- Load and Preprocess Test Data ---
        if not test_data_path.exists():
            logger.error(f"Test data file not found: {test_data_path}")
            return 0.0

        test_df = pd.read_csv(test_data_path, low_memory=False)
        logger.info(f"Loaded test data with {len(test_df)} samples")
        
        # Handle different column names
        if 'message' in test_df.columns and 'text' not in test_df.columns:
            test_df.rename(columns={'message': 'text'}, inplace=True)
            
        if 'text' not in test_df.columns:
            logger.error("Test data must contain a 'text' or 'message' column")
            return 0.0
            
        test_texts = test_df['text'].astype(str).tolist()

        all_labels_map = get_all_labels()
        label_columns = sorted(list(all_labels_map.keys()))
        
        # Preprocess label columns
        for col_name in label_columns:
            if col_name in test_df.columns:
                # This mapping converts string labels (e.g., 'yes') to integers (e.g., 1)
                value_map = {str(v).lower(): k for k, v in all_labels_map[col_name].items()}
                
                def safe_mapper(x):
                    if pd.isna(x): 
                        return 0
                    key = str(x).lower().strip()
                    # Handles cases where the column is already numeric or a string
                    if key in value_map:
                        return value_map[key]
                    elif str(key).isdigit() and int(key) in value_map.values():
                        return int(key)
                    else:
                        logger.warning(f"Unknown label value '{x}' in column '{col_name}', defaulting to 0")
                        return 0
                        
                test_df[col_name] = test_df[col_name].apply(safe_mapper).astype(int)

        # --- Get Predictions ---
        logger.info("Generating predictions for the test set...")
        try:
            predictions = model.predict(test_texts)
            if not predictions:
                logger.warning("The model's predict method returned no predictions. Skipping evaluation.")
                return 0.0
        except NotImplementedError:
            logger.warning("The model's `predict` method is not implemented. Skipping final evaluation.")
            return 0.0
        except Exception as e:
            logger.error(f"An error occurred during prediction: {e}")
            return 0.0

        # --- Collate True and Predicted Labels ---
        true_labels = {task: [] for task in label_columns}
        pred_labels = {task: [] for task in label_columns}

        for i, pred_dict in enumerate(predictions):
            if i >= len(test_df):
                logger.warning(f"More predictions ({len(predictions)}) than test samples ({len(test_df)})")
                break
                
            for task in label_columns:
                if task in test_df.columns and pd.notna(test_df.loc[i, task]):
                    true_labels[task].append(int(test_df.loc[i, task]))
                    
                    # Assumes prediction format is {'task_name': {'prediction': 'yes'/'no'}}
                    model_pred_label = pred_dict.get(task, {}).get('prediction', 'no') # Default to 'no'
                    pred_labels[task].append(1 if model_pred_label == 'yes' else 0)

        # --- Calculate and Report Metrics ---
        all_task_reports, all_f1_scores = {}, []
        
        logger.info("Calculating metrics for each task...")
        for task in label_columns:
            if not true_labels[task]:
                logger.info(f"No valid labels found for task: {task}. Skipping.")
                continue
            
            # Get label names (e.g., {0: 'no', 1: 'yes'}) for plotting
            task_label_map = all_labels_map.get(task, {})
            
            try:
                report = classification_report(
                    true_labels[task], pred_labels[task],
                    target_names=[task_label_map.get(i, str(i)) for i in sorted(task_label_map.keys())],
                    output_dict=True, zero_division=0
                )
                all_task_reports[task] = report
                task_f1 = f1_score(true_labels[task], pred_labels[task], average='macro')
                all_f1_scores.append(task_f1)
                
                logger.info(f"--- Task: {task} | Macro F1-Score: {task_f1:.4f} ---")
                
                plot_and_save_confusion_matrix(
                    true_labels[task], pred_labels[task], task_label_map, task, output_dir
                )
            except Exception as e:
                logger.error(f"Failed to calculate metrics for task {task}: {e}")
        
        # --- Calculate and Log Overall Performance ---
        if all_f1_scores:
            overall_macro_f1 = np.mean(all_f1_scores)
            logger.info("=" * 50)
            logger.info(f"OVERALL MODEL PERFORMANCE")
            logger.info(f"Average Macro F1-Score across {len(all_f1_scores)} tasks: {overall_macro_f1:.4f}")
            logger.info("=" * 50)
            all_task_reports['overall_average_macro_f1'] = overall_macro_f1
        else:
            logger.warning("No tasks were evaluated. Cannot compute overall performance.")
            overall_macro_f1 = 0.0

        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_task_reports, f, indent=4)
        logger.info(f"Full evaluation report saved to: {results_file}")

        return overall_macro_f1
        
    except Exception as e:
        logger.exception(f"An error occurred during evaluation: {e}")
        return 0.0


def main():
    """Main function to run the training and evaluation pipeline."""
    try:
        logger.info("--- Starting Model Training and Evaluation Pipeline ---")
        
        # Load configuration from the YAML file
        config_path = project_root / 'config' / 'model_config.yaml'
        if not config_path.exists():
            logger.error(f"Configuration file not found at: {config_path}")
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")

        # Log GPU/CPU availability
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA is available. Using GPU: {device_name}")
        else:
            logger.info("CUDA not available. Using CPU.")

        # Extract model configuration - handle both nested and flat structures
        model_name = config.get('model_name') or config.get('model', {}).get('name')
        if not model_name:
            logger.error("Model name not found in configuration")
            sys.exit(1)
            
        # Prepare model configuration
        model_config = {
            'training': config.get('training', {}),
            'hub': config.get('hub', {}),
            'model': config.get('model', {})
        }
        
        logger.info(f"Initializing multi-label classification model: {model_name}")
        model = TunedLLM(
            model_name=model_name,
            model_config=model_config
        )
        
        # Set additional model attributes if they exist in config
        max_length = (config.get('training', {}).get('max_length') or 
                     config.get('model', {}).get('max_length') or 256)
        model.max_length = max_length
        logger.info(f"Set model max_length to: {max_length}")

        # Start training
        logger.info("Starting model training...")
        model.train()
        logger.info("Model training completed")
        
        # Evaluation
        logger.info("--- Starting Final Model Evaluation ---")
        test_data_path = project_root / 'data' / 'raw' / 'test1.csv'
        evaluation_output_dir = project_root / 'evaluation' / f"results_{current_time}"
        
        if test_data_path.exists():
            overall_f1 = evaluate_model(model, test_data_path, evaluation_output_dir)
            logger.info(f"Final evaluation completed. Overall F1 Score: {overall_f1:.4f}")
        else:
            logger.warning(f"Test data not found at {test_data_path}. Skipping evaluation.")
        
        logger.info("--- Pipeline Finished Successfully ---")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unhandled error occurred in the main pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()