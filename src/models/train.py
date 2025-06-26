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
    format='%(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def plot_and_save_confusion_matrix(y_true, y_pred, labels, task_name, output_dir):
    """Plots and saves a confusion matrix."""
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
    plt.savefig(output_dir / f"{task_name}_confusion_matrix.png")
    plt.close()


def evaluate_model(model: TunedLLM, test_data_path: Path, output_dir: Path):
    """Evaluates the trained multi-label model on the test dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Starting evaluation on test data: {test_data_path}")

    # --- Load and Preprocess Test Data ---
    try:
        test_df = pd.read_csv(test_data_path, low_memory=False)
        if 'message' in test_df.columns and 'text' not in test_df.columns:
            test_df.rename(columns={'message': 'text'}, inplace=True)
        test_texts = test_df['text'].astype(str).tolist()

        all_labels_map = get_all_labels()
        label_columns = sorted(list(all_labels_map.keys()))
        for col_name in label_columns:
            if col_name in test_df.columns:
                # This mapping converts string labels (e.g., 'yes') to integers (e.g., 1)
                value_map = {str(v).lower(): k for k, v in all_labels_map[col_name].items()}
                def safe_mapper(x):
                    if pd.isna(x): return 0
                    key = str(x).lower().strip()
                    # Handles cases where the column is already numeric or a string
                    return value_map.get(key, int(key) if str(key).isdigit() and int(key) in value_map.values() else 0)
                test_df[col_name] = test_df[col_name].apply(safe_mapper).astype(int)

    except Exception as e:
        logging.error(f"Failed to load or process test data: {e}")
        return

    # --- Get Predictions ---
    logging.info("Generating predictions for the test set...")
    # The `predict` method is a placeholder, so this part won't work until it's implemented.
    # For now, we will skip this evaluation until `predict` is ready.
    try:
        predictions = model.predict(test_texts)
        if not predictions:
            logging.warning("The model's predict method returned no predictions. Skipping evaluation.")
            return
    except NotImplementedError:
        logging.warning("The model's `predict` method is not implemented. Skipping final evaluation.")
        return
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
        return

    # --- Collate True and Predicted Labels ---
    true_labels = {task: [] for task in label_columns}
    pred_labels = {task: [] for task in label_columns}

    for i, pred_dict in enumerate(predictions):
        for task in label_columns:
            if task in test_df.columns and pd.notna(test_df.loc[i, task]):
                true_labels[task].append(int(test_df.loc[i, task]))
                
                # Assumes prediction format is {'task_name': {'prediction': 'yes'/'no'}}
                model_pred_label = pred_dict.get(task, {}).get('prediction', 'no') # Default to 'no'
                pred_labels[task].append(1 if model_pred_label == 'yes' else 0)

    # --- Calculate and Report Metrics ---
    all_task_reports, all_f1_scores = {}, []
    
    logging.info("Calculating metrics for each task...")
    for task in label_columns:
        if not true_labels[task]:
            continue
        
        # Get label names (e.g., {0: 'no', 1: 'yes'}) for plotting
        task_label_map = all_labels_map.get(task, {})
        
        report = classification_report(
            true_labels[task], pred_labels[task],
            target_names=[task_label_map.get(i, str(i)) for i in sorted(task_label_map.keys())],
            output_dict=True, zero_division=0
        )
        all_task_reports[task] = report
        task_f1 = f1_score(true_labels[task], pred_labels[task], average='macro')
        all_f1_scores.append(task_f1)
        
        logging.info(f"--- Task: {task} | Macro F1-Score: {task_f1:.4f} ---")
        
        plot_and_save_confusion_matrix(
            true_labels[task], pred_labels[task], task_label_map, task, output_dir
        )
    
    # --- Calculate and Log Overall Performance ---
    if all_f1_scores:
        overall_macro_f1 = np.mean(all_f1_scores)
        logging.info("=" * 50)
        logging.info(f"OVERALL MODEL PERFORMANCE")
        logging.info(f"Average Macro F1-Score across {len(all_f1_scores)} tasks: {overall_macro_f1:.4f}")
        logging.info("=" * 50)
        all_task_reports['overall_average_macro_f1'] = overall_macro_f1
    else:
        logging.warning("No tasks were evaluated. Cannot compute overall performance.")
        overall_macro_f1 = 0.0

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_task_reports, f, indent=4)
    logging.info(f"Full evaluation report saved to: {results_file}")

    return overall_macro_f1


def main():
    """Main function to run the training and evaluation pipeline."""
    try:
        logging.info("--- Starting Model Training and Evaluation Pipeline ---")
        
        # Load configuration from the YAML file
        config_path = project_root / 'config' / 'model_config.yaml'
        if not config_path.exists():
            logging.error(f"Configuration file not found at: {config_path}")
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")

        if torch.cuda.is_available():
            logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logging.info("CUDA not available. Using CPU.")

        # The TunedLLM class expects a dictionary where it can find the 'training' key.
        # Passing the whole config dictionary is a common and clean pattern.
        model_config = {
            'training': config.get('training', {}),
        }
        
        logging.info("Initializing multi-label classification model...")
        model = TunedLLM(
            model_name=config['model']['name'], 
            model_config=model_config
        )
        
        # Manually set attributes not passed in the constructor config if needed
        model.max_length = config.get('model', {}).get('max_length', 256)

        # The train method will now use the config passed during initialization
        model.train()
        
        logging.info("--- Starting Final Model Evaluation ---")
        test_data_path = project_root / 'data' / 'raw' / 'test1.csv'
        evaluation_output_dir = project_root / 'evaluation' / f"results_{current_time}"
        
        evaluate_model(model, test_data_path, evaluation_output_dir)
        
        logging.info("--- Pipeline Finished ---")
        
    except Exception as e:
        logging.exception(f"An unhandled error occurred in the main pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()