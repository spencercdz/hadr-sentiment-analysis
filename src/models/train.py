# train.py

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

# Define the project root robustly.
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.llm.tuned import TunedLLM
from src.models.llm.labels import get_all_labels

# --- Setup Logging ---
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = logs_dir / f"training_{current_time}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def plot_and_save_confusion_matrix(y_true, y_pred, labels, task_name, output_dir):
    """Plots and saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {task_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / f"{task_name}_confusion_matrix.png")
    plt.close()


def evaluate_model(model: TunedLLM, test_data_path: Path, output_dir: Path):
    """Evaluates the trained multi-label model on the test dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting evaluation on test data: {test_data_path}")

    # --- Load and Preprocess Test Data ---
    try:
        test_df = pd.read_csv(test_data_path, low_memory=False)
        if 'message' in test_df.columns and 'text' not in test_df.columns:
            test_df.rename(columns={'message': 'text'}, inplace=True)
        test_texts = test_df['text'].tolist()

        # === FIX FOR 'ValueError: invalid literal for int...' ERROR ===
        # Apply the same preprocessing as in the training data preparation.
        all_labels_map = get_all_labels()
        label_columns = sorted(list(all_labels_map.keys()))
        for col_name in label_columns:
            if col_name in test_df.columns:
                value_map = {str(v).lower(): k for k, v in all_labels_map[col_name].items()}
                def safe_mapper(x):
                    if pd.isna(x): return 0
                    key = str(x).lower().strip()
                    return value_map.get(key, int(key) if key.isdigit() and int(key) in value_map.values() else 0)
                test_df[col_name] = test_df[col_name].apply(safe_mapper).astype(int)
        # =============================================================

    except Exception as e:
        logger.error(f"Failed to load or process test data: {e}")
        return

    # --- Get Predictions ---
    logger.info("Generating predictions for the test set...")
    predictions = model.predict(test_texts)
    
    # --- Collate True and Predicted Labels ---
    true_labels = {task: [] for task in label_columns}
    pred_labels = {task: [] for task in label_columns}

    for i, pred_dict in enumerate(predictions):
        for task in label_columns:
            if task in test_df.columns and pd.notna(test_df.loc[i, task]):
                true_labels[task].append(int(test_df.loc[i, task]))
                model_pred_label = pred_dict.get(task, {}).get('prediction', 'no')
                pred_labels[task].append(1 if model_pred_label == 'yes' else 0)

    # --- Calculate and Report Metrics ---
    all_task_reports, all_f1_scores = {}, []
    
    logger.info("Calculating metrics for each task...")
    for task in label_columns:
        if not true_labels[task]:
            continue
        
        # Get the human-readable labels for the confusion matrix
        task_label_names = list(all_labels_map[task].values())

        report = classification_report(
            true_labels[task], pred_labels[task],
            target_names=task_label_names if len(task_label_names) > 1 else ['No', 'Yes'],
            output_dict=True, zero_division=0
        )
        all_task_reports[task] = report
        task_f1 = f1_score(true_labels[task], pred_labels[task], average='macro')
        all_f1_scores.append(task_f1)
        
        logger.info(f"--- Task: {task} | Macro F1-Score: {task_f1:.4f} ---")
        
        plot_and_save_confusion_matrix(
            true_labels[task], pred_labels[task], task_label_names, task, output_dir
        )
    
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


def main():
    """Main function to run the training and evaluation pipeline."""
    try:
        logger.info("--- Starting Model Training and Evaluation Pipeline ---")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA not available. Using CPU.")

        model_config = {
            'training': {
                'num_epochs': 5,
                'learning_rate': 2e-5,
                'weight_decay': 0.01,
                'warmup_steps': 100,
                'gradient_accumulation_steps': 2,
                'logging_steps': 100,
                'eval_steps': 500,
                'save_steps': 500,
                'early_stopping_patience': 3,
                'batch_size': 16,
                'max_length': 128,
            }
        }
        
        logger.info("Initializing multi-label classification model...")
        model = TunedLLM(
            model_name='cardiffnlp/twitter-xlm-roberta-base-sentiment', 
            model_config=model_config
        )
        
        model.train() # This will now run without crashing
        
        logger.info("--- Starting Final Model Evaluation ---")
        test_data_path = project_root / 'data' / 'raw' / 'test1.csv'
        evaluation_output_dir = project_root / 'evaluation' / f"results_{current_time}"
        
        evaluate_model(model, test_data_path, evaluation_output_dir)
        
        logger.info("--- Pipeline Finished ---")
        
    except Exception as e:
        logger.exception(f"An unhandled error occurred in the main pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()