# test.py
import gc
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

# --- Project Setup ---
# Define the project root robustly to ensure modules are found.
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from llm.tuned import TunedLLM
from llm.labels import get_all_labels

# --- CONFIGURATION ---
HUB_REPO_ID = "spencercdz/xlm-roberta-sentiment-requests"
HF_TOKEN = os.environ.get("HF_TOKEN", None)
BATCH_SIZE = 32  # Use a larger batch size for faster evaluation
logging.basicConfig(level=logging.INFO)


def load_model_from_hub(repo_id: str, token: str = None) -> TunedLLM:
    """
    Downloads the model repository from the Hugging Face Hub to a local cache
    and then loads it using the TunedLLM.load_from_disk classmethod.

    This is the canonical way to load a pre-trained TunedLLM instance for inference.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub (e.g., 'user/repo-name').
        token (str, optional): Hugging Face API token for private repos. Defaults to None.

    Returns:
        TunedLLM: An initialized instance of the TunedLLM class, ready for prediction.
    """
    logging.info(f"Downloading model repository '{repo_id}' from the Hub...")
    # snapshot_download downloads the entire repository subfolder to a local cache
    # directory and returns the path to it. This is ideal for our use case.
    model_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        allow_patterns="final_model/*",  # Only download the final model artifacts
    )
    logging.info(f"Model downloaded to cache: {model_path}")

    # Use the class's own loading method on the now-local files.
    # The 'final_model' subfolder contains all necessary files.
    final_model_path = Path(model_path) / "final_model"
    return TunedLLM.load_from_disk(final_model_path)


def normalize_ground_truth(raw_value: Any, label_map: Dict[int, str]) -> str:
    """Converts a ground truth value from the CSV to its canonical string representation."""
    if pd.isna(raw_value):
        return label_map.get(0, "unknown")

    # Create a map that handles int, float, and string keys from the CSV.
    value_map = {v: v for k, v in label_map.items()}
    value_map.update({str(k): v for k, v in label_map.items()})
    key = str(raw_value).lower().strip()
    if key.endswith(".0"):
        key = key[:-2]
    return value_map.get(key, label_map.get(0, "unknown"))


def process_batch(args: tuple):
    """Processes a single batch of predictions in parallel to calculate stats."""
    predictions_batch, test_data_slice, all_labels_map = args
    local_task_stats = {
        task: {"correct": 0, "total": 0, "distribution": {}}
        for task in all_labels_map.keys()
    }
    comparison_records = []

    for i, prediction_dict in enumerate(predictions_batch):
        ground_truth_row = test_data_slice.iloc[i]
        record = {"id": ground_truth_row.get("id", test_data_slice.index[i])}
        for task_name, pred_info in prediction_dict.items():
            if task_name not in ground_truth_row:
                continue

            model_prediction_str = pred_info["prediction"]
            model_confidence = pred_info["confidence"]
            raw_truth = ground_truth_row[task_name]
            ground_truth_str = normalize_ground_truth(raw_truth, all_labels_map[task_name])

            record[f"{task_name}_true"] = ground_truth_str
            record[f"{task_name}_pred"] = model_prediction_str
            record[f"{task_name}_confidence"] = model_confidence

            stats = local_task_stats[task_name]
            stats["total"] += 1
            stats["distribution"][model_prediction_str] = (
                stats["distribution"].get(model_prediction_str, 0) + 1
            )
            if ground_truth_str == model_prediction_str:
                stats["correct"] += 1
        comparison_records.append(record)

    return comparison_records, local_task_stats


def main():
    """Main function to load a trained model from the Hub and evaluate it."""
    try:
        # The single, clean entry point for loading the model.
        model = load_model_from_hub(HUB_REPO_ID, token=HF_TOKEN)
    except Exception as e:
        logging.error(f"❌ Failed to load model. Aborting. Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Set the batch size on the loaded instance for efficient prediction.
    model.batch_size = BATCH_SIZE

    test_data_path = project_root / "data" / "raw" / "test1.csv"
    logging.info(f"✅ Loading test data from: {test_data_path}")
    test_data = pd.read_csv(test_data_path, low_memory=False).rename(
        columns={"message": "text"}
    )
    test_texts = test_data["text"].fillna("").tolist()
    logging.info(f"\nLoaded {len(test_texts)} test samples.")

    all_labels_map = get_all_labels()

    logging.info("\n🚀 Generating predictions...")
    all_predictions = model.predict(test_texts)

    logging.info("\n📊 Processing predictions and calculating stats...")
    num_workers = min(os.cpu_count() or 1, 8)
    chunk_size = max(1, len(all_predictions) // num_workers)
    tasks = []
    for i in range(0, len(all_predictions), chunk_size):
        tasks.append(
            (
                all_predictions[i : i + chunk_size],
                test_data.iloc[i : i + chunk_size],
                all_labels_map,
            )
        )

    aggregated_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results_iterator = list(
            tqdm(executor.map(process_batch, tasks), total=len(tasks), desc="Comparing")
        )
        aggregated_results.extend(results_iterator)

    all_comparison_records = []
    final_task_stats = {
        task: {"correct": 0, "total": 0, "distribution": {}}
        for task in all_labels_map.keys()
    }
    for records, local_stats in aggregated_results:
        all_comparison_records.extend(records)
        for task_name, stats in local_stats.items():
            final_task_stats[task_name]["correct"] += stats["correct"]
            final_task_stats[task_name]["total"] += stats["total"]
            for pred_label, count in stats["distribution"].items():
                dist = final_task_stats[task_name]["distribution"]
                dist[pred_label] = dist.get(pred_label, 0) + count

    comparison_df = pd.DataFrame(all_comparison_records)
    output_dir = project_root / "models" / "tuned"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f'test_results_{HUB_REPO_ID.replace("/", "_")}.csv'
    comparison_df.to_csv(output_path, index=False)
    logging.info(f"\n✅ Detailed test results saved to: {output_path}")

    print("\n--- Test Results Summary ---")
    for task, stats in sorted(final_task_stats.items()):
        if stats["total"] > 0:
            accuracy = (stats["correct"] / stats["total"] * 100)
            print(f"\nTASK: {task}")
            print(f"  - Accuracy: {accuracy:.2f}% ({stats['correct']}/{stats['total']})")
            sorted_dist = dict(sorted(stats["distribution"].items()))
            print(f"  - Prediction Distribution: {sorted_dist}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()