import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from .llm.untuned import UntunedLLM
from .llm.labels import get_all_labels

def main():
    # Initialize model configuration
    model_config = {
        'preprocessing': {
            'max_length': 128,
            'padding': 'max_length',
            'truncation': True
        },
        'batch_size': 32
    }
    
    print("Loading model...")
    model = UntunedLLM(
        model_name='spencercdz/twitter_disaster_sentiment',
        model_config=model_config
    )
    
    # Load test data
    test_data_path = project_root / 'data' / 'raw' / 'test1.csv'
    print(f"\nLoading test data from: {test_data_path}")
    test_data = pd.read_csv(test_data_path)
    print("\nTest data columns:", list(test_data.columns))
    
    # Extract texts for prediction
    test_texts = test_data['message'].tolist()
    print(f"\nLoaded {len(test_texts)} test samples")
    
    # Generate predictions using the untuned model
    print("\nGenerating predictions...")
    predictions = model.predict(test_texts)
    
    # Debug: Print sample prediction keys (should include every task per get_all_labels())
    if predictions and isinstance(predictions, list):
        sample_pred = predictions[0]
        if not sample_pred:
            print("Warning: Sample prediction is empty. Check your UntunedLLM.predict() implementation.")
        else:
            print("Sample prediction keys:", list(sample_pred.keys()))
    else:
        print("No predictions returned!")
        return
    
    # Load label mappings (dictionary with task names as keys)
    label_mappings = get_all_labels()
    print("Label mapping keys:", list(label_mappings.keys()))
    
    # Initialize statistics counters for each task
    task_stats = {task: {'correct': 0, 'total': 0} for task in label_mappings.keys()}
    comparison_data = []
    
    # Iterate through each test sample and its corresponding prediction
    for i, (text, pred) in enumerate(zip(test_texts, predictions)):
        row = {
            'id': i,
            'text': text,
            'text_clean': text  # You can replace this if you perform additional cleaning
        }
        # For each task, check if CSV ground truth is available and compare to prediction
        for task in label_mappings.keys():
            if task in test_data.columns:
                if task not in pred:
                    print(f"Warning: Sample {i} prediction does not include task '{task}'")
                    continue
                try:
                    # For the task 'genre', use string comparisons; for all others, convert to int (0/1)
                    if task == 'genre':
                        ground_truth = str(test_data[task].iloc[i]).strip().lower()
                        model_pred = str(pred[task]['prediction']).strip().lower()
                    else:
                        ground_truth = int(test_data[task].iloc[i])
                        model_pred = int(pred[task]['prediction'])
                    
                    row[task] = ground_truth
                    row[f'{task}_predicted'] = model_pred
                    row[f'{task}_confidence'] = pred[task].get('confidence', None)
                    
                    # Update accuracy counters
                    if ground_truth == model_pred:
                        task_stats[task]['correct'] += 1
                    task_stats[task]['total'] += 1
                except (ValueError, TypeError) as e:
                    print(f"Error processing sample {i} for task '{task}': {e}")
                    continue
        comparison_data.append(row)
    
    # Create a DataFrame from the comparison data and reorder columns
    comparison_df = pd.DataFrame(comparison_data)
    base_columns = ['id', 'text', 'text_clean']
    task_columns = [col for col in comparison_df.columns if col not in base_columns and not col.endswith('_predicted') and not col.endswith('_confidence')]
    column_order = base_columns + task_columns
    comparison_df = comparison_df[column_order]
    
    # Save the detailed comparison to CSV
    output_csv = project_root / 'data' / 'processed' / 'prediction_comparison.csv'
    comparison_df.to_csv(output_csv, index=False)
    print(f"\nDetailed prediction comparison saved to: {output_csv}")
    
    # Print Accuracy Statistics
    print("\nAccuracy Statistics:")
    total_samples = len(predictions)
    print(f"Total samples: {total_samples}")
    for task, stats in task_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            print(f"\n{task.capitalize()} Accuracy:")
            print(f"  Correct: {stats['correct']}")
            print(f"  Total: {stats['total']}")
            print(f"  Accuracy: {accuracy:.4f}")
        else:
            print(f"\nNo ground truth available for task: {task}")
    
    # Print Distribution Statistics per task
    print("\nDistribution Statistics:")
    for task in label_mappings.keys():
        if predictions and task in predictions[0]:
            task_counts = {}
            for pred in predictions:
                if task in pred and 'prediction' in pred[task]:
                    try:
                        if task == 'genre':
                            p_val = str(pred[task]['prediction']).strip().lower()
                        else:
                            p_val = str(int(pred[task]['prediction']))
                    except Exception:
                        p_val = str(pred[task]['prediction']).strip().lower()
                    task_counts[p_val] = task_counts.get(p_val, 0) + 1
            if task_counts:
                print(f"\n{task.capitalize()} Distribution:")
                for label, count in task_counts.items():
                    percentage = (count / total_samples) * 100
                    print(f"  {label}: {count} ({percentage:.2f}%)")
            else:
                print(f"\nNo predictions available to compute distribution for task: {task}")
        else:
            print(f"\nNo predictions available to compute distribution for task: {task}")

if __name__ == "__main__":
    main()
