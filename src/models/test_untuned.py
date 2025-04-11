import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from llm.untuned import UntunedLLM
from llm.utils import format_prediction_output

def main():
    # Initialize the model
    model_config = {
        'preprocessing': {
            'max_length': 128,
            'padding': 'max_length',
            'truncation': True
        },
        'batch_size': 16
    }
    
    model = UntunedLLM(
        model_name='aellxx/disaster-tweet-classification',
        model_config=model_config
    )
    
    # Load test data
    test_data_path = project_root / 'data' / 'raw' / 'test1.csv'
    print(f"\nLoading test data from: {test_data_path}")
    test_data = pd.read_csv(test_data_path)
    
    # Get texts to predict
    test_texts = test_data['message'].tolist()
    print(f"\nLoaded {len(test_texts)} test samples")
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = model.predict(test_texts)
    
    # Initialize counters for accuracy calculation
    task_stats = {
        'sentiment': {'correct': 0, 'total': 0},
        'genre': {'correct': 0, 'total': 0},
        'related': {'correct': 0, 'total': 0},
        'request': {'correct': 0, 'total': 0},
        'aid_related': {'correct': 0, 'total': 0},
        'medical_help': {'correct': 0, 'total': 0},
        'medical_products': {'correct': 0, 'total': 0},
        'search_and_rescue': {'correct': 0, 'total': 0},
        'security': {'correct': 0, 'total': 0},
        'military': {'correct': 0, 'total': 0},
        'child_alone': {'correct': 0, 'total': 0},
        'water': {'correct': 0, 'total': 0},
        'food': {'correct': 0, 'total': 0},
        'shelter': {'correct': 0, 'total': 0},
        'clothing': {'correct': 0, 'total': 0},
        'money': {'correct': 0, 'total': 0},
        'missing_people': {'correct': 0, 'total': 0},
        'refugees': {'correct': 0, 'total': 0},
        'deaths': {'correct': 0, 'total': 0},
        'weather': {'correct': 0, 'total': 0},
        'flood': {'correct': 0, 'total': 0},
        'storm': {'correct': 0, 'total': 0},
        'fire': {'correct': 0, 'total': 0},
        'earthquake': {'correct': 0, 'total': 0},
        'cold': {'correct': 0, 'total': 0},
        'other_weather': {'correct': 0, 'total': 0},
        'direct_report': {'correct': 0, 'total': 0}
    }
    
    # Calculate accuracy for each task
    for i, (text, pred) in enumerate(zip(test_texts, predictions)):
        # Compare predictions with ground truth
        for task in task_stats.keys():
            if task in test_data.columns and task in pred:
                # Get ground truth value (True/False)
                ground_truth = bool(test_data[task].iloc[i])
                # Get model prediction (True/False)
                model_pred = bool(pred[task]['prediction'])
                
                if ground_truth == model_pred:
                    task_stats[task]['correct'] += 1
                task_stats[task]['total'] += 1
    
    print("Task stats:\n", task_stats)

    # Print accuracy statistics
    print("\nAccuracy Statistics:")
    print(f"Total samples: {len(predictions)}")
    
    for task, stats in task_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            print(f"\n{task.capitalize()} Accuracy:")
            print(f"Correct: {stats['correct']}")
            print(f"Total: {stats['total']}")
            print(f"Accuracy: {accuracy:.4f}")
    
    # Print distribution statistics
    print("\nDistribution Statistics:")
    for task in task_stats.keys():
        if task in predictions[0]:
            task_counts = {
                'True': 0,
                'False': 0
            }
            for pred in predictions:
                task_pred = str(bool(pred[task]['prediction']))
                task_counts[task_pred] = task_counts.get(task_pred, 0) + 1
            
            print(f"\n{task.capitalize()} Distribution:")
            for label, count in task_counts.items():
                percentage = (count / len(predictions)) * 100
                print(f"{label}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    main() 