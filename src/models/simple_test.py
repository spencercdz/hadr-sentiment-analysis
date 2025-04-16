"""
Simple test file that doesn't require pandas - allows testing the model directly
"""
from pathlib import Path
import torch
import gc
import json
from tqdm import tqdm

from .llm.untuned import UntunedLLM
from .llm.labels import get_all_labels

# Test messages to predict
test_messages = [
    "Typhoon Haiyan has caused severe damage in the Philippines. 10,000 people are feared dead.",
    "We need food and water supplies in Tacloban immediately. Children are hungry and thirsty.",
    "The Red Cross is sending emergency response teams to the affected areas with medical supplies.",
    "Roads are blocked by debris, making it difficult for rescue teams to reach remote villages.",
    "I'm offering shelter for 5 families affected by the earthquake in my hotel.",
    "Can someone help locate my missing relatives in Cebu City after the typhoon?",
    "The military has been deployed to maintain order and prevent looting in disaster areas.",
    "The storm is expected to hit the coast tomorrow evening with winds up to 120 mph.",
    "Donations of blankets, clothes and medicine are urgently needed at the evacuation center.",
    "The earthquake has damaged the water supply system in several cities."
]

def main():
    # Initialize the model
    model_config = {
        'preprocessing': {
            'max_length': 128,
            'padding': 'max_length',
            'truncation': True
        },
        'batch_size': 5  # Small batch size
    }
    
    print("Loading model...")
    model = UntunedLLM(
        model_name='spencercdz/twitter_disaster_sentiment',
        model_config=model_config
    )
    
    # Get all label dictionaries
    label_mappings = get_all_labels()
    print("Label mapping keys:", list(label_mappings.keys()))
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = []
    
    try:
        for i in tqdm(range(0, len(test_messages), model_config['batch_size']), desc="Predicting batches"):
            batch_texts = test_messages[i:i + model_config['batch_size']]
            batch_preds = model.predict(batch_texts)
            predictions.extend(batch_preds)
            
            # Force garbage collection after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
    
    # Analyze diversity of predictions
    results = {}
    for task in label_mappings.keys():
        results[task] = {'distribution': {}}
    
    # Count prediction distribution
    for pred in predictions:
        for task, values in pred.items():
            prediction = values['prediction']
            pred_key = str(prediction)
            if pred_key not in results[task]['distribution']:
                results[task]['distribution'][pred_key] = 0
            results[task]['distribution'][pred_key] += 1
    
    # Print results
    print("\nPrediction Distribution:")
    for task, stats in results.items():
        print(f"\n{task}:")
        print(f"  Distribution: {stats['distribution']}")
    
    # Print detailed predictions for the first few examples
    print("\nDetailed predictions for sample messages:")
    for i, (text, pred) in enumerate(zip(test_messages[:3], predictions[:3])):
        print(f"\nSample {i+1}: '{text[:50]}...'")
        
        # Get selected predictions
        important_tasks = ['genre', 'related', 'request', 'offer', 'aid_related']
        for task in important_tasks:
            if task in pred:
                print(f"  {task}: {pred[task]['prediction']}")
                print(f"    Scores: {json.dumps(pred[task]['scores'], indent=2)}")

if __name__ == "__main__":
    main()
