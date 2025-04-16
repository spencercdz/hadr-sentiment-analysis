from typing import Dict, List, Union, Any, Optional
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch 
import yaml
import json
from transformers import PreTrainedTokenizer
from .labels import get_all_labels

# Initialize the project root directory
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent

# load config
def load_config(config_path: str = project_root / 'config' / 'model_config.yaml') -> Dict:
    """Loads the configuration from YAML file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        Dict: The configuration dictionary.
    """
    # Load the configuration from the YAML file 
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Save metadata
def save_metadata(metadata: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save model metadata to a JSON file.
    
    Args:
        metadata: Dictionary containing model metadata
        save_path: Path to save the metadata
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    with open(save_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)


# Batch encode data
def batch_encode(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    model_config: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """Tokenize and encode a batch of texts.
    
    Args:
        texts: List of texts to encode
        tokenizer: Tokenizer to use for encoding
        model_config: Model configuration dictionary
    
    Returns:
        Dictionary of encoded inputs
    """
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=model_config.get('max_length', 512),
        return_tensors='pt'
    )

# Format prediction results
def format_prediction_output(prediction: Dict[str, Union[Dict[str, Any], np.ndarray]]) -> Dict[str, Any]:
    """Format a single prediction output into a standardized format."""
    # Get mappings for all tasks
    all_label_mapping = get_all_labels()
    
    formatted_prediction = {}
    
    # Process each task's prediction
    for task, pred in prediction.items():
        # Get the task-specific mapping for label conversion
        task_label_mapping = all_label_mapping.get(task, {})
        
        # Prepare a standardized prediction structure
        if isinstance(pred, dict):
            # Case 1: Prediction is already a dictionary with scores and prediction
            scores = pred.get('scores', {})
            prediction_value = pred.get('prediction', None)
            context = pred.get('context', "")
            
            # Use the provided prediction directly - no conversion to True/False
            formatted_prediction[task] = {
                'prediction': prediction_value,
                'confidence': max(scores.values()) if scores else 0.0,
                'scores': scores,
                'context': context
            }
            
        elif isinstance(pred, np.ndarray):
            # Case 2: Prediction is raw numpy array of scores
            # Convert array to dictionary using label mapping
            scores = {}
            
            # Different handling based on task type
            if task in {'genre', 'related'}:
                # For multi-class tasks
                for i, score in enumerate(pred):
                    if i < len(task_label_mapping):
                        label = task_label_mapping[i]
                        scores[label] = float(score)
                
                # Get the label with highest score
                max_label = max(scores.items(), key=lambda x: x[1])[0] if scores else None
                
                formatted_prediction[task] = {
                    'prediction': max_label,
                    'confidence': scores.get(max_label, 0.0) if max_label else 0.0,
                    'scores': scores
                }
            else:
                # For binary tasks
                if len(pred) >= 2:
                    scores = {
                        'no': float(pred[0]),
                        'yes': float(pred[1])
                    }
                    
                    # Determine prediction based on higher score
                    max_label = 'yes' if pred[1] > pred[0] else 'no'
                    
                    formatted_prediction[task] = {
                        'prediction': max_label,
                        'confidence': scores.get(max_label, 0.0),
                        'scores': scores
                    }
                else:
                    # Handle unexpected array size
                    formatted_prediction[task] = {
                        'prediction': 'no',
                        'confidence': 0.0,
                        'scores': {'no': 1.0, 'yes': 0.0}
                    }
        else:
            # Fallback for unexpected prediction type
            formatted_prediction[task] = {
                'prediction': 'no',
                'confidence': 0.0,
                'scores': {'no': 1.0, 'yes': 0.0}
            }
    
    return formatted_prediction

# Calculate sequence length stats
def calculate_sequence_length_stats(
        text: List[int],
        tokenizer
) -> Dict[str, Union[int, float]]:
    """
    Calculate sequence length statistics for a dataset.

    Args:
        text: List of text data
        tokenizer: Tokenizer to use

    Returns:
        Dict containing sequence length statistics.
    """
    lengths = []
    for text in tqdm(text, desc="Calculating sequence lengths"):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        lengths.append(len(tokens))
    
    return {
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "std_length": float(np.std(lengths)),
        "p95_length": float(np.percentile(lengths, 95)),
        "p99_length": float(np.percentile(lengths, 99))
    }

# Set up deterministic mode
def setup_deterministic_mode(seed: int = 42) -> None:
    """
    Set up deterministic mode for reproducibility.

    Args:
        seed (int): The random seed to use.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False