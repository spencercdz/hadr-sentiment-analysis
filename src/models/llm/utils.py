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
        padding=True,
        truncation=True,
        max_length=model_config.get('max_length', 512),
        return_tensors='pt'
    )

# REMOVE: Get sentiment labels
def get_sentiment_labels() -> Dict[int, str]:
    """
    Get mappings of sentiment label ids to human-readable labels.

    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
    """
    return {
        0: 'non-negative',
        1: 'negative'
    }

# REMOVE: Get sentiment labels
def get_event_type_labels() -> Dict[int, str]:
    """
    Get mappings of event type label ids to human-readable labels.
    """
    return {
        0: 'unknown',
        1: 'storm',
        2: 'flood',
        3: 'earthquake',
        4: 'fire',
        5: 'meteor',
        6: 'volcano',
        7: 'landslide',
        8: 'haze',
    }

# REMOVE: Get sentiment labels
def get_event_type_detail_labels() -> Dict[int, str]:
    """
    Get mappings of event type detail label ids to human-readable labels.
    """
    return {
        0: 'unknown',
        1: 'avalanche',
        2: 'blizzard',
        3: 'bush_fire',
        4: 'cyclone',
        5: 'dust_storm',
        6: 'earthquake',
        7: 'flood',
        8: 'forest_fire',
        9: 'haze',
        10: 'hurricane',
        11: 'landslide',
        12: 'meteor',
        13: 'storm',
        14: 'tornado',
        15: 'tsunami',
        16: 'typhoon',
        17: 'volcano',
        18: 'wildfire'
    }

# REMOVE: Get sentiment labels
def get_label_labels() -> Dict[int, str]:
    """
    Get mappings of label label ids to human-readable labels.
    """
    return {
        0: 'irrelvant',
        1: 'dont_know'
    }

# Format prediction results
def format_prediction_output(prediction: Dict[str, Union[Dict[str, Any], np.ndarray]]) -> Dict[str, Any]:
    """Format a single prediction output into a standardized format.
    
    Args:
        prediction: Raw prediction output from the model for all tasks
        
    Returns:
        Formatted prediction with all task predictions and scores
    """
    # Get all label mappings
    all_label_mapping = get_all_labels()
    
    formatted_prediction = {}
    
    # Process each task's prediction
    for task, pred in prediction.items():
        # Handle different prediction formats
        if isinstance(pred, dict):
            if 'scores' in pred:
                scores = pred['scores']
            else:
                scores = pred
        elif isinstance(pred, np.ndarray):
            if task == 'sentiment':
                # For sentiment, combine positive and neutral into non-negative
                if len(pred) == 3:  # twitter-roberta-base-sentiment output
                    non_negative_score = float(pred[1] + pred[2])  # neutral + positive
                    negative_score = float(pred[0])
                    scores = {
                        'non-negative': non_negative_score,
                        'negative': negative_score
                    }
                else:
                    scores = {all_label_mapping[i]: float(score) for i, score in enumerate(pred)}
            else:
                # For other tasks, use the disaster classification model output
                scores = {all_label_mapping[i]: float(score) for i, score in enumerate(pred)}
        else:
            scores = {k: float(v) for k, v in pred.items()}
        
        # Get the label with highest score
        max_label = max(scores.items(), key=lambda x: x[1])[0]
        max_score = scores[max_label]
        
        # Convert prediction to True/False based on task
        if task == 'sentiment':
            # For sentiment, keep the original labels
            prediction_value = max_label
        else:
            # For other tasks, convert to True/False
            # If the label contains 'not' or is 'unknown', it's False
            # Otherwise, it's True
            prediction_value = not ('not' in max_label.lower() or max_label.lower() == 'unknown')
        
        formatted_prediction[task] = {
            'prediction': prediction_value,
            'confidence': float(max_score),
            'scores': scores
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