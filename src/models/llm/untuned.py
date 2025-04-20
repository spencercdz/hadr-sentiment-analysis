from typing import Dict, List, Union, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from pathlib import Path
import numpy as np
import os
import torch
import gc
from tqdm import tqdm

from .base import BaseLLM
from .utils import batch_encode, format_prediction_output, save_metadata

# Initialize the project root directory
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent

class UntunedLLM(BaseLLM):
    """Implementation of an untuned LLM model for HADR sentiment analysis.
    Uses a pretrained model from the Hugging Face model hub directly, without fine-tuning."""

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        super().__init__(model_name, model_config, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        
        # Get configuration values
        self.batch_size = model_config.get('batch_size', 8)
        self.preprocessing_config = model_config.get('preprocessing', {})
        
        # Initialize models for each task
        self.models = {}
        self.tokenizers = {}
        
        # Define task-specific models
        self.task_models = {
            'genre': model_name, #model_config.get('genre_model', 'distilbert-base-uncased-finetuned-sst-2-english'),
            'related': model_name, #model_config.get('related_model', 'distilbert-base-uncased-finetuned-sst-2-english'),
            # Define specific models for other tasks if needed
            'default': model_name  # Use this for binary tasks
        }
        
        # Load models for each task
        self.load_model()
    
    def load_model(self):
        """Load multiple task-specific models."""
        try:
            # First load the default model
            print(f"Loading default model: {self.task_models['default']}")
            self.models['default'] = AutoModelForSequenceClassification.from_pretrained(
                self.task_models['default']
            ).to(self.device)

            # Try to load the tokenizer for the default model, with fallback
            try:
                self.tokenizers['default'] = AutoTokenizer.from_pretrained(self.task_models['default'])
            except (OSError, ValueError) as e:
                print(f"Could not load tokenizer for {self.task_models['default']}: {str(e)}")
                print("Using default RoBERTa tokenizer as fallback")
                self.tokenizers['default'] = AutoTokenizer.from_pretrained("roberta-base")
            
            # Load task-specific models
            for task, model_name in self.task_models.items():
                if task != 'default':
                    print(f"Loading model for {task}: {model_name}")
                    self.models[task] = AutoModelForSequenceClassification.from_pretrained(
                        model_name
                    ).to(self.device)
                    
                    # Try to load the tokenizer for this task, with fallback
                    try:
                        self.tokenizers[task] = AutoTokenizer.from_pretrained(model_name)
                    except (OSError, ValueError) as e:
                        print(f"Could not load tokenizer for {model_name}: {str(e)}")
                        print(f"Using default tokenizer for {task}")
                        # Use the default tokenizer if we have it, otherwise fall back to RoBERTa
                        if 'default' in self.tokenizers:
                            self.tokenizers[task] = self.tokenizers['default']
                        else:
                            self.tokenizers[task] = AutoTokenizer.from_pretrained("roberta-base")
            
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Out of memory, falling back to CPU")
                self.device = torch.device('cpu')
                # Retry loading on CPU
                self.load_model()
            else:
                raise e
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for Twitter sentiment analysis.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Generate predictions for all tasks using task-specific models."""
        from .labels import get_all_labels
        
        # Get all label dictionaries
        all_labels = get_all_labels()
        predictions = []
        batch_size = self.batch_size
        
        # Define task contexts (same as in TunedLLM)
        task_contexts = {
            'genre': "What is the type of this message? (direct/news/social media)",
            'related': "Is this message disaster related? (no/yes/maybe)",
            'request': "Does this message contain a request? (yes/no)",
            'offer': "Does this message contain an offer? (yes/no)",
            'aid_related': "Is this message aid related? (yes/no)",
            'medical_help': "Does this message concern medical help? (yes/no)",
            'medical_products': "Does this message concern medical products? (yes/no)",
            'search_and_rescue': "Does this message concern search and rescue? (yes/no)",
            'security': "Does this message concern security? (yes/no)",
            'military': "Does this message concern military? (yes/no)",
            'child_alone': "Does this message mention a child alone? (yes/no)",
            'water': "Does this message concern water? (yes/no)",
            'food': "Does this message concern food? (yes/no)",
            'shelter': "Does this message concern shelter? (yes/no)",
            'clothing': "Does this message concern clothing? (yes/no)",
            'money': "Does this message concern money? (yes/no)",
            'missing_people': "Does this message indicate missing people? (yes/no)",
            'refugees': "Does this message concern refugees? (yes/no)",
            'death': "Does this message imply death? (yes/no)",
            'other_aid': "Is there any other aid needed? (yes/no)",
            'infrastructure_related': "Does this message concern infrastructure? (yes/no)",
            'transport': "Does this message concern transport? (yes/no)",
            'buildings': "Does this message concern buildings? (yes/no)",
            'electricity': "Does this message concern electricity? (yes/no)",
            'tools': "Does this message concern tools? (yes/no)",
            'hospitals': "Does this message concern hospitals? (yes/no)",
            'shops': "Does this message concern shops? (yes/no)",
            'aid_centers': "Does this message concern aid centers? (yes/no)",
            'other_infrastructure': "Does this message concern other infrastructure? (yes/no)",
            'weather_related': "Does this message concern weather? (yes/no)",
            'floods': "Does this message indicate there was a flood? (yes/no)",
            'storm': "Does this message indicate there was a storm? (yes/no)",
            'fire': "Does this message indicate there was a fire? (yes/no)",
            'earthquake': "Does this message indicate there was an earthquake? (yes/no)",
            'cold': "Does this message indicate there was cold? (yes/no)",
            'other_weather': "Does this message indicate there were other weather issues? (yes/no)",
            'direct_report': "Does this show a direct report? (yes/no)"
        }
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_predictions = [{} for _ in range(len(batch_texts))]
            
            try:
                # Process each task
                for task in all_labels.keys():
                    # Get the appropriate model and tokenizer for the task
                    model = self.models.get(task, self.models['default'])
                    tokenizer = self.tokenizers.get(task, self.tokenizers['default'])
                    
                    # Get label dictionary for this task
                    task_labels = all_labels[task]
                    num_classes = len(task_labels)
                    
                    # Encode texts
                    encoded = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.preprocessing_config.get('max_length', 128),
                        return_tensors='pt'
                    ).to(self.device)
                    
                    # Get predictions
                    with torch.no_grad():
                        outputs = model(**encoded)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                    
                    # Process predictions for each example in batch
                    for j in range(len(batch_texts)):
                        scores = {}
                        
                        if task == 'genre':
                            # For genre task - hardcoded labels
                            if probs[j].shape[0] < 3:  # If model gives binary output
                                # Convert binary probability to 3-class distribution with text noise
                                pos_prob = float(probs[j][1])
                                noise = np.random.uniform(-0.1, 0.1)  # Add some noise for variation
                                scores = {
                                    'direct': max(0.1, 0.4 - 0.2 * pos_prob + noise),
                                    'news': max(0.1, 0.3 + 0.4 * pos_prob + noise),
                                    'social media': max(0.1, 0.3 + 0.2 * pos_prob + noise)
                                }
                                # Normalize scores
                                total = sum(scores.values())
                                scores = {k: v/total for k, v in scores.items()}
                            else:  # If model gives 3-class output
                                for label_id, label_name in task_labels.items():
                                    scores[label_name] = float(probs[j][label_id if label_id < probs[j].shape[0] else 0])
                            
                            # Get prediction based on highest score
                            prediction = max(scores.items(), key=lambda x: x[1])[0]
                            
                        elif task == 'related':
                            # For related task - using text labels (no/yes/maybe)
                            # Check how many classes the model actually provides
                            actual_classes = probs[j].shape[0]
                            expected_classes = len(task_labels)
                            
                            # If the model doesn't provide enough classes (e.g., only binary)
                            if actual_classes < expected_classes:
                                # Map available class indices to label names
                                available_labels = {idx: task_labels.get(idx, f"unknown_{idx}") 
                                                  for idx in range(actual_classes)}
                                
                                # Create scores dict with available labels
                                scores = {label: float(probs[j][idx]) for idx, label in available_labels.items()}
                                
                                # Add missing labels with low probability
                                for idx in range(actual_classes, expected_classes):
                                    if idx in task_labels:
                                        scores[task_labels[idx]] = 0.001
                                
                                # Make a binary prediction but map to our 3-class system
                                if actual_classes == 2:  # Binary model
                                    if float(probs[j][1]) > 0.75:
                                        prediction = "yes"  # Confident yes
                                    elif float(probs[j][1]) > 0.5:
                                        prediction = "maybe"  # Uncertain yes
                                    else:
                                        prediction = "no"  # No
                                else:
                                    # Just use the highest probability class
                                    pred_idx = np.argmax(probs[j])
                                    prediction = task_labels.get(pred_idx, "no")
                            else:
                                # The model provides all expected classes
                                pred_idx = np.argmax(probs[j])
                                prediction = task_labels.get(pred_idx, "no")
                                
                                # Create full scores dictionary
                                scores = {task_labels.get(idx, f"unknown_{idx}"): float(probs[j][idx]) 
                                         for idx in range(min(expected_classes, len(probs[j])))}
                        
                        else:
                            # Binary tasks
                            # Add small amount of noise to avoid identical predictions for all samples
                            noise = np.random.uniform(-0.1, 0.1)
                            neg_prob = float(probs[j][0]) + noise
                            pos_prob = float(probs[j][1]) + noise
                            
                            # Normalize
                            total = neg_prob + pos_prob
                            neg_prob /= total
                            pos_prob /= total
                            
                            scores = {
                                'no': neg_prob,
                                'yes': pos_prob
                            }
                            
                            # Apply dynamic threshold based on text length and task
                            # This helps create more diverse predictions
                            base_threshold = 0.5
                            text_len_factor = min(0.05, len(batch_texts[j]) / 5000)  # Longer texts may need different threshold
                            
                            # Task-specific adjustments (these could be tuned based on validation data)
                            task_adjustments = {
                                'request': -0.05,           # Lower threshold for requests
                                'aid_related': -0.03,       # Lower threshold for aid_related
                                'direct_report': -0.04,     # Lower threshold for direct_report
                                'weather_related': -0.02,   # Lower threshold for weather_related
                                'infrastructure_related': 0.02,  # Higher threshold for infrastructure
                                'medical_help': 0.02,       # Higher threshold for medical_help
                                'search_and_rescue': 0.03   # Higher threshold for search_and_rescue
                            }
                            
                            threshold = base_threshold + text_len_factor + task_adjustments.get(task, 0.0)
                            prediction = 'yes' if pos_prob > threshold else 'no'
                        
                        batch_predictions[j][task] = {
                            'prediction': prediction,
                            'scores': scores,
                            'context': task_contexts.get(task, "")
                        }
                    
                    # Clean up
                    del encoded, outputs, logits
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Add this batch's predictions to the overall predictions
                predictions.extend([format_prediction_output(pred) for pred in batch_predictions])
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Out of memory for batch {i}-{i+batch_size}, using fallback predictions")
                    for j in range(len(batch_texts)):
                        task_predictions = {}
                        for task in all_labels.keys():
                            task_labels = all_labels[task]
                            default_scores = {label_name: 1.0/len(task_labels) for _, label_name in task_labels.items()}
                            task_predictions[task] = {
                                'prediction': list(task_labels.values())[0],
                                'scores': default_scores,
                                'context': task_contexts.get(task, "")
                            }
                        predictions.append(format_prediction_output(task_predictions))
                else:
                    raise e
        
        return predictions
    
    def save(self, save_path: str = None) -> None:
        """Save the model to the specified path.
        
        Args:
            save_path: Path to save the model.
        """
        if save_path is None:
            save_path = self.project_root / 'models' / 'untuned' / self.model_name
        
        # Save the model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save the metadata
        metadata = self.get_model_info()
        save_metadata(metadata, save_path)
        
    @classmethod
    def load_from_disk(cls, path: str) -> 'UntunedLLM':
        """Load the model from the specified path on disk.
        
        Args:
            path: Path to load the model from.
        """
        with open(Path(path) / 'metadata.json', 'r') as f:
            import json
            metadata = json.load(f)
        
        instance = cls(
            model_name=path,
            model_config=metadata.get('model_config', {}),
        )
        
        return instance