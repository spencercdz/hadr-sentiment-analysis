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
        """Initialize the untuned LLM model.
        
        Args:
            model_name: Name of the pre-trained model to use
            model_config: Configuration dictionary for the model
        """
        super().__init__(model_name, model_config, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        
        # Get configuration values
        self.batch_size = model_config.get('batch_size', 8)
        self.preprocessing_config = model_config.get('preprocessing', {})
        
        # Initialize models for each task
        self.models = {}
        self.tokenizers = {}
        
        # Load models for each task
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained model and tokenizer from Hugging Face."""
        try:
            print(f"Loading model: {self.model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Out of memory, falling back to CPU")
                self.device = torch.device('cpu')
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
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
        """Generate predictions for all tasks.
        
        Args:
            texts: List of texts to predict
            
        Returns:
            List of dictionaries containing predictions for each task
        """
        predictions = []
        batch_size = self.batch_size
        
        # Process texts in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating predictions"):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Tokenize the batch
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.preprocessing_config['max_length'],
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = torch.tensor(encoded['input_ids']).to(self.device)
                attention_mask = torch.tensor(encoded['attention_mask']).to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                    scores = torch.softmax(logits, dim=1).cpu().numpy()
                
                # Format predictions for each task
                for j in range(len(batch_texts)):
                    task_predictions = {}
                    for task in [
                        'sentiment', 'genre', 'related', 'request', 'aid_related',
                        'medical_help', 'medical_products', 'search_and_rescue',
                        'security', 'military', 'child_alone', 'water', 'food',
                        'shelter', 'clothing', 'money', 'missing_people', 'refugees',
                        'deaths', 'weather', 'flood', 'storm', 'fire', 'earthquake',
                        'cold', 'other_weather', 'direct_report'
                    ]:
                        task_predictions[task] = scores[j]
                    predictions.append(format_prediction_output(task_predictions))
                
                # Clear memory
                del input_ids, attention_mask, outputs, logits
                torch.cuda.empty_cache()
                gc.collect()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("Out of memory, skipping batch")
                    # Add default predictions for skipped batch
                    for _ in range(len(batch_texts)):
                        task_predictions = {}
                        for task in [
                            'sentiment', 'genre', 'related', 'request', 'aid_related',
                            'medical_help', 'medical_products', 'search_and_rescue',
                            'security', 'military', 'child_alone', 'water', 'food',
                            'shelter', 'clothing', 'money', 'missing_people', 'refugees',
                            'deaths', 'weather', 'flood', 'storm', 'fire', 'earthquake',
                            'cold', 'other_weather', 'direct_report'
                        ]:
                            task_predictions[task] = np.zeros(2)  # Default prediction
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