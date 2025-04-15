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

            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except OSError:
                print("Tokenizer files not found in model repository, using base model tokenizer.")
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
            
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
        """
        Generate predictions for all tasks using the untuned model.
        Since the untuned model is a plain pretrained model with 2 output logits,
        we replicate its binary prediction for every task in the label mapping.
        """
        from .labels import get_all_labels  # ensure import
        from .utils import format_prediction_output  # ensure import

        predictions = []
        batch_size = self.batch_size
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # Encode texts without token_type_ids since the model does not accept them
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.preprocessing_config.get('max_length', 128),
                return_tensors='pt',
                return_token_type_ids=False  # FIX: Do not return token_type_ids.
            ).to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits  # shape: (batch_size, 2)
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # shape: (batch_size, 2)
            
            # For each sample in the batch, assign the same binary prediction to every task
            for j in range(len(batch_texts)):
                sample_pred = {}
                # Calculate binary prediction and confidence
                pred_binary = int(probs[j][1] > probs[j][0])
                confidence = float(probs[j][1] if pred_binary == 1 else probs[j][0])
                # Build prediction dictionary for each task based on label mappings
                for task in get_all_labels().keys():
                    sample_pred[task] = {
                        'prediction': pred_binary,
                        'confidence': confidence,
                        'scores': {'0': float(probs[j][0]), '1': float(probs[j][1])}
                    }
                predictions.append(format_prediction_output(sample_pred))
            
            # Clean up
            del encoded, outputs, logits
            torch.cuda.empty_cache()
            gc.collect()
        
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