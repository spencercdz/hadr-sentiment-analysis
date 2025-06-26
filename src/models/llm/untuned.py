# src/models/llm/untuned.py

import torch
import gc
import json
from pathlib import Path
from typing import Dict, List, Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import BaseLLM
from .labels import get_all_labels
from .utils import save_metadata

class UntunedLLM(BaseLLM):
    """
    A simple baseline model that uses a single, off-the-shelf, pre-trained
    sentiment model to generate predictions for all tasks.

    This model is NOT trained. Its purpose is to provide a performance
    benchmark against which to compare the fine-tuned model.
    """
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        super().__init__(model_name, model_config)
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        
        # Configuration
        self.batch_size = model_config.get('batch_size', 16)
        self.max_length = model_config.get('max_length', 128)

        # This will hold the label mapping from the loaded model (e.g., {0: 'negative', 1: 'neutral', 2: 'positive'})
        self.id2label = {}

        # Load the model upon initialization
        self.load_model()

    def load_model(self):
        """Load a single, pre-trained model and tokenizer from the Hugging Face Hub."""
        print(f"Loading single baseline model: {self.model_name}")
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            # Store the model's label mapping for intelligent prediction
            self.id2label = self.model.config.id2label
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generates predictions using a consistent and logical heuristic.
        - For the 'sentiment' task, it uses the model's actual output.
        - For all other tasks, it predicts 'yes' if the sentiment is positive,
          and 'no' otherwise.
        """
        all_task_names = sorted(list(get_all_labels().keys()))
        final_predictions = [{} for _ in texts]

        print("Generating baseline predictions...")
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

            for j, pred_idx in enumerate(predictions):
                # Get the actual predicted label string (e.g., 'positive')
                pred_label = self.id2label[pred_idx.item()]
                confidence = probabilities[j][pred_idx].item()
                
                original_text_idx = i + j

                # Apply the prediction logic for every task
                for task in all_task_names:
                    if task == 'sentiment':
                        # For the sentiment task, use the direct output
                        final_predictions[original_text_idx][task] = {
                            'prediction': pred_label,
                            'confidence': confidence,
                        }
                    else:
                        # For all other tasks, use the "positive" sentiment as a proxy for "yes"
                        is_positive = pred_label == 'positive'
                        final_predictions[original_text_idx][task] = {
                            'prediction': 'yes' if is_positive else 'no',
                            'confidence': confidence if is_positive else (1.0 - confidence),
                        }
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        return final_predictions

    def save(self, save_path: Path) -> None:
        """
        Saves all necessary model artifacts to the specified directory.
        """
        print(f"Saving untuned model and tokenizer to {save_path}...")
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save the model and tokenizer so it can be fully reloaded
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save metadata for context
        save_metadata(self.get_model_info(), save_path)
        print("Save complete.")

    @classmethod
    def load_from_disk(cls, load_path: Path) -> 'UntunedLLM':
        """
        Loads a complete UntunedLLM instance from a directory on disk.
        """
        print(f"Loading UntunedLLM from disk: {load_path}")
        if not load_path.exists():
            raise FileNotFoundError(f"Directory not found: {load_path}")

        # Load metadata to get the original model name and config
        with open(load_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        # Create a new instance of the class
        instance = cls(
            model_name=metadata['model_name'],
            model_config=metadata.get('model_config', {}),
        )
        
        # Overwrite the model and tokenizer with the saved local versions
        instance.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        instance.tokenizer = AutoTokenizer.from_pretrained(load_path)
        instance.model.to(instance.device)
        instance.model.eval()
        instance.id2label = instance.model.config.id2label
        
        return instance