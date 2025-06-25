# untuned.py

"""
Contains the UntunedLLM class, a simple baseline that uses a single,
off-the-shelf model to predict all 38 tasks sequentially. This serves
as a performance and efficiency benchmark against the TunedLLM.
"""

import torch
import gc
from pathlib import Path
from typing import Dict, List, Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import BaseLLM
from .labels import get_all_labels
from .utils import save_metadata


class UntunedLLM(BaseLLM):
    """
    Implementation of a simple, untuned LLM baseline.
    """
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        super().__init__(model_name, model_config)
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        
        # --- Configuration ---
        self.batch_size = model_config.get('batch_size', 16)
        self.max_length = self.training_config.get('max_length', 128)

        # --- Model Initialization ---
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load a single, pre-trained model and tokenizer."""
        print(f"Loading single baseline model: {self.model_name}")
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generates predictions for all tasks by running the same model repeatedly.
        """
        all_task_names = sorted(list(get_all_labels().keys()))
        
        final_predictions_for_all_texts = [{} for _ in texts]

        print(f"Predicting for {len(all_task_names)} tasks one by one (this will be slow)...")
        for task in all_task_names:
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]

                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()

                for j, text_probs in enumerate(probabilities):
                    # Naive logic: Assume the second output class (index 1) corresponds to "positive" or "yes".
                    if text_probs.shape[0] >= 2:
                        confidence_yes = float(text_probs[1])
                    else: # Fallback for models with only one output
                        confidence_yes = float(text_probs[0])
                    
                    prediction_label = "yes" if confidence_yes > 0.5 else "no"
                    original_text_idx = i + j
                    
                    final_predictions_for_all_texts[original_text_idx][task] = {
                        'prediction': prediction_label,
                        'confidence': confidence_yes,
                    }
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        return final_predictions_for_all_texts

    def save(self, save_path: str = None) -> None:
        """Saves only metadata, as this model is not trained."""
        if save_path is None:
            model_dir_name = self.model_name.replace("/", "_")
            save_path = self.project_root / 'models' / 'untuned' / model_dir_name
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        metadata = self.get_model_info()
        save_metadata(metadata, save_path)
        print(f"Untuned model metadata saved to {save_path}")

    @classmethod
    def load_from_disk(cls, path: str) -> 'UntunedLLM':
        with open(Path(path) / 'metadata.json', 'r') as f:
            import json
            metadata = json.load(f)
        
        instance = cls(
            model_name=metadata['model_name'],
            model_config=metadata.get('model_config', {}),
        )
        return instance