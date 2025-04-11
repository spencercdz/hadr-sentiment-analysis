from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

class BaseLLM(ABC):
    """Base class for LLM models in the HADR sentiment analysis system."""
    
    def __init__(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.model_config = model_config
        self.device = device
        self.model: PreTrainedModel = None
        self.tokenizer: PreTrainedTokenizer = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def predict(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """Run inference on a batch of texts."""
        pass
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text before prediction.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Basic text preprocessing
        text = text.strip()
        return text
    
    def format_predictions(self, predictions: List[Dict[str, float]]) -> List[Dict[str, Union[str, float]]]:
        """Format raw prediction scores into a standardized format.
        
        Args:
            predictions: List of dictionaries containing raw prediction scores
            
        Returns:
            List of dictionaries with formatted predictions
        """
        formatted_predictions = []
        for pred in predictions:
            # Get the label with highest score
            max_label = max(pred.items(), key=lambda x: x[1])[0]
            max_score = pred[max_label]
            
            # Create formatted prediction
            formatted_pred = {
                'sentiment': max_label,
                'confidence': max_score,
                'scores': pred  # Include all scores for reference
            }
            formatted_predictions.append(formatted_pred)
        
        return formatted_predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for metadata.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'model_config': self.model_config,
            'device': self.device
        }