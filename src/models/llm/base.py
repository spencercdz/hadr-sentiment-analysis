# src/models/llm/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path
import torch
import json
from transformers import PreTrainedModel, PreTrainedTokenizer

class BaseLLM(ABC):
    """
    Abstract Base Class for all LLM models in the system.

    This class defines a contract that all model implementations must follow,
    ensuring they have methods for loading, predicting, saving, and being
    loaded from disk.
    """
    
    def __init__(
        self,
        model_name: str,
        model_config: Dict[str, Any]
    ):
        self.model_name = model_name
        self.model_config = model_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: PreTrainedModel = None
        self.tokenizer: PreTrainedTokenizer = None
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model and tokenizer into memory.
        This can be from the Hugging Face Hub or from a local directory.
        """
        pass
    
    @abstractmethod
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of texts and return structured predictions.
        """
        pass
    
    @abstractmethod
    def save(self, save_path: Path) -> None:
        """
        Save all necessary artifacts for this model to a directory.
        This MUST include the model weights, config, and tokenizer.
        """
        pass

    ### FIX: The order of decorators has been swapped.
    @classmethod
    @abstractmethod
    def load_from_disk(cls, load_path: Path) -> 'BaseLLM':
        """
        A class method to load a complete model instance from a directory.
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information for metadata.
        """
        return {
            'model_class': self.__class__.__name__,
            'model_name': self.model_name,
            'model_config': self.model_config,
            'device': self.device
        }