# Import the LLM models
from .base import BaseLLM
from .tuned import TunedLLM
from .untuned import UntunedLLM

# Import the model registration module first to ensure proper registration
from .model_registration import register_multi_head_model

# Register the model with the Transformers library
try:
    # Register the model when the module is imported
    config_class = register_multi_head_model()
    print(f"Successfully registered MultiHeadXLMRoberta model with config class: {config_class.__name__}")
except Exception as e:
    print(f"Warning: Could not register model with Auto classes: {str(e)}")
    # Continue without registration - will be handled during model loading

# Import the MultiHeadXLMRoberta class after registration to avoid circular imports
from .multi_head import MultiHeadXLMRoberta