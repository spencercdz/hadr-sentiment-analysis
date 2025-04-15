import sys
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
import logging
from transformers import Trainer, TrainingArguments

from .tuned import TunedLLM

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting model training...")
        
        # Check CUDA availability
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
        
        # Initialize model configuration
        model_config = {
            'preprocessing': {
                'max_length': 128,
                'padding': 'max_length',
                'truncation': True
            },
            'training': {
                'num_epochs': 50,
                'warmup_steps': 500,
                'weight_decay': 0.01,
                'learning_rate': 2e-5,
                'gradient_accumulation_steps': 2,
                'fp16': torch.cuda.is_available(),  # Use mixed precision if CUDA is available
                'logging_steps': 10,
                'save_steps': 100,
                'evaluation_strategy': 'steps',
                'eval_steps': 100,
                'save_total_limit': 2,
                'load_best_model_at_end': True,
                'metric_for_best_model': 'eval_loss',
                'greater_is_better': False
            },
            'batch_size': 32
        }
        
        # Create and train the model
        logger.info("Initializing model...")
        model = TunedLLM(
            model_name='aellxx/disaster-tweet-classification',
            model_config=model_config
        )
        
        # Train the model
        logger.info("Starting training...")
        model.train()
        
        logger.info("Training complete!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error("Training failed. Please check the error message above.")
        raise

if __name__ == "__main__":
    main() 