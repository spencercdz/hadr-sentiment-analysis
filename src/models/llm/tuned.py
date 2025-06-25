import os
import sys
import math
import random
import logging
import traceback
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    AutoConfig,
    EvalPrediction,
    TrainerCallback
)
from sklearn.metrics import accuracy_score, f1_score
from concurrent.futures import ThreadPoolExecutor
from googletrans import Translator
from nltk.corpus import wordnet
from nltk import download, data
from nltk import word_tokenize
from .multi_label_model import MultiLabelModel
from .labels import get_all_labels
from .utils import format_prediction_output, save_metadata, batch_encode
from .base import BaseLLM

# Download required NLTK resources
try:
    download('punkt', quiet=True)
    download('wordnet', quiet=True)
    download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK resources: {str(e)}")

# Define Focal Loss for handling class imbalance
class FocalLoss(torch.nn.Module):
    """Focal Loss implementation for handling imbalanced datasets.
    
    Focal loss applies a modulating factor to the standard cross entropy loss,
    reducing the relative loss for well-classified examples and focusing more
    on hard, misclassified examples.
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # For multi-class classification
        if len(inputs.shape) > 1 and inputs.shape[1] > 1:
            # Apply softmax and compute cross entropy
            ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            # Apply focal modulation
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            # For binary classification
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TunedLLM(BaseLLM):
    """Implementation of a fine-tuned LLM model for HADR sentiment analysis.
    Uses a pretrained model from the Hugging Face model hub and fine-tunes it on the training data.
    Supports multi-head fine-tuning by extending a pre-trained sentiment model with additional task heads."""

    def __init__(self, model_name: str, model_config: Dict[str, Any], pretrained_sentiment_model: str = None):
        """Initialize the tuned LLM model.
        
        Args:
            model_name: Name of the pre-trained model to use
            model_config: Configuration dictionary for the model
            pretrained_sentiment_model: Path or name of a pre-trained sentiment model to extend
        """
        super().__init__(model_name, model_config, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        
        # Get configuration values - optimize batch size based on GPU memory
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                # Get GPU memory in MB
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                # Adjust batch size based on available memory
                if gpu_memory < 8000:  # Less than 8GB
                    default_batch = 8
                elif gpu_memory < 12000:  # Less than 12GB
                    default_batch = 16
                else:  # 12GB or more
                    default_batch = 32
                print(f"Using GPU with {gpu_memory:.2f} MB memory")
            except (AssertionError, RuntimeError) as e:
                print(f"CUDA is available but device access failed: {e}")
                default_batch = 8  # Conservative default
        else:
            default_batch = 8  # Conservative default for CPU
            print("Using CPU for model training/inference")
            
        self.batch_size = model_config.get('batch_size', default_batch)
        self.preprocessing_config = model_config.get('preprocessing', {})
        self.training_config = model_config.get('training', {})
        self.data_augmentation_config = model_config.get('data_augmentation', {})
        
        # Store the pretrained sentiment model path
        self.pretrained_sentiment_model = pretrained_sentiment_model
        
        # Initialize translator for back translation - but only if needed
        self.translator = None
        if self.data_augmentation_config.get('enabled', False) and \
           self.data_augmentation_config.get('back_translation_prob', 0) > 0:
            try:
                self.translator = Translator()
            except Exception as e:
                print(f"Warning: Could not initialize translator: {str(e)}")
        
        # Define all labels
        self.all_labels_map = get_all_labels()
        self.num_labels = self.calculate_num_labels()

        # Initialize model components
        self.model = None
        self.tokenizer = None
        
        # Load model using the updated method
        self.load_model()
        
        # Load data paths
        self.train_path = self.project_root / 'data' / 'raw' / 'test1.csv'
        self.validation_path = self.project_root / 'data' / 'raw' / 'validation1.csv'
        
        # Clear GPU cache after initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # For data augmentation: Check for punkt, wordnet, etc.
        try:
            data.find('tokenizers/punkt')
            data.find('corpora/wordnet')
            self.nltk_available = True
        except LookupError:
            logging.warning("NLTK resources not found: falling back to simple augmentations")
            self.nltk_available = False
        
    def calculate_num_labels(self):
        """Calculate the total number of labels for the multi-label model."""
        all_labels_flat = [label for task_labels in self.all_labels_map.values() for label in task_labels.keys()]
        unique_labels = sorted(list(set(all_labels_flat)))
        self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        self.id_to_label = {i: label for i, label in enumerate(unique_labels)}
        return len(unique_labels)
            
    def load_model(self):
        """Load the pre-trained model and tokenizer for multi-label classification."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = MultiLabelModel(
            backbone_model_name=self.pretrained_sentiment_model or self.model_name,
            num_labels=self.num_labels,
            freeze_backbone=self.training_config.get('freeze_backbone', True)
        )
        self.model.to(self.device)

    def _prepare_data(self, df):
        """Prepare the dataset for training."""
        # Create a single set of labels for multi-label classification
        labels = df[self.all_labels].values.tolist()
        
        # Create a new dataframe with the preprocessed text and labels
        prepared_df = pd.DataFrame({
            'text': df['text'],
            'labels': labels
        })
        
        # Convert to a Hugging Face Dataset
        dataset = Dataset.from_pandas(prepared_df)
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples['text'], 
                truncation=True, 
                padding='max_length',
                max_length=self.model_config.get('max_length', 512)
            ),
            batched=True
        )
        
        # Set the format to PyTorch tensors
        tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return tokenized_dataset
    
    def train(self):
        """Fine-tune the multi-label model on the training data."""
        try:
            # Clear GPU cache before loading data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load data
            train_df = pd.read_csv(self.train_path)
            train_dataset = self._prepare_data(train_df)

            validation_df = pd.read_csv(self.validation_path)
            validation_dataset = self._prepare_data(validation_df)

            # Create checkpoint directory structure
            model_dir = self.project_root / 'models' / 'tuned' / self.model_name
            os.makedirs(model_dir, exist_ok=True)

            # Create a unique output directory for this training run
            from datetime import datetime
            run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            output_dir = model_dir / f"run-{run_timestamp}"
            os.makedirs(output_dir, exist_ok=True)

            # Data collator for padding
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True,
                pad_to_multiple_of=8 if self.training_config.get('fp16', True) else None
            )

            # Define training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.training_config.get('num_epochs', 3),
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                warmup_steps=self.training_config.get('warmup_steps', 100),
                weight_decay=self.training_config.get('weight_decay', 0.01),
                logging_dir=output_dir / 'logs',
                logging_steps=self.training_config.get('logging_steps', 10),
                evaluation_strategy="steps",
                eval_steps=self.training_config.get('eval_steps', 50),
                save_strategy="steps",
                save_steps=self.training_config.get('save_steps', 100),
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                fp16=self.training_config.get('fp16', True),
                report_to=self.training_config.get('report_to', 'none')
            )

            # Function to compute metrics for multi-label classification
            def compute_metrics(p: EvalPrediction):
                logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
                preds = (torch.sigmoid(torch.from_numpy(logits)) > 0.5).int().numpy()
                f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='samples', zero_division=0)
                accuracy = accuracy_score(y_true=p.label_ids, y_pred=preds)
                return {"f1": f1, "accuracy": accuracy}

            # Create a trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=self.training_config.get('early_stopping_patience', 3))]
            )

            # Train the model
            trainer.train()

            # Save the best model
            final_model_dir = model_dir / 'final'
            trainer.save_model(final_model_dir)
            self.tokenizer.save_pretrained(final_model_dir)
            print(f"Saved final model to {final_model_dir}")

            # Save metadata
            metadata = {
                'model_name': self.model_name,
                'training_timestamp': run_timestamp,
                'config': self.model_config
            }
            save_metadata(metadata, final_model_dir)

        except Exception as e:
            logging.error(f"An error occurred during training: {str(e)}")
            traceback.print_exc()
            
    def predict(self, text: str, **kwargs) -> List[str]:
        """Predict labels for a given text using the multi-label model.
        
        Args:
            text: Input text to analyze
            **kwargs: Additional arguments (not used in this implementation)
            
        Returns:
            A list of predicted labels
        """
        
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Process outputs
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            predictions = (probabilities > 0.5).astype(int)
            
            # Map predictions to labels
            predicted_labels = []
            for i, p in enumerate(predictions):
                if p == 1:
                    predicted_labels.append(self.id_to_label[i])
            
            return predicted_labels
        

        """Speed-optimized prediction method.
        
        Optimizations include:
        1. Parallel processing with ThreadPoolExecutor
        2. ONNX Runtime for inference (when available)
        3. Dynamic batching based on available resources
        4. Half-precision for faster computation
        5. Early stopping for certain tasks
        """
        import concurrent.futures
        from time import time
        
        all_labels = get_all_labels()
        predictions = []
        start_time = time()
        
        task_contexts = {
            'genre': "What is the type of this message? (direct/news/social media)",
            'related': "Is this message disaster related? (no/yes/maybe)",
            'request': "Does this message contain a request? (yes/no)",
            'offer': "Does this message contain an offer? (yes/no)",
            'aid_related': "Is this message aid related? (yes/no)",
            'direct_report': "Does this show a direct report? (yes/no)"
        }
        
        avg_length = sum(len(text) for text in texts) / len(texts)
        dynamic_batch_size = max(1, min(self.batch_size, int(4096 / avg_length)))
        
        def preprocess_batch(text_batch):
            return [self.preprocess_text(text) for text in text_batch]
            
        def process_batch(batch_idx):
            start_idx = batch_idx * dynamic_batch_size
            end_idx = min(start_idx + dynamic_batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            try:
                batch_texts = preprocess_batch(batch_texts)
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.preprocessing_config.get('max_length', 128),
                    return_tensors='pt'
                )
                
                if self.device.type == 'cuda':
                    inputs = {k: v.to(self.device).half() for k, v in inputs.items()}
                else:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                
                batch_predictions = []
                
                for j, text in enumerate(batch_texts):
                    probs = torch.sigmoid(logits[j]).cpu().numpy()
                    predictions = (probs > 0.5).astype(int)
                    batch_predictions.append({
                        'labels': self.labels,
                        'probabilities': probs.tolist(),
                        'predictions': predictions.tolist()
                    })
                
                del inputs, outputs, logits
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                return batch_predictions
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                return [{task: {'prediction': 'unknown', 'scores': {}, 'context': ''} 
                        for task in self.all_tasks} for _ in range(len(batch_texts))]
        
        num_batches = (len(texts) + dynamic_batch_size - 1) // dynamic_batch_size
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count() or 2, 4)) as executor:
            future_to_batch = {executor.submit(process_batch, i): i for i in range(num_batches)}
            
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_predictions = future.result()
                    predictions.extend(batch_predictions)
                except Exception as e:
                    print(f"Batch {batch_idx} generated an exception: {str(e)}")
        
        end_time = time()
        print(f"Prediction completed in {end_time - start_time:.2f} seconds for {len(texts)} items")
        
        return predictions

    def load_data(self) -> Tuple[Dataset, Dataset, Dict[str, torch.Tensor]]:
        """Load and preprocess the training data."""
        try:
            # Load the training data
            train_data_path = self.project_root / 'data' / 'raw' / 'train1.csv'
            validation_data_path = self.project_root / 'data' / 'raw' / 'validation1.csv'
            
            if not train_data_path.exists():
                raise FileNotFoundError(f"Training data not found at {train_data_path}")
                
            print(f"Loading training data from {train_data_path}")
            train_data = pd.read_csv(train_data_path)

            print(f"Loading validation data from {validation_data_path}")
            validation_data = pd.read_csv(validation_data_path)
            
            # Determine which tasks are available in the data
            self.available_tasks = []
            for task in self.training_tasks:
                if task in train_data.columns:
                    self.available_tasks.append(task)
                else:
                    print(f"Warning: Task '{task}' not found in data columns")
            
            print(f"Available tasks for training: {self.available_tasks}")
            
            # Create clean DataFrames for Dataset conversion
            train_clean = pd.DataFrame()
            val_clean = pd.DataFrame()
            
            # Add text column
            train_clean['text'] = train_data['message'].astype(str)
            val_clean['text'] = validation_data['message'].astype(str)
            
            # Add task columns with proper type conversion
            for task in self.available_tasks:
                if task in train_data.columns:
                    # Convert values to consistent format
                    train_clean[task] = train_data[task].apply(self._normalize_label_value)
                    val_clean[task] = validation_data[task].apply(self._normalize_label_value)
            
            # Create Datasets
            train_dataset = Dataset.from_pandas(train_clean)
            val_dataset = Dataset.from_pandas(val_clean)
            
            # Map function to process the datasets
            train_dataset = train_dataset.map(
                self.preprocess_function,
                batched=True,
                batch_size=100,
                remove_columns=train_clean.columns.tolist(),
                desc="Preprocessing training data"
            )
            
            val_dataset = val_dataset.map(
                self.preprocess_function,
                batched=True,
                batch_size=100,
                remove_columns=train_clean.columns.tolist(),
                desc="Preprocessing validation data"
            )
            
            return train_dataset, val_dataset, None
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            traceback.print_exc()
            raise
            
    def save(self, save_path: str = None) -> None:
        """Save the model to the specified path with all components for a complete checkpoint.
        
        This method ensures that all necessary components are saved for a complete checkpoint,
        including model weights, configuration, tokenizer, and training state information.
        The saved model can be loaded for inference or to resume training from this exact point.
        
        Args:
            save_path: Path to save the model.
            
        Returns:
            Path where the model was saved
        """
        if save_path is None:
            save_path = self.project_root / 'models' / 'tuned' / self.model_name
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        
        # Save the tokenizer with all necessary files (vocabulary, special tokens, etc.)
        self.tokenizer.save_pretrained(save_path)
        
        # Save comprehensive metadata with model information
        metadata = self.get_model_info()
        
        # Add training configuration to metadata
        metadata['training_config'] = self.training_config
        metadata['preprocessing_config'] = self.preprocessing_config
        metadata['data_augmentation_config'] = self.data_augmentation_config
        
        # Add task information
        metadata['task_label_counts'] = self.task_label_counts
        metadata['available_tasks'] = self.available_tasks
        
        # Save the metadata
        save_metadata(metadata, save_path)
        
        # Save task label information separately for easier access
        with open(os.path.join(save_path, "task_labels.json"), 'w') as f:
            import json
            json.dump(self.task_label_counts, f, indent=2)
        
        print(f"Model successfully saved to {save_path} with complete checkpoint information")
        return save_path
        
    @classmethod
    def load_from_disk(cls, path: str) -> 'TunedLLM':
        """Load the model from the specified path on disk.
        
        Args:
            path: Path to load the model from.
        """
        path_obj = Path(path)
        metadata_path = path_obj / 'metadata.json'
        
        # Load metadata if it exists
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                import json
                metadata = json.load(f)
            model_config = metadata.get('model_config', {})
        else:
            print(f"Warning: No metadata.json found at {metadata_path}. Using default configuration.")
            model_config = {}
        
        # Create the instance
        instance = cls(
            model_name=path,
            model_config=model_config,
        )
        
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        
        return instance

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'model_config': {
                'batch_size': self.batch_size,
                'preprocessing': self.preprocessing_config,
                'training': self.training_config
            },
            'device': str(self.device),
            'tasks': self.all_tasks,
            'available_tasks': self.available_tasks,
            'task_label_counts': self.task_label_counts,
            'num_labels': self.num_labels if hasattr(self, 'num_labels') else None
        }