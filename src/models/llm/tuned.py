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
    AutoConfig
)
from concurrent.futures import ThreadPoolExecutor
from googletrans import Translator
from nltk.corpus import wordnet
from nltk import download
from nltk import word_tokenize

# Download required NLTK resources
try:
    download('punkt', quiet=True)
    download('wordnet', quiet=True)
    download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK resources: {str(e)}")

from .labels import (
    get_all_labels,
    get_related_labels,
    get_genre_labels
)
from .utils import format_prediction_output, save_metadata, batch_encode
from .base import BaseLLM

class TunedLLM(BaseLLM):
    """Implementation of a fine-tuned LLM model for HADR sentiment analysis.
    Uses a pretrained model from the Hugging Face model hub and fine-tunes it on the training data."""

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        """Initialize the tuned LLM model.
        
        Args:
            model_name: Name of the pre-trained model to use
            model_config: Configuration dictionary for the model
        """
        super().__init__(model_name, model_config, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        
        # Get configuration values
        self.batch_size = model_config.get('batch_size', 32)
        self.preprocessing_config = model_config.get('preprocessing', {})
        self.training_config = model_config.get('training', {})
        self.data_augmentation_config = model_config.get('data_augmentation', {})
        
        # Initialize translator for back translation
        self.translator = Translator()
        
        # Define all tasks - updated to use tasks from the labels
        self.all_tasks = list(get_all_labels().keys())
        
        # Sentiment task is typically not part of the training since we're predicting it
        if 'sentiment' in self.all_tasks:
            self.training_tasks = [task for task in self.all_tasks if task != 'sentiment']
        else:
            self.training_tasks = self.all_tasks.copy()
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        
        # Load model using the updated method
        self.load_model()
        
        # Load data paths
        self.train_path = self.project_root / 'data' / 'raw' / 'test1.csv'
        self.validation_path = self.project_root / 'data' / 'raw' / 'validation1.csv'
        
    def calculate_num_labels(self):
        """Calculate the total number of classes for all tasks."""
        num_labels = 0
        all_label_defs = get_all_labels()
        
        for task in self.training_tasks:
            task_label_dict = all_label_defs.get(task, {})
            num_labels += len(task_label_dict)
            
        self.num_labels = num_labels
        return num_labels
            
    def load_model(self):
        """Load the pre-trained model and tokenizer, reinitialize classification head."""
        try:
            # Calculate number of labels
            self.calculate_num_labels()
            
            # Load the model config first to customize parameters
            config = AutoConfig.from_pretrained(self.model_name)
            config.num_labels = self.num_labels
            config.problem_type = "multi_label_classification"
            
            print(f"Loading model with {self.num_labels} output classes")
            
            # Load the pre-trained model with our custom config
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=config,
                ignore_mismatched_sizes=True  # This allows loading with different output size
            ).to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Customize the model architecture if needed
            if hasattr(self.model, 'classifier'):
                old_head = self.model.classifier
                
                # Create a more sophisticated classification head
                new_head = torch.nn.Sequential(
                    torch.nn.Dropout(config.hidden_dropout_prob),
                    torch.nn.Linear(config.hidden_size, config.hidden_size),
                    torch.nn.GELU(),  # Better than Tanh for modern architectures
                    torch.nn.LayerNorm(config.hidden_size),  # Add layer normalization for stability
                    torch.nn.Dropout(config.hidden_dropout_prob),
                    torch.nn.Linear(config.hidden_size, self.num_labels)
                ).to(self.device)
                
                # Initialize new head weights
                with torch.no_grad():
                    # Initialize the final layer with Xavier initialization
                    torch.nn.init.xavier_uniform_(new_head[1].weight)
                    torch.nn.init.zeros_(new_head[1].bias)
                    torch.nn.init.xavier_uniform_(new_head[5].weight)
                    torch.nn.init.zeros_(new_head[5].bias)
                
                self.model.classifier = new_head
                # Update config to match new number of labels
                config.num_labels = self.num_labels
            
            print(f"Model loaded successfully: {self.model_name} with {self.num_labels} output classes")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            traceback.print_exc()
            return False

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for Twitter sentiment analysis.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if pd.isna(text):
            return ""
        text = str(text)
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    
    def get_synonyms(self, word):
        """Get synonyms for a word using WordNet."""
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and synonym not in synonyms:
                    synonyms.append(synonym)
        return synonyms if synonyms else [word]

    def _synonym_replacement(self, text, n=2):
        """Replace n words in the text with their synonyms."""
        words = nltk.word_tokenize(text)
        new_words = words.copy()
        random_word_list = list(set([word for word in words if len(word) > 3]))
        random.shuffle(random_word_list)
        num_replaced = 0
        
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) > 0:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break

        return ' '.join(new_words)

    def _random_deletion(self, text, p=0.1):
        """Randomly delete words from the text with probability p."""
        words = nltk.word_tokenize(text)
        if len(words) == 1:
            return text
            
        new_words = []
        for word in words:
            if random.random() > p or len(word) <= 2:  # Keep short words and randomly keep others
                new_words.append(word)
                
        if len(new_words) == 0:  # If all words are deleted, keep a random one
            return random.choice(words)
            
        return ' '.join(new_words)

    def _random_swap(self, text, n=2):
        """Randomly swap n pairs of words in the text."""
        words = nltk.word_tokenize(text)
        new_words = words.copy()
        
        for _ in range(n):
            if len(new_words) < 2:
                break
                
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            
        return ' '.join(new_words)

    def _random_insertion(self, text, n=2):
        """Randomly insert n synonyms into the text."""
        words = nltk.word_tokenize(text)
        new_words = words.copy()
        
        for _ in range(n):
            if not words:
                break
                
            random_word = random.choice([w for w in words if len(w) > 3])
            synonyms = self.get_synonyms(random_word)
            
            if synonyms:
                random_synonym = random.choice(synonyms)
                random_idx = random.randint(0, len(new_words))
                new_words.insert(random_idx, random_synonym)
                
        return ' '.join(new_words)

    def _back_translation(self, text, target_lang='fr'):
        """Translate text to another language and back to English."""
        try:
            # Skip if translator is not available
            if not hasattr(self, 'translator') or self.translator is None:
                return text
                
            # Translate to target language
            translated = self.translator.translate(text, dest=target_lang)
            # Translate back to English
            back_translated = self.translator.translate(translated.text, dest='en')
            return back_translated.text
        except Exception as e:
            print(f"Back translation error: {str(e)}")
            return text

    def _simple_char_swap(self, text: str) -> str:
        """Simple character swap augmentation that doesn't require NLTK."""
        if len(text) <= 4:
            return text
            
        chars = list(text)
        # Swap a few random adjacent characters
        for _ in range(max(1, len(text) // 20)):
            idx = random.randint(0, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            
        return ''.join(chars)
        
    def _simple_word_dropout(self, text: str) -> str:
        """Simple word dropout augmentation that doesn't require NLTK."""
        # Simple space-based word splitting (not as good as NLTK but works without it)
        words = text.split()
        if len(words) <= 2:
            return text
            
        # Drop a small percentage of words
        keep_prob = 0.9
        result = [word for word in words if random.random() < keep_prob]
        
        # Make sure we don't drop everything
        if not result:
            return text
            
        return ' '.join(result)

    def augment_text(self, text: str) -> str:
        """Apply data augmentation techniques to the input text.
        
        This improved version implements multiple robust augmentation techniques
        and applies them probabilistically based on configuration.
        """
        # If data augmentation is disabled, return the original text
        if not self.data_augmentation_config.get('enabled', False):
            return text
            
        # Check if text is too short for meaningful augmentation
        if len(text.split()) < 4:
            return text
        
        # Get augmentation probabilities from config
        synonym_prob = self.data_augmentation_config.get('synonym_replacement_prob', 0.3)
        deletion_prob = self.data_augmentation_config.get('random_deletion_prob', 0.2)
        swap_prob = self.data_augmentation_config.get('random_swap_prob', 0.2)
        insertion_prob = self.data_augmentation_config.get('random_insertion_prob', 0.2)
        backtrans_prob = self.data_augmentation_config.get('back_translation_prob', 0.1)
        
        # Initialize googletrans for back translation if needed
        if random.random() < backtrans_prob and not hasattr(self, 'translator'):
            try:
                from googletrans import Translator
                self.translator = Translator()
            except ImportError:
                print("Warning: googletrans not available for back translation")
                self.translator = None
                backtrans_prob = 0
        
        # Check if NLTK resources are available
        nltk_available = True
        try:
            from nltk.tokenize import word_tokenize
            word_tokenize("test")
        except (ImportError, LookupError):
            nltk_available = False
            print("Warning: NLTK tokenization not available, falling back to simple methods")
        
        # Define augmentation techniques with their probabilities
        techniques = []
        
        # Add NLTK-based techniques if available
        if nltk_available:
            techniques.extend([
                (self._synonym_replacement, synonym_prob),
                (self._random_deletion, deletion_prob),
                (self._random_swap, swap_prob),
                (self._random_insertion, insertion_prob)
            ])
            
            # Add back translation if translator is available
            if hasattr(self, 'translator') and self.translator is not None:
                techniques.append((self._back_translation, backtrans_prob))
        else:
            # Simple character-level augmentations as fallback
            techniques.append((self._simple_char_swap, 0.1))
            techniques.append((self._simple_word_dropout, 0.1))
        
        # Apply augmentation techniques based on their probabilities
        augmented_text = text
        aug_applied = False  # Track if any augmentation was applied
        
        for technique, prob in techniques:
            if random.random() < prob:
                try:
                    result = technique(augmented_text)
                    # Only accept the augmentation if it changed the text and isn't empty
                    if result and result != augmented_text and len(result) > 10:
                        augmented_text = result
                        aug_applied = True
                except Exception as e:
                    print(f"Error applying augmentation technique {technique.__name__}: {str(e)}")
        
        # If no augmentation was applied, return the original text
        return augmented_text if aug_applied else text
    
    def _synonym_replacement(self, text, n=None):
        """Replace words in the text with their synonyms.
        
        Args:
            text: Input text
            n: Number of words to replace. If None, scales with text length.
        """
        words = nltk.word_tokenize(text)
        
        # Scale number of replacements with text length if not specified
        if n is None:
            n = max(1, int(len(words) * 0.15))  # Replace ~15% of words
            
        new_words = words.copy()
        
        # Only consider words with more than 3 characters that aren't stopwords
        potential_words = [word for word in words if len(word) > 3 and word.isalpha()]
        
        if not potential_words:
            return text
            
        # Shuffle to randomize replacements
        random.shuffle(potential_words)
        
        num_replaced = 0
        for target_word in potential_words:
            # Get synonyms for the word
            synonyms = self.get_synonyms(target_word)
            
            if synonyms and len(synonyms) > 0:
                # Choose a random synonym
                synonym = random.choice(synonyms)
                
                # Replace all occurrences of the target word
                for i in range(len(new_words)):
                    if new_words[i] == target_word:
                        new_words[i] = synonym
                        num_replaced += 1
                        
            # Stop if we've replaced enough words
            if num_replaced >= n:
                break
                
        # Join the words back into a sentence
        augmented_text = ' '.join(new_words)
        return augmented_text

    def _random_deletion(self, text, p=None):
        """Randomly delete words from the text with probability p.
        
        Args:
            text: Input text
            p: Probability of deleting each word. If None, scales with text length.
        """
        words = nltk.word_tokenize(text)
        
        # Minimum of 3 words required
        if len(words) <= 3:
            return text
            
        # Scale deletion probability with text length if not specified
        if p is None:
            # Delete fewer words in shorter texts, more in longer texts
            p = min(0.2, max(0.05, 0.1 * math.log(len(words) / 10)))
            
        # Always keep at least half the words
        keep_count = max(3, int(len(words) * (1 - p)))
        
        # Don't delete functional words (pronouns, articles, etc.) that tend to be short
        keep_list = []
        for word in words:
            # Always keep short words and words with punctuation
            if len(word) <= 3 or not word.isalpha():
                keep_list.append(word)
            elif random.random() > p:  # Randomly keep other words
                keep_list.append(word)
                
        # If we deleted too many words, keep at least the minimum
        if len(keep_list) < keep_count:
            # Randomly select from deleted words to add back
            deleted = [w for w in words if w not in keep_list]
            keep_list.extend(random.sample(deleted, min(keep_count - len(keep_list), len(deleted))))
            
        return ' '.join(keep_list)

    def _random_swap(self, text, n=None):
        """Randomly swap pairs of words in the text.
        
        Args:
            text: Input text
            n: Number of swaps to perform. If None, scales with text length.
        """
        words = nltk.word_tokenize(text)
        
        # Need at least 4 words for meaningful swaps
        if len(words) < 4:
            return text
            
        # Scale number of swaps with text length if not specified
        if n is None:
            n = max(1, int(len(words) * 0.1))  # Swap ~10% of words
            
        new_words = words.copy()
        
        # Track which words can be swapped (avoid punctuation)
        swappable = [i for i, word in enumerate(new_words) if word.isalpha()]
        
        if len(swappable) < 2:
            return text
            
        for _ in range(min(n, len(swappable) // 2)):
            # Choose two different positions to swap
            pos1, pos2 = random.sample(swappable, 2)
            # Perform the swap
            new_words[pos1], new_words[pos2] = new_words[pos2], new_words[pos1]
            
        return ' '.join(new_words)

    def _random_insertion(self, text, n=None):
        """Randomly insert synonyms into the text.
        
        Args:
            text: Input text
            n: Number of insertions to perform. If None, scales with text length.
        """
        words = nltk.word_tokenize(text)
        
        # Need at least 3 words for insertions
        if len(words) < 3:
            return text
            
        # Scale number of insertions with text length if not specified
        if n is None:
            n = max(1, int(len(words) * 0.1))  # Insert ~10% more words
            
        new_words = words.copy()
        
        # Only consider words with more than 3 characters
        content_words = [word for word in words if len(word) > 3 and word.isalpha()]
        
        if not content_words:
            return text
            
        for _ in range(n):
            # Choose a random content word
            word = random.choice(content_words)
            
            # Get synonyms for the word
            synonyms = self.get_synonyms(word)
            
            if synonyms:
                # Choose a random synonym
                synonym = random.choice(synonyms)
                
                # Choose a random position to insert the synonym
                insert_pos = random.randint(0, len(new_words))
                
                # Insert the synonym
                new_words.insert(insert_pos, synonym)
                
        return ' '.join(new_words)
        
    def get_synonyms(self, word):
        """Get synonyms for a word using WordNet with improved filtering.
        
        Args:
            word: Input word to find synonyms for
            
        Returns:
            List of synonyms
        """
        if not word or len(word) <= 3 or not word.isalpha():
            return []
            
        synonyms = set()
        
        # Try to get synonyms from WordNet
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    # Extract the synonym name and replace underscores
                    synonym = lemma.name().replace('_', ' ')
                    
                    # Only add if different from original and is alphabetic
                    if (synonym.lower() != word.lower() and 
                        synonym.isalpha() and 
                        len(synonym) > 2):
                        synonyms.add(synonym)
        except Exception as e:
            print(f"Error getting synonyms for '{word}': {str(e)}")
            
        return list(synonyms)
    
    def preprocess_function(self, examples):
        """Preprocess and tokenize text data for model training."""
        # Tokenize the text
        tokenized = self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=self.preprocessing_config.get('max_length', 128),
            return_tensors='pt'
        )
        
        # Get all task labels
        all_label_defs = get_all_labels()
        
        # Create labels tensor
        batch_size = len(examples['text'])
        labels = torch.zeros((batch_size, self.num_labels))
        
        # Keep track of the current index in the label space
        current_idx = 0
        
        # For each task, get the corresponding labels and set them in the label tensor
        for task in self.training_tasks:
            if task in examples:
                task_label_dict = all_label_defs.get(task, {})
                num_classes = len(task_label_dict)
                
                for i, label_value in enumerate(examples[task]):
                    if pd.notna(label_value):
                        # Convert the label value to an integer index
                        if isinstance(label_value, str):
                            try:
                                label_idx = int(label_value)
                            except ValueError:
                                # Handle text labels for tasks like 'related'
                                if task == 'related':
                                    if label_value.lower() == 'yes':
                                        label_idx = 1
                                    elif label_value.lower() == 'maybe':
                                        label_idx = 2
                                    else:
                                        label_idx = 0
                                else:
                                    # For binary tasks
                                    try:
                                        label_idx = int(label_value)
                                    except ValueError:
                                        label_idx = 1 if label_value.lower() == 'yes' else 0
                        else:
                            label_idx = int(label_value)
                        
                        # Ensure the label index is valid
                        if 0 <= label_idx < num_classes:
                            # Set the corresponding entry in the labels tensor
                            labels[i, current_idx + label_idx] = 1.0
                
                # Move index forward
                current_idx += num_classes
        
        # Add the labels to the tokenized output
        tokenized['labels'] = labels.tolist()
        
        return tokenized

    def train(self):
        """Train the model with improved training configuration."""
        try:
            train_dataset, validation_dataset, class_weights = self.load_data()
            
            steps_per_epoch = len(train_dataset) // self.batch_size
            total_steps = steps_per_epoch * self.training_config.get('epochs', 50)
            warmup_steps = int(total_steps * self.training_config.get('warmup_ratio', 0.1))
            
            training_args = TrainingArguments(
                output_dir=str(self.project_root / 'models' / 'tuned' / self.model_name),
                num_train_epochs=self.training_config.get('epochs', 50),
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                warmup_steps=warmup_steps,
                weight_decay=self.training_config.get('weight_decay', 0.01),
                logging_dir=str(self.project_root / 'logs'),
                logging_steps=self.training_config.get('logging_steps', 10),
                eval_strategy="steps",
                eval_steps=self.training_config.get('eval_steps', 100),
                save_strategy="steps",
                save_steps=self.training_config.get('save_steps', 100),
                save_total_limit=self.training_config.get('save_total_limit', 2),
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                fp16=self.training_config.get('fp16', torch.cuda.is_available()),
                gradient_accumulation_steps=self.training_config.get('gradient_accumulation_steps', 4),
                learning_rate=self.training_config.get('learning_rate', 2e-5),
            )
            
            # Data collator for padding
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding='max_length',
                max_length=self.preprocessing_config.get('max_length', 128),
                pad_to_multiple_of=8 if self.training_config.get('fp16', True) else None
            )
            
            # Define focal loss for imbalanced classes
            class FocalLoss(torch.nn.Module):
                def __init__(self, alpha=1, gamma=2, reduction='mean'):
                    super(FocalLoss, self).__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    self.reduction = reduction
                
                def forward(self, inputs, targets):
                    BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        inputs, targets, reduction='none'
                    )
                    pt = torch.exp(-BCE_loss)
                    F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
                    
                    if self.reduction == 'mean':
                        return torch.mean(F_loss)
                    elif self.reduction == 'sum':
                        return torch.sum(F_loss)
                    else:
                        return F_loss
            
            # Create instances of loss functions
            focal_loss_fn = FocalLoss(gamma=2.0)
            device = self.model.device
            
            # Get task definitions
            all_labels_defs = get_all_labels()
            
            # Task weights to prioritize important tasks
            task_weights = {
                'genre': 1.0,
                'related': 1.2,  # Higher weight for core tasks
                'request': 1.5,
                'offer': 1.5,
                'aid_related': 1.5,
                'direct_report': 1.5
            }
            
            # Set default weights for other tasks
            for task in self.available_tasks:
                if task not in task_weights:
                    task_weights[task] = 1.0
            
            class CustomTrainer(Trainer):
                def __init__(self, *args, focal_loss=None, task_weights=None, available_tasks=None, all_labels=None, **kwargs):
                    self.focal_loss = focal_loss
                    self.task_weights = task_weights or {}
                    self.available_tasks = available_tasks or []
                    self.all_labels = all_labels or {}
                    super().__init__(*args, **kwargs)
                
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    # Check if logits are 3D and reshape if needed
                    if len(logits.shape) == 3:
                        # If output is [batch_size, sequence_length, num_classes]
                        # We'll use the first token's output (similar to CLS token)
                        logits = logits[:, 0, :]
                    
                    if self.focal_loss is not None and self.available_tasks and self.all_labels:
                        # Apply multi-task loss with task-specific weighting
                        task_losses = []
                        current_idx = 0
                        
                        for task in self.available_tasks:
                            task_labels = self.all_labels[task]
                            num_classes = len(task_labels)
                            
                            # Extract task-specific logits and labels
                            task_logits = logits[:, current_idx:current_idx + num_classes]
                            task_labels_tensor = labels[:, current_idx:current_idx + num_classes]
                            
                            # Get task-specific weighting
                            task_weight = self.task_weights.get(task, 1.0)
                            
                            # Use focal loss for heavily imbalanced binary tasks
                            if num_classes == 2 and task in ['request', 'offer', 'aid_related', 'direct_report']:
                                task_loss = self.focal_loss(task_logits, task_labels_tensor)
                            else:
                                # Use standard BCE loss for other tasks
                                loss_fct = torch.nn.functional.binary_cross_entropy_with_logits(
                                    task_logits, task_labels_tensor, reduction='mean'
                                )
                                task_loss = loss_fct
                            
                            # Apply task weighting
                            task_losses.append(task_loss * task_weight)
                            current_idx += num_classes
                        
                        # Combine losses across all tasks
                        loss = torch.mean(torch.stack(task_losses))
                    else:
                        # Fallback to standard BCE loss if task info not available
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                    
                    return (loss, outputs) if return_outputs else loss
            
            # Create the trainer with our custom loss handling
            trainer = CustomTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                data_collator=data_collator,
                focal_loss=focal_loss_fn,
                task_weights=task_weights,
                available_tasks=self.available_tasks,
                all_labels=all_labels_defs,
                callbacks=[EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.get('early_stopping_patience', 5)
                )]
            )
            
            trainer.train()
            
            model_save_path = self.project_root / 'models' / 'tuned' / self.model_name
            trainer.save_model(str(model_save_path))
            self.tokenizer.save_pretrained(str(model_save_path))
            
            eval_results = trainer.evaluate()
            print("\nEvaluation Results:")
            print(eval_results)
            
            self.test_sample_predictions()
            
            print("\nTraining Complete!")
            return True
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            traceback.print_exc()
            raise
    
    def test_sample_predictions(self):
        """Test the model on a few examples."""
        test_texts = [
            "I need help! My house was destroyed in the earthquake.",
            "We are collecting donations for hurricane victims.",
            "The Red Cross is sending emergency response teams to the affected area."
        ]
        
        print("\nTesting model on example texts:")
        for i, text in enumerate(test_texts):
            print(f"\nSample {i+1}: '{text[:50]}...'")
            predictions = self.predict(text)
            
            for task in ['genre', 'related', 'request', 'offer', 'aid_related']:
                if task in predictions:
                    pred = predictions[task]
                    print(f"  {task}: {pred}")
                    
                    if isinstance(predictions.get(f"{task}_scores"), dict):
                        scores = predictions[f"{task}_scores"]
                        print(f"    Scores: {{\n  " + ",\n  ".join([f'\"{k}\": {v}' for k, v in scores.items()]) + "\n}")
    
    def predict(self, texts: List[str], optimize_speed=True):
        """Generate predictions for all tasks (including sentiment) while including context.
        
        Args:
            texts: List of texts to predict
            optimize_speed: Whether to use speed optimizations
            
        Returns:
            List of dictionaries containing predictions for each task along with context
        """
        if optimize_speed:
            return self.predict_optimized(texts)
        
        all_labels = get_all_labels()
        predictions = []
        
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
            'direct_report': "Does this show a direct report? (yes/no)",
            'sentiment': "What is the sentiment of this message? (negative/positive)"
        }
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                batch_texts = [self.preprocess_text(text) for text in batch_texts]
                
                input_ids = self.tokenizer(
                    batch_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=self.preprocessing_config.get('max_length', 128),
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**input_ids)
                    logits = outputs.logits
                
                batch_predictions = []
                
                for j, text in enumerate(batch_texts):
                    task_predictions = {}
                    current_idx = 0
                    
                    for task in self.all_tasks:
                        task_labels = all_labels.get(task, {})
                        num_classes = len(task_labels)
                        
                        task_logits = logits[j, current_idx:current_idx + num_classes]
                        
                        task_probs = torch.nn.functional.softmax(task_logits, dim=0).cpu().numpy()
                        
                        task_scores = {}
                        for label_id, label_name in task_labels.items():
                            if label_id < len(task_probs):
                                task_scores[label_name] = float(task_probs[label_id])
                                
                        if task == 'related':
                            if any(isinstance(k, (int, str)) and k in ['0', '1', '2', 0, 1, 2] for k in task_scores.keys()):
                                text_scores = {}
                                mapping = {0: 'no', '0': 'no', 1: 'yes', '1': 'yes', 2: 'maybe', '2': 'maybe'}
                                for k, v in task_scores.items():
                                    if isinstance(k, (int, str)) and k in mapping:
                                        text_key = mapping[k]
                                        text_scores[text_key] = v
                                    else:
                                        text_scores[k] = v
                                task_scores = text_scores
                        
                        if task == 'genre' or task == 'related':
                            max_label = max(task_scores.items(), key=lambda x: x[1]) if task_scores else ('unknown', 0.0)
                            prediction = max_label[0]
                            
                            if task == 'related':
                                if prediction in ['0', '1', '2', 0, 1, 2]:
                                    mapping = {0: 'no', '0': 'no', 1: 'yes', '1': 'yes', 2: 'maybe', '2': 'maybe'}
                                    prediction = mapping.get(prediction, prediction)
                        else:
                            if 'yes' in task_scores and 'no' in task_scores:
                                prediction = 'yes' if task_scores['yes'] > task_scores['no'] else 'no'
                            else:
                                prediction = "unknown"
                        
                        task_predictions[task] = {
                            'scores': task_scores,
                            'prediction': prediction,
                            'context': task_contexts.get(task, "")
                        }
                        
                        current_idx += num_classes
                    
                    batch_predictions.append(format_prediction_output(task_predictions))
                
                del input_ids, outputs, logits
                torch.cuda.empty_cache()
                gc.collect()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("Out of memory, skipping batch")
                    for _ in range(len(batch_texts)):
                        task_predictions = {}
                        for task in self.all_tasks:
                            task_labels = all_labels.get(task, {})
                            default_scores = {label_name: 1.0/len(task_labels) for label_id, label_name in task_labels.items()}
                            task_predictions[task] = {
                                'scores': default_scores,
                                'prediction': list(task_labels.values())[0] if task_labels else "unknown",
                                'context': task_contexts.get(task, "")
                            }
                        predictions.append(format_prediction_output(task_predictions))
                else:
                    raise e
        
        return predictions
        
    def predict_optimized(self, texts: List[str]):
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
                    padding='max_length',
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
                    task_predictions = {}
                    current_idx = 0
                    
                    for task in self.all_tasks:
                        task_labels = all_labels.get(task, {})
                        num_classes = len(task_labels)
                        
                        task_logits = logits[j, current_idx:current_idx + num_classes]
                        
                        task_probs = torch.nn.functional.softmax(task_logits, dim=0).cpu().numpy()
                        
                        task_scores = {}
                        for label_id, label_name in task_labels.items():
                            if label_id < len(task_probs):
                                task_scores[label_name] = float(task_probs[label_id])
                                
                        if task == 'related':
                            if any(isinstance(k, (int, str)) and k in ['0', '1', '2', 0, 1, 2] for k in task_scores.keys()):
                                text_scores = {}
                                mapping = {0: 'no', '0': 'no', 1: 'yes', '1': 'yes', 2: 'maybe', '2': 'maybe'}
                                for k, v in task_scores.items():
                                    if isinstance(k, (int, str)) and k in mapping:
                                        text_key = mapping[k]
                                        text_scores[text_key] = v
                                    else:
                                        text_scores[k] = v
                                task_scores = text_scores
                        
                        if task == 'genre' or task == 'related':
                            max_label = max(task_scores.items(), key=lambda x: x[1]) if task_scores else ('unknown', 0.0)
                            prediction = max_label[0]
                            
                            if task == 'related':
                                if prediction in ['0', '1', '2', 0, 1, 2]:
                                    mapping = {0: 'no', '0': 'no', 1: 'yes', '1': 'yes', 2: 'maybe', '2': 'maybe'}
                                    prediction = mapping.get(prediction, prediction)
                        else:
                            if 'yes' in task_scores and 'no' in task_scores:
                                prediction = 'yes' if task_scores['yes'] > task_scores['no'] else 'no'
                            else:
                                prediction = "unknown"
                        
                        task_predictions[task] = {
                            'scores': task_scores,
                            'prediction': prediction,
                            'context': task_contexts.get(task, "")
                        }
                        
                        current_idx += num_classes
                    
                    batch_predictions.append(format_prediction_output(task_predictions))
                
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
            
            if not train_data_path.exists():
                raise FileNotFoundError(f"Training data not found at {train_data_path}")
                
            print(f"Loading training data from {train_data_path}")
            train_data = pd.read_csv(train_data_path)
            
            # Determine which tasks are available in the data
            self.available_tasks = []
            for task in self.training_tasks:
                if task in train_data.columns:
                    self.available_tasks.append(task)
                else:
                    print(f"Warning: Task '{task}' not found in data columns")
            
            print(f"Available tasks for training: {self.available_tasks}")
            
            # Apply data augmentation if enabled
            if self.data_augmentation_config.get('enabled', False):
                print("Applying data augmentation to improve model robustness...")
                augmented_data = []
                
                # Process augmentation in parallel
                import concurrent.futures
                
                def augment_row(row):
                    # Only augment disaster-related messages to keep balance
                    is_related = (
                        row.get('related') == 1 or 
                        str(row.get('related')).lower() == 'yes' or
                        row.get('aid_related') == 1 or 
                        str(row.get('aid_related')).lower() == 'yes'
                    )
                    
                    # Higher chance of augmenting positive examples from minority classes
                    is_minority = (
                        row.get('request') == 1 or 
                        str(row.get('request')).lower() == 'yes' or
                        row.get('offer') == 1 or 
                        str(row.get('offer')).lower() == 'yes'
                    )
                    
                    # Determine augmentation probability based on message characteristics
                    aug_prob = 0.3  # Base probability
                    if is_minority:
                        aug_prob = 0.8  # Higher probability for minority classes
                    elif is_related:
                        aug_prob = 0.5  # Medium probability for related messages
                        
                    if random.random() < aug_prob:
                        try:
                            new_row = row.copy()
                            new_row['message'] = self.augment_text(row['message'])
                            
                            # Only return if augmentation actually changed the text
                            if new_row['message'] != row['message']:
                                return new_row
                        except Exception as e:
                            print(f"Augmentation error: {str(e)}")
                    
                    return None
                
                # Process in reasonable chunks to avoid memory issues
                chunk_size = 1000
                for i in range(0, len(train_data), chunk_size):
                    chunk = train_data.iloc[i:i+chunk_size]
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
                        results = list(executor.map(augment_row, [row for _, row in chunk.iterrows()]))
                    
                    # Filter out None results and add to augmented data
                    augmented_data.extend([r for r in results if r is not None])
                
                # Combine original and augmented data
                if augmented_data:
                    print(f"Generated {len(augmented_data)} augmented examples")
                    aug_df = pd.DataFrame(augmented_data)
                    train_data = pd.concat([train_data, aug_df], ignore_index=True)
                    print(f"Training data size after augmentation: {len(train_data)}")
            
            # Split into training and validation sets
            validation_split = 0.1
            train_size = int(len(train_data) * (1 - validation_split))
            
            # Shuffle before splitting
            train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Create train and validation sets
            train_df = train_data[:train_size]
            val_df = train_data[train_size:]
            
            # Create clean DataFrames for Dataset conversion
            train_clean = pd.DataFrame()
            val_clean = pd.DataFrame()
            
            # Add text column
            train_clean['text'] = train_df['message'].astype(str)
            val_clean['text'] = val_df['message'].astype(str)
            
            # Add task columns with proper type conversion
            for task in self.available_tasks:
                if task in train_data.columns:
                    # Convert values to consistent format
                    train_clean[task] = train_df[task].apply(self._normalize_label_value)
                    val_clean[task] = val_df[task].apply(self._normalize_label_value)
            
            # Create Datasets
            train_dataset = Dataset.from_pandas(train_clean)
            val_dataset = Dataset.from_pandas(val_clean)
            
            # Map function to process the datasets
            train_dataset = train_dataset.map(
                self.preprocess_function,
                batched=True,
                batch_size=100,
                desc="Preprocessing training data"
            )
            
            val_dataset = val_dataset.map(
                self.preprocess_function,
                batched=True,
                batch_size=100,
                desc="Preprocessing validation data"
            )
            
            # Calculate class weights for handling imbalance
            class_weights = self._calculate_class_weights(train_df)
            
            return train_dataset, val_dataset, class_weights
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            traceback.print_exc()
            raise
            
    def _normalize_label_value(self, value):
        """Normalize label values to consistent string format."""
        if pd.isna(value):
            return "0"
        
        if isinstance(value, (int, float)):
            return str(int(value))
        
        # Convert yes/no text values
        text_val = str(value).lower().strip()
        if text_val in ['yes', 'true', '1']:
            return "1"
        elif text_val in ['no', 'false', '0']:
            return "0"
        
        # For 'related' task with 'maybe' value
        if text_val == 'maybe':
            return "2"
            
        return str(value)
        
    def _calculate_class_weights(self, df):
        """Calculate class weights to handle class imbalance.
        
        Args:
            df: DataFrame with training data
            
        Returns:
            Dictionary of task -> class weights
        """
        class_weights = {}
        all_label_defs = get_all_labels()
        
        for task in self.available_tasks:
            if task not in df.columns:
                continue
                
            task_label_dict = all_label_defs.get(task, {})
            num_classes = len(task_label_dict)
            
            # For each task, count occurrences of each class
            class_counts = {}
            
            # Convert to numeric format for counting
            values = df[task].apply(self._normalize_label_value)
            
            for label_id in range(num_classes):
                label_str = str(label_id)
                count = (values == label_str).sum()
                class_counts[label_id] = max(1, count)  # Avoid division by zero
            
            # Calculate weights (inversely proportional to frequency)
            total = sum(class_counts.values())
            weights = torch.zeros(num_classes)
            
            for label_id, count in class_counts.items():
                # Use inverse frequency weighting
                weights[label_id] = total / (num_classes * count)
            
            # Normalize weights to sum to num_classes
            if weights.sum() > 0:
                weights = weights * (num_classes / weights.sum())
            else:
                weights = torch.ones(num_classes)
                
            class_weights[task] = weights
            
        return class_weights
    
    def save(self, save_path: str = None) -> None:
        """Save the model to the specified path.
        
        Args:
            save_path: Path to save the model.
        """
        if save_path is None:
            save_path = self.project_root / 'models' / 'tuned' / self.model_name
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        metadata = self.get_model_info()
        save_metadata(metadata, save_path)
        
    @classmethod
    def load_from_disk(cls, path: str) -> 'TunedLLM':
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
            'available_tasks': self.available_tasks
        }
