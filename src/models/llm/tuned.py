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
from .multi_head import MultiHeadXLMRoberta
from .labels import (
    get_all_labels,
    get_related_labels,
    get_genre_labels
)
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
        
        # Define all tasks - updated to use tasks from the labels
        self.all_tasks = list(get_all_labels().keys())
        
        # Sentiment task is typically not part of the training since we're extending a pre-trained sentiment model
        if 'sentiment' in self.all_tasks:
            self.training_tasks = [task for task in self.all_tasks if task != 'sentiment']
        else:
            self.training_tasks = self.all_tasks.copy()
        
        # Initialize model components
        self.model = None  # Main model with shared encoder
        self.tokenizer = None
        self.task_heads = {}  # Dictionary to store task-specific classification heads
        
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
        
    def calculate_task_labels(self):
        """Calculate the number of labels for each task and store in a dictionary."""
        self.task_label_counts = {}
        all_label_defs = get_all_labels()
        
        # Calculate number of labels for each task
        for task in self.all_tasks:
            task_label_dict = all_label_defs.get(task, {})
            self.task_label_counts[task] = len(task_label_dict)
        
        # Calculate total number of labels (excluding sentiment if using pretrained model)
        num_labels = 0
        for task in self.training_tasks:
            num_labels += self.task_label_counts.get(task, 0)
            
        self.num_labels = num_labels
        return self.task_label_counts
            
    def load_model(self):
        """Load the pre-trained model and tokenizer, create multi-head architecture.
        
        Uses the MultiHeadXLMRoberta architecture with a frozen backbone and separate
        classification heads for each task. Ensures proper registration with Hugging Face's
        Auto classes for API compatibility.
        """
        try:
            # Calculate number of labels for each task
            self.calculate_task_labels()
            
            # Determine which model to load as base
            base_model_name = self.pretrained_sentiment_model if self.pretrained_sentiment_model else self.model_name
            print(f"Loading base model from: {base_model_name}")
            
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
            
            # Get freeze_backbone setting from config (default to True)
            freeze_backbone = self.training_config.get('freeze_backbone', True)
            
            # Create the multi-head model
            print(f"Creating MultiHeadXLMRoberta model with frozen backbone: {freeze_backbone}")
            
            # Import and call the enhanced model registration function to ensure proper registration with Hugging Face
            try:
                from .model_registration import ensure_model_registered, MultiHeadXLMRobertaConfig
                # Ensure the model is registered with Hugging Face's Auto classes
                if ensure_model_registered():
                    print("Successfully registered MultiHeadXLMRoberta with Hugging Face's Auto classes")
                
                # Use the registered configuration class for better compatibility
                config = MultiHeadXLMRobertaConfig(
                    backbone_model=base_model_name,
                    task_labels=self.task_label_counts,
                    freeze_backbone=freeze_backbone
                )
            except Exception as e:
                print(f"Warning: Could not register model with Auto classes: {str(e)}")
                print("Model may not be compatible with Hugging Face API requests")
                # Fall back to standard config
                config = AutoConfig.from_pretrained(base_model_name)
                config.model_type = "multi_head_xlm_roberta"
                config.architectures = ["MultiHeadXLMRoberta"]
                config.task_labels = self.task_label_counts
                config.backbone_model = base_model_name
                config.freeze_backbone = freeze_backbone
                
            # Config is now initialized in the try/except block above
            
            self.model = MultiHeadXLMRoberta(
                model_name=base_model_name,
                task_labels=self.task_label_counts,
                freeze_backbone=freeze_backbone
            ).to(self.device)
            
            # Store the task heads for easy access
            self.task_heads = self.model.heads
            
            # If we're using a pretrained sentiment model, we might want to initialize
            # the sentiment head with the pretrained weights
            if self.pretrained_sentiment_model and 'sentiment' in self.task_label_counts:
                try:
                    print(f"Loading pretrained sentiment head from: {self.pretrained_sentiment_model}")
                    pretrained_model = AutoModelForSequenceClassification.from_pretrained(
                        self.pretrained_sentiment_model
                    )
                    
                    # Check if the pretrained model has a classifier
                    if hasattr(pretrained_model, 'classifier'):
                        # Get the sentiment head from our model
                        sentiment_head = self.task_heads['sentiment']
                        
                        # Check if the architectures are compatible
                        if isinstance(pretrained_model.classifier, torch.nn.Linear) and isinstance(sentiment_head[-1], torch.nn.Linear):
                            # Copy the weights and biases of the final layer
                            with torch.no_grad():
                                sentiment_head[-1].weight.copy_(pretrained_model.classifier.weight)
                                sentiment_head[-1].bias.copy_(pretrained_model.classifier.bias)
                            print(f"Successfully initialized sentiment head with pretrained weights")
                        else:
                            print(f"Architectures not compatible for weight transfer")
                    
                    # Clean up
                    del pretrained_model
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Could not load pretrained sentiment head: {str(e)}")
            
            # Print model statistics
            trainable_params = self.model.get_trainable_parameters()
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model loaded successfully with {len(self.task_heads)} task heads")
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
            
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
        
    def unfreeze_backbone_layers(self, num_layers=None):
        """Unfreeze specific layers of the backbone for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the top. If None, unfreeze all layers.
        """
        if isinstance(self.model, MultiHeadXLMRoberta):
            self.model.unfreeze_backbone(num_layers)
            trainable_params = self.model.get_trainable_parameters()
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Unfrozen backbone layers. Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
        else:
            print("Model is not a MultiHeadXLMRoberta instance, cannot unfreeze backbone layers.")
        
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
        if not self.data_augmentation_config.get('enabled', False):
            return text
        words = text.split()
        if len(words) < 4:
            return text

        # pull probs out
        p = self.data_augmentation_config
        techniques = []

        if self.nltk_available:
            techniques = [
                (self._synonym_replacement,    p.get('synonym_replacement_prob', 0.3)),
                (self._random_deletion,        p.get('random_deletion_prob',    0.2)),
                (self._random_swap,            p.get('random_swap_prob',        0.2)),
                (self._random_insertion,       p.get('random_insertion_prob',   0.2)),
            ]
            if p.get('back_translation_prob', 0) > 0 and hasattr(self, 'translator'):
                techniques.append((self._back_translation, p.get('back_translation_prob', 0.1)))
        else:
            # fallback without NLTK
            techniques = [
                (self._simple_char_swap, 0.1),
                (self._simple_word_dropout, 0.1),
            ]

        augmented = text
        for fn, prob in techniques:
            if random.random() < prob:
                out = fn(augmented)
                # accept any non-empty change
                if out and out != augmented:
                    augmented = out

        return augmented
    
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
        # 1) tokenize - use a smaller max_length to reduce memory usage
        max_length = self.preprocessing_config.get('max_length', 128)
        # Use a more efficient tokenization approach
        tokenized = self.tokenizer(
            examples['text'],
            padding=False,  # We'll pad in the data collator
            truncation=True,
            max_length=self.preprocessing_config.get('max_length', 128),
            return_attention_mask=True,
            return_tensors=None
        )

        # 2) prepare empty one‐hot label matrix [batch_size x total_num_labels]
        all_label_defs = get_all_labels()
        batch_size = len(examples['text'])
        labels = torch.zeros((batch_size, self.num_labels), dtype=torch.float)

        current_idx = 0
        for task in self.training_tasks:
            if task not in examples:
                continue

            task_label_dict = all_label_defs.get(task, {})
            num_classes = len(task_label_dict)

            # build string versions of keys & values for isdigit checks
            keys_str = [str(k).strip() for k in task_label_dict.keys()]
            vals_str = [str(v).strip() for v in task_label_dict.values()]

            # decide how to map label‐names → integer IDs
            if all(vs.isdigit() for vs in vals_str):
                # values are digit‐strings:  name→id
                name_to_idx = {
                    str(k).strip().lower(): int(str(v).strip())
                    for k, v in task_label_dict.items()
                }
            elif all(ks.isdigit() for ks in keys_str):
                # keys are digit‐strings:  id→name
                name_to_idx = {
                    str(v).strip().lower(): int(str(k).strip())
                    for k, v in task_label_dict.items()
                }
            else:
                # fallback: neither side is numeric, assume values are the label‐names
                # and enumerate them in insertion order
                vals = [str(v).strip() for v in task_label_dict.values()]
                name_to_idx = {lbl.lower(): idx for idx, lbl in enumerate(vals)}

            # now for this batch, fill the one‐hot at the right offsets
            for i, raw in enumerate(examples[task]):
                if pd.isna(raw):
                    continue

                # a) if it’s already numeric, use it
                if isinstance(raw, (int, float)) or (isinstance(raw, str) and raw.isdigit()):
                    idx = int(raw)
                else:
                    # b) normalize and lookup
                    idx = name_to_idx.get(str(raw).strip().lower(), 0)

                # c) sanity‐check & write one‐hot
                if 0 <= idx < num_classes:
                    labels[i, current_idx + idx] = 1.0

            current_idx += num_classes

        tokenized['labels'] = labels.tolist()
        return tokenized
    
    def train(self):
        """Train the model with improved training configuration.
        
        Implements sequential training of each task head with early stopping for each task.
        This ensures that each head is fully trained before moving to the next one.
        """
        try:
            # Clear GPU cache before loading data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            train_dataset, validation_dataset, class_weights = self.load_data()
            
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
                padding=True, # For now pad to the largest text in the batch
                pad_to_multiple_of=8 if self.training_config.get('fp16', True) else None
            )
            
            # Create instances of loss functions
            focal_loss_fn = FocalLoss(gamma=2.0)
            device = self.model.device
            
            # Get task definitions
            all_labels_defs = get_all_labels()
            training_tasks = self.training_tasks
            
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
            
            # Define a custom trainer for single-task training
            class SingleTaskTrainer(Trainer):
                def __init__(self, *args, task=None, task_idx=0, focal_loss=None, all_labels_defs=None, class_weights_dict=None, **kwargs):
                    self.task = task
                    self.task_idx = task_idx
                    self.focal_loss = focal_loss
                    self.all_labels_defs = all_labels_defs or {}
                    self.class_weights_dict = class_weights_dict or {}
                    
                    # Handle the tokenizer to processing_class conversion for future compatibility
                    if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
                        kwargs['processing_class'] = kwargs.pop('tokenizer')
                        
                    super().__init__(*args, **kwargs)
                
                def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                    # Extract the full labels tensor
                    full_labels = inputs.pop("labels")  # shape [batch, total_labels]
                    
                    # Calculate the start and end indices for this task's labels
                    start_idx = sum(len(self.all_labels_defs[t]) for t in training_tasks[:self.task_idx])
                    end_idx = start_idx + len(self.all_labels_defs[self.task])
                    
                    # Extract just this task's labels (one-hot)
                    task_one_hot_labels = full_labels[:, start_idx:end_idx]
                    
                    # Convert one-hot labels to class indices
                    target_indices = torch.argmax(task_one_hot_labels, dim=1)
                    
                    # Run the model with only this task
                    outputs = model(**inputs, task=self.task, return_dict=True)
                    logits = outputs.logits  # shape [batch, num_classes_for_task]
                    
                    # Determine if FocalLoss should be used for this task
                    use_focal_loss = self.task in ['request', 'offer', 'aid_related', 'direct_report'] and self.focal_loss is not None
                    
                    if use_focal_loss:
                        # FocalLoss expects class indices as targets
                        loss = self.focal_loss(logits, target_indices)
                    else:
                        # Use CrossEntropyLoss with class weights for other tasks
                        task_class_weights = self.class_weights_dict.get(self.task)
                        current_device = logits.device
                        weights_on_device = task_class_weights.to(current_device) if task_class_weights is not None else None
                        
                        loss_fct = torch.nn.CrossEntropyLoss(weight=weights_on_device)
                        loss = loss_fct(logits, target_indices)
                    
                    return (loss, outputs) if return_outputs else loss
            
            # Function to compute metrics for a single task
            def compute_single_task_metrics(eval_pred, task, task_idx, all_labels_defs):
                logits, labels = eval_pred.predictions, eval_pred.label_ids
                
                # If the model returned a tuple (loss, logits), grab logits[0]
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                # Calculate the start and end indices for this task's labels
                start_idx = sum(len(all_labels_defs[t]) for t in training_tasks[:task_idx])
                end_idx = start_idx + len(all_labels_defs[task])
                
                # Extract just this task's labels (one-hot)
                task_one_hot_labels = labels[:, start_idx:end_idx]
                
                # Convert one-hot labels to class indices
                label_indices = np.argmax(task_one_hot_labels, axis=1)
                # Convert logits to predicted class indices
                pred_indices = np.argmax(logits, axis=1)
                
                # Get the label mapping for this task
                label_mapping = all_labels_defs[task]
                num_classes = len(label_mapping)
                
                metrics = {}
                
                # Calculate metrics (unified for binary and multi-class)
                accuracy = accuracy_score(label_indices, pred_indices)
                # For binary tasks, 'weighted' f1 is fine, or can specify positive label if needed
                f1_average_mode = 'binary' if num_classes == 2 and 1 in label_indices else 'weighted'
                if num_classes == 2 and f1_average_mode == 'binary':
                    # Ensure pos_label is correctly identified if not always 1
                    # This assumes the positive class is encoded as 1 after argmax
                    f1 = f1_score(label_indices, pred_indices, average='binary', pos_label=1, zero_division=0)
                else:
                    f1 = f1_score(label_indices, pred_indices, average='weighted', zero_division=0)

                if num_classes == 2: # Optional: print mapping for binary tasks
                    label_names = list(label_mapping.values())
                    if len(label_names) == 2:
                         # Assuming 0 and 1 are the class indices after argmax
                        print(f"Task {task} binary mapping (indices): 0 -> {label_names[0]}, 1 -> {label_names[1]}")

                # Add metrics with both regular and eval_ prefix to ensure compatibility
                metrics[f"{task}_accuracy"] = accuracy
                metrics[f"{task}_f1"] = f1
                metrics[f"eval_{task}_accuracy"] = accuracy
                metrics[f"eval_{task}_f1"] = f1
                
                return metrics
            
            # Train each task head sequentially
            print(f"Starting sequential training of {len(training_tasks)} task heads")

            # Learning rate and unfreezing schedule configurations
            base_lr = self.training_config.get('learning_rate', 2e-5)
            backbone_lr_factor = self.training_config.get('backbone_learning_rate_factor', 0.1) # Factor to multiply base_lr for backbone
            backbone_lr = base_lr * backbone_lr_factor
            unfreeze_incrementally = self.training_config.get('unfreeze_backbone_incrementally', False)
            unfreeze_schedule_config = self.training_config.get('unfreeze_schedule', {})
            initial_unfreeze_layers = unfreeze_schedule_config.get('initial_unfreeze_layers', 0)
            layers_per_step = unfreeze_schedule_config.get('layers_per_step', 2)
            tasks_per_step = unfreeze_schedule_config.get('tasks_per_step', 1) # Unfreeze after this many tasks

            num_currently_unfrozen_layers = 0
            if isinstance(self.model, MultiHeadXLMRoberta) and self.model.backbone is not None:
                total_backbone_layers = len(self.model.backbone.encoder.layer) if hasattr(self.model.backbone, 'encoder') and hasattr(self.model.backbone.encoder, 'layer') else 12 # Default for RoBERTa-base
            else:
                total_backbone_layers = 12 # Default

            # Initial unfreezing if configured
            if unfreeze_incrementally and initial_unfreeze_layers > 0:
                self.unfreeze_backbone_layers(min(initial_unfreeze_layers, total_backbone_layers))
                num_currently_unfrozen_layers = min(initial_unfreeze_layers, total_backbone_layers)
                print(f"Initially unfroze {num_currently_unfrozen_layers} backbone layers.")
            elif not self.training_config.get('freeze_backbone', True): # If backbone is not frozen from the start
                self.unfreeze_backbone_layers(None) # Unfreeze all
                num_currently_unfrozen_layers = total_backbone_layers
                print(f"Backbone initially unfrozen ({num_currently_unfrozen_layers} layers).")


            for task_idx, task in enumerate(training_tasks):
                print(f"\n{'='*50}\nTraining task {task_idx+1}/{len(training_tasks)}: {task}\n{'='*50}")
                
                if task not in self.model.heads:
                    print(f"Task {task} not found in model heads, skipping...")
                    continue

                # Determine effective learning rate for this task
                # Use backbone_lr if any backbone layers are unfrozen, otherwise base_lr for head-only training
                effective_lr = backbone_lr if num_currently_unfrozen_layers > 0 else base_lr
                print(f"Using learning rate: {effective_lr} for task {task} (Backbone LR: {backbone_lr}, Base LR: {base_lr}, Unfrozen Layers: {num_currently_unfrozen_layers})")

                steps_per_epoch = len(train_dataset) // self.batch_size
                task_epochs = self.training_config.get('epochs_per_task', self.training_config.get('epochs', 20))
                total_steps = steps_per_epoch * task_epochs
                warmup_steps = int(total_steps * self.training_config.get('warmup_ratio', 0.1))
                
                task_output_dir = output_dir / task
                os.makedirs(task_output_dir, exist_ok=True)
                
                task_training_args = TrainingArguments(
                    output_dir=str(task_output_dir),
                    num_train_epochs=task_epochs,
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
                    save_total_limit=3,
                    load_best_model_at_end=True,
                    metric_for_best_model=f"eval_{task}_f1",
                    greater_is_better=True,
                    fp16=self.training_config.get('fp16', torch.cuda.is_available()),
                    gradient_accumulation_steps=self.training_config.get('gradient_accumulation_steps', 4),
                    learning_rate=effective_lr, # Use adjusted learning rate
                    dataloader_num_workers=2,
                    dataloader_pin_memory=True,
                    optim="adamw_torch",
                )
                
                task_compute_metrics = lambda eval_pred: compute_single_task_metrics(
                    eval_pred, task, task_idx, all_labels_defs
                )
                
                task_trainer = SingleTaskTrainer(
                    model=self.model,
                    args=task_training_args,
                    train_dataset=train_dataset,
                    eval_dataset=validation_dataset,
                    processing_class=self.tokenizer,
                    data_collator=data_collator,
                    compute_metrics=task_compute_metrics,
                    task=task,
                    task_idx=task_idx,
                    focal_loss=focal_loss_fn,
                    all_labels_defs=all_labels_defs,
                    class_weights_dict=class_weights,
                    callbacks=[EarlyStoppingCallback(
                        early_stopping_patience=self.training_config.get('early_stopping_patience', 5),
                        early_stopping_threshold=self.training_config.get('early_stopping_threshold', 0.01)
                    )]
                )
                
                train_result = task_trainer.train()
                task_trainer.save_model(str(task_output_dir / "best"))
                metrics = train_result.metrics
                task_trainer.log_metrics("train", metrics)
                task_trainer.save_metrics("train", metrics)
                eval_metrics = task_trainer.evaluate()
                task_trainer.log_metrics("eval", eval_metrics)
                task_trainer.save_metrics("eval", eval_metrics)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(f"Completed training for task: {task}")
                print(f"Final evaluation metrics: {eval_metrics}")

                # Incremental unfreezing after this task is done
                if unfreeze_incrementally and (task_idx + 1) % tasks_per_step == 0:
                    if num_currently_unfrozen_layers < total_backbone_layers:
                        num_to_unfreeze_now = num_currently_unfrozen_layers + layers_per_step
                        self.unfreeze_backbone_layers(min(num_to_unfreeze_now, total_backbone_layers))
                        num_currently_unfrozen_layers = min(num_to_unfreeze_now, total_backbone_layers)
                        print(f"Unfrozen up to {num_currently_unfrozen_layers} backbone layers after task {task_idx + 1} ({task}).")
                    else:
                        print(f"All {total_backbone_layers} backbone layers already unfrozen.")
            
            final_model_path = output_dir / "final_model"
            os.makedirs(final_model_path, exist_ok=True)
            self.model.save_pretrained(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            # Save metadata about the training process
            metadata = {
                "model_name": self.model_name,
                "training_tasks": self.training_tasks,
                "batch_size": self.batch_size,
                "training_config": self.training_config,
                "timestamp": run_timestamp,
            }
            save_metadata(metadata, final_model_path / "training_metadata.json")
            
            print(f"\nTraining completed successfully. Model saved to {final_model_path}")
            return str(final_model_path)

            def compute_metrics(eval_pred: EvalPrediction):
                logits, labels = eval_pred.predictions, eval_pred.label_ids
                # If your model returned a tuple (loss, logits), grab logits[0]
                if isinstance(logits, tuple):
                    logits = logits[0]

                all_labels_defs = get_all_labels()
                metrics = {}
                offset = 0

                # Debug information
                print(f"Computing metrics for tasks: {self.training_tasks}")
                print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")

                # Use training_tasks instead of available_tasks to match the order of logits
                for task in self.training_tasks:              
                    if task not in self.model.heads:
                        continue
                        
                    n_classes = len(all_labels_defs[task])
                    task_logits = logits[:, offset:offset + n_classes]
                    task_labels = labels[:, offset:offset + n_classes]
                    offset += n_classes

                    if task_logits.shape[1] == 0:               # nothing to score
                        print(f"Skipping {task} - no logits available")
                        continue

                    print(f"Computing metrics for {task} - classes: {n_classes}, logits shape: {task_logits.shape}")
                    
                    if n_classes > 2:
                        preds = np.argmax(task_logits, axis=1)
                        truth = np.argmax(task_labels, axis=1)
                        metrics[f"eval_{task}_acc"] = accuracy_score(truth, preds)
                        metrics[f"eval_{task}_f1"]  = f1_score(truth, preds, average="weighted")
                    else:
                        probs = torch.sigmoid(torch.from_numpy(task_logits)).numpy().reshape(-1)
                        preds = (probs > 0.5).astype(int)
                        truth = task_labels.reshape(-1).astype(int)
                        metrics[f"eval_{task}_acc"] = accuracy_score(truth, preds)
                        metrics[f"eval_{task}_f1"]  = f1_score(truth, preds)
                    
                    print(f"Added metrics for {task}: acc={metrics[f'eval_{task}_acc']:.4f}, f1={metrics[f'eval_{task}_f1']:.4f}")

                print(f"Final metrics: {metrics}")
                return metrics

            # Create a custom callback to properly save checkpoints
            class SaveMultiHeadModelCallback(TrainerCallback):
                def __init__(self, model, tokenizer, metadata_func, save_metadata_func, tuned_llm_instance=None):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.metadata_func = metadata_func
                    self.save_metadata_func = save_metadata_func
                    self.tuned_llm_instance = tuned_llm_instance
                    
                def on_save(self, args, state, control, **kwargs):
                    # Get the checkpoint directory
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                    
                    # If the model is MultiHeadXLMRoberta, use its save_pretrained method
                    if isinstance(self.model, MultiHeadXLMRoberta):
                        # Store optimizer and scheduler states if available
                        if hasattr(kwargs, 'optimizer') and kwargs['optimizer'] is not None:
                            self.model.optimizer_state = kwargs['optimizer'].state_dict()
                        
                        if hasattr(kwargs, 'scheduler') and kwargs['scheduler'] is not None:
                            self.model.scheduler_state = kwargs['scheduler'].state_dict()
                        
                        # Try to use enhanced model registration for saving
                        try:
                            from .model_registration import register_multi_head_model, save_model_with_auto_registration
                            # Register the model with Auto classes
                            register_multi_head_model()
                            
                            # Save the model with Auto class registration
                            save_model_with_auto_registration(self.model, checkpoint_dir, self.tokenizer)
                            print(f"Checkpoint saved with Auto class registration at {checkpoint_dir}")
                        except Exception as e:
                            print(f"Warning: Could not save with Auto registration: {str(e)}")
                            # Fall back to standard save_pretrained
                            self.model.save_pretrained(checkpoint_dir)
                            # Save the tokenizer with all necessary files
                            self.tokenizer.save_pretrained(checkpoint_dir)
                        
                        # Save metadata with training information
                        metadata = self.metadata_func()
                        # Add current training state information
                        metadata['training_state'] = {
                            'global_step': state.global_step,
                            'epoch': state.epoch,
                            'max_steps': state.max_steps,
                            'best_metric': state.best_metric if hasattr(state, 'best_metric') else None,
                            'best_model_checkpoint': state.best_model_checkpoint if hasattr(state, 'best_model_checkpoint') else None,
                        }
                        # Add information about backbone freezing state
                        metadata['backbone_frozen'] = not any(p.requires_grad for p in self.model.backbone.parameters())
                        metadata['trainable_parameters'] = self.model.get_trainable_parameters()
                        metadata['total_parameters'] = sum(p.numel() for p in self.model.parameters())
                        self.save_metadata_func(metadata, checkpoint_dir)
                        
                        # Save complete training state if TunedLLM instance is available
                        if self.tuned_llm_instance is not None:
                            # Create a training_state subdirectory in the checkpoint directory
                            training_state_dir = os.path.join(checkpoint_dir, 'training_state')
                            # Save the complete training state (optimizer, scheduler, RNG states, etc.)
                            trainer = kwargs.get('trainer', None)
                            if trainer is not None:
                                self.tuned_llm_instance.save_training_state(trainer, training_state_dir)
                            else:
                                print("Warning: Trainer not available in kwargs, cannot save complete training state")
                        
                        print(f"Saved complete checkpoint to {checkpoint_dir} with config, weights, and training state")
                    
                    return control
            
            # Create the trainer with our custom loss handling and optimized settings
            trainer = CustomTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                data_collator=data_collator,
                compute_loss_func=multi_head_compute_loss if isinstance(self.model, MultiHeadXLMRoberta) else None,
                compute_metrics=compute_metrics,
                focal_loss=focal_loss_fn,
                task_weights=task_weights,
                available_tasks=self.available_tasks,
                all_labels=all_labels_defs,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=self.training_config.get('early_stopping_patience', 5)
                    ),
                    SaveMultiHeadModelCallback(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        metadata_func=self.get_model_info,
                        save_metadata_func=save_metadata,
                        tuned_llm_instance=self  # Pass the TunedLLM instance for complete checkpoint saving
                    )
                ]
            )
            
            # Clear memory before training
            torch.cuda.empty_cache()
            
            trainer.train()
            
            # Save the final model after training
            # This ensures all necessary files (config.json, pytorch_model.bin, etc.) are saved
            model_save_path = self.project_root / 'models' / 'tuned' / self.model_name
            os.makedirs(model_save_path, exist_ok=True)
            
            # Get the best model from training
            if training_args.load_best_model_at_end:
                print(f"Using best model from training")
            else:
                print(f"Using final model from training")
                
            # Import the model registration module
            try:
                from .model_registration import register_multi_head_model, save_model_with_auto_registration
                # Register the model with Auto classes
                register_multi_head_model()
            except ImportError as e:
                print(f"Warning: Could not import model_registration module: {str(e)}")
                
            # Save the model using enhanced registration if available
            if isinstance(self.model, MultiHeadXLMRoberta):
                print(f"Saving MultiHeadXLMRoberta model to {model_save_path}")
                try:
                    # Try to use the enhanced registration method
                    save_model_with_auto_registration(self.model, str(model_save_path), self.tokenizer)
                    print(f"Model saved with Auto class registration")
                except Exception as e:
                    print(f"Warning: Could not save with Auto registration: {str(e)}")
                    # Fall back to standard save_pretrained
                    self.model.save_pretrained(model_save_path)
                    # Save the tokenizer
                    self.tokenizer.save_pretrained(str(model_save_path))
            else:
                print(f"Saving standard model to {model_save_path}")
                trainer.save_model(str(model_save_path))
                # Save the tokenizer
                self.tokenizer.save_pretrained(str(model_save_path))
            
            # Save metadata
            metadata = self.get_model_info()
            # Add information about backbone freezing state
            if isinstance(self.model, MultiHeadXLMRoberta):
                metadata['backbone_frozen'] = not any(p.requires_grad for p in self.model.backbone.parameters())
                metadata['trainable_parameters'] = self.model.get_trainable_parameters()
                metadata['total_parameters'] = sum(p.numel() for p in self.model.parameters())
            save_metadata(metadata, model_save_path)
            
            print(f"Model successfully saved to {model_save_path} with full configuration and weights")
            
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
        """Generate predictions for all tasks using the multi-head model.
        
        Uses the MultiHeadXLMRoberta architecture with a frozen backbone and separate
        classification heads for each task.
        
        Args:
            texts: List of texts to predict
            optimize_speed: Whether to use speed optimizations
            
        Returns:
            List of dictionaries containing predictions for each task along with context
        """
        if optimize_speed and not isinstance(self.model, MultiHeadXLMRoberta):
            # Only use optimized version if not using multi-head architecture
            return self.predict_optimized(texts)
        
        all_labels = get_all_labels()
        predictions = []
        
        # Convert single text to list for consistent handling
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        # Preprocess texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # Define task contexts
        task_contexts = {
            'sentiment': "What is the sentiment of this message? (positive/negative/neutral)",
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
            'direct_report': "Does this show a direct report? (yes/no)"
        }
        
        # Check if we're using the multi-head architecture
        if hasattr(self, 'task_heads') and self.task_heads:
            # Multi-head prediction approach
            batch_predictions = [{} for _ in range(len(texts))]
            
            # Tokenize texts once for efficiency
            encoded_inputs = self.tokenizer(
                preprocessed_texts,
                padding=True,
                truncation=True,
                max_length=self.preprocessing_config.get('max_length', 512),
                return_tensors='pt'
            ).to(self.device)
            
            # Process each task with its specific head
            for task in self.all_tasks:
                # Skip tasks that don't have a head
                if task not in self.task_heads:
                    continue
                    
                # Get label dictionary for this task
                task_labels = all_labels.get(task, {})
                
                # Run inference with the task-specific head
                with torch.no_grad():
                    # Call the model with the task parameter to use the correct head
                    outputs = self.model(
                        **encoded_inputs,
                        task=task,
                        return_dict=True
                    )
                    logits = outputs.logits
                    
                    # Process predictions based on task type
                    if task == 'genre' or task == 'related':
                        # Multi-class classification
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                        
                        # Process each example in the batch
                        for i in range(len(texts)):
                            scores = {}
                            for label_id, label_name in task_labels.items():
                                if label_id < probs[i].shape[0]:
                                    scores[label_name] = float(probs[i][label_id])
                            
                            # Get prediction based on highest score
                            prediction = max(scores.items(), key=lambda x: x[1])[0] if scores else list(task_labels.values())[0]
                            
                            # Store prediction and scores
                            batch_predictions[i][task] = {
                                'prediction': prediction,
                                'scores': scores,
                                'context': task_contexts.get(task, "")
                            }
                    else:
                        # Binary classification
                        probs = torch.sigmoid(logits).cpu().numpy()
                        
                        # Process each example in the batch
                        for i in range(len(texts)):
                            # For binary tasks, we have yes/no predictions
                            # Map 0/1 to no/yes explicitly
                            label_names = list(task_labels.values())
                            if len(label_names) == 2:
                                # If we have explicit label names, use them
                                no_label = label_names[0]  # 0 index maps to 'no'
                                yes_label = label_names[1]  # 1 index maps to 'yes'
                            else:
                                # Default to standard yes/no
                                no_label = 'no'
                                yes_label = 'yes'
                                
                            scores = {
                                no_label: 1.0 - float(probs[i][0]),
                                yes_label: float(probs[i][0])
                            }
                            
                            # Get prediction based on threshold
                            prediction = yes_label if scores[yes_label] > 0.5 else no_label
                            
                            # Store prediction and scores
                            batch_predictions[i][task] = {
                                'prediction': prediction,
                                'scores': scores,
                                'context': task_contexts.get(task, "")
                            }
            
            # Format predictions for output
            predictions = [format_prediction_output(pred) for pred in batch_predictions]
        else:
            # Standard prediction approach (single head model)
            # Tokenize texts
            encoded_inputs = batch_encode(preprocessed_texts, self.tokenizer, self.model_config)
            
            # Create dataset
            dataset = Dataset.from_dict({
                'input_ids': encoded_inputs['input_ids'].tolist(),
                'attention_mask': encoded_inputs['attention_mask'].tolist()
            })
            
            # Create data loader
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            
            # Generate predictions
            all_logits = []
            
            # Run inference
            self.model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    all_logits.append(outputs.logits.cpu())
            
            # Concatenate all batch logits
            all_logits = torch.cat(all_logits, dim=0)
            
            # Process predictions for each example
            start_idx = 0
            for i, text in enumerate(texts):
                task_predictions = {}
                
                for task in self.all_tasks:
                    task_labels = all_labels.get(task, {})
                    num_classes = len(task_labels)
                    
                    # Extract logits for this task
                    task_logits = all_logits[i, start_idx:start_idx + num_classes]
                    
                    # Process based on task type
                    if task == 'genre' or task == 'related':
                        # Multi-class classification
                        probs = torch.softmax(task_logits, dim=0).numpy()
                        scores = {}
                        
                        for label_id, label_name in task_labels.items():
                            if label_id < len(probs):
                                scores[label_name] = float(probs[label_id])
                        
                        # Get prediction based on highest score
                        prediction = max(scores.items(), key=lambda x: x[1])[0] if scores else list(task_labels.values())[0]
                    else:
                        # Binary classification
                        prob = torch.sigmoid(task_logits[0]).item()
                        
                        # Map 0/1 to no/yes explicitly
                        label_names = list(task_labels.values())
                        if len(label_names) == 2:
                            # If we have explicit label names, use them
                            no_label = label_names[0]  # 0 index maps to 'no'
                            yes_label = label_names[1]  # 1 index maps to 'yes'
                        else:
                            # Default to standard yes/no
                            no_label = 'no'
                            yes_label = 'yes'
                            
                        scores = {no_label: 1.0 - prob, yes_label: prob}
                        prediction = yes_label if prob > 0.5 else no_label
                    
                    # Store prediction and scores
                    task_predictions[task] = {
                        'prediction': prediction,
                        'scores': scores,
                        'context': task_contexts.get(task, "")
                    }
                    
                    # Update start index for next task
                    start_idx += num_classes
                
                # Add formatted prediction
                predictions.append(format_prediction_output(task_predictions))
        
        # Return single prediction if input was a single string
        if single_input:
            return predictions[0] if predictions else {}
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
            
            else:
                print("Data augmentation is disabled.")
            
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
            
            # Calculate class weights for handling imbalance
            class_weights = self._calculate_class_weights(train_data)
            
            return train_dataset, val_dataset, class_weights
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            traceback.print_exc()
            raise
            
    def _normalize_label_value(self, value):
        if pd.isna(value):
            return "0"
        if isinstance(value, (int, float)):
            return str(int(value))
        text_val = str(value).lower().strip()
        # binary tasks
        if text_val in ("yes","true","1"):
            return "1"
        if text_val in ("no","false","0"):
            return "0"
        # related has maybe
        if text_val == "maybe":
            return "2"
        # genre
        if text_val in ("direct","news","social"):
            return str({"direct":0,"news":1,"social":2}[text_val])
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
        
        # Ensure model is properly registered with Hugging Face's Auto classes before saving
        try:
            from .model_registration import register_multi_head_model
            # Register the model with Hugging Face's Auto classes
            register_multi_head_model()
            print("Ensured MultiHeadXLMRoberta is registered with Hugging Face's Auto classes")
        except Exception as e:
            print(f"Warning: Could not register model with Auto classes: {str(e)}")
            
        # Save the model using the appropriate method
        if isinstance(self.model, MultiHeadXLMRoberta):
            print(f"Saving MultiHeadXLMRoberta model to {save_path}")
            
            # Create/update config.json with required fields for Hugging Face compatibility
            config = self.model.config.to_dict() if hasattr(self.model, 'config') else {}
            config["model_type"] = "multi_head_xlm_roberta"  # Critical for Auto classes
            config["architectures"] = ["MultiHeadXLMRoberta"]  # Required for Auto classes
            config["backbone_model"] = self.model.model_name if hasattr(self.model, 'model_name') else self.model_name
            config["task_labels"] = self.task_label_counts
            config["is_multi_head"] = True
            
            # Save the enhanced config
            import json
            import os
            with open(os.path.join(save_path, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
                
            # Save the model with its save_pretrained method
            self.model.save_pretrained(save_path)
            
            # Create a special file that indicates this is a MultiHeadXLMRoberta model
            # This helps with auto-detection when loading the model via Hugging Face API
            with open(os.path.join(save_path, "multi_head_model.txt"), 'w') as f:
                f.write("This is a MultiHeadXLMRoberta model for HADR sentiment analysis.\n")
                f.write(f"Backbone model: {self.model_name}\n")
                f.write(f"Tasks: {', '.join(self.task_label_counts.keys())}\n")
        else:
            print(f"Saving standard model to {save_path}")
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
        
        # Check if this is a MultiHeadXLMRoberta model by looking for task_labels.json or multi_head_model.txt
        task_labels_path = path_obj / 'task_labels.json'
        multi_head_marker = path_obj / 'multi_head_model.txt'
        is_multi_head = task_labels_path.exists() or multi_head_marker.exists() or (path_obj / 'config.json').exists()
        
        # Create the instance
        instance = cls(
            model_name=path,
            model_config=model_config,
        )
        
        # If it's a MultiHeadXLMRoberta model, load it using from_pretrained
        if is_multi_head:
            print(f"Loading MultiHeadXLMRoberta model from {path}")
            try:
                # Try to import the model registration module
                try:
                    from .model_registration import register_multi_head_model, load_model_with_auto_registration
                    # Register the model with Auto classes
                    register_multi_head_model()
                except ImportError as e:
                    print(f"Warning: Could not import model_registration module: {str(e)}")
                
                # Load task labels if available
                if task_labels_path.exists():
                    with open(task_labels_path, 'r') as f:
                        task_labels = json.load(f)
                else:
                    # Try to extract from config.json
                    config_path = path_obj / 'config.json'
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            task_labels = config.get('task_labels', None)
                    else:
                        task_labels = None
                
                # Determine if backbone should be frozen from training config or metadata
                freeze_backbone = instance.training_config.get('freeze_backbone', True)
                
                # Check if metadata has backbone_frozen information
                if metadata and 'backbone_frozen' in metadata:
                    freeze_backbone = metadata['backbone_frozen']
                    print(f"Using backbone_frozen={freeze_backbone} from metadata")
                
                # Try to use the enhanced loading method if available
                try:
                    if 'load_model_with_auto_registration' in locals():
                        instance.model = load_model_with_auto_registration(
                            path,
                            task_labels=task_labels,
                            freeze_backbone=freeze_backbone
                        )
                        print(f"Loaded model using Auto registration")
                    else:
                        # Fall back to standard from_pretrained
                        instance.model = MultiHeadXLMRoberta.from_pretrained(
                            path, 
                            task_labels=task_labels,
                            freeze_backbone=freeze_backbone
                        )
                        print(f"Loaded model using standard from_pretrained")
                except Exception as e:
                    print(f"Warning: Enhanced loading failed, falling back to standard method: {str(e)}")
                    # Fall back to standard from_pretrained
                    instance.model = MultiHeadXLMRoberta.from_pretrained(
                        path, 
                        task_labels=task_labels,
                        freeze_backbone=freeze_backbone
                    )
                
                # Load the tokenizer
                instance.tokenizer = AutoTokenizer.from_pretrained(path)
                
                # Update task heads
                instance.task_heads = instance.model.heads
                
                # Update instance with backbone freezing information
                instance.training_config['freeze_backbone'] = freeze_backbone
                
                print(f"Successfully loaded model with {len(instance.task_heads)} task heads")
                print(f"Backbone frozen: {freeze_backbone}")
                print(f"Trainable parameters: {instance.model.get_trainable_parameters():,}")
            except Exception as e:
                print(f"Error loading MultiHeadXLMRoberta model: {str(e)}")
                traceback.print_exc()
        
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
        
    def save_training_state(self, trainer, save_path: str = None):
        """Save the current training state for resuming training later.
        
        This method saves the optimizer state, scheduler state, and other training
        information needed to resume training from this exact point.
        
        Args:
            trainer: The Trainer object with current training state
            save_path: Path to save the training state. If None, use the model path.
            
        Returns:
            Path where the training state was saved
        """
        if save_path is None:
            save_path = self.project_root / 'models' / 'tuned' / self.model_name / 'training_state'
            
        os.makedirs(save_path, exist_ok=True)
        
        # Save optimizer state
        if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
            torch.save(trainer.optimizer.state_dict(), os.path.join(save_path, 'optimizer.pt'))
            
        # Save scheduler state
        if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler is not None:
            torch.save(trainer.lr_scheduler.state_dict(), os.path.join(save_path, 'scheduler.pt'))
            
        # Save RNG states for reproducibility
        rng_states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states['cuda'] = torch.cuda.get_rng_state_all()
            
        torch.save(rng_states, os.path.join(save_path, 'rng_states.pt'))
        
        # Save training arguments and state
        if hasattr(trainer, 'args'):
            trainer.args.save_to_json(os.path.join(save_path, 'training_args.json'))
            
        if hasattr(trainer, 'state'):
            with open(os.path.join(save_path, 'trainer_state.json'), 'w') as f:
                import json
                json.dump(trainer.state.__dict__, f, indent=2)
                
        print(f"Training state successfully saved to {save_path}")
        return save_path
        
    def load_training_state(self, trainer, load_path: str = None):
        """Load a previously saved training state to resume training.
        
        Args:
            trainer: The Trainer object to update with loaded state
            load_path: Path to load the training state from. If None, use the model path.
            
        Returns:
            True if successful, False otherwise
        """
        if load_path is None:
            load_path = self.project_root / 'models' / 'tuned' / self.model_name / 'training_state'
            
        if not os.path.exists(load_path):
            print(f"No training state found at {load_path}")
            return False
            
        try:
            # Load optimizer state
            optimizer_path = os.path.join(load_path, 'optimizer.pt')
            if os.path.exists(optimizer_path) and hasattr(trainer, 'optimizer'):
                trainer.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
                print("Loaded optimizer state")
                
            # Load scheduler state
            scheduler_path = os.path.join(load_path, 'scheduler.pt')
            if os.path.exists(scheduler_path) and hasattr(trainer, 'lr_scheduler'):
                trainer.lr_scheduler.load_state_dict(torch.load(scheduler_path))
                print("Loaded learning rate scheduler state")
                
            # Load RNG states
            rng_path = os.path.join(load_path, 'rng_states.pt')
            if os.path.exists(rng_path):
                rng_states = torch.load(rng_path)
                random.setstate(rng_states['python'])
                np.random.set_state(rng_states['numpy'])
                torch.set_rng_state(rng_states['torch'])
                if torch.cuda.is_available() and 'cuda' in rng_states:
                    torch.cuda.set_rng_state_all(rng_states['cuda'])
                print("Loaded random number generator states")
                
            # Load trainer state
            state_path = os.path.join(load_path, 'trainer_state.json')
            if os.path.exists(state_path) and hasattr(trainer, 'state'):
                with open(state_path, 'r') as f:
                    import json
                    state_dict = json.load(f)
                    for key, value in state_dict.items():
                        setattr(trainer.state, key, value)
                print("Loaded trainer state")
                
            print(f"Successfully loaded training state from {load_path}")
            return True
            
        except Exception as e:
            print(f"Error loading training state: {str(e)}")
            import traceback
            traceback.print_exc()
            return False