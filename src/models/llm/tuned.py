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
        classification heads for each task.
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
        """Train the model with improved training configuration."""
        try:
            # Clear GPU cache before loading data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            train_dataset, validation_dataset, class_weights = self.load_data()
            
            steps_per_epoch = len(train_dataset) // self.batch_size
            total_steps = steps_per_epoch * self.training_config.get('epochs', 50)
            warmup_steps = int(total_steps * self.training_config.get('warmup_ratio', 0.1))
            
            # Create checkpoint directory structure
            model_dir = self.project_root / 'models' / 'tuned' / self.model_name
            os.makedirs(model_dir, exist_ok=True)
            
            # Create a unique output directory for this training run
            from datetime import datetime
            run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            output_dir = model_dir / f"run-{run_timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Optimize training arguments for better performance
            training_args = TrainingArguments(
                output_dir=str(output_dir),
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
                save_total_limit=self.training_config.get('save_total_limit', 5),  # Keep more checkpoints
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                fp16=self.training_config.get('fp16', torch.cuda.is_available()),
                gradient_accumulation_steps=self.training_config.get('gradient_accumulation_steps', 4),
                learning_rate=self.training_config.get('learning_rate', 2e-5),
                dataloader_num_workers=2,  # Use multiprocessing for data loading
                dataloader_pin_memory=True,  # Speed up data transfer to GPU
                optim="adamw_torch",  # Use PyTorch's AdamW implementation
            )
            
            # Data collator for padding
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True, # For now pad to the largest text in the batch
                #max_length=self.preprocessing_config.get('max_length', 128),
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
            
            class CustomTrainer(Trainer):
                def __init__(self, *args, focal_loss=None, task_weights=None, available_tasks=None, all_labels=None, **kwargs):
                    self.focal_loss = focal_loss
                    self.task_weights = task_weights or {}
                    self.available_tasks = available_tasks or []
                    self.all_labels = all_labels or {}
                    super().__init__(*args, **kwargs)
                
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    if isinstance(model, MultiHeadXLMRoberta):
                        return multi_head_compute_loss(model, inputs, return_outputs)
                    else:
                        labels = inputs.pop("labels")
                        outputs = model(**inputs, return_dict=True)
                        
                        # Handle different output formats from the model
                        if isinstance(outputs, dict):
                            # MultiHeadXLMRoberta returns a dictionary of task outputs
                            # We need to combine the logits from all tasks
                            logits = []
                            for task_name, task_logits in outputs.items():
                                logits.append(task_logits)
                            # Concatenate all logits along the last dimension
                            if logits:
                                logits = torch.cat(logits, dim=1)
                            else:
                                raise ValueError("No task outputs found in model output dictionary")
                        else:
                            # Standard model returns an object with logits attribute
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
            
            # Create a custom compute_loss function for MultiHeadXLMRoberta
            def multi_head_compute_loss(model, inputs, return_outputs=False):
                labels = inputs.pop("labels")                      # shape [batch, total_labels]
                task_losses = []
                all_logits = []

                # for each task, run the model with that head and compute its loss
                for task_idx, task in enumerate(self.training_tasks):
                    if task not in model.heads:
                        continue

                    # run only that head
                    outputs = model(**inputs, task=task, return_dict=True)
                    logits = outputs.logits                          # shape [batch, num_classes]
                    all_logits.append(logits)

                    # figure out which slice of the big label vector belongs to this task
                    start = sum(len(all_labels_defs[t]) for t in self.training_tasks[:task_idx])
                    end   = start + len(all_labels_defs[task])
                    label_slice = labels[:, start:end]               # one‐hot

                    num_classes = end - start
                    weight      = task_weights.get(task, 1.0)

                    if num_classes > 2:
                        # multi‐class: CE on the full logits
                        target = torch.argmax(label_slice, dim=1)   # shape [batch]
                        loss_fct = torch.nn.CrossEntropyLoss()
                        task_loss = loss_fct(logits, target)
                    else:
                        # binary as 2-way classification: CE on 2-logits
                        target = torch.argmax(label_slice, dim=1)   # shape [batch]
                        loss_fct = torch.nn.CrossEntropyLoss()
                        task_loss = loss_fct(logits, target)

                    task_losses.append(weight * task_loss)

                if task_losses:
                    loss = torch.stack(task_losses).mean()
                else:
                    loss = torch.tensor(0.0, device=model.device)
                
                # Concatenate all logits for evaluation
                if all_logits:
                    combined_logits = torch.cat(all_logits, dim=1)
                    # Create a SequenceClassifierOutput with the combined logits
                    from transformers.modeling_outputs import SequenceClassifierOutput
                    all_outputs = SequenceClassifierOutput(logits=combined_logits)
                else:
                    all_outputs = None

                return (loss, all_outputs) if return_outputs else loss

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
                        metrics[f"{task}_acc"] = accuracy_score(truth, preds)
                        metrics[f"{task}_f1"]  = f1_score(truth, preds, average="weighted")
                    else:
                        probs = torch.sigmoid(torch.from_numpy(task_logits)).numpy().reshape(-1)
                        preds = (probs > 0.5).astype(int)
                        truth = task_labels.reshape(-1).astype(int)
                        metrics[f"{task}_acc"] = accuracy_score(truth, preds)
                        metrics[f"{task}_f1"]  = f1_score(truth, preds)
                    
                    print(f"Added metrics for {task}: acc={metrics[f'{task}_acc']:.4f}, f1={metrics[f'{task}_f1']:.4f}")

                print(f"Final metrics: {metrics}")
                return metrics

            # Create a custom callback to properly save checkpoints
            class SaveMultiHeadModelCallback(TrainerCallback):
                def __init__(self, model, tokenizer, metadata_func, save_metadata_func):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.metadata_func = metadata_func
                    self.save_metadata_func = save_metadata_func
                    
                def on_save(self, args, state, control, **kwargs):
                    # Get the checkpoint directory
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                    
                    # If the model is MultiHeadXLMRoberta, use its save_pretrained method
                    if isinstance(self.model, MultiHeadXLMRoberta):
                        # Save the model with all necessary files
                        self.model.save_pretrained(checkpoint_dir)
                        
                        # Save the tokenizer
                        self.tokenizer.save_pretrained(checkpoint_dir)
                        
                        # Save metadata
                        metadata = self.metadata_func()
                        self.save_metadata_func(metadata, checkpoint_dir)
                        
                        print(f"Saved complete checkpoint to {checkpoint_dir} with config and weights")
                    
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
                        save_metadata_func=save_metadata
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
                
            # Save the model using MultiHeadXLMRoberta's save_pretrained method
            # This ensures all necessary files are saved properly
            if isinstance(self.model, MultiHeadXLMRoberta):
                print(f"Saving MultiHeadXLMRoberta model to {model_save_path}")
                self.model.save_pretrained(model_save_path)
            else:
                print(f"Saving standard model to {model_save_path}")
                trainer.save_model(str(model_save_path))
                
            # Save the tokenizer
            self.tokenizer.save_pretrained(str(model_save_path))
            
            # Save metadata
            metadata = self.get_model_info()
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
                            scores = {
                                'no': 1.0 - float(probs[i][0]),
                                'yes': float(probs[i][0])
                            }
                            
                            # Get prediction based on threshold
                            prediction = 'yes' if scores['yes'] > 0.5 else 'no'
                            
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
                        scores = {'no': 1.0 - prob, 'yes': prob}
                        prediction = 'yes' if prob > 0.5 else 'no'
                    
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
        """Save the model to the specified path.
        
        Args:
            save_path: Path to save the model.
        """
        if save_path is None:
            save_path = self.project_root / 'models' / 'tuned' / self.model_name
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save the model using the appropriate method
        if isinstance(self.model, MultiHeadXLMRoberta):
            print(f"Saving MultiHeadXLMRoberta model to {save_path}")
            self.model.save_pretrained(save_path)
        else:
            print(f"Saving standard model to {save_path}")
            self.model.save_pretrained(save_path)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save metadata
        metadata = self.get_model_info()
        save_metadata(metadata, save_path)
        
        print(f"Model successfully saved to {save_path} with full configuration and weights")
        
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
        
        # Check if this is a MultiHeadXLMRoberta model by looking for task_labels.json
        task_labels_path = path_obj / 'task_labels.json'
        is_multi_head = task_labels_path.exists() or (path_obj / 'config.json').exists()
        
        # Create the instance
        instance = cls(
            model_name=path,
            model_config=model_config,
        )
        
        # If it's a MultiHeadXLMRoberta model, load it using from_pretrained
        if is_multi_head:
            print(f"Loading MultiHeadXLMRoberta model from {path}")
            try:
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
                
                # Load the model using MultiHeadXLMRoberta's from_pretrained
                instance.model = MultiHeadXLMRoberta.from_pretrained(
                    path, 
                    task_labels=task_labels,
                    freeze_backbone=instance.training_config.get('freeze_backbone', True)
                )
                
                # Load the tokenizer
                instance.tokenizer = AutoTokenizer.from_pretrained(path)
                
                # Update task heads
                instance.task_heads = instance.model.heads
                
                print(f"Successfully loaded model with {len(instance.task_heads)} task heads")
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
            'available_tasks': self.available_tasks
        }
