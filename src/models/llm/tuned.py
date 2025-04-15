from typing import Dict, List, Union, Any, Optional
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    AutoConfig,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from datasets import Dataset
import pandas as pd
import numpy as np
import torch
import gc
from pathlib import Path
import os
from tqdm import tqdm
import random
from googletrans import Translator
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')

from .base import BaseLLM
from .utils import batch_encode, format_prediction_output, save_metadata
from .labels import get_all_labels

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
        
        # Initialize translator for back translation
        self.translator = Translator()
        
        # Define all tasks
        self.all_tasks = [
            'sentiment', 'genre', 'related', 'request', 'offer', 'aid_related',
            'medical_help', 'medical_products', 'search_and_rescue',
            'security', 'military', 'child_alone', 'water', 'food',
            'shelter', 'clothing', 'money', 'missing_people', 'refugees',
            'death', 'other_aid', 'infrastructure_related', 'other_infrastructure', 
            'weather_related', 'floods', 'storm', 'fire', 'earthquake',
            'cold', 'other_weather', 'direct_report'
        ]
        
        self.training_tasks = [task for task in self.all_tasks if task != 'sentiment']
        
        # Initialize model and tokenizer (and classification_heads remain unused in the new training configuration)
        self.model = None
        self.tokenizer = None
        
        # Load model using the updated method
        self.load_model()
        
        # Load data paths
        self.train_path = self.project_root / 'data' / 'raw' / 'test1.csv'
        self.validation_path = self.project_root / 'data' / 'raw' / 'validation1.csv'
        
    def load_model(self):
        """Load the pre-trained model and tokenizer from Hugging Face and reinitialize the classification head.
        This version creates a new head whose output dimension is based on the total number of classes,
        which is assumed to be:
            num_classes = ( (number of binary tasks) * 2 ) + 3
        where the 'related' task uses 3 classes.
        Note: Initially, available tasks are assumed to be all training_tasks. If later the data shows a subset,
        the head is updated in load_data.
        """
        try:
            print(f"Loading model: {self.model_name}")
            # Here we use self.model_name as provided (it might be "aellxx/disaster-tweet-classification")
            # Calculate total number of classes based on all training tasks (assuming all are available)
            num_binary_tasks = len(self.training_tasks) - 1  # all tasks except 'related'
            num_classes = (num_binary_tasks * 2) + 3  # 2 classes per binary task + 3 for 'related'
            
            # Load with original configuration
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,  # original number of labels
                problem_type="multi_label_classification"
            ).to(self.device)
            
            # Get the model's config
            config = self.model.config
            
            # Create a new classification head for our tasks
            old_head = self.model.classifier
            new_head = torch.nn.Sequential(
                torch.nn.Dropout(config.hidden_dropout_prob),
                torch.nn.Linear(config.hidden_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Dropout(config.hidden_dropout_prob),
                torch.nn.Linear(config.hidden_size, num_classes)
            ).to(self.device)
            
            # Initialize the new head weights
            with torch.no_grad():
                # Copy weights from the old head's dense layer if available
                if hasattr(old_head, "dense"):
                    new_head[1].weight = old_head.dense.weight
                    new_head[1].bias = old_head.dense.bias
                # Initialize the final layer
                torch.nn.init.xavier_uniform_(new_head[4].weight)
                torch.nn.init.zeros_(new_head[4].bias)
            
            # Replace the classifier and update configuration
            self.model.classifier = new_head
            config.num_labels = num_classes
            config.problem_type = "multi_label_classification"
            
            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except OSError:
                print("Tokenizer files not found in model repository, using base model tokenizer.")
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
            
            torch.cuda.empty_cache()
            gc.collect()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Out of memory, falling back to CPU")
                self.device = torch.device('cpu')
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=2,
                    problem_type="multi_label_classification"
                ).to(self.device)
                config = self.model.config
                old_head = self.model.classifier
                new_head = torch.nn.Sequential(
                    torch.nn.Dropout(config.hidden_dropout_prob),
                    torch.nn.Linear(config.hidden_size, config.hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Dropout(config.hidden_dropout_prob),
                    torch.nn.Linear(config.hidden_size, num_classes)
                ).to(self.device)
                with torch.no_grad():
                    if hasattr(old_head, "dense"):
                        new_head[1].weight = old_head.dense.weight
                        new_head[1].bias = old_head.dense.bias
                    torch.nn.init.xavier_uniform_(new_head[4].weight)
                    torch.nn.init.zeros_(new_head[4].bias)
                self.model.classifier = new_head
                config.num_labels = num_classes
                config.problem_type = "multi_label_classification"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            else:
                raise e

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
                synonyms.append(lemma.name())
        return list(set(synonyms))

    def augment_text(self, text):
        """Apply various text augmentation techniques without using Google Translate."""
        words = text.split()
        augmented_texts = []
        if random.random() < 0.3:
            n_words = max(1, int(0.1 * len(words)))
            indices = random.sample(range(len(words)), min(n_words, len(words)))
            new_words = words.copy()
            for idx in indices:
                synonyms = self.get_synonyms(words[idx])
                if synonyms:
                    new_words[idx] = random.choice(synonyms)
            augmented_texts.append(' '.join(new_words))
        if random.random() < 0.3 and words:
            n_words = max(1, int(0.1 * len(words)))
            new_words = words.copy()
            for _ in range(n_words):
                idx = random.randint(0, len(new_words))
                word = random.choice(words)
                new_words = new_words[:idx] + [word] + new_words[idx:]
            augmented_texts.append(' '.join(new_words))
        if random.random() < 0.3 and len(words) > 3:
            n_words = max(1, int(0.1 * len(words)))
            indices = random.sample(range(len(words)), min(n_words, len(words)))
            new_words = [word for i, word in enumerate(words) if i not in indices]
            if new_words:
                augmented_texts.append(' '.join(new_words))
        return augmented_texts

    # --- UPDATED METHODS FOR TRAINING AND FORMATTING (Based on sample.py) ---

    def load_data(self) -> tuple:
        """Load and preprocess training and validation data with augmentation."""
        train_data = pd.read_csv(self.train_path)
        validation_data = pd.read_csv(self.validation_path)
        
        print("\nAvailable columns in training data:")
        print(train_data.columns.tolist())
        
        column_mapping = {
            'genre': 'genre',
            'related': 'related',
            'request': 'request',
            'offer': 'offer',
            'aid_related': 'aid_related',
            'medical_help': 'medical_help',
            'medical_products': 'medical_products',
            'search_and_rescue': 'search_and_rescue',
            'security': 'security',
            'military': 'military',
            'child_alone': 'child_alone',
            'water': 'water',
            'food': 'food',
            'shelter': 'shelter',
            'clothing': 'clothing',
            'money': 'money',
            'missing_people': 'missing_people',
            'refugees': 'refugees',
            'death': 'death',
            'other_aid': 'other_aid',
            'infrastructure_related': 'infrastructure_related',
            'other_infrastructure': 'other_infrastructure',
            'weather_related': 'weather_related',
            'floods': 'floods',
            'storm': 'storm',
            'fire': 'fire',
            'earthquake': 'earthquake',
            'cold': 'cold',
            'other_weather': 'other_weather',
            'direct_report': 'direct_report'
        }
        
        for col in column_mapping.values():
            if col in train_data.columns:
                train_data[col] = pd.to_numeric(train_data[col], errors='coerce').fillna(0).astype(int)
                validation_data[col] = pd.to_numeric(validation_data[col], errors='coerce').fillna(0).astype(int)
        
        self.available_tasks = []
        for task in self.training_tasks:
            if task in column_mapping and column_mapping[task] in train_data.columns:
                self.available_tasks.append(task)
            else:
                print(f"Warning: Task '{task}' not found in data columns")
        
        print(f"\nAvailable tasks for training: {self.available_tasks}")
        
        class_weights = {}
        for task in self.available_tasks:
            col_name = column_mapping[task]
            class_counts = train_data[col_name].value_counts()
            total_samples = len(train_data)
            weights = {}
            if task == 'related':
                for class_label in [0, 1, 2]:
                    count = class_counts.get(class_label, 0)
                    weights[class_label] = total_samples / (3 * count) if count > 0 else 1.0
            else:
                for class_label in [0, 1]:
                    count = class_counts.get(class_label, 0)
                    weights[class_label] = total_samples / (2 * count) if count > 0 else 1.0
            class_weights[task] = weights
            print(f"\n{task} class distribution:")
            if task == 'related':
                for class_label in [0, 1, 2]:
                    count = class_counts.get(class_label, 0)
                    print(f"Class {class_label} samples: {count}")
                    print(f"Class {class_label} weight: {weights[class_label]:.2f}")
            else:
                for class_label in [0, 1]:
                    count = class_counts.get(class_label, 0)
                    print(f"Class {class_label} samples: {count}")
                    print(f"Class {class_label} weight: {weights[class_label]:.2f}")
        
        print("\nApplying data augmentation...")
        augmented_data = []
        for _, row in tqdm(train_data.iterrows(), total=len(train_data)):
            text = row['message']
            augmented_texts = self.augment_text(text)
            for aug_text in augmented_texts:
                new_row = row.copy()
                new_row['message'] = aug_text
                augmented_data.append(new_row)
        
        train_data = pd.concat([train_data, pd.DataFrame(augmented_data)], ignore_index=True)
        
        train_data = train_data.fillna({'message': ''})
        validation_data = validation_data.fillna({'message': ''})
        
        train_data['message'] = train_data['message'].apply(self.preprocess_text)
        validation_data['message'] = validation_data['message'].apply(self.preprocess_text)
        
        # Recompute number of classes based on available_tasks
        num_binary_tasks = len(self.available_tasks) - 1 if len(self.available_tasks) > 0 else 0
        new_num_classes = (num_binary_tasks * 2) + 3 if num_binary_tasks > 0 else 3
        
        # Update model head if the number of labels differs from current config
        if self.model.config.num_labels != new_num_classes:
            config = self.model.config
            old_head = self.model.classifier
            new_head = torch.nn.Sequential(
                torch.nn.Dropout(config.hidden_dropout_prob),
                torch.nn.Linear(config.hidden_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Dropout(config.hidden_dropout_prob),
                torch.nn.Linear(config.hidden_size, new_num_classes)
            ).to(self.device)
            with torch.no_grad():
                # Attempt to copy available weights if dimensions match; otherwise, initialize
                try:
                    new_head[1].weight = old_head[1].weight
                    new_head[1].bias = old_head[1].bias
                except Exception:
                    pass
                torch.nn.init.xavier_uniform_(new_head[4].weight)
                torch.nn.init.zeros_(new_head[4].bias)
            self.model.classifier = new_head
            config.num_labels = new_num_classes
        
        train_dataset = Dataset.from_pandas(train_data)
        validation_dataset = Dataset.from_pandas(validation_data)
        
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples['message'],
                padding='max_length',
                truncation='longest_first',
                max_length=self.preprocessing_config.get('max_length', 128),
                return_tensors=None
            )
            batch_size = len(examples['message'])
            # Here we use available_tasks (which might be a subset) to compute total classes
            num_binary_tasks = len(self.available_tasks) - 1 if len(self.available_tasks) > 0 else 0
            num_classes = (num_binary_tasks * 2) + 3 if num_binary_tasks > 0 else 3
            labels = torch.zeros((batch_size, num_classes), dtype=torch.float)
            current_idx = 0
            for task in self.available_tasks:
                col_name = column_mapping[task]
                if task == 'related':
                    for i, label in enumerate(examples[col_name]):
                        label_idx = current_idx + int(label)
                        labels[i, label_idx] = 1.0
                    current_idx += 3
                else:
                    for i, label in enumerate(examples[col_name]):
                        label_idx = current_idx + int(label)
                        labels[i, label_idx] = 1.0
                    current_idx += 2
            return {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': labels.tolist()
            }
        
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in train_dataset.column_names if col != 'message']
        )
        validation_dataset = validation_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in validation_dataset.column_names if col != 'message']
        )
        
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return train_dataset, validation_dataset, class_weights

    def train(self):
        """Train the model with improved training configuration."""
        try:
            train_dataset, validation_dataset, class_weights = self.load_data()
            
            steps_per_epoch = len(train_dataset) // (self.batch_size * self.training_config.get('gradient_accumulation_steps', 1))
            
            training_args = TrainingArguments(
                output_dir=str(self.project_root / 'models' / 'tuned' / self.model_name),
                num_train_epochs=50,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                warmup_steps=self.training_config.get('warmup_steps', 500),
                weight_decay=self.training_config.get('weight_decay', 0.01),
                logging_dir=str(self.project_root / 'logs'),
                logging_steps=10,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_steps=steps_per_epoch,
                save_total_limit=None,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                fp16=self.training_config.get('fp16', True),
                gradient_accumulation_steps=self.training_config.get('gradient_accumulation_steps', 4),
                learning_rate=self.training_config.get('learning_rate', 2e-5),
                remove_unused_columns=False,
                label_smoothing_factor=0.1
            )
            
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding='max_length',
                max_length=self.preprocessing_config.get('max_length', 128),
                pad_to_multiple_of=8 if self.training_config.get('fp16', True) else None
            )
            
            class CustomTrainer(Trainer):
                def __init__(self, *args, available_tasks=None, class_weights=None, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.available_tasks = available_tasks
                    self.class_weights = class_weights
                    self.device = self.model.device
                
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    
                    if isinstance(outputs, tuple):
                        if len(outputs) == 2:
                            loss, logits = outputs
                        else:
                            logits = outputs[0]
                            loss = None
                    else:
                        logits = outputs.logits
                        loss = None
                    
                    if len(logits.shape) == 3:
                        logits = logits[:, 0, :]
                    
                    if loss is None:
                        task_losses = []
                        current_idx = 0
                        for task in self.available_tasks:
                            if task == 'related':
                                task_logits = logits[:, current_idx:current_idx + 3]
                                task_labels = labels[:, current_idx:current_idx + 3]
                                current_idx += 3
                            else:
                                task_logits = logits[:, current_idx:current_idx + 2]
                                task_labels = labels[:, current_idx:current_idx + 2]
                                current_idx += 2
                            
                            pos_weight = torch.tensor([self.class_weights[task][1]]).to(self.device)
                            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                            task_loss = loss_fct(task_logits, task_labels)
                            task_losses.append(task_loss)
                        
                        loss = torch.mean(torch.stack(task_losses))
                    
                    return (loss, outputs) if return_outputs else loss
            
            trainer = CustomTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
                available_tasks=self.available_tasks,
                class_weights=class_weights
            )
            
            trainer.train()
            self.save()
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Generate predictions for all tasks (including sentiment) while including context.
        
        Args:
            texts: List of texts to predict
            
        Returns:
            List of dictionaries containing predictions for each task along with context
        """
        predictions = []
        batch_size = self.batch_size
        
        # Define context for tasks (can be expanded as needed)
        task_contexts = {
            'genre': "What is the type of this message? (direct/news/social media)",
            'related': "Is this message disaster related? (yes/no/maybe)",
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
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating predictions"):
            batch_texts = texts[i:i + batch_size]
            try:
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.preprocessing_config['max_length'],
                    return_tensors='pt'
                )
                input_ids = torch.tensor(encoded['input_ids']).to(self.device)
                attention_mask = torch.tensor(encoded['attention_mask']).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                    scores = torch.softmax(logits, dim=1).cpu().numpy()
                
                for j in range(len(batch_texts)):
                    task_predictions = {}
                    for task_idx, task in enumerate(self.all_tasks):
                        if task == 'sentiment':
                            sentiment_scores = {
                                'negative': float(scores[j][0]),
                                'positive': float(scores[j][1])
                            }
                            task_predictions[task] = {
                                'scores': sentiment_scores,
                                'prediction': bool(scores[j][1] > scores[j][0]),
                                'context': task_contexts.get(task, "")
                            }
                        else:
                            t_idx = self.training_tasks.index(task) if task in self.training_tasks else 0
                            # Using 3 outputs per non-sentiment task
                            task_scores = {
                                'negative': float(scores[j][t_idx * 3]),
                                'positive': float(scores[j][t_idx * 3 + 1]),
                                'neutral': float(scores[j][t_idx * 3 + 2])
                            }
                            task_predictions[task] = {
                                'scores': task_scores,
                                'prediction': bool(scores[j][t_idx * 3 + 1] > scores[j][t_idx * 3]),
                                'context': task_contexts.get(task, "")
                            }
                    predictions.append(format_prediction_output(task_predictions))
                
                del input_ids, attention_mask, outputs, logits
                torch.cuda.empty_cache()
                gc.collect()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("Out of memory, skipping batch")
                    for _ in range(len(batch_texts)):
                        task_predictions = {}
                        for task in self.all_tasks:
                            task_predictions[task] = {
                                'scores': {'negative': 0.5, 'positive': 0.5},
                                'prediction': False,
                                'context': task_contexts.get(task, "")
                            }
                        predictions.append(format_prediction_output(task_predictions))
                else:
                    raise e
        
        return predictions

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
