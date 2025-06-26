# src/models/llm/tuned.py

import os
import logging
import traceback
import json
from typing import Dict, List, Any
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import csv
import random 
from datasets import Dataset
from transformers import (
    AutoConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    XLMRobertaTokenizer, 
    AutoTokenizer,
    TrainerCallback,
    EvalPrediction,
    default_data_collator,
)
from sklearn.metrics import f1_score, accuracy_score

import nltk
import nlpaug.augmenter.word as naw

from .base import BaseLLM
from .labels import get_all_labels, get_sentiment_labels
from .utils import save_metadata
from .multi_head_model import MultiHeadClassificationModel
from .custom_trainer import CustomTrainer

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.WARNING)

def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    """
    Computes and returns F1 (micro and macro) and subset accuracy for the multi-label head.
    """
    # The CustomTrainer is configured to output a tuple of (sentiment_logits, multilabel_logits).
    # We are interested in the second element (index 1) for this metric calculation.
    logits = p.predictions[1] 
    labels = p.label_ids

    # Convert logits to probabilities using the sigmoid function, as this is a multi-label problem.
    probs = 1 / (1 + np.exp(-logits))
    
    # Get binary predictions (0 or 1) by thresholding the probabilities at 0.5.
    y_pred = (probs > 0.5).astype(int)
    
    # Ensure the true labels are also integers.
    y_true = labels.astype(int)
    
    # Calculate F1-score with 'micro' averaging. This metric aggregates the contributions
    # of all classes to compute the average metric. It's useful for imbalanced datasets.
    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    
    # Calculate F1-score with 'macro' averaging. This calculates the metric independently
    # for each class and then takes the average, treating all classes equally.
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    
    # Calculate subset accuracy. This is a strict metric that only considers a prediction
    # correct if the entire set of labels for a given sample is predicted correctly.
    subset_accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    
    return {
        'f1_micro': f1_micro, 
        'f1_macro': f1_macro, 
        'subset_accuracy': subset_accuracy
    }

class CsvLoggingCallback(TrainerCallback):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.is_initialized = False
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or 'eval_loss' not in logs: return
        log_data = {'step': state.global_step, **logs}
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_data.keys())
                if not self.is_initialized:
                    writer.writeheader()
                    self.is_initialized = True
                writer.writerow(log_data)
        except Exception as e:
            logging.error(f"Could not write to CSV log file: {e}")

class TunedLLM(BaseLLM):
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        super().__init__(model_name, model_config)
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This model requires a GPU.")
        self.device = "cuda"
        logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")

        self.training_config = model_config.get('training', {})
        self.augmentation_config = self.training_config.get('augmentation', {})
        self.batch_size = self.training_config.get('batch_size', 16)
        self.max_length = self.training_config.get('max_length', 256)

        all_labels_map = get_all_labels()
        self.sentiment_task_name = 'sentiment'
        self.binary_tasks = []
        self.multiclass_tasks = {}
        for task, labels in all_labels_map.items():
            if task == self.sentiment_task_name: continue
            if len(labels) > 2: self.multiclass_tasks[task] = len(labels)
            else: self.binary_tasks.append(task)
        self.multilabel_column_names = []
        self.num_multilabels = len(self.binary_tasks) + sum(self.multiclass_tasks.values())
        self.num_sentiment_labels = len(get_sentiment_labels())
        
        logging.info(f"Sentiment head will predict {self.num_sentiment_labels} classes.")
        logging.info(f"Multi-label head will be trained on {len(self.binary_tasks)} binary and {len(self.multiclass_tasks)} multi-class tasks.")
        logging.info(f"Total multi-label output neurons (after one-hot encoding): {self.num_multilabels}")

        self.model = None
        self.tokenizer = None

    def load_model(self):
        logging.info(f"Loading tokenizer for '{self.model_name}'.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        logging.info("Initializing custom MultiHeadClassificationModel for feature extraction.")
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_sentiment_labels = self.num_sentiment_labels
        self.model = MultiHeadClassificationModel(config=config, model_name=self.model_name, num_multilabels=self.num_multilabels)
        self.model.to(self.device)

    def _create_dynamic_augment_transform(self):
        logging.info("Creating dynamic augmentation transform...")
        
        try:
            nltk.data.find('corpora/wordnet.zip')
            nltk.data.find('taggers/averaged_perceptron_tagger.zip')
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            logging.warning("Downloading necessary NLTK data for augmentation...")
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            logging.info("NLTK data download complete.")
        
        strength = self.augmentation_config.get('strength', 0.1)
        rate = self.augmentation_config.get('rate', 1.0)
        augmenter = naw.SynonymAug(aug_src='wordnet', aug_p=strength)
        
        logging.info(f"Dynamic augmentation configured with strength={strength} and rate={rate}")
        
        def transform(examples):
            # Handle both single examples and batches
            if isinstance(examples['text'], str):
                # Single example
                original_texts = [examples['text']]
                sentiment_labels = [examples['sentiment_labels']]
                multilabel_labels = [examples['multilabel_labels']]
            else:
                # Batch of examples
                original_texts = examples['text']
                sentiment_labels = examples['sentiment_labels']
                multilabel_labels = examples['multilabel_labels']
            
            augmented_texts = []
            for text in original_texts:
                # Ensure text is a string and handle None/NaN values
                if text is None or pd.isna(text):
                    text = ""
                text = str(text)
                
                if random.random() < rate and text.strip():  # Only augment non-empty texts
                    try:
                        augmented_text = augmenter.augment(text)
                        # Handle case where augmenter returns a list
                        if isinstance(augmented_text, list):
                            augmented_text = augmented_text[0] if augmented_text else text
                        augmented_texts.append(str(augmented_text))
                    except Exception as e:
                        logging.warning(f"Augmentation failed for text: {text[:50]}... Error: {e}")
                        augmented_texts.append(text)
                else:
                    augmented_texts.append(text)
            
            # Tokenize the texts - ensure they are strings
            tokenized = self.tokenizer(
                augmented_texts, 
                truncation=True, 
                padding='max_length', 
                max_length=self.max_length,
                return_tensors=None  # Return lists, not tensors
            )
            
            # Add labels back to tokenized output
            tokenized['sentiment_labels'] = sentiment_labels
            tokenized['multilabel_labels'] = multilabel_labels
            
            return tokenized

        return transform

    def _prepare_data(self, df: pd.DataFrame) -> Dataset:
        all_labels_map = get_all_labels()
        for col_name in all_labels_map.keys():
            if col_name in df.columns:
                value_map = {str(v).lower(): k for k, v in all_labels_map[col_name].items()}
                def safe_mapper(x):
                    if pd.isna(x): return 0
                    key = str(x).lower().strip()
                    return value_map.get(key, int(key) if key.isdigit() and int(key) in value_map.values() else 0)
                df[col_name] = df[col_name].apply(safe_mapper).astype(int)
            else:
                df[col_name] = 0

        final_multilabel_df = pd.DataFrame(index=df.index)
        for task, num_classes in self.multiclass_tasks.items():
            dummies = pd.get_dummies(df[task], prefix=task, dtype=float)
            for i in range(num_classes):
                col_name = f"{task}_{i}"
                if col_name not in dummies.columns: dummies[col_name] = 0.0
            final_multilabel_df = pd.concat([final_multilabel_df, dummies], axis=1)
        for task in self.binary_tasks:
            final_multilabel_df[task] = df[task].astype(float)
        
        if not self.multilabel_column_names:
            self.multilabel_column_names = sorted(final_multilabel_df.columns.tolist())

        df['multilabel_labels'] = final_multilabel_df[self.multilabel_column_names].values.tolist()
        df['sentiment_labels'] = df[self.sentiment_task_name]
        
        return Dataset.from_pandas(df)

    def train(self):
        try:
            self.load_model()
            logging.info("Loading train, validation, and test datasets...")
            train_path = self.project_root / 'data' / 'raw' / 'train1.csv'
            validation_path = self.project_root / 'data' / 'raw' / 'validation1.csv'
            test_path = self.project_root / 'data' / 'raw' / 'test1.csv'
            
            train_df = pd.read_csv(train_path, low_memory=False)
            validation_df = pd.read_csv(validation_path, low_memory=False)

            def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
                if 'message' in df.columns: 
                    df.rename(columns={'message': 'text'}, inplace=True)
                df['text'] = df['text'].fillna("").astype(str)  # Handle NaN values
                return df

            train_df = preprocess_df(train_df)
            validation_df = preprocess_df(validation_df)

            train_dataset = self._prepare_data(train_df)
            validation_dataset = self._prepare_data(validation_df)

            # Apply tokenization/augmentation
            if self.augmentation_config.get('enabled', False):
                augment_transform = self._create_dynamic_augment_transform()
                train_dataset.set_transform(augment_transform)
            else:
                def tokenize_only(examples):
                    tokenized = self.tokenizer(
                        examples['text'], 
                        truncation=True, 
                        padding='max_length', 
                        max_length=self.max_length,
                        return_tensors=None
                    )
                    tokenized['sentiment_labels'] = examples['sentiment_labels']
                    tokenized['multilabel_labels'] = examples['multilabel_labels']
                    return tokenized
                
                train_dataset.set_transform(tokenize_only)

            def tokenize_validation(examples):
                tokenized = self.tokenizer(
                    examples['text'], 
                    truncation=True, 
                    padding='max_length', 
                    max_length=self.max_length,
                    return_tensors=None
                )
                tokenized['sentiment_labels'] = examples['sentiment_labels']
                tokenized['multilabel_labels'] = examples['multilabel_labels']
                return tokenized
            
            validation_dataset.set_transform(tokenize_validation)

            model_dir = self.project_root / 'models' / 'tuned' / self.model_name.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            log_csv_path = model_dir / "training_log.csv"
            csv_logger = CsvLoggingCallback(csv_path=log_csv_path)
            logging.info(f"Detailed training logs will be saved to: {log_csv_path}")

            training_args = TrainingArguments(
                output_dir=str(model_dir),
                remove_unused_columns=False,
                num_train_epochs=self.training_config.get('num_epochs', 500),
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                learning_rate=float(self.training_config.get('learning_rate', 2e-5)),
                logging_strategy="epoch",
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1_micro",
                greater_is_better=True,
                save_total_limit=100,
                fp16=True,
                report_to="none"
            )
            
            early_stopping_patience = self.training_config.get('early_stopping_patience', 50)
            
            trainer = CustomTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                compute_metrics=compute_metrics,
                callbacks=[csv_logger, EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
                data_collator=default_data_collator,
            )

            logging.info("Starting training...")
            trainer.train()
            logging.info("Training finished.")

            if test_path.exists():
                logging.info("--- Evaluating on Test Set ---")
                test_df = pd.read_csv(test_path, low_memory=False)
                test_df = preprocess_df(test_df)
                test_dataset = self._prepare_data(test_df)
                test_dataset = test_dataset.map(tokenize_validation, batched=True, remove_columns=test_df.columns.tolist())
                test_results = trainer.evaluate(eval_dataset=test_dataset)
                logging.info(f"--- Test Results: {test_results} ---")
                csv_logger.on_log(training_args, trainer.state, None, logs={'step': 'test', **test_results})
            
            final_model_dir = model_dir / 'final_model'
            self.save(final_model_dir)

        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            traceback.print_exc()

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Please call load_model() or load_from_disk() first.")

        self.model.eval()
        
        sentiment_labels = get_sentiment_labels()
        idx_to_sentiment = {v: k for k, v in sentiment_labels.items()}
        all_labels_map = get_all_labels()
        
        all_predictions = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            sentiment_preds = torch.argmax(outputs.sentiment_logits, dim=1).cpu().numpy()
            multilabel_probs = torch.sigmoid(outputs.multilabel_logits).cpu().numpy()
            multilabel_preds = (multilabel_probs > 0.5).astype(int)

            for j in range(len(batch_texts)):
                single_prediction = {}
                
                sentiment_idx = sentiment_preds[j]
                single_prediction['sentiment'] = {
                    'prediction': idx_to_sentiment.get(sentiment_idx, "unknown"),
                    'confidence': torch.softmax(outputs.sentiment_logits[j], dim=0)[sentiment_idx].item()
                }

                pred_vector = multilabel_preds[j]
                
                for task_name in self.binary_tasks:
                    pred = pred_vector[self.multilabel_column_names.index(task_name)]
                    single_prediction[task_name] = 'yes' if pred == 1 else 'no'

                for task_name, num_classes in self.multiclass_tasks.items():
                    task_cols = [f"{task_name}_{k}" for k in range(num_classes)]
                    start_idx = self.multilabel_column_names.index(task_cols[0])
                    end_idx = start_idx + num_classes
                    
                    task_preds = pred_vector[start_idx:end_idx]
                    predicted_class_idx = np.argmax(task_preds) if np.sum(task_preds) > 0 else 0
                    
                    idx_to_label = {v: k for k, v in all_labels_map[task_name].items()}
                    single_prediction[task_name] = idx_to_label.get(predicted_class_idx, "unknown")
                
                all_predictions.append(single_prediction)

        return all_predictions

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info['multilabel_column_names'] = self.multilabel_column_names
        return info

    def save(self, save_path: Path) -> None:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving.")
        
        logging.info(f"Saving tuned model, tokenizer, and metadata to {save_path}")
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        save_metadata(self.get_model_info(), save_path)
        logging.info(f"Save complete: {save_path}")

    @classmethod
    def load_from_disk(cls, load_path: Path) -> 'TunedLLM':
        logging.info(f"Loading TunedLLM from disk: {load_path}")
        if not load_path.exists():
            raise FileNotFoundError(f"Directory not found: {load_path}")

        with open(load_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        instance = cls(
            model_name=metadata['model_name'],
            model_config=metadata.get('model_config', {}),
        )
        
        instance.tokenizer = AutoTokenizer.from_pretrained(load_path)
        instance.model = MultiHeadClassificationModel.from_pretrained(load_path)
        instance.model.to(instance.device)
        
        instance.multilabel_column_names = metadata.get('multilabel_column_names', [])
        
        return instance