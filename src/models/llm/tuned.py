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
from datasets import Dataset
from transformers import (
    AutoConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    XLMRobertaTokenizer, 
    AutoTokenizer,
    TrainerCallback,
    EvalPrediction
)
from sklearn.metrics import f1_score, accuracy_score

### NEW: Imports for data augmentation
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
    logits = p.predictions[1] 
    labels = p.label_ids
    probs = 1 / (1 + np.exp(-logits))
    y_pred = (probs > 0.5).astype(int)
    y_true = labels.astype(int)
    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    subset_accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    return {'f1_micro': f1_micro, 'f1_macro': f1_macro, 'subset_accuracy': subset_accuracy}

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
        ### NEW: Read augmentation config from the model config
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
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            logging.error(f"Failed to load tokenizer for {self.model_name}. Error: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        logging.info("Initializing custom MultiHeadClassificationModel for feature extraction.")
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_sentiment_labels = self.num_sentiment_labels
        self.model = MultiHeadClassificationModel(config=config, model_name=self.model_name, num_multilabels=self.num_multilabels)
        self.model.to(self.device)

    def _apply_augmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies synonym replacement augmentation to a portion of the dataframe."""
        
        logging.info(f"Checking augmentation config: {self.augmentation_config}")

        if not self.augmentation_config.get('enabled', False):
            logging.warning("Augmentation is disabled in the configuration. Skipping.")
            return df

        logging.info("Applying data augmentation to the training set...")
        
        # --- FIX: Added the new '_eng' package to the check and download list ---
        try:
            # Check for all required packages. If one is missing, it will raise the error.
            nltk.data.find('corpora/wordnet.zip')
            nltk.data.find('taggers/averaged_perceptron_tagger.zip')
            nltk.data.find('taggers/averaged_perceptron_tagger_eng') # <-- NEW CHECK

        except LookupError:
            logging.warning("Downloading necessary NLTK data for augmentation...")
            # Download all required packages if any are missing.
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            logging.info("NLTK data download complete.")

        rate = self.augmentation_config.get('rate', 0.5)
        strength = self.augmentation_config.get('strength', 0.1)

        df_to_augment = df.sample(frac=rate)
        if df_to_augment.empty:
            logging.warning("No data selected for augmentation. The 'rate' might be too low or the dataset too small.")
            return df
        
        texts_to_augment = df_to_augment['text'].tolist()
        
        logging.info(f"Augmenting {len(texts_to_augment)} samples with synonym replacement (strength={strength})...")
        # This is the line that requires the POS tagger
        augmenter = naw.SynonymAug(aug_src='wordnet', aug_p=strength)
        augmented_texts = augmenter.augment(texts_to_augment)

        logging.info("--- AUGMENTATION SANITY CHECK ---")
        for i in range(min(3, len(texts_to_augment))):
            logging.info(f"Original  : {texts_to_augment[i]}")
            logging.info(f"Augmented : {augmented_texts[i]}")
            logging.info("-" * 10)
        logging.info("--- END SANITY CHECK ---")
        
        augmented_df = df_to_augment.copy()
        augmented_df['text'] = augmented_texts

        final_df = pd.concat([df, augmented_df], ignore_index=True)
        logging.info(f"Augmentation complete. Training set size increased from {len(df)} to {len(final_df)}.")
        
        return final_df

    ### MODIFIED: This method now takes a DataFrame instead of a file path.
    def _prepare_data(self, df: pd.DataFrame) -> Dataset:
        """Processes a DataFrame to create labels and tokenize for the model."""
        original_columns = df.columns.tolist()

        # The 'text' column is assumed to exist and be ready for processing.
        # Renaming from 'message' now happens in the `train` method.
        
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
        
        dataset = Dataset.from_pandas(df)
        def tokenize_and_format(examples):
            tokenized = self.tokenizer(examples['text'], truncation=True, padding='max_length', max_length=self.max_length)
            tokenized['sentiment_labels'] = examples['sentiment_labels']
            tokenized['multilabel_labels'] = examples['multilabel_labels']
            return tokenized
        return dataset.map(tokenize_and_format, batched=True, remove_columns=original_columns)

    ### MODIFIED: The train method now orchestrates loading, augmenting, and preparing data.
    def train(self):
        try:
            self.load_model()
            logging.info("Loading train, validation, and test datasets...")
            train_path = self.project_root / 'data' / 'raw' / 'train1.csv'
            validation_path = self.project_root / 'data' / 'raw' / 'validation1.csv'
            test_path = self.project_root / 'data' / 'raw' / 'test1.csv'
            
            # --- 1. Load data into DataFrames ---
            if not train_path.exists(): raise FileNotFoundError(f"Train data not found: {train_path}")
            train_df = pd.read_csv(train_path, low_memory=False)
            
            if not validation_path.exists(): raise FileNotFoundError(f"Validation data not found: {validation_path}")
            validation_df = pd.read_csv(validation_path, low_memory=False)

            # --- 2. Pre-process DataFrames (e.g., rename columns) ---
            def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
                if 'message' in df.columns:
                    df.rename(columns={'message': 'text'}, inplace=True)
                df['text'] = df['text'].astype(str)
                return df

            train_df = preprocess_df(train_df)
            validation_df = preprocess_df(validation_df)

            # --- 3. Apply augmentation ONLY to the training DataFrame ---
            train_df = self._apply_augmentation(train_df)

            # --- 4. Prepare datasets for the Trainer ---
            train_dataset = self._prepare_data(train_df)
            validation_dataset = self._prepare_data(validation_df)

            model_dir = self.project_root / 'models' / 'tuned' / self.model_name.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            log_csv_path = model_dir / "training_log.csv"
            csv_logger = CsvLoggingCallback(csv_path=log_csv_path)
            logging.info(f"Detailed training logs will be saved to: {log_csv_path}")

            training_args = TrainingArguments(
                output_dir=str(model_dir),
                remove_unused_columns=False,
                num_train_epochs=self.training_config.get('num_epochs', 3),
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                learning_rate=float(self.training_config.get('learning_rate', 2e-5)),
                logging_steps=200,
                eval_strategy="steps",
                eval_steps=200,
                save_strategy="steps",
                save_steps=200,
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1_micro",
                greater_is_better=True,
                fp16=True,
                report_to="none"
            )
            
            trainer = CustomTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[csv_logger]
            )

            logging.info("Starting training...")
            trainer.train()
            logging.info("Training finished.")

            if test_path.exists():
                logging.info("--- Evaluating on Test Set ---")
                test_df = pd.read_csv(test_path, low_memory=False)
                test_df = preprocess_df(test_df)
                test_dataset = self._prepare_data(test_df)
                test_results = trainer.evaluate(eval_dataset=test_dataset)
                logging.info(f"--- Test Results: {test_results} ---")
                csv_logger.on_log(training_args, trainer.state, None, logs={'step': 'test', **test_results})
            
            final_model_dir = model_dir / 'final_model'
            self.save(final_model_dir)

        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            traceback.print_exc()

    def predict(self, texts: List[str]):
        # TODO: Implement prediction logic that decodes the one-hot encoded output
        pass

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
        logging.info("Save complete.")

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