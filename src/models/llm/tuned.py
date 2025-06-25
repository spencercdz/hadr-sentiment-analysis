# src/models/llm/tuned.py

import os
import logging
import traceback
from typing import Dict, List, Any
from pathlib import Path
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    # === FIX: Import the specific tokenizer class ===
    XLMRobertaTokenizer, 
    AutoTokenizer 
)
from sklearn.metrics import f1_score, roc_auc_score

from .base import BaseLLM
from .labels import get_all_labels, get_sentiment_labels
from .utils import save_metadata
from .multi_head_model import MultiHeadClassificationModel
from .custom_trainer import CustomTrainer

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.WARNING)

class TunedLLM(BaseLLM):
    """
    Extends a pre-trained sentiment model by adding a new head for other tasks.
    """
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        super().__init__(model_name, model_config)
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent

        self.training_config = model_config.get('training', {})
        self.batch_size = self.training_config.get('batch_size', 8)
        self.max_length = self.training_config.get('max_length', 128)

        all_tasks = sorted(list(get_all_labels().keys()))
        self.sentiment_task_name = 'sentiment'
        self.multilabel_tasks = [task for task in all_tasks if task != self.sentiment_task_name]
        
        self.num_sentiment_labels = len(get_sentiment_labels())
        self.num_multilabels = len(self.multilabel_tasks)
        
        logging.info(f"Sentiment head will predict {self.num_sentiment_labels} classes.")
        logging.info(f"Multi-label head will predict {self.num_multilabels} tasks.")

        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Loads the tokenizer and our custom MultiHeadClassificationModel."""
        logging.info(f"Loading tokenizer for '{self.model_name}'.")

        # === FIX for tricky SentencePiece models ===
        # If the model is an XLM-Roberta model, use its specific tokenizer class
        # directly to avoid auto-detection errors. Otherwise, use AutoTokenizer.
        try:
            if 'xlm-roberta' in self.model_name:
                logging.info("Using explicit XLMRobertaTokenizer for this model.")
                self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_name)
            else:
                logging.info("Using AutoTokenizer.")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            logging.error(f"Failed to load tokenizer for {self.model_name}. Error: {e}")
            logging.info("Attempting to fall back to AutoTokenizer with use_fast=False.")
            # Fallback for other tricky models
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        # ============================================

        logging.info("Initializing custom MultiHeadClassificationModel.")
        config = AutoConfig.from_pretrained(self.model_name)
        
        config.num_sentiment_labels = self.num_sentiment_labels
        
        self.model = MultiHeadClassificationModel(
            config=config,
            model_name=self.model_name,
            num_multilabels=self.num_multilabels
        )
        self.model.to(self.device)

    # The rest of the file remains the same...
    def _prepare_data(self, file_path: Path) -> Dataset:
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found at: {file_path}")
            
        df = pd.read_csv(file_path, low_memory=False)

        if 'message' in df.columns:
            df.rename(columns={'message': 'text'}, inplace=True)
        df['text'] = df['text'].astype(str)

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

        df['sentiment_labels'] = df[self.sentiment_task_name]
        df['multilabel_labels'] = df[self.multilabel_tasks].values.tolist()
        
        dataset = Dataset.from_pandas(df)

        def tokenize_and_format(examples):
            tokenized = self.tokenizer(
                examples['text'], truncation=True, padding='max_length', max_length=self.max_length
            )
            tokenized['sentiment_labels'] = examples['sentiment_labels']
            tokenized['multilabel_labels'] = [list(map(float, ls)) for ls in examples['multilabel_labels']]
            return tokenized

        return dataset.map(tokenize_and_format, batched=True, remove_columns=df.columns.tolist())


    def train(self):
        try:
            train_path = self.project_root / 'data' / 'raw' / 'train1.csv'
            validation_path = self.project_root / 'data' / 'raw' / 'validation1.csv'
            
            train_dataset = self._prepare_data(train_path)
            validation_dataset = self._prepare_data(validation_path)

            model_dir = self.project_root / 'models' / 'tuned' / self.model_name.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(model_dir),
                num_train_epochs=self.training_config.get('num_epochs', 3),
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                learning_rate=self.training_config.get('learning_rate', 2e-5),
                logging_steps=100,
                evaluation_strategy="steps",
                eval_steps=200,
                save_strategy="steps",
                save_steps=200,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                fp16=torch.cuda.is_available(),
                report_to="none"
            )
            
            trainer = CustomTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
            )

            logging.info("Starting training on multi-head model...")
            trainer.train()
            logging.info("Training finished successfully.")

            final_model_dir = model_dir / 'final_model'
            trainer.save_model(final_model_dir)
            logging.info(f"Saved final multi-head model to {final_model_dir}")

        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            traceback.print_exc()

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        self.model.eval()
        all_predictions = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = [str(text) for text in texts[i:i + self.batch_size]]
            inputs = self.tokenizer(
                batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                
                sentiment_probs = torch.softmax(outputs.sentiment_logits, dim=-1).cpu().numpy()
                multilabel_probs = torch.sigmoid(outputs.multilabel_logits).cpu().numpy()

            for j in range(len(batch_texts)):
                preds_for_one_text = {}
                
                sentiment_pred_idx = sentiment_probs[j].argmax()
                sentiment_id2label = get_sentiment_labels()
                preds_for_one_text[self.sentiment_task_name] = {
                    "prediction": sentiment_id2label.get(sentiment_pred_idx, "unknown"),
                    "confidence": float(sentiment_probs[j][sentiment_pred_idx])
                }
                
                for k, task_name in enumerate(self.multilabel_tasks):
                    prob = multilabel_probs[j][k]
                    preds_for_one_text[task_name] = {
                        "prediction": "yes" if prob > 0.5 else "no",
                        "confidence": float(prob)
                    }
                all_predictions.append(preds_for_one_text)
                
        return all_predictions