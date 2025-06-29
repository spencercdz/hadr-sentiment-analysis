# src/models/llm/tuned.py

import argparse
import csv
import json
import logging
import os
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import nltk
import nlpaug.augmenter.word as naw
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download, login, whoami
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from .base import BaseLLM
from .custom_trainer import CustomTrainer
from .labels import get_all_labels, get_sentiment_labels
from .multi_head_model import MultiHeadClassificationModel
from .utils import save_metadata

logging.basicConfig(level=logging.INFO)


def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    """
    Computes and returns F1 (micro and macro) and subset accuracy for the multi-label head.
    """
    logits = p.predictions[1]
    labels = p.label_ids
    probs = 1 / (1 + np.exp(-logits))
    y_pred = (probs > 0.5).astype(int)
    y_true = labels.astype(int)

    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average="macro", zero_division=0)
    subset_accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "subset_accuracy": subset_accuracy,
    }


class CsvLoggingCallback(TrainerCallback):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.is_initialized = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "eval_loss" not in logs:
            return

        log_data = {
            "step": state.global_step,
            **{k: v for k, v in logs.items() if k.startswith("eval_") or k == "epoch"},
        }

        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(log_data.keys()))
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

        self.training_config = model_config.get("training", {})
        self.augmentation_config = self.training_config.get("augmentation", {})
        self.batch_size = self.training_config.get("batch_size", 16)
        self.max_length = self.training_config.get("max_length", 256)

        self.hub_config = model_config.get("hub", {})
        self.enable_hub_upload = self.hub_config.get("enabled", False)
        self.hub_repo_id = self.hub_config.get("repo_id", None)
        self.hub_private = self.hub_config.get("private", True)
        self.hub_token = self.hub_config.get("token", None)

        logging.info(
            f"Hub configuration: enabled={self.enable_hub_upload}, repo_id={self.hub_repo_id}, private={self.hub_private}"
        )

        all_labels_map = get_all_labels()
        self.sentiment_task_name = "sentiment"
        self.binary_tasks = []
        self.multiclass_tasks = {}
        for task, labels in all_labels_map.items():
            if task == self.sentiment_task_name:
                continue
            if len(labels) > 2:
                self.multiclass_tasks[task] = len(labels)
            else:
                self.binary_tasks.append(task)

        self.multilabel_column_names: List[str] = []
        self.num_multilabels = len(self.binary_tasks) + sum(
            self.multiclass_tasks.values()
        )
        self.num_sentiment_labels = len(get_sentiment_labels())

        logging.info(
            f"Sentiment head will predict {self.num_sentiment_labels} classes."
        )
        logging.info(
            f"Multi-label head will be trained on {len(self.binary_tasks)} binary and {len(self.multiclass_tasks)} multi-class tasks."
        )
        logging.info(
            f"Total multi-label output neurons (after one-hot encoding): {self.num_multilabels}"
        )

        self.model = None
        self.tokenizer = None

        self._prediction_decoders_built = False
        self.binary_task_decoders: Dict[str, Any] = {}
        self.multiclass_task_decoders: Dict[str, Any] = {}
        self.sentiment_decoder: Dict[int, str] = {}

    def setup_huggingface_login(self):
        if not self.enable_hub_upload:
            logging.info("Hub upload disabled, skipping authentication")
            return

        logging.info("Setting up Hugging Face authentication...")

        if not self.hub_repo_id:
            raise ValueError(
                "Hub upload is enabled, but 'hub.repo_id' is not set in the model config."
            )

        try:
            if self.hub_token:
                login(token=self.hub_token)
                logging.info("✅ Logged in to Hugging Face Hub using provided token")
            else:
                user_info = whoami()
                logging.info(
                    f"✅ Already logged in to Hugging Face Hub as: {user_info.get('name', 'unknown')}"
                )
        except Exception as e:
            logging.error(f"❌ Hugging Face authentication failed: {e}")
            logging.error(
                "Please run 'huggingface-cli login' or provide a 'hub.token' in config."
            )
            logging.error("Alternatively, set hub.enabled=false to disable Hub upload.")
            raise RuntimeError("Hugging Face authentication required")

    def load_model(self):
        logging.info(f"Loading tokenizer for '{self.model_name}'.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        logging.info("Initializing custom MultiHeadClassificationModel.")
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_sentiment_labels = self.num_sentiment_labels
        self.model = MultiHeadClassificationModel(
            config=config,
            model_name=self.model_name,
            num_multilabels=self.num_multilabels,
        )
        self.model.to(self.device)

    def _create_dynamic_augment_transform(self):
        logging.info("Creating dynamic augmentation transform...")

        try:
            nltk.data.find("corpora/wordnet.zip")
        except LookupError:
            logging.warning("Downloading 'wordnet' for NLTK augmentation...")
            nltk.download("wordnet", quiet=True)
            logging.info("NLTK data download complete.")

        strength = self.augmentation_config.get("strength", 0.1)
        rate = self.augmentation_config.get("rate", 1.0)
        augmenter = naw.SynonymAug(aug_src="wordnet", aug_p=strength)

        logging.info(
            f"Dynamic augmentation configured with strength={strength} and rate={rate}"
        )

        def transform(examples):
            original_texts = examples["text"]
            augmented_texts = []
            for text in original_texts:
                text = str(text) if text and not pd.isna(text) else ""
                if random.random() < rate and text.strip():
                    try:
                        augmented_text = augmenter.augment(text)
                        if isinstance(augmented_text, list):
                            augmented_text = augmented_text[0] if augmented_text else text
                        augmented_texts.append(str(augmented_text))
                    except Exception as e:
                        logging.warning(
                            f"Augmentation failed for text: {text[:50]}... Error: {e}"
                        )
                        augmented_texts.append(text)
                else:
                    augmented_texts.append(text)

            tokenized = self.tokenizer(
                augmented_texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors=None,
            )
            tokenized["sentiment_labels"] = examples["sentiment_labels"]
            tokenized["multilabel_labels"] = examples["multilabel_labels"]
            return tokenized

        return transform

    def _prepare_data(self, df: pd.DataFrame) -> Dataset:
        all_labels_map = get_all_labels()
        for col_name, labels in all_labels_map.items():
            if col_name in df.columns:
                value_map = {str(v).lower(): k for k, v in labels.items()}
                df[col_name] = df[col_name].apply(lambda x: value_map.get(str(x).lower().strip(), 0) if pd.notna(x) else 0).astype(int)
            else:
                df[col_name] = 0

        final_multilabel_df = pd.DataFrame(index=df.index)
        for task, num_classes in self.multiclass_tasks.items():
            dummies = pd.get_dummies(df[task], prefix=task, dtype=float)
            for i in range(num_classes):
                col = f"{task}_{i}"
                if col not in dummies.columns:
                    dummies[col] = 0.0
            final_multilabel_df = pd.concat([final_multilabel_df, dummies], axis=1)

        for task in self.binary_tasks:
            final_multilabel_df[task] = df[task].astype(float)

        if not self.multilabel_column_names:
            self.multilabel_column_names = sorted(final_multilabel_df.columns.tolist())
            logging.info(
                f"Established multi-label column order: {self.multilabel_column_names}"
            )

        df["multilabel_labels"] = final_multilabel_df[self.multilabel_column_names].values.tolist()
        df["sentiment_labels"] = df[self.sentiment_task_name]
        return Dataset.from_pandas(df)

    def train(self):
        try:
            self.setup_huggingface_login()
            self.load_model()
            logging.info("Loading train, validation, and test datasets...")
            data_dir = self.project_root / "data" / "raw"
            train_df = pd.read_csv(data_dir / "train1.csv", low_memory=False)
            validation_df = pd.read_csv(data_dir / "validation1.csv", low_memory=False)
            test_path = data_dir / "test1.csv"

            def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
                if "message" in df.columns:
                    df.rename(columns={"message": "text"}, inplace=True)
                df["text"] = df["text"].fillna("").astype(str)
                return df

            train_df, validation_df = preprocess_df(train_df), preprocess_df(validation_df)

            train_dataset = self._prepare_data(train_df)
            validation_dataset = self._prepare_data(validation_df)

            def tokenize_func(examples):
                tokenized = self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                )
                tokenized["sentiment_labels"] = examples["sentiment_labels"]
                tokenized["multilabel_labels"] = examples["multilabel_labels"]
                return tokenized

            if self.augmentation_config.get("enabled", False):
                train_dataset.set_transform(self._create_dynamic_augment_transform())
            else:
                train_dataset.set_transform(tokenize_func)
            validation_dataset.set_transform(tokenize_func)

            model_dir = self.project_root / "models" / "tuned" / self.model_name.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)
            log_csv_path = model_dir / "training_log.csv"
            
            callbacks = [
                CsvLoggingCallback(csv_path=log_csv_path),
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.get("early_stopping_patience", 10)
                ),
            ]
            logging.info(f"Detailed training logs will be saved to: {log_csv_path}")
            if self.enable_hub_upload:
                logging.info(f"🚀 Hub auto-upload enabled. Pushing to: {self.hub_repo_id}")

            training_args = TrainingArguments(
                output_dir=str(model_dir),
                num_train_epochs=self.training_config.get("num_epochs", 50),
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                learning_rate=float(self.training_config.get("learning_rate", 2e-5)),
                weight_decay=self.training_config.get("weight_decay", 0.01),
                logging_strategy="epoch",
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1_micro",
                greater_is_better=True,
                save_total_limit=self.training_config.get("save_total_limit", 2),
                fp16=True,
                push_to_hub=self.enable_hub_upload,
                hub_model_id=self.hub_repo_id,
                hub_token=self.hub_token,
                hub_private_repo=self.hub_private,
                report_to="all" if self.enable_hub_upload else "none",
            )

            trainer = CustomTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
                tokenizer=self.tokenizer,
            )

            logging.info("🏃 Starting training...")
            trainer.train()
            logging.info("🎯 Training finished.")

            if test_path.exists():
                logging.info("--- Evaluating on Test Set ---")
                test_df = preprocess_df(pd.read_csv(test_path, low_memory=False))
                test_dataset = self._prepare_data(test_df)
                test_dataset.set_transform(tokenize_func)
                test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
                logging.info(f"--- Test Results: {test_results} ---")
                callbacks[0].on_log(training_args, trainer.state, None, logs={"step": "test", **test_results})
            
            final_model_dir = model_dir / "final_model"
            trainer.save_model(final_model_dir)
            logging.info(f"Final best model saved locally to {final_model_dir}")
            save_metadata(self.get_model_info(), final_model_dir)

        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            traceback.print_exc()
    
    def _prepare_for_prediction(self):
        if self._prediction_decoders_built:
            return
        logging.info("Building prediction decoders...")
        if not self.multilabel_column_names:
            raise RuntimeError("Cannot build decoders: model not trained or loaded.")
        
        all_labels_map = get_all_labels()
        self.sentiment_decoder = all_labels_map[self.sentiment_task_name]
        
        for task_name in self.binary_tasks:
            try:
                self.binary_task_decoders[task_name] = {
                    "index": self.multilabel_column_names.index(task_name),
                    "labels": all_labels_map[task_name],
                }
            except (ValueError, KeyError) as e:
                logging.warning(f"Could not create decoder for binary task '{task_name}': {e}")
        
        for task_name, num_classes in self.multiclass_tasks.items():
            try:
                start_idx = self.multilabel_column_names.index(f"{task_name}_0")
                self.multiclass_task_decoders[task_name] = {
                    "slice": slice(start_idx, start_idx + num_classes),
                    "labels": all_labels_map[task_name],
                }
            except (ValueError, KeyError) as e:
                logging.warning(f"Could not create decoder for multi-class task '{task_name}': {e}")
        
        self._prediction_decoders_built = True
        logging.info("✅ Prediction decoders built successfully.")

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded. Please call train() or load_from_disk() first.")
        
        self._prepare_for_prediction()
        self.model.eval()
        all_predictions = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                sentiment_probs = torch.softmax(outputs["sentiment_logits"], dim=-1).cpu().numpy()
                multilabel_probs = torch.sigmoid(outputs["multilabel_logits"]).cpu().numpy()

            for j in range(len(batch_texts)):
                single_pred = {}
                sentiment_pred_idx = np.argmax(sentiment_probs[j])
                single_pred[self.sentiment_task_name] = {
                    "prediction": self.sentiment_decoder.get(sentiment_pred_idx, "unknown"),
                    "confidence": sentiment_probs[j, sentiment_pred_idx].item(),
                    "scores": sentiment_probs[j].tolist(),
                }

                prob_vector = multilabel_probs[j]
                single_pred["multilabel_scores"] = prob_vector.tolist()
                for task, decoder in self.binary_task_decoders.items():
                    prob = prob_vector[decoder["index"]]
                    pred_idx = 1 if prob > 0.5 else 0
                    single_pred[task] = {
                        "prediction": decoder["labels"][pred_idx],
                        "confidence": (prob if pred_idx == 1 else 1 - prob).item(),
                    }

                for task, decoder in self.multiclass_task_decoders.items():
                    task_probs = prob_vector[decoder["slice"]]
                    pred_idx = np.argmax(task_probs)
                    single_pred[task] = {
                        "prediction": decoder["labels"].get(pred_idx, "unknown"),
                        "confidence": task_probs[pred_idx].item(),
                    }
                all_predictions.append(single_pred)
        
        return all_predictions

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "multilabel_column_names": self.multilabel_column_names,
            "binary_tasks": self.binary_tasks,
            "multiclass_tasks": self.multiclass_tasks,
        })
        return info

    def save(self, save_path: Path) -> None:
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded before saving.")
        
        logging.info(f"Saving tuned model, tokenizer, and metadata to {save_path}")
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        save_metadata(self.get_model_info(), save_path)
        logging.info(f"Save complete: {save_path}")

    @classmethod
    def load_from_disk(cls, load_path: Path) -> "TunedLLM":
        logging.info(f"Loading TunedLLM from disk: {load_path}")
        if not load_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {load_path}")

        metadata_path = load_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {load_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        instance = cls(
            model_name=metadata["model_name"],
            model_config=metadata.get("model_config", {}),
        )

        instance.tokenizer = AutoTokenizer.from_pretrained(load_path)
        config = AutoConfig.from_pretrained(load_path)
        instance.multilabel_column_names = metadata.get("multilabel_column_names", [])
        
        instance.model = MultiHeadClassificationModel(
            config=config,
            model_name=metadata["model_name"],
            num_multilabels=len(instance.multilabel_column_names),
        )
        
        weights_path = load_path / "model.safetensors"
        if not weights_path.exists():
             weights_path = load_path / "pytorch_model.bin"
        state_dict = load_file(weights_path, device="cpu")
        instance.model.load_state_dict(state_dict)
        
        instance.model.to(instance.device)
        
        instance.binary_tasks = metadata.get("binary_tasks", [])
        instance.multiclass_tasks = metadata.get("multiclass_tasks", {})
        
        if not instance.multilabel_column_names:
            logging.warning("Warning: 'multilabel_column_names' not found in metadata.")
        
        instance._prepare_for_prediction()
        logging.info("✅ TunedLLM loaded successfully from disk.")
        return instance

def run_inference_from_hub(
    text: str, hub_repo_id: str = "spencercdz/xlm-roberta-sentiment-requests"
):
    """
    Downloads the model from the Hub and runs inference on a single piece of text.
    """
    subfolder = "final_model"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Running inference on device: {device}")

    metadata_path = hf_hub_download(repo_id=hub_repo_id, filename="metadata.json", subfolder=subfolder)
    with open(metadata_path, "r") as f:
        file_metadata = json.load(f)

    all_labels_map = get_all_labels()
    num_multilabels = len(file_metadata["multilabel_column_names"])
    num_sentiment_labels = len(all_labels_map["sentiment"])
    base_model_name = file_metadata["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(hub_repo_id, subfolder=subfolder)
    config = AutoConfig.from_pretrained(hub_repo_id, subfolder=subfolder)
    config.num_sentiment_labels = num_sentiment_labels

    model_shell = MultiHeadClassificationModel(
        config=config, model_name=base_model_name, num_multilabels=num_multilabels
    )
    weights_path = hf_hub_download(repo_id=hub_repo_id, filename="model.safetensors", subfolder=subfolder)
    state_dict = load_file(weights_path, device="cpu")
    model_shell.load_state_dict(state_dict, strict=False)

    model = model_shell.to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    sentiment_probs = torch.softmax(outputs["sentiment_logits"], dim=-1).cpu().numpy()[0]
    multilabel_probs = torch.sigmoid(outputs["multilabel_logits"]).cpu().numpy()[0]

    results = {}
    sentiment_decoder = all_labels_map["sentiment"]
    sentiment_pred_idx = sentiment_probs.argmax()
    results["sentiment"] = {
        "prediction": sentiment_decoder.get(sentiment_pred_idx, "unknown"),
        "confidence": sentiment_probs[sentiment_pred_idx].item(),
    }

    for task_name in file_metadata["binary_tasks"]:
        idx = file_metadata["multilabel_column_names"].index(task_name)
        prob = multilabel_probs[idx]
        pred = 1 if prob > 0.5 else 0
        results[task_name] = {
            "prediction": all_labels_map[task_name][pred],
            "confidence": (prob if pred == 1 else 1 - prob).item(),
        }

    for task_name, num_classes in file_metadata["multiclass_tasks"].items():
        start_idx = file_metadata["multilabel_column_names"].index(f"{task_name}_0")
        task_probs = multilabel_probs[start_idx : start_idx + num_classes]
        pred_idx = task_probs.argmax()
        results[task_name] = {
            "prediction": all_labels_map[task_name].get(pred_idx, "unknown"),
            "confidence": task_probs[pred_idx].item(),
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a new model or run inference with a pre-trained one from the Hub.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["train", "predict"],
        default="predict",
        help="The action to perform.\n"
        "'train': Starts the full training process (requires local data and config).\n"
        "'predict': Downloads the final model from the Hub and runs inference (default).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="I need food, water, and shelter. Help me! People are dying.",
        help="The text to classify when using --action 'predict'.",
    )
    args = parser.parse_args()

    if args.action == "predict":
        logging.info("--- Running in Inference Mode ---")
        if not args.text:
            raise ValueError("The --text argument cannot be empty for prediction.")
        
        predictions = run_inference_from_hub(text=args.text)
        
        print("\n--- Prediction Results ---")
        print(f'Input Text: "{args.text}"')
        print(json.dumps(predictions, indent=2))

    elif args.action == "train":
        logging.info("--- Running in Training Mode ---")
        logging.warning("Training mode is not fully implemented in this example CLI.")
        logging.warning("To train, you would typically load a full config file, instantiate TunedLLM, and call .train().")
        # Example of how you *would* run training:
        # from .utils import load_config
        # config_path = Path('path/to/your/model_config.yaml')
        # model_config = load_config(config_path)
        # tuned_model = TunedLLM(model_name=model_config['model_name'], model_config=model_config)
        # tuned_model.train()
        pass