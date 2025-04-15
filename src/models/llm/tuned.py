from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
import evaluate
import numpy as np
import pandas as pd
import json
from pathlib import Path
from transformers import DataCollatorWithPadding
import torch
import gc
from transformers import set_seed, RobertaConfig
from torch import nn
from transformers import RobertaPreTrainedModel, RobertaModel
import matplotlib.pyplot as plt
import os

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# Paths
current_dir = Path(__file__).resolve().parent  # Get the directory containing this script
project_root = current_dir.parent.parent.parent  # Go up to project root (models -> src -> root)
train_path = project_root / 'data' / 'processed' / 'train.csv'
test_path = project_root / 'data' / 'processed' / 'test.csv'
validation_path = project_root / 'data' / 'processed' / 'validation.csv'

# Load data
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
validation_data = pd.read_csv(validation_path)

# Create label mappings for all tasks
sentiment_labels = {0: 'negative', 1: 'positive'}
sentiment_label2id = {'negative': 0, 'positive': 1}

# Create mappings for other categories
event_types = {i: label for i, label in enumerate(train_data['event_type'].unique())}
event_type_details = {i: label for i, label in enumerate(train_data['event_type_detail'].unique())}
labels = {i: label for i, label in enumerate(train_data['label'].unique())}

# Create reverse mappings
event_type2id = {v: k for k, v in event_types.items()}
event_type_detail2id = {v: k for k, v in event_type_details.items()}
label2id = {v: k for k, v in labels.items()}

# Convert all labels to integers
train_data['sentiment'] = train_data['sentiment'].astype(int)
train_data['event_type_id'] = train_data['event_type'].map(event_type2id)
train_data['event_type_detail_id'] = train_data['event_type_detail'].map(event_type_detail2id)
train_data['label_id'] = train_data['label'].map(label2id)

test_data['sentiment'] = test_data['sentiment'].astype(int)
test_data['event_type_id'] = test_data['event_type'].map(event_type2id)
test_data['event_type_detail_id'] = test_data['event_type_detail'].map(event_type_detail2id)
test_data['label_id'] = test_data['label'].map(label2id)

validation_data['sentiment'] = validation_data['sentiment'].astype(int)
validation_data['event_type_id'] = validation_data['event_type'].map(event_type2id)
validation_data['event_type_detail_id'] = validation_data['event_type_detail'].map(event_type_detail2id)
validation_data['label_id'] = validation_data['label'].map(label2id)

# Load tokenizer
model_path = 'aellxx/disaster-tweet-classification'
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create custom multi-task model with regularization
class MultiTaskRoberta(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # Add dropout
        
        # Task weights (can be adjusted based on task importance)
        self.task_weights = {
            'sentiment': 1.0,
            'event_type': 1.0,
            'event_detail': 1.0,
            'label': 1.0
        }
        
        # Classifiers for each task using config attributes
        self.sentiment_classifier = nn.Linear(config.hidden_size, config.sentiment_num_labels)
        self.event_type_classifier = nn.Linear(config.hidden_size, config.event_type_num_labels)
        self.event_detail_classifier = nn.Linear(config.hidden_size, config.event_detail_num_labels)
        self.label_classifier = nn.Linear(config.hidden_size, config.label_num_labels)
        
        # Loss function
        self.loss_fct = nn.CrossEntropyLoss()
        
    def forward(self, 
                input_ids=None, 
                attention_mask=None, 
                token_type_ids=None,
                sentiment_labels=None,
                event_type_labels=None,
                event_detail_labels=None,
                label_labels=None,
                **kwargs):
    
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        
        pooled_output = outputs[1] if outputs[1] is not None else outputs[0][:, 0]
        pooled_output = self.dropout(pooled_output)  # Apply dropout to pooled output
        
        # Label classification
        label_logits = self.label_classifier(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)
        event_type_logits = self.event_type_classifier(pooled_output)
        event_detail_logits = self.event_detail_classifier(pooled_output)
        
        loss = 0
        
        if sentiment_labels is not None:
            if isinstance(sentiment_labels, list):
                sentiment_labels = torch.tensor(sentiment_labels, device=pooled_output.device, dtype=torch.long)
            sentiment_loss = self.loss_fct(sentiment_logits, sentiment_labels.view(-1))
            loss += self.task_weights['sentiment'] * sentiment_loss
        
        if event_type_labels is not None:
            if isinstance(event_type_labels, list):
                event_type_labels = torch.tensor(event_type_labels, device=pooled_output.device, dtype=torch.long)
            event_type_loss = self.loss_fct(event_type_logits, event_type_labels.view(-1))
            loss += self.task_weights['event_type'] * event_type_loss
            
        if event_detail_labels is not None:
            if isinstance(event_detail_labels, list):
                event_detail_labels = torch.tensor(event_detail_labels, device=pooled_output.device, dtype=torch.long)
            event_detail_loss = self.loss_fct(event_detail_logits, event_detail_labels.view(-1))
            loss += self.task_weights['event_detail'] * event_detail_loss
        
        if label_labels is not None:
            if isinstance(label_labels, list):
                label_labels = torch.tensor(label_labels, device=pooled_output.device, dtype=torch.long)
            label_loss = self.loss_fct(label_logits, label_labels.view(-1))
            loss += self.task_weights['label'] * label_loss
        
        return {
            'loss': loss,
            'sentiment_logits': sentiment_logits,
            'event_type_logits': event_type_logits,
            'event_detail_logits': event_detail_logits,
            'label_logits': label_logits
        }

# LR Finder class
class LRFinderTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def find_lr(self, start_lr=1e-7, end_lr=10, num_iter=100):
        # Save original parameters to restore them later
        orig_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Set up optimizer with start_lr
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=start_lr)
        self.optimizer = optimizer
        
        # Log for recording lr and loss
        lrs, losses = [], []
        
        # Start at the beginning of the dataset
        self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'sentiment_labels', 'event_type_labels', 'event_detail_labels', 'label_labels'])
        dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=32)
        
        # Compute multiplicative factor
        mult = (end_lr / start_lr) ** (1 / num_iter)
        
        # Training loop
        for i, batch in enumerate(dataloader):
            if i >= num_iter:
                break
                
            # Move batch to device
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
            
            # Backward pass
            loss.backward()
            
            # Log
            lrs.append(optimizer.param_groups[0]['lr'])
            losses.append(loss.item())
            
            # Update
            optimizer.step()
            optimizer.zero_grad()
            
            # Update lr
            for param_group in optimizer.param_groups:
                param_group['lr'] *= mult
                
        # Restore original parameters
        for name, param in self.model.named_parameters():
            param.data = orig_params[name].data
            
        # Plot
        plt.figure(figsize=(10, 6))
        plt.semilogx(lrs, losses)
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.savefig('lr_finder.png')
        plt.close()
        
        # Find the optimal learning rate (where loss decreases the most)
        min_grad_idx = 0
        max_grad = 0
        for i in range(1, len(lrs)):
            if i > 1:
                grad = (losses[i] - losses[i-1]) / (lrs[i] - lrs[i-1])
                if -grad > max_grad:
                    max_grad = -grad
                    min_grad_idx = i
        
        optimal_lr = lrs[min_grad_idx] / 10
        print(f"Suggested optimal learning rate: {optimal_lr:.8f}")
        
        return lrs, losses, optimal_lr

# Get the base configuration
config = RobertaConfig.from_pretrained(model_path)

# Add custom configuration for multi-task learning
config.sentiment_num_labels = 2
config.event_type_num_labels = len(event_types)
config.event_detail_num_labels = len(event_type_details)
config.label_num_labels = len(labels)
config.hidden_dropout_prob = 0.2  # Increase dropout for more regularization

# Initialize the model with the modified config
model = MultiTaskRoberta.from_pretrained(model_path, config=config)

# Convert pandas DataFrames to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)
validation_dataset = Dataset.from_pandas(validation_data)

# Preprocess function to handle all tasks
def preprocess_function(examples):
    tokenized = tokenizer(
        examples['clean_text'],
        truncation=True,
        padding='max_length',
        max_length=256,
    )
    
    # Add all labels as integers
    tokenized['sentiment_labels'] = [int(label) if label is not None else 0 for label in examples['sentiment']]
    tokenized['event_type_labels'] = [int(label) if label is not None else 0 for label in examples['event_type_id']]
    tokenized['event_detail_labels'] = [int(label) if label is not None else 0 for label in examples['event_type_detail_id']]
    tokenized['label_labels'] = [int(label) if label is not None else 0 for label in examples['label_id']]
    
    return tokenized

# Tokenize the datasets
train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
test_tokenized = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)
validation_tokenized = validation_dataset.map(preprocess_function, batched=True, remove_columns=validation_dataset.column_names)

# Define evaluation metrics
accuracy = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    if isinstance(predictions, dict):
        sentiment_preds = np.argmax(predictions['sentiment_logits'], axis=1)
        event_type_preds = np.argmax(predictions['event_type_logits'], axis=1)
        event_detail_preds = np.argmax(predictions['event_detail_logits'], axis=1)
        label_preds = np.argmax(predictions['label_logits'], axis=1)
    else:
        sentiment_preds = np.argmax(predictions[0], axis=1)
        event_type_preds = np.argmax(predictions[1], axis=1)
        event_detail_preds = np.argmax(predictions[2], axis=1)
        label_preds = np.argmax(predictions[3], axis=1)

    # Check if labels are dict or tuple/list and extract accordingly
    if isinstance(labels, dict):
        sentiment_labels = labels['sentiment_labels']
        event_type_labels = labels['event_type_labels']
        event_detail_labels = labels['event_detail_labels']
        label_labels = labels['label_labels']
    else:
        sentiment_labels, event_type_labels, event_detail_labels, label_labels = labels

    metrics = {}
    metrics['sentiment_accuracy'] = accuracy.compute(predictions=sentiment_preds, references=sentiment_labels)['accuracy']
    metrics['event_type_accuracy'] = accuracy.compute(predictions=event_type_preds, references=event_type_labels)['accuracy']
    metrics['event_detail_accuracy'] = accuracy.compute(predictions=event_detail_preds, references=event_detail_labels)['accuracy']
    metrics['label_accuracy'] = accuracy.compute(predictions=label_preds, references=label_labels)['accuracy']

    metrics['avg_accuracy'] = np.mean([
        metrics['sentiment_accuracy'], 
        metrics['event_type_accuracy'], 
        metrics['event_detail_accuracy'], 
        metrics['label_accuracy']
    ])
    
    return metrics

# Custom callback to save detailed metrics
class DetailedMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            metrics_dir = os.path.join(args.output_dir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            with open(os.path.join(metrics_dir, f'metrics_step_{state.global_step}.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
                
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Completed epoch {state.epoch}")

# Modified data collator to ensure labels are tensors
class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        for key in ['sentiment_labels', 'event_type_labels', 'event_detail_labels', 'label_labels']:
            if key in batch and not isinstance(batch[key], torch.Tensor):
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
        return batch

data_collator = CustomDataCollator(tokenizer=tokenizer, padding=True)

# Custom trainer for multi-task model
class SimpleMultiTaskTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            if prediction_loss_only:
                return (loss, None, None)
                
            sentiment_logits = outputs["sentiment_logits"]
            event_type_logits = outputs["event_type_logits"]
            event_detail_logits = outputs["event_detail_logits"]
            label_logits = outputs["label_logits"]
            
            # Get labels from inputs
            sentiment_labels = inputs["sentiment_labels"]
            event_type_labels = inputs["event_type_labels"]
            event_detail_labels = inputs["event_detail_labels"]
            label_labels = inputs["label_labels"]
            
            preds = (sentiment_logits, event_type_logits, event_detail_logits, label_logits)
            labels = (sentiment_labels, event_type_labels, event_detail_labels, label_labels)
            
            return (loss, preds, labels)

# Check GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")

torch.cuda.empty_cache()
gc.collect()

# Set random seed for reproducibility
set_seed(42)

# Move model to GPU
model = model.to(device)

# First, find the optimal learning rate (optional step)
training_args_lr_finder = TrainingArguments(
    output_dir='lr_finder_output',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_dir='lr_finder_logs',
    report_to="none",
)

lr_finder = LRFinderTrainer(
    model=model,
    args=training_args_lr_finder,
    train_dataset=train_tokenized,
    data_collator=data_collator,
)

lrs, losses, optimal_lr = lr_finder.find_lr(start_lr=1e-7, end_lr=1, num_iter=100)

# Set up training arguments with optimal learning rate (or your chosen learning rate)
training_args = TrainingArguments(
    output_dir='multi-task-disaster-classifier',
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
    logging_strategy='epoch',
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='avg_accuracy',
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    weight_decay=0.01,
    save_total_limit=3,
    lr_scheduler_type='linear',
    report_to="none",
    logging_dir='./logs',
    logging_steps=50,
)

trainer = SimpleMultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=validation_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
        DetailedMetricsCallback(),
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)
    ],
)

# Uncomment the next two lines to start training.
trainer.train()
# 
# After training, save the model:
output_dir = 'multi-task-disaster-classifier-final'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
#
# Apply model to test set
test_results = trainer.predict(test_tokenized)
#
# Calculate and print test metrics
test_metrics = compute_metrics(test_results)
print("\nFinal Test metrics:")
for metric_name, value in test_metrics.items():
    print(f"{metric_name}: {value:.4f}")
#
with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
     json.dump(test_metrics, f, indent=2)
#
# Plot the training history if needed (this section depends on the logs)
plt.figure(figsize=(12, 8))
train_logs = [log for log in trainer.state.log_history if 'loss' in log and 'eval_loss' not in log]
eval_logs = [log for log in trainer.state.log_history if 'eval_loss' in log]
#
if train_logs:
    train_steps = [log.get('step', 0) for log in train_logs]
    train_losses = [log.get('loss', 0) for log in train_logs]
    plt.plot(train_steps, train_losses, label='Training Loss')
#
if eval_logs:
    eval_steps = [log.get('step', 0) for log in eval_logs]
    eval_losses = [log.get('eval_loss', 0) for log in eval_logs]
    plt.plot(eval_steps, eval_losses, label='Validation Loss')
    
    for metric in ['eval_sentiment_accuracy', 'eval_event_type_accuracy', 'eval_event_detail_accuracy', 'eval_label_accuracy', 'eval_avg_accuracy']:
        if all(metric in log for log in eval_logs):
            metric_values = [log.get(metric, 0) for log in eval_logs]
            plt.plot(eval_steps, metric_values, label=metric.replace('eval_', ''))
#
plt.title('Training Progress')
plt.xlabel('Training Steps')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.close()
#
print("\nModel training and evaluation complete!")
print(f"Final model saved to: {output_dir}")
