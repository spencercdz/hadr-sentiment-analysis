from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np
import pandas as pd
import json
from pathlib import Path
from transformers import DataCollatorWithPadding
import torch
import gc
from transformers import set_seed
from torch import nn
from transformers import RobertaPreTrainedModel, RobertaModel

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# Paths
current_dir = Path(__file__).resolve().parent  # Get the directory containing this script
project_root = current_dir.parent.parent  # Go up to project root (3 levels: preprocessing -> data -> src -> root)
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
model_path = 'distilroberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create custom multi-task model
class MultiTaskRoberta(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        
        # Classifiers for each task
        self.sentiment_classifier = nn.Linear(config.hidden_size, 2)
        self.event_type_classifier = nn.Linear(config.hidden_size, len(event_types))
        self.event_detail_classifier = nn.Linear(config.hidden_size, len(event_type_details))
        self.label_classifier = nn.Linear(config.hidden_size, len(labels))
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Use pooled output for classification
        
        # Get logits for each task
        sentiment_logits = self.sentiment_classifier(pooled_output)
        event_type_logits = self.event_type_classifier(pooled_output)
        event_detail_logits = self.event_detail_classifier(pooled_output)
        label_logits = self.label_classifier(pooled_output)
        
        return {
            'sentiment': sentiment_logits,
            'event_type': event_type_logits,
            'event_detail': event_detail_logits,
            'label': label_logits
        }

# Initialize the multi-task model
model = MultiTaskRoberta.from_pretrained(
    model_path,
    num_labels={
        'sentiment': 2,
        'event_type': len(event_types),
        'event_detail': len(event_type_details),
        'label': len(labels)
    }
)

# Convert pandas DataFrames to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)
validation_dataset = Dataset.from_pandas(validation_data)

# Modify the preprocess function to handle all tasks
def preprocess_function(examples):
    tokenized = tokenizer(
        examples['clean_text'],
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors='pt'
    )
    
    # Add all labels
    tokenized['sentiment_labels'] = examples['sentiment']
    tokenized['event_type_labels'] = examples['event_type_id']
    tokenized['event_detail_labels'] = examples['event_type_detail_id']
    tokenized['label_labels'] = examples['label_id']
    
    return tokenized

# Tokenize the datasets
train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
test_tokenized = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)
validation_tokenized = validation_dataset.map(preprocess_function, batched=True, remove_columns=validation_dataset.column_names)

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define evaluation metrics
accuracy = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Split predictions for each task
    sentiment_preds = predictions['sentiment']
    event_type_preds = predictions['event_type']
    event_detail_preds = predictions['event_detail']
    label_preds = predictions['label']
    
    # Calculate metrics for each task
    metrics = {}
    
    # Sentiment metrics
    sentiment_accuracy = accuracy.compute(
        predictions=np.argmax(sentiment_preds, axis=1),
        references=labels['sentiment_labels']
    )['accuracy']
    metrics['sentiment_accuracy'] = sentiment_accuracy
    
    # Event type metrics
    event_type_accuracy = accuracy.compute(
        predictions=np.argmax(event_type_preds, axis=1),
        references=labels['event_type_labels']
    )['accuracy']
    metrics['event_type_accuracy'] = event_type_accuracy
    
    # Event detail metrics
    event_detail_accuracy = accuracy.compute(
        predictions=np.argmax(event_detail_preds, axis=1),
        references=labels['event_detail_labels']
    )['accuracy']
    metrics['event_detail_accuracy'] = event_detail_accuracy
    
    # Label metrics
    label_accuracy = accuracy.compute(
        predictions=np.argmax(label_preds, axis=1),
        references=labels['label_labels']
    )['accuracy']
    metrics['label_accuracy'] = label_accuracy
    
    return metrics

# Set up training arguments
lr = 2e-4
batch_size = 32
epochs = 10

# Check GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")

# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

# Set random seed for reproducibility
set_seed(42)

# Move model to GPU
model = model.to(device)

training_args = TrainingArguments(
    output_dir='multi-task-disaster-classifier',
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    logging_strategy='epoch',
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    fp16=True,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    weight_decay=0.01,
    save_total_limit=2,
    report_to="none",
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=validation_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Save model
model.save_pretrained('multi-task-disaster-classifier')

# Apply model to test set
test_results = trainer.predict(test_tokenized)

# Extract predictions for each task
test_predictions = test_results.predictions
test_labels = test_results.label_ids

# Calculate and print test metrics
test_metrics = compute_metrics((test_predictions, test_labels))
print(test_metrics)

# Save test metrics to file
with open('test_metrics.json', 'w') as f:
    json.dump(test_metrics, f)