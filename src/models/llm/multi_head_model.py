# src/models/llm/multi_head_model.py

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoConfig, AutoModelForSequenceClassification
from transformers.modeling_outputs import ModelOutput
import logging

@dataclass
class MultiHeadModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    sentiment_logits: torch.FloatTensor = None
    multilabel_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class MultiHeadClassificationModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config, model_name: str, num_multilabels: int):
        super().__init__(config)
        
        logging.info(f"Loading original pre-trained sentiment model from '{model_name}' to extract its layers.")
        original_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.backbone = original_model.base_model
        original_output_layer = original_model.classifier.out_proj
        
        dropout_prob = config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1
        self.dropout = nn.Dropout(dropout_prob)
        
        self.sentiment_classifier = nn.Linear(config.hidden_size, config.num_sentiment_labels)
        
        logging.info("Copying weights from the original model's final output layer...")
        self.sentiment_classifier.load_state_dict(original_output_layer.state_dict())

        # This line now creates a Linear layer with the correct number of outputs (e.g., 41)
        self.multilabel_classifier = nn.Linear(config.hidden_size, num_multilabels)
        
        logging.info("Freezing backbone parameters...")
        for param in self.backbone.parameters():
            param.requires_grad = False

        logging.info("Freezing original sentiment classifier parameters...")
        for param in self.sentiment_classifier.parameters():
            param.requires_grad = False
            
        logging.info("Multi-label classifier head is initialized and remains trainable.")
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # ... other arguments are the same
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> MultiHeadModelOutput:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)

        sentiment_logits = self.sentiment_classifier(pooled_output)
        multilabel_logits = self.multilabel_classifier(pooled_output)

        loss = None

        if not return_dict:
            return (sentiment_logits, multilabel_logits) + outputs[1:]

        return MultiHeadModelOutput(
            loss=loss,
            sentiment_logits=sentiment_logits,
            multilabel_logits=multilabel_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )