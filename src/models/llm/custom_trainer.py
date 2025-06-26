# src/models/llm/custom_trainer.py

import torch
import torch.nn as nn
from transformers import Trainer
from typing import Dict, Any, List, Optional

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        This is called during the TRAINING loop.
        It computes loss ONLY for the multi-label head.
        """
        local_inputs = inputs.copy()
        multilabel_labels = local_inputs.pop("multilabel_labels")
        local_inputs.pop("sentiment_labels", None)
        local_inputs.pop("sentiment", None)
        
        outputs = model(**local_inputs)
        
        loss_fct_multilabel = nn.BCEWithLogitsLoss()
        labels = multilabel_labels.to(outputs.multilabel_logits.dtype)
        
        total_loss = loss_fct_multilabel(outputs.multilabel_logits, labels)
        
        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        This is called during the EVALUATION loop.
        We override it to ensure inputs are cleaned before the model call and
        that the returned loss is detached from the computation graph.
        """
        model_inputs = inputs.copy()
        true_labels = model_inputs.pop("multilabel_labels")
        model_inputs.pop("sentiment_labels", None)
        model_inputs.pop("sentiment", None)

        with torch.no_grad():
            outputs = model(**model_inputs)
            
            # Re-use our custom loss calculation
            loss = self.compute_loss(model, inputs, return_outputs=False)
            
            # Package the logits for the `compute_metrics` function
            logits = (outputs.sentiment_logits, outputs.multilabel_logits)

        # === THE FIX: Detach the loss from the computation graph ===
        # The Trainer needs a plain tensor to collate and average the loss,
        # so we detach it from the graph to remove its `requires_grad` property.
        return (loss.detach(), logits, true_labels)