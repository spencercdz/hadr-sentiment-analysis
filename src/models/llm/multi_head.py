import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Dict, Any

class MultiHeadXLMRoberta(nn.Module):
    """Implementation of a multi-head XLM-RoBERTa model for HADR sentiment analysis.
    Uses a frozen backbone with separate classification heads for each task.
    """
    
    def __init__(self, model_name: str, task_labels: dict, freeze_backbone: bool = True):
        """
        Initialize the multi-head XLM-RoBERTa model.
        
        Args:
            model_name: Name of the pre-trained model to use as backbone
            task_labels: Dictionary mapping task names to number of labels
            freeze_backbone: Whether to freeze the backbone parameters
        """
        super().__init__()
        
        # 1) Load only the backbone (without classification head)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = self.backbone.config
        hidden_size = self.config.hidden_size
        
        # 2) Freeze the backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 3) Add one classification head per task
        self.heads = nn.ModuleDict({
            task: self._create_classification_head(hidden_size, n_labels)
            for task, n_labels in task_labels.items()
        })
        
        # Store task information
        self.task_labels = task_labels
        self.model_name = model_name
        
        # Store the device (will be updated when .to() is called)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _create_classification_head(self, hidden_size: int, num_labels: int) -> nn.Module:
        """Create a classification head with improved architecture.
        
        Args:
            hidden_size: Size of the hidden representation
            num_labels: Number of output labels
            
        Returns:
            Classification head module
        """
        head = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(hidden_size, num_labels)
        )
        
        # Initialize weights
        with torch.no_grad():
            nn.init.xavier_uniform_(head[1].weight)
            nn.init.zeros_(head[1].bias)
            nn.init.xavier_uniform_(head[5].weight)
            nn.init.zeros_(head[5].bias)
            
        return head
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None
    ):
        # 1) Backbone forward - optimize memory usage
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            backbone_outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=False,  # Disable attention outputs to save memory
                output_hidden_states=False,  # Disable hidden states to save memory
                return_dict=return_dict,
            )

        # 2) Get the pooled [CLS] representation
        #    (depending on model, it's either outputs[1] or outputs[0][:,0])
        if hasattr(backbone_outputs, "pooler_output"):
            pooled_output = backbone_outputs.pooler_output
        else:
            sequence_output = backbone_outputs[0]
            pooled_output = sequence_output[:, 0]

        # --------------------------------------------------------------------------------
        # Single‐task mode: if a `task` name is provided, just run that head
        # --------------------------------------------------------------------------------
        if task is not None and task in self.heads:
            logits = self.heads[task](pooled_output)

            loss = None
            if labels is not None:
                if task in ("genre", "related"):
                    # multi‐class
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.task_labels[task]), labels.view(-1))
                else:
                    # binary
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1), labels.float().view(-1))

            if return_dict:
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=backbone_outputs.hidden_states if hasattr(backbone_outputs, "hidden_states") else None,
                    attentions=backbone_outputs.attentions if hasattr(backbone_outputs, "attentions") else None,
                )
            # tuple output for backward compatibility
            output = (logits,) + backbone_outputs[2:]
            return (loss, *output) if loss is not None else output

        # --------------------------------------------------------------------------------
        # Multi‐task mode: no `task` specified, so concatenate every head’s logits
        # --------------------------------------------------------------------------------
        all_logits = []
        for name, head in self.heads.items():
            all_logits.append(head(pooled_output))
        # [batch_size, sum_of_all_num_labels]
        concat_logits = torch.cat(all_logits, dim=1)

        if return_dict:
            return SequenceClassifierOutput(logits=concat_logits)
        return (concat_logits,) + backbone_outputs[2:]
    
    def unfreeze_backbone(self, num_layers=None):
        """Unfreeze the backbone or specific layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the top. If None, unfreeze all.
        """
        if num_layers is None:
            # Unfreeze all backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = True
            print(f"Unfrozen all backbone layers")
        else:
            # Keep most of the backbone frozen, but unfreeze the top N layers
            # This is useful for fine-tuning with limited data
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
                
            # Unfreeze the embedding layer if requested
            if num_layers >= self.config.num_hidden_layers + 1:
                for name, param in self.backbone.embeddings.named_parameters():
                    param.requires_grad = True
                print(f"Unfrozen embedding layer")
                num_layers -= 1
                
            # Unfreeze the top N encoder layers
            if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
                layers_to_unfreeze = min(num_layers, len(self.backbone.encoder.layer))
                for i in range(len(self.backbone.encoder.layer) - layers_to_unfreeze, len(self.backbone.encoder.layer)):
                    for param in self.backbone.encoder.layer[i].parameters():
                        param.requires_grad = True
                print(f"Unfrozen top {layers_to_unfreeze} encoder layers")
            
            # Unfreeze the pooler
            if hasattr(self.backbone, 'pooler'):
                for param in self.backbone.pooler.parameters():
                    param.requires_grad = True
                print(f"Unfrozen pooler layer")
    
    def get_trainable_parameters(self):
        """Get the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    @property
    def device(self):
        """Get the device where the model is located."""
        return self._device
        
    def to(self, device):
        """Move the model to the specified device and update the internal device attribute."""
        self._device = device
        return super().to(device)
    
    def save_pretrained(self, save_path):
        """Save the model to the specified path."""
        # Save the backbone
        self.backbone.save_pretrained(save_path)
        
        # Save the heads
        torch.save(self.heads.state_dict(), f"{save_path}/task_heads.pt")
        
        # Save the task labels
        import json
        with open(f"{save_path}/task_labels.json", 'w') as f:
            json.dump(self.task_labels, f)
    
    @classmethod
    def from_pretrained(cls, model_path, task_labels=None, freeze_backbone=True):
        """Load a pretrained model from the specified path."""
        # Load the backbone
        backbone = AutoModel.from_pretrained(model_path)
        config = backbone.config
        
        # Load the task labels if not provided
        if task_labels is None:
            import json
            try:
                with open(f"{model_path}/task_labels.json", 'r') as f:
                    task_labels = json.load(f)
            except FileNotFoundError:
                raise ValueError("task_labels must be provided if not found in the model path")
        
        # Create the model
        model = cls(model_path, task_labels, freeze_backbone)
        
        # Load the heads
        try:
            model.heads.load_state_dict(torch.load(f"{model_path}/task_heads.pt"))
        except FileNotFoundError:
            print("No task heads found in the model path. Using newly initialized heads.")
        
        return model