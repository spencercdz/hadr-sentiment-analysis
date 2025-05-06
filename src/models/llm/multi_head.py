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
        """Save the model to the specified path.
        
        This method saves the complete model state including all training weights and configuration,
        ensuring full compatibility with Hugging Face's transformers library for checkpoint loading.
        The saved model can be loaded with from_pretrained() to resume training or for inference.
        """
        import os
        import json
        
        # Create the directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save the backbone configuration with additional model parameters
        config = self.backbone.config.to_dict()
        
        # Add multi-head specific configuration
        config["model_type"] = "multi_head_xlm_roberta"
        config["backbone_model"] = self.model_name
        config["task_labels"] = self.task_labels
        config["is_multi_head"] = True
        
        # Save the enhanced config
        with open(os.path.join(save_path, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save the entire model state dict (backbone + heads) as pytorch_model.bin
        # This ensures all training weights are preserved
        torch.save(self.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        
        # Save the task heads separately for easier loading in our custom from_pretrained
        torch.save(self.heads.state_dict(), os.path.join(save_path, "task_heads.pt"))
        
        # Save the task labels
        with open(os.path.join(save_path, "task_labels.json"), 'w') as f:
            json.dump(self.task_labels, f, indent=2)
            
        # Save comprehensive model architecture info with training details
        model_info = {
            "model_type": "multi_head_xlm_roberta",
            "backbone_model": self.model_name,
            "task_labels": self.task_labels,
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "intermediate_size": self.config.intermediate_size if hasattr(self.config, "intermediate_size") else None,
            "hidden_dropout_prob": self.config.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.config.attention_probs_dropout_prob if hasattr(self.config, "attention_probs_dropout_prob") else None,
            "max_position_embeddings": self.config.max_position_embeddings if hasattr(self.config, "max_position_embeddings") else None,
            "type_vocab_size": self.config.type_vocab_size if hasattr(self.config, "type_vocab_size") else None,
            "initializer_range": self.config.initializer_range if hasattr(self.config, "initializer_range") else None,
            "layer_norm_eps": self.config.layer_norm_eps if hasattr(self.config, "layer_norm_eps") else None,
            "trainable_parameters": self.get_trainable_parameters()
        }
        with open(os.path.join(save_path, "model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=2)
            
        print(f"Model successfully saved to {save_path} with full weights and configuration")
    
    @classmethod
    def from_pretrained(cls, model_path, task_labels=None, freeze_backbone=True):
        """Load a pretrained model from the specified path.
        
        This method loads a model saved with save_pretrained, handling both the enhanced format
        (with complete model state and configuration) and the legacy format.
        
        Args:
            model_path: Path to the saved model directory
            task_labels: Dictionary mapping task names to number of labels (optional if saved in model)
            freeze_backbone: Whether to freeze the backbone parameters
            
        Returns:
            Loaded MultiHeadXLMRoberta model with complete training weights and configuration
        """
        import os
        import json
        
        # First, try to load the task labels from the saved model
        if task_labels is None:
            try:
                # Try to load from task_labels.json first (preferred)
                task_labels_path = os.path.join(model_path, "task_labels.json")
                if os.path.exists(task_labels_path):
                    with open(task_labels_path, 'r') as f:
                        task_labels = json.load(f)
                        print(f"Loaded task labels from {task_labels_path}")
                # If not found, try to load from config.json
                else:
                    config_path = os.path.join(model_path, "config.json")
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            if "task_labels" in config:
                                task_labels = config["task_labels"]
                                print(f"Loaded task labels from config.json")
            except Exception as e:
                print(f"Error loading task labels: {str(e)}")
                if task_labels is None:
                    raise ValueError("task_labels must be provided if not found in the model path")
        
        # Determine backbone model name
        backbone_model = model_path
        try:
            # Try to get backbone model name from model_info.json
            model_info_path = os.path.join(model_path, "model_info.json")
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                    if "backbone_model" in model_info:
                        backbone_model = model_info["backbone_model"]
                        print(f"Using backbone model: {backbone_model}")
            # If not found, try config.json
            elif os.path.exists(os.path.join(model_path, "config.json")):
                with open(os.path.join(model_path, "config.json"), 'r') as f:
                    config = json.load(f)
                    if "backbone_model" in config:
                        backbone_model = config["backbone_model"]
                        print(f"Using backbone model from config: {backbone_model}")
        except Exception as e:
            print(f"Warning: Could not determine backbone model, using model_path: {str(e)}")
        
        # Create a new instance of the model
        model = cls(backbone_model, task_labels, freeze_backbone)
        
        # Try to load the full model state dict (preferred method)
        pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            try:
                # Load the full state dict
                state_dict = torch.load(pytorch_model_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
                print(f"Successfully loaded complete model state from {pytorch_model_path}")
            except Exception as e:
                print(f"Warning: Could not load full model state: {str(e)}")
                print("Falling back to loading components separately...")
                
                # Try to load task heads separately
                try:
                    task_heads_path = os.path.join(model_path, "task_heads.pt")
                    if os.path.exists(task_heads_path):
                        model.heads.load_state_dict(torch.load(task_heads_path, map_location="cpu"))
                        print(f"Successfully loaded task heads from {task_heads_path}")
                except Exception as e:
                    print(f"Warning: Could not load task heads: {str(e)}")
        else:
            print(f"Warning: pytorch_model.bin not found at {pytorch_model_path}")
            print("Loading components separately...")
            
            # Try to load task heads separately
            try:
                task_heads_path = os.path.join(model_path, "task_heads.pt")
                if os.path.exists(task_heads_path):
                    model.heads.load_state_dict(torch.load(task_heads_path, map_location="cpu"))
                    print(f"Successfully loaded task heads from {task_heads_path}")
            except Exception as e:
                print(f"Warning: Could not load task heads: {str(e)}")
        
        print(f"Model loaded with {len(model.task_labels)} tasks and {model.get_trainable_parameters():,} trainable parameters")
        return model