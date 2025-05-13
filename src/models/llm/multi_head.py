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
        """Save the model to the specified path as a single consolidated checkpoint file.

        This method saves the model's configuration, state dictionary, and optionally
        optimizer and scheduler states into a single 'model_checkpoint.pt' file.
        This facilitates easier model sharing and sequential training by ensuring all
        necessary components are bundled together.

        Args:
            save_path (str): Directory where the 'model_checkpoint.pt' file will be saved.

        Returns:
            str: The path to the saved checkpoint file.
        """
        import os
        import torch
        import json # Added import for json

        os.makedirs(save_path, exist_ok=True)

        # Prepare config dictionary
        config = self.backbone.config.to_dict()
        config["model_type"] = "multi_head_xlm_roberta"
        config["architectures"] = ["MultiHeadXLMRoberta"]
        config["backbone_model"] = self.model_name
        config["task_labels"] = self.task_labels
        config["is_multi_head"] = True
        config["freeze_backbone_at_save_time"] = not any(p.requires_grad for p in self.backbone.parameters())

        # Data to save in the single checkpoint file
        checkpoint_data = {
            'config': config,
            'model_state_dict': self.state_dict()
        }

        if hasattr(self, 'optimizer_state') and self.optimizer_state is not None:
            checkpoint_data['optimizer_state_dict'] = self.optimizer_state
        if hasattr(self, 'scheduler_state') and self.scheduler_state is not None:
            checkpoint_data['scheduler_state_dict'] = self.scheduler_state

        # Save the consolidated checkpoint
        checkpoint_file_path = os.path.join(save_path, "model_checkpoint.pt")
        torch.save(checkpoint_data, checkpoint_file_path)

        # Save the config.json separately as well for Hugging Face compatibility / inspection
        with open(os.path.join(save_path, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Model successfully saved to {checkpoint_file_path} (consolidated) and config.json.")
        return checkpoint_file_path
    
    @classmethod
    def from_pretrained(cls, model_path, task_labels=None, freeze_backbone_override=None, **kwargs):
        """Load a pretrained model from a single consolidated checkpoint file or a standard directory.

        This method first attempts to load a model from a 'model_checkpoint.pt' file if it exists.
        If not found, it falls back to loading from a standard Hugging Face model directory structure
        (expecting 'config.json' and 'pytorch_model.bin').

        Args:
            model_path (str): Path to the directory containing 'model_checkpoint.pt' or 'config.json'/'pytorch_model.bin',
                              OR path to the 'model_checkpoint.pt' file itself.
            task_labels (dict, optional): Dictionary mapping task names to number of labels.
                                         If provided, overrides task_labels from the saved config.
            freeze_backbone_override (bool, optional): Whether to freeze the backbone parameters upon loading.
                                                     If None, uses the 'freeze_backbone_at_save_time' from config (if loading checkpoint)
                                                     or 'freeze_backbone' from config (if loading standard dir), or defaults to True.
            **kwargs: Additional arguments (e.g., map_location for torch.load).

        Returns:
            Loaded MultiHeadXLMRoberta model.
        """
        import os
        import torch
        import json # Added import for json
        import traceback # Added import for traceback

        # Determine if model_path is a directory or a direct file path to the checkpoint
        if os.path.isdir(model_path):
            checkpoint_file_path = os.path.join(model_path, "model_checkpoint.pt")
            config_file_path = os.path.join(model_path, "config.json")
            pytorch_model_bin_path = os.path.join(model_path, "pytorch_model.bin")
        else: # Assumed to be a direct path to model_checkpoint.pt
            checkpoint_file_path = model_path
            # Derive potential directory for config.json if needed for fallback
            model_dir_for_fallback = os.path.dirname(model_path)
            config_file_path = os.path.join(model_dir_for_fallback, "config.json")
            pytorch_model_bin_path = os.path.join(model_dir_for_fallback, "pytorch_model.bin")

        map_location = kwargs.pop('map_location', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Attempt to load from consolidated checkpoint first
        if os.path.exists(checkpoint_file_path):
            print(f"Attempting to load from consolidated checkpoint: {checkpoint_file_path}")
            try:
                checkpoint_data = torch.load(checkpoint_file_path, map_location=map_location)
                config_data = checkpoint_data['config']
                model_state_dict = checkpoint_data['model_state_dict']

                backbone_model_name = config_data['backbone_model']
                
                task_labels_from_config = config_data.get('task_labels')
                task_labels_to_use = task_labels if task_labels is not None else task_labels_from_config
                if task_labels_to_use is None:
                    raise ValueError("task_labels must be provided or be in the saved config.")

                if freeze_backbone_override is not None:
                    freeze_backbone_on_load = freeze_backbone_override
                else:
                    freeze_backbone_on_load = config_data.get('freeze_backbone_at_save_time', True)

                model = cls(
                    model_name=backbone_model_name,
                    task_labels=task_labels_to_use,
                    freeze_backbone=freeze_backbone_on_load,
                    **kwargs
                )
                model.load_state_dict(model_state_dict)
                model.to(map_location)

                if 'optimizer_state_dict' in checkpoint_data:
                    model.optimizer_state = checkpoint_data['optimizer_state_dict']
                if 'scheduler_state_dict' in checkpoint_data:
                    model.scheduler_state = checkpoint_data['scheduler_state_dict']

                print(f"Model successfully loaded from consolidated checkpoint: {checkpoint_file_path}.")
                return model
            except Exception as e:
                print(f"Failed to load from consolidated checkpoint: {str(e)}. Traceback: {traceback.format_exc()}")
                print("Falling back to standard directory loading if possible.")
        
        # Fallback to standard Hugging Face directory loading (config.json + pytorch_model.bin)
        print(f"Attempting to load from standard directory: {os.path.dirname(config_file_path) if os.path.isdir(model_path) else model_path}")
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"'config.json' not found at {config_file_path}. Cannot load model.")
        if not os.path.exists(pytorch_model_bin_path):
            raise FileNotFoundError(f"'pytorch_model.bin' not found at {pytorch_model_bin_path}. Cannot load model.")

        with open(config_file_path, 'r') as f:
            config_data = json.load(f)

        backbone_model_name = config_data.get('backbone_model', config_data.get('_name_or_path'))
        
        task_labels_from_config = config_data.get('task_labels')
        task_labels_to_use = task_labels if task_labels is not None else task_labels_from_config
        if task_labels_to_use is None:
            raise ValueError("task_labels must be provided or be in the saved config.json.")

        if freeze_backbone_override is not None:
            freeze_backbone_on_load = freeze_backbone_override
        else:
            # Use 'freeze_backbone' from config if present, else default to True
            freeze_backbone_on_load = config_data.get('freeze_backbone', True) 

        model = cls(
            model_name=backbone_model_name,
            task_labels=task_labels_to_use,
            freeze_backbone=freeze_backbone_on_load,
            **kwargs
        )
        
        state_dict = torch.load(pytorch_model_bin_path, map_location=map_location)
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"Successfully loaded model state from {pytorch_model_bin_path} (strict mode).")
        except RuntimeError as e:
            print(f"Strict loading failed: {e}. Attempting non-strict loading from {pytorch_model_bin_path}.")
            model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded model state from {pytorch_model_bin_path} (non-strict mode). Review warnings.")
        
        model.to(map_location)

        # Load optimizer/scheduler if they exist as separate .pt files (legacy or specific save pattern)
        optimizer_path = os.path.join(os.path.dirname(pytorch_model_bin_path), "optimizer.pt")
        if os.path.exists(optimizer_path):
            try:
                model.optimizer_state = torch.load(optimizer_path, map_location=map_location)
                print(f"Loaded optimizer state from {optimizer_path}")
            except Exception as e_opt:
                print(f"Warning: Could not load optimizer state from {optimizer_path}: {e_opt}")
        
        scheduler_path = os.path.join(os.path.dirname(pytorch_model_bin_path), "scheduler.pt")
        if os.path.exists(scheduler_path):
            try:
                model.scheduler_state = torch.load(scheduler_path, map_location=map_location)
                print(f"Loaded scheduler state from {scheduler_path}")
            except Exception as e_sch:
                print(f"Warning: Could not load scheduler state from {scheduler_path}: {e_sch}")

        print(f"Model successfully loaded from standard directory: {os.path.dirname(config_file_path)}.")
        return model