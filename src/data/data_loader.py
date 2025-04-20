# data_loader.py: Data Loader Utilities
import os
import json
from datasets import load_dataset
from huggingface_hub import login

# Authenticate to HuggingFace if you are not using the CLI to login
# login(token="hf_REPLACEWITHYOURTOKEN")

# Create target path
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/data/
project_root = os.path.dirname(os.path.dirname(current_dir))  # Goes up to project
raw_path = os.path.join(project_root, 'data', 'raw')  # project/data/raw/
metadata_path = os.path.join(project_root, 'data', 'metadata') # project/data/metadata/

# Create directories if they don't exist
raw_path.mkdir(parents=True, exist_ok=True)
metadata_path.mkdir(parents=True, exist_ok=True)

# Load Hugging Face dataset (Natural Disasters) in data/raw
ds = load_dataset(
    "melisekm/natural-disasters-from-social-media",
    )

# Save dataset to my folder
for split in ["train", "test", "validation"]:
    file_path = os.path.join(raw_path, f"{split}.csv")
    ds[split].to_csv(file_path, index=False)
    print(f"Saved {split} to: {file_path}")  # Verification

# Save metadata
metadata = {
    "dataset_name": "natural-disasters-from-social-media",
    "description": ds['train'].info.description,
    "features": str(ds["train"].features),
    "num_samples": {split: len(ds[split]) for split in ds.keys()}
}

with open(os.path.join(metadata_path, "dataset_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)