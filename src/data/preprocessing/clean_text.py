# clean_text.py: Data Cleaner Utilities 
import pandas as pd
from text_utils import TextCleaner
from datetime import datetime
from pathlib import Path
import os

# Create paths using the current file's location
current_dir = Path(__file__).resolve().parent  # Get the directory containing this script
project_root = current_dir.parent.parent.parent  # Go up to project root (3 levels: preprocessing -> data -> src -> root)

# Define main paths
raw_path = project_root / 'data' / 'raw'
processed_path = project_root / 'data' / 'processed'

print(f"Raw data path: {raw_path}")
print(f"Processed data path: {processed_path}")

# Create directories if they don't exist
raw_path.mkdir(parents=True, exist_ok=True)
processed_path.mkdir(parents=True, exist_ok=True)

# Create TextCleaner object
cleaner = TextCleaner()

# Keep important columns and information
columns_to_keep = ['clean_text', 'sentiment', 'event_type', 'event_type_detail', 'label'] 

# Open csv files and clean
try:
    for file_name in ["test.csv", "train.csv", "validation.csv"]:
        print(f'{datetime.now()} - Cleaning file: {file_name}')
        file_path = raw_path / file_name # Initialize file path
        output_path = processed_path / file_name # Initialize output path

        print(f'{datetime.now()} - Reading file: {file_path}')
        df = pd.read_csv(file_path) # Read file into a pandas array
        df.dropna(inplace=True) # Remove missing values

        print(f'{datetime.now()} - Renaming columns')
        df.rename(columns={'target':'sentiment'}, inplace=True) # Rename columns

        print(f'{datetime.now()} - Cleaning text')
        df['clean_text'] = df['text'].apply(cleaner.clean) # Clean Text
        df = df[columns_to_keep] # Only keep columns that we want

        print(f'{datetime.now()} - Saving cleaned file to {output_path}')
        df.to_csv(output_path, index=False) # Export as csv to data/processed/
        print(f'{datetime.now()} - Cleaned file saved to {output_path}')
        
except Exception as e:
    print(f"Output {file_name} was not saved. An exception has occured while trying to clean file")