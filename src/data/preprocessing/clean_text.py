# clean_text.py: Data Cleaner Utilities 
import pandas as pd
from text_utils import TextCleaner
from pathlib import Path

# Create paths
project_root = Path.cwd() # Project is in hadr/src/data/preprocessing

# Define main paths
raw_path = project_root / 'data' / 'raw'
processed_path = project_root / 'data' / 'processed'

# Create TextCleaner object
cleaner = TextCleaner()

# Keep important columns and information
columns_to_keep = ['clean_text', 'sentiment', 'event_type', 'event_type_detail', 'label'] 

# Open csv files and clean
try:
    for file_name in ["test.csv", "train.csv", "validation.csv"]:
        file_path = raw_path / file_name # Initialize file path
        output_path = processed_path / file_name # Initialize output path

        df = pd.read_csv(file_path) # Read file into a pandas array
        df.dropna(inplace=True) # Remove missing values

        df.rename(columns={'target':'sentiment'}, inplace=True) # Rename columns

        df['clean_text'] = df['text'].apply(cleaner.clean) # Clean Text

        df = df[columns_to_keep] # Only keep columns that we want

        df.to_csv(output_path, index=False) # Export as csv to data/processed/
        
except Exception as e:
    print(f"An exception has occured while trying to clean file")