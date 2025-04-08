# clean_text.py: Data Cleaner Utilities 
from emoji import demojize
import pandas as pd
import numpy as np
import os
import re

# Create paths
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/data/preprocessing
project_root = os.path.dirname(os.path.dirname(current_dir)) # Goes up to project
raw_path = os.path.join(project_root, 'data', 'raw') # project/data/raw
processed_path = os.path.join(project_root, 'data', 'processed') # project/data/processed

# Handling tags and encoding emojis
def clean_keep_crisis_tags(text):
    pass

def encode_emojis(text):
    return demojize(text, delimiters=(" ", " "))

# Final data cleaner function
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) # Remove all URLs

    text = re.sub(r'@\w+', '', text) # Remove all mentions (@username)

    text = re.sub(r'[^\w\s.,!?]', '', text) # Remove special chars and keep basic letters and basic punctuation

    text = encode_emojis(text) # Encode emojis

    text = text.lower() # Convert to lowercase

    text = ' '.join(text.split()) # Remove extra whitespaces

    return text

# Open csv files
try:
    for file_name in ["test.csv", "train.csv", "validation.csv"]:
        file_path = os.path.join(processed_path, 'file_name') # Locate file path
        df = pd.read_csv(file_path, ) # Read path into a pandas array
        
except Exception as e:
    print(f"An exception has occured while trying to clean file")