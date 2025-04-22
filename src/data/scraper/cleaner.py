#!/usr/bin/env python
"""
Data cleaner script for Myanmar Earthquake 2025 dataset.
This script:
1. Removes duplicate tweets based on tweet_id
2. Standardizes the query column to lowercase "myanmar earthquake 2025"
3. Sorts tweets chronologically by tweet_id
4. Fixes whitespace issues in tweet text
"""

import csv
import os
import pandas as pd
from pathlib import Path

def clean_myanmar_data():
    """Clean the Myanmar earthquake dataset"""
    # File paths
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = project_root / 'data' / 'raw' / 'myanmar_earthquake_2025'
    input_file = data_dir / 'myanmar_earthquake_2025.csv'
    
    # Create a backup of the original file
    backup_file = data_dir / 'myanmar_earthquake_2025_backup.csv'
    
    print(f"Loading data from {input_file}")
    
    # Read the CSV file with pandas
    df = pd.read_csv(input_file)
    original_count = len(df)
    print(f"Original dataset contains {original_count} rows")
    
    # Create backup
    df.to_csv(backup_file, index=False)
    print(f"Created backup at {backup_file}")
    
    # Step 1: Make query column lowercase but preserve the original search terms
    # Convert all query values to lowercase without standardizing them
    df['query'] = df['query'].str.lower()
    # Verify the unique query values after update
    unique_queries = df['query'].unique()
    print(f"Converted all queries to lowercase. Unique values after update: {unique_queries}")
    
    # Step 2: Remove duplicates based on tweet_id
    duplicates = df.duplicated(subset=['tweet_id'], keep='first')
    duplicate_count = duplicates.sum()
    print(f"Found {duplicate_count} duplicate tweets")
    
    # Keep only the first occurrence of each tweet_id
    df = df.drop_duplicates(subset=['tweet_id'], keep='first')
    
    # Step 3: Fix whitespace issues in tweet text
    if 'text' in df.columns:
        # Replace newlines with spaces and normalize multiple spaces
        df['text'] = df['text'].apply(lambda text: ' '.join(str(text).replace('\n', ' ').replace('\r', ' ').split()))
        print("Fixed whitespace issues in tweet text")
    
    # Ensure 'verified' column format is consistent (True/False, not true/false)
    if 'verified' in df.columns:
        df['verified'] = df['verified'].apply(lambda v: str(v).capitalize())
        print("Standardized 'verified' column to use True/False capitalization")
        
    # Ensure location is never None
    if 'location' in df.columns:
        df['location'] = df['location'].fillna("")
        print("Ensured location is never None")
    
    # Step 4: Sort by tweet_id for chronological order (smallest/earliest first)
    df['tweet_id'] = df['tweet_id'].astype(str)  # Ensure string format
    df = df.sort_values(by='tweet_id')
    
    # Save the cleaned data back to the original file
    df.to_csv(input_file, index=False)
    
    # Print summary
    print(f"Cleaned dataset contains {len(df)} rows")
    print(f"Removed {original_count - len(df)} duplicate entries")
    print(f"Sorted tweets chronologically by tweet_id")
    print(f"Fixed whitespace formatting in tweet text")
    print(f"Saved cleaned data to {input_file}")

if __name__ == "__main__":
    clean_myanmar_data()
    print("Data cleaning complete!")
