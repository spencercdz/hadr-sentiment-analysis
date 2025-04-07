import pandas as pd
import numpy as np
import os

# Create paths
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/data/preprocessing
project_root = os.path.dirname(os.path.dirname(current_dir)) # Goes up to project
raw_path = os.path.join(project_root, 'data', 'raw') # project/data/raw
processed_path = os.path.join(project_root, 'data', 'processed')

# Cleaner function
