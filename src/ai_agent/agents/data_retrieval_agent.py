"""
DataRetrievalAgent - Responsible for retrieving raw data from files or APIs
"""
import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from .base import ReportWorkflowState, TWITTER_DATA_DIR

class DataRetrievalAgent:
    """Agent responsible for retrieving raw data from files."""
    def __init__(self, llm, workflow_state: ReportWorkflowState, raw_data_path: Optional[Path] = None):
        self.llm = llm
        self.workflow_state = workflow_state
        self.raw_data_path = raw_data_path or TWITTER_DATA_DIR
    
    def retrieve_data(self):
        """Retrieve raw data from files based on the query."""
        try:
            print(f"Step 1: Retrieving data for {self.workflow_state.query}")
            
            # Extract disaster info using natural language understanding
            disaster_data = self._extract_disaster_info(self.workflow_state.query)
            
            # Create a standardized query based on disaster info
            disaster_query = self._create_standardized_query(disaster_data)
            
            # Find matching data files based on the standardized query
            data_file = self._find_matching_data_file(disaster_query)
            
            if not data_file:
                return {"success": False, "message": f"No data found for {disaster_query}"}
            
            # Load and process the data
            data = self._load_data_file(data_file)
            
            # Store in workflow state
            disaster_info = {
                "location": disaster_data.get("location", ""),
                "type": disaster_data.get("type", ""),
                "year": disaster_data.get("year", ""),
                "query": disaster_query
            }
            
            self.workflow_state.raw_data = {
                "disaster_info": disaster_info,
                "tweets": data
            }
            
            self.workflow_state.data_loaded = True
            return {"success": True, "message": f"Successfully loaded data for {disaster_query}"}
        except Exception as e:
            import traceback
            print(f"Error retrieving data: {str(e)}")
            print(traceback.format_exc())
            return {"success": False, "message": f"Error retrieving data: {str(e)}"}
    
    def _extract_disaster_info(self, query):
        """Extract disaster location, type, and year from the query."""
        # Use simple string matching for now
        query_lower = query.lower()
        
        # Default values
        disaster_info = {
            "location": "",
            "type": "",
            "year": ""
        }
        
        # Look for common disaster types
        disaster_types = ["earthquake", "flood", "hurricane", "typhoon", "tsunami", 
                         "wildfire", "tornado", "drought", "cyclone", "landslide"]
        for disaster_type in disaster_types:
            if disaster_type in query_lower:
                disaster_info["type"] = disaster_type
                break
        
        # Look for years (2020-2030)
        import re
        year_match = re.search(r"20[2-3][0-9]", query)
        if year_match:
            disaster_info["year"] = year_match.group(0)
        
        # Extract location (simplistic approach)
        # Try to find the location before the disaster type
        if disaster_info["type"]:
            parts = query_lower.split(disaster_info["type"])
            if len(parts) > 1 and parts[0].strip():
                words = parts[0].strip().split()
                if words:
                    # Use the last word before the disaster type as location
                    disaster_info["location"] = words[-1].capitalize()
        
        # If no location found yet, try to extract any capitalized words
        if not disaster_info["location"]:
            capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
            if capitalized_words:
                disaster_info["location"] = capitalized_words[0]
        
        # Print the extracted information
        print(f"Extracted disaster info: {disaster_info}")
        return disaster_info
    
    def _create_standardized_query(self, disaster_data):
        """Create a standardized query for file matching."""
        parts = []
        if disaster_data.get("location"):
            parts.append(disaster_data["location"].lower())
        if disaster_data.get("type"):
            parts.append(disaster_data["type"].lower())
        if disaster_data.get("year"):
            parts.append(disaster_data["year"])
        
        if not parts:
            # If no specific parts were extracted, use the original query
            return self.workflow_state.query.replace(" ", "_").lower()
        
        # Combine the parts into a standardized query
        return "_".join(parts)
    
    def _find_matching_data_file(self, query):
        """Find a data file that matches the query."""
        # Normalize the query for matching
        query_normalized = query.lower().replace(" ", "_")
        
        # Check if the directory exists
        if not self.raw_data_path.exists():
            print(f"Raw data path does not exist: {self.raw_data_path}")
            return None
        
        # List all files in the directory
        print(f"Looking for files in {self.raw_data_path}")
        for file_path in self.raw_data_path.iterdir():
            # Only consider regular files
            if not file_path.is_file():
                continue
            
            # Check if filename contains the query
            filename_lower = file_path.stem.lower()
            if query_normalized in filename_lower:
                print(f"Found matching file: {file_path}")
                return file_path
        
        # If no exact match found, try more flexible matching
        for file_path in self.raw_data_path.iterdir():
            if not file_path.is_file():
                continue
            
            # Split query into parts and check if any part matches
            query_parts = query_normalized.split("_")
            filename_lower = file_path.stem.lower()
            
            for part in query_parts:
                if len(part) > 3 and part in filename_lower:  # Require at least 4 chars for matching
                    print(f"Found partial match: {file_path}")
                    return file_path
        
        print(f"No matching file found for query: {query}")
        return None
    
    def _load_data_file(self, file_path):
        """Load and process data from a file."""
        suffix = file_path.suffix.lower()
        
        # Handle different file formats
        if suffix == '.json':
            return self._load_json_file(file_path)
        elif suffix == '.csv':
            return self._load_csv_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_json_file(self, file_path):
        """Load data from a JSON file."""
        print(f"Loading JSON file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if the data is already in the expected format
        if isinstance(data, list):
            # Assume it's a list of tweets
            return data
        elif isinstance(data, dict):
            # If it's a dictionary, look for tweet data
            for key in ['tweets', 'twitter_data', 'social_media_data']:
                if key in data and data[key]:
                    return data[key]
            
            # If no specific tweet key found, return the whole dict
            return data
        else:
            raise ValueError(f"Unexpected JSON data format in {file_path}")
    
    def _load_csv_file(self, file_path):
        """Load data from a CSV file."""
        print(f"Loading CSV file: {file_path}")
        try:
            # Try pandas first if available
            if 'pd' in globals():
                df = pd.read_csv(file_path)
                return df.to_dict('records')
        except Exception as e:
            print(f"Error using pandas: {str(e)}, falling back to csv module")
        
        # Fallback to standard csv module
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
        
        # If all else fails
        print("Could not load CSV file with standard methods, using manual parsing")
        tweets = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                return tweets
            
            # Assume first line is header
            header = lines[0].strip().split(',')
            for line in lines[1:]:
                values = line.strip().split(',')
                if len(values) >= len(header):
                    tweet = {}
                    for i, field in enumerate(header):
                        tweet[field.strip()] = values[i].strip()
                    tweets.append(tweet)
        
        return tweets
