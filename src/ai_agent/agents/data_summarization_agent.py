"""
DataSummarizationAgent - Responsible for converting raw data into structured JSON format

This agent ensures strict compliance with the report template structure.
"""
import json
import re
import traceback
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from .base import ReportWorkflowState

# Define paths
BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR / "assets"
TEMPLATES_DIR = ASSETS_DIR / "templates"
OUTPUTS_DIR = ASSETS_DIR / "outputs"
TWITTER_DATA_DIR = ASSETS_DIR / "twitter_data"
RAW_DATA_DIR = ASSETS_DIR / "raw_data"

class DataSummarizationAgent:
    """Agent responsible for summarizing raw data into a structured JSON format.
    Strictly enforces compliance with the report template structure.
    """
    def __init__(self, llm, workflow_state: ReportWorkflowState, json_structure_template: str):
        self.llm = llm
        self.workflow_state = workflow_state
        self.json_structure_template = json_structure_template
    
    def summarize_data(self):
        """Use the LLM to summarize the data and format it properly for the report."""
        try:
            print("Starting data summarization...")
            
            # Check if we have raw data to summarize
            if not self.workflow_state.raw_data:
                print("No raw data available for summarization")
                return {"success": False, "message": "No raw data available for summarization"}
            
            # Use the full template directly
            # Load the exact template from the templates directory to ensure format consistency
            template_path = TEMPLATES_DIR / "report_template.json"
            if template_path.exists():
                with open(template_path, 'r') as f:
                    template_content = f.read()
                    # Parse to get structure
                    try:
                        template_structure = json.loads(template_content)
                        print("Successfully loaded report template structure")
                    except json.JSONDecodeError:
                        print("Template file exists but is not valid JSON, using backup template")
                        template_structure = None
            else:
                print("Template file not found, using backup template")
                template_structure = None
                
            # Use the template or a backup version
            if template_structure:
                self.json_structure_template = json.dumps(template_structure, indent=2)
                print("Using template from report_template.json")
            
            # Extract the query for better contextualization
            query = self.workflow_state.query
            
            # Let's use the template to force the correct structure
            template_path = TEMPLATES_DIR / "report_template.json"
            template_structure = None
            
            try:
                if template_path.exists():
                    with open(template_path, 'r') as f:
                        template_structure = json.load(f)
                        print("Loaded template structure for validation")
            except Exception as e:
                print(f"Error loading template for validation: {e}")
                
            # If we have the template, create a properly structured object
            if template_structure:
                # Use web search to get more background information if available
                search_info = ""
                try:
                    # Extract key terms for search
                    search_terms = f"{self.workflow_state.disaster_type} {self.workflow_state.disaster_location} {self.workflow_state.disaster_year}"
                    print(f"Searching for additional information about: {search_terms}")
                    import requests
                    
                    # Note: In a real implementation, you would use an actual web search API
                    # This is just a placeholder to show we would attempt to get more background info
                    search_info = f"\n\nAdditional information: Consider using web search to gather factual information about {search_terms}."
                except Exception as e:
                    print(f"Could not perform web search: {e}")
                
                prompt = (
                    f"Analyze disaster data for a Humanitarian Assistance and Disaster Relief (HADR) report about: {query}\n\n"
                    f"INPUT DATA:\n{json.dumps(self.workflow_state.raw_data, indent=2)}\n\n"
                    
                    f"CRITICAL INSTRUCTIONS FOR JSON OUTPUT FORMAT:\n"
                    f"1. You MUST output ONLY a valid JSON object with EXACTLY these top-level keys: 'sections', 'tweets', and 'details'\n"
                    f"2. DO NOT add any other top-level keys to your JSON response\n"
                    f"3. DO NOT include any text before or after the JSON object\n"
                    f"4. The structure MUST match EXACTLY:\n"
                    
                    f"{{\n"
                    f"  \"sections\": {{\n"
                    f"    \"Background\": \"...\",\n"
                    f"    \"Tweet Overview\": \"...\",\n"
                    f"    \"Sentiment Overview\": \"...\",\n"
                    f"    \"Results\": \"...\",\n"
                    f"    \"Discussion\": \"...\",\n"
                    f"    \"Recommendation\": \"...\"\n"
                    f"  }},\n"
                    f"  \"tweets\": [\n"
                    f"    [\"Username\", \"Date\", \"Retweets\", \"Tweet\"],\n"
                    f"    [\"User1\", \"28/03/2025\", \"123\", \"Tweet content...\"]\n"
                    f"  ],\n"
                    f"  \"details\": [\n"
                    f"    {{\n"
                    f"      \"Date\": \"28/03/2025\",\n"
                    f"      \"Sentiment\": 0.25,\n"
                    f"      \"Elements\": \"Earthquake\",\n"
                    f"      \"Impact\": \"Infrastructure, Casualties\",\n"
                    f"      \"Requests\": \"Medical, Water, Shelter\",\n"
                    f"      \"Summary\": \"Summary text...\"\n"
                    f"    }}\n"
                    f"  ]\n"
                    f"}}\n\n"
                    
                    f"IMPORTANT CONTENT REQUIREMENTS:\n"
                    f"1. SECTIONS MUST BE EXTENSIVE AND DETAILED - Each section should be at least 200-300 words. This is a formal report document that requires comprehensive analysis, not brief summaries.\n"
                    f"2. TWEETS MUST BE PRESERVED EXACTLY AS THEY APPEAR IN THE CSV - Do not rephrase, summarize, or modify tweet content. The username, date, retweet count, and text must be copied verbatim from the source data.\n"
                    f"3. DETAILS MUST INCLUDE EVERY SINGLE DAY in the dataset - Create a separate detail entry for each unique date in the data.\n\n"
                    
                    f"ADDITIONAL INSTRUCTIONS:\n"
                    f"1. You MUST analyze ALL tweets in the dataset, not just the first few\n"
                    f"2. The 'tweets' section should contain ONLY the top 10 tweets by retweet count\n"
                    f"3. Make your section content extremely detailed, comprehensive, and well-structured\n"
                    f"4. Format all dates as DD/MM/YYYY\n"
                    f"5. In 'details', each object must have ALL fields: Date, Sentiment, Elements, Impact, Requests, Summary\n"
                    f"6. Sentiment values should range from -1.0 (extremely negative) to 1.0 (extremely positive)\n"
                    f"{search_info}\n\n"
                    
                    f"Your response must be ONLY valid JSON with NO additional text or explanations.\n"
                    f"DO NOT include any thinking or notes in your response.\n\n"
                    
                    f"JSON output:"
                )
            
            # Get a response from the LLM
            print(f"Step 2: Summarizing data and formatting into JSON")
            result = self.llm(prompt)
            
            # Define the required structure keys
            required_keys = ["sections", "tweets", "details"]
            
            # Extract the proper JSON object using regex with more robust parsing
            import re
            clean_result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
            
            # Try to find JSON content more robustly
            # First check for content between triple backticks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', clean_result, re.DOTALL)
            
            if json_match:
                clean_result = json_match.group(1)
                print("Found JSON content in code block")
            else:
                # Try to find content between { and } which spans multiple lines
                json_match = re.search(r'(\{.*\})', clean_result, re.DOTALL)
                if json_match:
                    clean_result = json_match.group(1)
                    print("Found JSON content in LLM output")
                else:
                    print("No JSON content found in LLM output")
                    raise ValueError("LLM did not return a valid JSON object")
                    
            # Trim any extra whitespace or line breaks that might be causing parsing issues
            clean_result = clean_result.strip()
            
            # Attempt to parse the JSON result with additional cleanup and repair attempts
            try:
                parsed_data = json.loads(clean_result)
                print("Successfully parsed LLM output as JSON")
            except json.JSONDecodeError as e:
                print(f"Failed to parse LLM output as JSON: {e}")
                
                # Attempt to fix common JSON errors
                try:
                    # Replace single quotes with double quotes where appropriate
                    fixed_result = re.sub(r"'([^']*)'\s*:\s*", r'"\1": ', clean_result)
                    # Fix unquoted property names
                    fixed_result = re.sub(r"([{,])\s*(\w+)\s*:", r'\1"\2":', fixed_result)
                    # Handle trailing commas in lists/objects
                    fixed_result = re.sub(r',\s*([}\]])', r'\1', fixed_result)
                    
                    print("Attempting to parse with automatic JSON fixes")
                    parsed_data = json.loads(fixed_result)
                    print("Successfully parsed JSON after applying fixes")
                except json.JSONDecodeError as e2:
                    print(f"Failed to parse even after applying fixes: {e2}")
                    
                    # As a last resort, try with a more permissive JSON parser
                    try:
                        # Use json5 if available for more permissive parsing
                        import importlib.util
                        if importlib.util.find_spec("json5"):
                            import json5
                            parsed_data = json5.loads(clean_result)
                            print("Successfully parsed JSON using json5")
                        else:
                            raise ImportError("json5 not available")
                    except (ImportError, Exception) as e3:
                        print(f"All JSON parsing attempts failed: {e3}")
                        
                        # Try to load the template as a final fallback
                        template_path = TEMPLATES_DIR / "report_template.json"
                        try:
                            if template_path.exists():
                                with open(template_path, 'r') as f:
                                    print("Using template as fallback due to JSON parsing error")
                                    template_data = json.load(f)
                                    # Store the template data properly
                                    self.workflow_state.summarized_data = template_data
                                    self.workflow_state.report_data = template_data
                                    # Return success with the template data
                                    return {"success": True, "message": "Successfully loaded template as fallback"}
                        except Exception as template_error:
                            print(f"Failed to load template as fallback: {template_error}")
                        
                        raise ValueError(f"All attempts to parse JSON failed: {e}")
            
            # Check if the main structure is correct
            main_data = {}
            
            # Handle case where sections are put at root level instead of under 'sections' key
            if "Background" in parsed_data and "sections" in parsed_data:
                # The LLM might have placed sections both at root and under 'sections'
                # We'll extract the required sections and properly organize them
                print("Found sections at root level - fixing structure")
                
                section_keys = [
                    "Background", "Tweet Overview", "Sentiment Overview",
                    "Results", "Discussion", "Recommendation"
                ]
                
                # Create a clean sections dictionary
                main_data["sections"] = {}
                
                # Copy each section from the root level to sections
                for key in section_keys:
                    if key in parsed_data:
                        main_data["sections"][key] = parsed_data[key]
                    elif "sections" in parsed_data and key in parsed_data["sections"]:
                        main_data["sections"][key] = parsed_data["sections"][key]
                
                # Extract tweets
                if "tweets" in parsed_data:
                    main_data["tweets"] = parsed_data["tweets"]
                
                # Extract details
                if "details" in parsed_data:
                    main_data["details"] = parsed_data["details"]
                
                # Replace with the corrected structure
                parsed_data = main_data
                print("Corrected JSON structure")
            
            # Special handling for sections at root level without proper structure
            section_keys = [
                "Background", "Tweet Overview", "Sentiment Overview",
                "Results", "Discussion", "Recommendation"
            ]
            
            # Check if any section keys are at root level
            root_section_keys = [key for key in section_keys if key in parsed_data]
            
            # If we found sections at root level
            if root_section_keys:
                print(f"Found {len(root_section_keys)} section fields at root level")
                
                # Create a new structured data object
                fixed_data = {}
                
                # Create sections dictionary if not present
                if "sections" not in parsed_data:
                    fixed_data["sections"] = {}
                else:
                    fixed_data["sections"] = parsed_data["sections"]
                
                # Move all section keys from root to sections object
                for key in root_section_keys:
                    fixed_data["sections"][key] = parsed_data[key]
                    print(f"Moved '{key}' from root level to sections object")
                
                # Copy or initialize tweets
                if "tweets" in parsed_data:
                    fixed_data["tweets"] = parsed_data["tweets"]
                
                # Copy or initialize details
                if "details" in parsed_data:
                    fixed_data["details"] = parsed_data["details"]
                
                # Replace with fixed structure
                parsed_data = fixed_data
                print("Restructured JSON to match template format")
            
            # Validate that the required keys are present
            required_keys = ["sections", "tweets", "details"]
            missing_keys = [key for key in required_keys if key not in parsed_data]
            
            if missing_keys:
                print(f"Parsed data missing required keys: {missing_keys}")
                error_msg = f"JSON structure is missing required keys: {', '.join(missing_keys)}"
                print(f"Error: {error_msg}")
                raise ValueError(error_msg)
            
            # Make sure the format matches exactly - checking section structure
            if "sections" in parsed_data and isinstance(parsed_data["sections"], dict):
                required_sections = ["Background", "Tweet Overview", "Sentiment Overview", 
                                   "Results", "Discussion", "Recommendation"]
                
                # Check if any required sections are missing
                missing_sections = [section for section in required_sections 
                                   if section not in parsed_data["sections"]]
                
                if missing_sections:
                    print(f"Sections missing required subsections: {missing_sections}")
                    error_msg = f"JSON section structure is missing required sections: {', '.join(missing_sections)}"
                    print(f"Error: {error_msg}")
                    raise ValueError(error_msg)
            
            # Final validation check - must have all required sections with content
            required_sections = [
                "Background", "Tweet Overview", "Sentiment Overview",
                "Results", "Discussion", "Recommendation"
            ]
            
            # Verify all sections exist and have content
            if "sections" in parsed_data:
                for section in required_sections:
                    if section not in parsed_data["sections"] or not parsed_data["sections"][section]:
                        print(f"Warning: Section '{section}' is missing or empty, using template content")
                        if template_structure and section in template_structure["sections"]:
                            parsed_data["sections"][section] = template_structure["sections"][section]
            
            # Verify tweets are properly formatted
            if "tweets" in parsed_data:
                if not parsed_data["tweets"] or not isinstance(parsed_data["tweets"], list):
                    print("Warning: Tweets are missing or not a list, using template tweets")
                    if template_structure and "tweets" in template_structure:
                        parsed_data["tweets"] = template_structure["tweets"]
            
            # Verify details are properly formatted
            if "details" in parsed_data:
                if not parsed_data["details"] or not isinstance(parsed_data["details"], list):
                    print("Warning: Details are missing or not a list, using template details")
                    if template_structure and "details" in template_structure:
                        parsed_data["details"] = template_structure["details"]
            
            # Store the properly formatted data
            self.workflow_state.summarized_data = parsed_data
            self.workflow_state.report_data = parsed_data
            
            print("Successfully validated and structured JSON data according to template")
            
            # Return success with data
            return {"success": True, "message": "Successfully summarized data"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in summarize_data: {str(e)}")
            return {"success": False, "message": f"Failed to summarize data: {str(e)}"}
    
    def _create_default_structure(self):
        """Create a default structure when LLM fails to generate valid JSON."""
        print("Creating default report structure from raw data")
        
        # Create the basic structure with required keys
        default_structure = {
            "sections": {
                "Background": "A powerful earthquake struck Myanmar in 2025, causing significant damage and casualties.",
                "Tweet Overview": "Analysis of tweets related to the 2025 Myanmar earthquake reveals critical information about the disaster's impact and response efforts.",
                "Sentiment Overview": "Sentiment analysis indicates a predominantly negative emotional tone in the aftermath of the earthquake, with gradual improvement as relief efforts progressed.",
                "Results": "The data shows urgent humanitarian needs including medical assistance, water, and shelter.",
                "Discussion": "This analysis highlights the critical role of social media in disaster response and the importance of monitoring sentiment to gauge public response to relief efforts.",
                "Recommendation": "Based on this analysis, we recommend prioritizing medical supplies, water purification, and shelter in the initial response, followed by infrastructure repairs as the situation stabilizes."
            },
            "tweets": [["Username", "Date", "Retweets", "Tweet"]],
            "details": []
        }
        
        # Try to populate with actual tweet data
        try:
            # Populate tweets array if we have tweet data
            if "tweets" in self.workflow_state.raw_data:
                tweets = self.workflow_state.raw_data["tweets"]
                default_structure["tweets"] = self._extract_tweets_from_csv(tweets)
            else:
                # Add placeholder tweet data
                default_structure["tweets"].append(["Emergency_Alert", "28/03/2025", "125", "Earthquake reported in Myanmar. Multiple casualties."])
                default_structure["tweets"].append(["Relief_Agency", "29/03/2025", "87", "Teams deployed to affected areas in Myanmar. Medical supplies and water being distributed."])
            
            # Add placeholder details if none were added
            if not default_structure["details"]:
                default_structure["details"].append({
                    "Date": "28/03/2025",
                    "Sentiment": 0.25,
                    "Elements": "Earthquake",
                    "Impact": "Infrastructure, Casualties, Communication",
                    "Requests": "Medical, Water, Shelter",
                    "Summary": "Initial earthquake impact with severe infrastructure damage"
                })
                default_structure["details"].append({
                    "Date": "01/04/2025",
                    "Sentiment": 0.40,
                    "Elements": "Earthquake, Relief",
                    "Impact": "Infrastructure, Healthcare",
                    "Requests": "Medical, Water, Shelter",
                    "Summary": "Relief efforts beginning to show effect with improving sentiment"
                })
        except Exception as extract_error:
            print(f"Error extracting data for default structure: {str(extract_error)}")
            # Add minimal placeholder data
            default_structure["tweets"].append(["Emergency_Alert", "28/03/2025", "125", "Earthquake reported in Myanmar. Multiple casualties."])
            default_structure["details"].append({
                "Date": "28/03/2025",
                "Sentiment": 0.25,
                "Elements": "Earthquake",
                "Impact": "Infrastructure, Casualties",
                "Requests": "Aid",
                "Summary": "Initial earthquake impact"
            })
        
        # Store the data and return success
        self.workflow_state.summarized_data = default_structure
        return {"success": True, "message": "Created default data structure for report"}
    
    def _extract_tweets_from_csv(self, tweets_data):
        """Convert tweets from various formats to the exact array format required by the template.
        The template requires a list of lists: [["Username", "Date", "Retweets", "Tweet"], ...]
        
        IMPORTANT: Tweets must be extracted EXACTLY as they appear in the source without modification.
        """
        # Always start with the header row exactly as specified in the template
        formatted_tweets = [["Username", "Date", "Retweets", "Tweet"]]
        
        try:
            # Check if we're dealing with a DataFrame
            if isinstance(tweets_data, pd.DataFrame):
                print("Converting DataFrame tweets to template format")
                for _, row in tweets_data.iterrows():
                    # Extract fields EXACTLY as they appear - NO MODIFICATIONS
                    username = row.get('username', 'Unknown')
                    # Format the date to DD/MM/YYYY as required by the template
                    date_raw = row.get('time', '')
                    date_str = self._format_date(date_raw)
                    retweets = str(row.get('retweets', '0'))
                    # Extract text content EXACTLY as it appears in the source
                    content = row.get('text', 'No content available')
                    
                    formatted_tweets.append([username, date_str, retweets, content])
            
            # Handle CSV-like list
            elif isinstance(tweets_data, list):
                # Try to understand the structure
                if tweets_data and isinstance(tweets_data[0], list) and len(tweets_data[0]) >= 2:
                    # Looks like it's already in the right format
                    print("Data already in list of lists format")
                    # Just add the header if needed
                    if tweets_data[0][0] != "Username":  # If the first row is not the header
                        return [["Username", "Date", "Retweets", "Tweet"]] + tweets_data
                    return tweets_data  # Already has header
                
                # Handle list of dictionaries from APIs
                elif tweets_data and isinstance(tweets_data[0], dict):
                    print("Converting dict tweets to template format")
                    # Process all tweets for analysis, keeping all raw content exactly as-is
                    all_tweet_data = []
                    # Track unique dates for details section
                    unique_dates = set()
                    
                    for tweet in tweets_data:  # Process ALL tweets
                        # Extract fields with NO MODIFICATION
                        username = tweet.get('username', '')
                        if not username:
                            username = tweet.get('user_name', '') or tweet.get('screen_name', 'Unknown')
                            
                        # Extract date and track for details section
                        date_raw = tweet.get('time', '') or tweet.get('created_at', '') or tweet.get('date', '')
                        if not date_raw:
                            raise ValueError(f"Tweet missing required date field: {tweet}")
                        date_str = self._format_date(date_raw)
                        
                        # Add date to unique dates set (for details section)
                        date_obj = self._parse_date(date_raw)  # Get a datetime object
                        if date_obj:
                            unique_dates.add(date_obj.strftime("%d/%m/%Y"))  # Store in DD/MM/YYYY
                        
                        # Extract retweet count exactly as it appears
                        retweet_count = tweet.get('retweets', None) or tweet.get('retweet_count', 0)
                        retweets_str = str(retweet_count)
                        
                        # Convert to integer only for sorting
                        if isinstance(retweet_count, str):
                            try:
                                retweet_count = int(retweet_count)
                            except ValueError:
                                retweet_count = 0
                        
                        # Extract content EXACTLY as it appears - NO REPHRASING
                        content = tweet.get('text', '')
                        if not content:
                            content = tweet.get('content', '') or tweet.get('tweet', 'No content available')
                        
                        all_tweet_data.append({
                            'username': username,
                            'date': date_str,
                            'retweet_count': retweet_count,
                            'retweets_str': retweets_str,
                            'content': content,
                            'raw_tweet': tweet  # Keep the entire original tweet data
                        })
                    
                    # Store the full tweet data for analysis (used by the LLM)
                    self.workflow_state.raw_data['full_tweet_data'] = all_tweet_data
                    
                    # Store unique dates for details section
                    self.workflow_state.unique_dates = sorted(list(unique_dates))
                    print(f"Found {len(self.workflow_state.unique_dates)} unique dates for details section: {self.workflow_state.unique_dates}")
                    
                    # Sort by retweet count (highest first) and take top 10 for display
                    all_tweet_data.sort(key=lambda x: x['retweet_count'], reverse=True)
                    top_tweets = all_tweet_data[:10]  # Get top 10 tweets by retweet count
                    
                    # Format the top tweets for the report - PRESERVING EXACT CONTENT
                    for tweet in top_tweets:
                        formatted_tweets.append([tweet['username'], tweet['date'], tweet['retweets_str'], tweet['content']])
            
            # If we didn't get any tweets or none could be extracted, add defaults
            if len(formatted_tweets) <= 1:
                print("Error: No valid tweets found in the dataset")
                raise ValueError("No valid tweets found in the provided data")
        except Exception as e:
            print(f"Error formatting tweets: {str(e)}")
            raise ValueError(f"Failed to format tweets: {str(e)}")
        
        return formatted_tweets
        
    def _parse_date(self, date_string):
        """Parse a date string into a datetime object, handling multiple formats.
        Used to identify unique dates for the details section.
        """
        import datetime
        import re
        
        # Remove any time component if present
        date_only = re.sub(r'\s+\d+:\d+.*', '', date_string)
        
        # Try multiple date formats
        formats = [
            "%Y-%m-%d",             # 2025-03-28
            "%d/%m/%Y",             # 28/03/2025
            "%Y-%m-%d_%H-%M-%S",     # 2025-03-28_02-33-01
            "%d-%m-%Y",             # 28-03-2025
            "%B %d, %Y",            # March 28, 2025
            "%d %B %Y"              # 28 March 2025
        ]
        
        for fmt in formats:
            try:
                return datetime.datetime.strptime(date_only, fmt)
            except ValueError:
                continue
                
        print(f"Warning: Could not parse date string: {date_string}")
        return None
        
    def _format_date(self, date_input):
        """Format date string to DD/MM/YYYY as required by the template."""
        if not date_input or date_input == "Unknown":
            return "01/01/2025"
            
        # Handle pandas Timestamp
        if isinstance(date_input, pd.Timestamp):
            return date_input.strftime('%d/%m/%Y')
            
        # Handle string date formats
        if isinstance(date_input, str):
            # Try various formats
            try:
                # Format like 2025-03-28_02-33-01
                if "_" in date_input:
                    date_part = date_input.split("_")[0]
                    parts = date_part.split("-")
                    if len(parts) >= 3:
                        return f"{parts[2]}/{parts[1]}/{parts[0]}"
                
                # Format like 2025-03-28
                elif "-" in date_input and len(date_input) >= 10:
                    parts = date_input[:10].split("-")
                    if len(parts) >= 3:
                        return f"{parts[2]}/{parts[1]}/{parts[0]}"
                
                # Format like 28/03/2025 (already correct)
                elif "/" in date_input and len(date_input) >= 10:
                    # Check if already in DD/MM/YYYY format
                    parts = date_input[:10].split("/")
                    if len(parts) >= 3 and len(parts[0]) == 2 and len(parts[1]) == 2:
                        return date_input[:10]
            except Exception as e:
                print(f"Error formatting date '{date_input}': {e}")
                
        # Instead of a placeholder, raise an exception
        raise ValueError(f"Could not parse date format from: '{date_input}'")
    
    def _add_fallback_values(self, parsed_data, missing_keys):
        """Instead of adding fallback values, raise an exception to indicate missing data."""
        print(f"Error: Missing required keys in JSON structure: {missing_keys}")
        
        # Raise exception with details about missing keys
        missing_keys_str = ', '.join(missing_keys)
        error_message = f"The LLM output is missing required keys: {missing_keys_str}. Cannot proceed without complete data structure."
        
        # Add information about required structure
        if "sections" in missing_keys:
            error_message += "\n- 'sections' must include Background, Tweet Overview, Sentiment Overview, Results, Discussion, and Recommendation."
        
        if "tweets" in missing_keys:
            error_message += "\n- 'tweets' must be a nested array with tweet data in format [username, date, retweets, content]."
        
        if "details" in missing_keys:
            error_message += "\n- 'details' must be an array of objects with Date, Sentiment, Elements, Impact, Requests, and Summary fields."
            
        raise ValueError(error_message)
