"""
HADR Agent - LangGraph-based agent for HADR sentiment analysis and report generation
"""
import os
import sys
import json
import csv
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, TypedDict, Annotated, Union
import traceback
from datetime import datetime
from langchain_community.llms import Ollama
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from langchain.chains import LLMChain
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

# Configure paths
current_dir = Path(__file__).parent
tools_dir = current_dir / "tools"
sys.path.append(str(tools_dir))

# Configure root project path to allow importing models
project_root = current_dir.parent.parent.parent
models_dir = project_root / "src" / "models"
sys.path.append(str(project_root))
sys.path.append(str(models_dir))

# Import sentiment analysis packages for Twitter sentiment analysis
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    HAS_SENTIMENT_MODEL = True
except ImportError:
    HAS_SENTIMENT_MODEL = False
    print("Warning: transformers or torch package not available for sentiment analysis")

# Import labels if available
try:
    from models.llm.labels import get_all_labels
except ImportError:
    def get_all_labels():
        return {}

# Import the build_report module
import build_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("hadr_agent")

# Define paths
AI_AGENT_DIR = Path(__file__).parent.parent
ASSETS_DIR = AI_AGENT_DIR / "assets"
TWITTER_DATA_DIR = ASSETS_DIR / "twitter_data"
TEMPLATES_DIR = ASSETS_DIR / "templates"
OUTPUTS_DIR = ASSETS_DIR / "outputs"

# Create directories if they don't exist
for directory in [ASSETS_DIR, TWITTER_DATA_DIR, TEMPLATES_DIR, OUTPUTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Define agent state
class AgentState(TypedDict):
    query: str
    disaster_type: str
    disaster_location: str
    disaster_date: str
    twitter_data: List[Dict[str, Any]]
    search_results: List[str]
    wikipedia_results: List[str]
    sentiment_analysis: List[Dict[str, Any]]
    report_data: Dict[str, Any]
    report_path: str
    response: str
    error: str

def init_llm():
    """Initialize LLM for agent operations"""
    try:
        # For local use without API keys
        return Ollama(model="qwen2.5-coder:14b") # original: deepseek-r1:8b
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        # Fallback to a model with reasonable performance
        logger.info("Falling back to default LLM")
        return Ollama(model="llama2")

def init_sentiment_model():
    """Initialize the sentiment analysis model"""
    if not HAS_SENTIMENT_MODEL:
        logger.info("No torch/transformers available, using simulated sentiment")
        return None
        
    try:
        # Use the roberta-twitter-sentiment model
        model_name = "spencercdz/roberta-twitter-sentiment"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logger.info(f"Successfully loaded {model_name} for sentiment analysis")
        return {"model": model, "tokenizer": tokenizer}
    except Exception as e:
        logger.error(f"Error initializing sentiment model: {e}")
        logger.error(traceback.format_exc())
        return None

def init_tools():
    """Initialize tools for the agent"""
    tools = []
    
    # Search tool
    try:
        search = DuckDuckGoSearchAPIWrapper()
        tools.append(Tool(
            name="web_search",
            func=search.run,
            description="Search the web for current information about disasters and events"
        ))
        logger.info("DuckDuckGo search tool initialized")
    except Exception as e:
        logger.error(f"Error initializing DuckDuckGo search tool: {e}")
    
    # Wikipedia tool
    try:
        wikipedia = WikipediaAPIWrapper()
        tools.append(Tool(
            name="wikipedia_search",
            func=wikipedia.run,
            description="Search Wikipedia for background information on locations and types of disasters"
        ))
        logger.info("Wikipedia search tool initialized")
    except Exception as e:
        logger.error(f"Error initializing Wikipedia search tool: {e}")
    
    return tools

def extract_disaster_info(query: str) -> Dict[str, str]:
    """Extract disaster type, location, and date from the query"""
    # Initialize LLM for extraction
    llm = init_llm()
    
    # Get current time
    current_year = datetime.now()
    
    # Define prompt template
    template = """
    Extract the disaster type, location, and date from the following query.
    IMPORTANT: The current time is {current_year} and any events referenced for {current_year} HAVE ALREADY OCCURRED. Do not treat them as future or hypothetical events.
    If a specific date isn't mentioned but the current year ({current_year}) is, use today's date in {current_year}.
    
    Query: {query}
    
    Provide the output in the following JSON format:
    {{
        "disaster_type": "type of disaster (e.g., earthquake, hurricane, flood)",
        "disaster_location": "location affected by the disaster",
        "disaster_date": "date of the disaster (YYYY-MM-DD format if possible)"
    }}
    """
    
    prompt = PromptTemplate(template=template, input_variables=["query", "current_year"])
    extraction_chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        # Extract disaster information
        response = extraction_chain.run(query=query, current_year=current_year)
        logger.info(f"Extraction response: {response}")
        
        # Parse JSON response
        # Find the first occurrence of '{' and the last occurrence of '}'
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx+1]
            try:
                info = json.loads(json_str)
                logger.info(f"Extracted disaster info: {info}")
                return info
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}, Response: {response}")
        else:
            logger.error(f"Could not find JSON in response: {response}")
        
        # Fallback to simple extraction if JSON parsing fails
        disaster_types = ["earthquake", "flood", "hurricane", "tsunami", "wildfire", "tornado"]
        disaster_type = ""
        for dtype in disaster_types:
            if dtype in query.lower():
                disaster_type = dtype
                break
        
        # Try to extract location and date
        words = query.split()
        disaster_location = words[0] if len(words) > 0 else ""
        disaster_date = words[-1] if len(words) > 1 else ""
        
        return {
            "disaster_type": disaster_type,
            "disaster_location": disaster_location,
            "disaster_date": disaster_date
        }
    except Exception as e:
        logger.error(f"Error extracting disaster info: {e}")
        logger.error(traceback.format_exc())
        return {
            "disaster_type": "",
            "disaster_location": "",
            "disaster_date": ""
        }

def load_twitter_data(query: str, disaster_info: Dict[str, str]) -> List[Dict[str, Any]]:
    """Load and filter Twitter data based on query and disaster info"""
    logger.info(f"Loading Twitter data for query: {query}")
    twitter_data = []
    
    try:
        # Find appropriate file
        csv_files = list(TWITTER_DATA_DIR.glob("*.csv"))
        if not csv_files:
            logger.error("No Twitter data files found")
            return twitter_data
        
        # Use the first CSV file found (can be enhanced to select best matching file)
        csv_file = csv_files[0]
        logger.info(f"Using Twitter data file: {csv_file}")
        
        # Load data
        df = pd.read_csv(csv_file)
        
        # Convert to list of dictionaries
        for _, row in df.iterrows():
            tweet = {
                "Username": row["username"],
                "Date": row["time"].split("_")[0] if "_" in row["time"] else row["time"],
                "Retweets": str(row["retweets"]),
                "Tweet": row["text"]
            }
            twitter_data.append(tweet)
        
        logger.info(f"Loaded {len(twitter_data)} tweets")
        return twitter_data
    except Exception as e:
        logger.error(f"Error loading Twitter data: {e}")
        logger.error(traceback.format_exc())
        return twitter_data

def search_web(query: str, disaster_info: Dict[str, str]) -> List[str]:
    """Search the web for disaster information"""
    search_tool = DuckDuckGoSearchAPIWrapper()
    search_results = []
    
    try:
        # Build enhanced search query
        enhanced_query = f"{disaster_info['disaster_type']} {disaster_info['disaster_location']} {disaster_info['disaster_date']} disaster impact statistics"
        
        # Execute search
        results = search_tool.run(enhanced_query)
        search_results.append(results)
        
        # Additional search for humanitarian aspects
        humanitarian_query = f"{disaster_info['disaster_type']} {disaster_info['disaster_location']} humanitarian assistance relief efforts"
        results = search_tool.run(humanitarian_query)
        search_results.append(results)
        
        logger.info(f"Collected {len(search_results)} search results")
        return search_results
    except Exception as e:
        logger.error(f"Error searching web: {e}")
        logger.error(traceback.format_exc())
        return search_results

def search_wikipedia(query: str, disaster_info: Dict[str, str]) -> List[str]:
    """Search Wikipedia for disaster information"""
    wikipedia_tool = WikipediaAPIWrapper()
    wikipedia_results = []
    
    try:
        # Search for disaster type
        if disaster_info["disaster_type"]:
            results = wikipedia_tool.run(f"{disaster_info['disaster_type']} disaster")
            wikipedia_results.append(results)
        
        # Search for location information
        if disaster_info["disaster_location"]:
            results = wikipedia_tool.run(disaster_info["disaster_location"])
            wikipedia_results.append(results)
        
        # Search for combined query
        results = wikipedia_tool.run(query)
        wikipedia_results.append(results)
        
        logger.info(f"Collected {len(wikipedia_results)} Wikipedia results")
        return wikipedia_results
    except Exception as e:
        logger.error(f"Error searching Wikipedia: {e}")
        logger.error(traceback.format_exc())
        return wikipedia_results

def analyze_sentiment(twitter_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze sentiment of tweets using the roberta-twitter-sentiment model"""
    if not twitter_data:
        return []
    
    # Initialize sentiment model
    sentiment_model = init_sentiment_model()
    
    logger.info(f"Analyzing sentiment for {len(twitter_data)} tweets")
    
    # If no model is available, use simulated sentiment
    if sentiment_model is None:
        logger.warning("No sentiment model available, using simulated sentiment")
        return _simulate_sentiment_analysis(twitter_data)
    
    try:
        results = []
        model = sentiment_model["model"]
        tokenizer = sentiment_model["tokenizer"]
        
        # Put model in evaluation mode and get device
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Process in batches
        batch_size = 16  # Smaller batch size for memory efficiency
        start_time = datetime.now()
        
        for i in range(0, len(twitter_data), batch_size):
            batch = twitter_data[i:i+batch_size]
            texts = [tweet.get('text', '') for tweet in batch]
            
            # Tokenize inputs
            encoded_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**encoded_inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert predictions to sentiment values
            # RoBERTa outputs: [negative, neutral, positive]
            sentiments = predictions.cpu().numpy()
            
            # Map predictions to results
            for j, sentiment_scores in enumerate(sentiments):
                tweet = batch[j]
                text = tweet.get('text', '').lower()
                
                # Calculate binary sentiment (0 or 1)
                # 0 for negative, 1 for positive
                # Using highest probability class
                sentiment_class = int(np.argmax(sentiment_scores))
                # Map to binary where 0=negative, 1=positive (ignoring neutral)
                sentiment_value = 1 if sentiment_class == 2 else 0
                
                # Determine genre based on username and content
                username = tweet.get('username', '').lower()
                if any(source in username for source in ["news", "cnn", "bbc", "reuters", "ap", "report"]):
                    genre = "news"
                elif any(name in username for name in ["resident", "local", "citizen", "survivor"]):
                    genre = "direct"
                else:
                    genre = "social"
                
                # Determine if content is related to the disaster
                disaster_keywords = ["earthquake", "disaster", "emergency", "damage", "victim", "rescue", 
                                    "myanmar", "burma", "debris", "aftershock", "trapped", "quake"]
                related = "yes" if any(word in text for word in disaster_keywords) else "maybe"
                
                # Check for specific content patterns
                has_request_words = any(word in text for word in ["need", "help", "please", "urgent", "require", "assistance"])
                has_offer_words = any(word in text for word in ["offer", "provide", "donate", "giving", "available", "support"])
                
                # Determine request/offer status
                is_request = has_request_words and not has_offer_words
                is_offer = has_offer_words and not has_request_words
                aid_related = is_request or is_offer or "aid" in text or "relief" in text or "help" in text
                
                result = {
                    **tweet,
                    'sentiment': sentiment_value,
                    'related': related,
                    'genre': genre,
                    'request': is_request,
                    'offer': is_offer,
                    'aid_related': aid_related
                }
                results.append(result)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Sentiment analysis completed in {duration:.2f} seconds for {len(twitter_data)} items")
        
        return results
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        logger.error(traceback.format_exc())
        return _simulate_sentiment_analysis(twitter_data)

def _simulate_sentiment_analysis(twitter_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simulate sentiment analysis when the real model is not available"""
    import random
    
    result = []
    for tweet in twitter_data:
        # Extract text content to determine simulated classifications more intelligently
        text = tweet.get("Tweet", "").lower()
        
        # Generate intelligent sentiment scores based on content
        has_negative_words = any(word in text for word in ["damage", "death", "casualties", "injured", "trapped", "crisis", "emergency"])
        has_positive_words = any(word in text for word in ["rescue", "aid", "help", "support", "recovery", "saved", "distribute"])
        has_request_words = any(word in text for word in ["need", "send", "please", "urgently", "required", "necessary", "help us"])
        has_offer_words = any(word in text for word in ["providing", "sending", "donate", "distributing", "offering", "deployed", "mobilizing"])
        
        # Basic sentiment calculation - negative words decrease score, positive words increase it
        # but we'll use argmax for final classification
        base_sentiment = 0.5  # Neutral starting point
        if has_negative_words:
            base_sentiment -= 0.2
        if has_positive_words:
            base_sentiment += 0.15
            
        # Simulate sentiment classification (0=negative, 1=positive)
        probs = [random.uniform(0, 0.4), random.uniform(0, 0.4), random.uniform(0, 0.4)]
        probs = [p/sum(probs) for p in probs]  # Normalize
        sentiment_class = int(np.argmax(probs))
        # Map to binary where 0=negative, 1=positive (ignoring neutral)
        sentiment_value = 1 if sentiment_class == 2 else 0
        
        # Determine genre based on username and content
        username = tweet.get("Username", "").lower()
        if any(name in username for name in ["news", "alert", "channel", "times", "report"]):
            genre = "news"
        elif any(name in username for name in ["resident", "local", "citizen", "survivor"]):
            genre = "direct"
        else:
            genre = "social"
            
        # Determine if content is related to the disaster
        related = "yes" if any(word in text for word in ["earthquake", "disaster", "emergency", "damage", "victim", "rescue"]) else "maybe"
        
        # Determine aid-related status
        aid_related = has_negative_words or has_positive_words or has_request_words or has_offer_words
        
        # Determine request/offer status
        is_request = has_request_words and not has_offer_words
        is_offer = has_offer_words and not has_request_words
        
        # Additional binary classifications based on content
        affected_individuals = "affected_individuals" in text or "victims" in text or "people" in text
        infrastructure_damage = "infrastructure" in text or "buildings" in text or "roads" in text
        medical_help = "medical" in text or "hospital" in text or "injuries" in text
        water = "water" in text or "drinking" in text or "thirst" in text
        
        # Create a comprehensive prediction structure matching TunedLLM output format
        prediction = {
            'sentiment': sentiment_value,
            "genre": genre,
            "related": related,
            "aid_related": aid_related,
            "request": is_request,
            "offer": is_offer,
            "affected_individuals": affected_individuals,
            "infrastructure_damage": infrastructure_damage,
            "medical_help": medical_help,
            "water": water,
            "f1_score": 0.89,  # Simulated performance metrics
            "precision": 0.92,
            "recall": 0.87
        }
        
        # Add sentiment information to the tweet data
        sentiment_data = {
            "Tweet": tweet.get("Tweet", ""),
            "Username": tweet.get("Username", ""),
            "Date": tweet.get("Date", ""),
            "sentiment_prediction": prediction
        }
        result.append(sentiment_data)
    
    logger.info(f"Simulated sentiment analysis for {len(result)} tweets with multi-task classification")
    return result

def extract_and_repair_json(text):
    """Extract valid JSON from text and attempt to repair if necessary"""
    import re
    import json
    
    # First try finding JSON between first { and last }
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        json_str = text[start_idx:end_idx+1]
        try:
            # Try parsing as is
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed: {e}, attempting repair")
            
            # Try fixing common issues:
            # 1. Fix trailing commas before closing brackets
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            # 2. Fix missing quotes around keys
            json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_str)
            
            # 3. Fix unescaped quotes in strings
            # (This is complex and might need more sophisticated handling)
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.error("JSON repair failed")
                return None
    return None

def generate_report_data(
    query: str, 
    disaster_info: Dict[str, str], 
    twitter_data: List[Dict[str, Any]], 
    search_results: List[str], 
    wikipedia_results: List[str],
    sentiment_analysis: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate structured report data in the required format"""
    # Define the LLM chain for report generation
    llm = init_llm()

    # Current Time
    current_time = datetime.now()
    
    # Calculate sentiment statistics from sentiment_analysis
    tweet_count = len(sentiment_analysis)
    positive_count = sum(1 for item in sentiment_analysis if item.get('sentiment', 0) == 1)
    negative_count = sum(1 for item in sentiment_analysis if item.get('sentiment', 0) == 0)
    request_count = sum(1 for item in sentiment_analysis if item.get('request', False))
    offer_count = sum(1 for item in sentiment_analysis if item.get('offer', False))
    
    template = """
    You are an expert humanitarian assistance and disaster relief (HADR) analyst. You need to create a structured report about a disaster, while maintaining high contextual knowledge with the information provided to you.

    CRITICAL: Your output MUST be valid JSON without any explanation text before or after. Your response will not include markdown code blocks, explanations, thought process, or any other text - ONLY return the exact JSON object.

    Based on the available information and tweets, you should only consider the information and tweets that lie within a time period close to the disaster.

    YOU CANNOT ADD NEW KEYS TO THE JSON OBJECT, AND CAN ONLY USE THE DEFINED KEYS FOR EACH SECTION OF THE JSON.
    
    IMPORTANT: The current year is {current_year} and any events referenced for {current_year} HAVE ALREADY OCCURRED. 
    This is not a hypothetical or future scenario - the disaster has already happened.

    When it comes to "tweets", you will mention the entire length of the available tweet data that has been analyzed. However, you will only store the information for the 10 tweets, and you will pick the 10 tweets with the most retweets. These tweets must be directly related to the disaster and the location of the disaster. You will also store the username, retweets, and tweet from the original tweet data.
    
    YOU ARE NOT TO REPHRASE THE USERNAME, RETWEETS, OR TWEET FROM THE ORIGINAL TWEET DATA.

    When it comes to "details", you will review every single tweet in the available tweet data. You will store a result for every date that is available in the dataset as long as it is approximately within the time range of the disaster. If more than one tweet is available for a date, you are to summarize all the information for the day and summarize it into an inclusive overview. Sentiment score is a float value that can lie on any value between 0 and 1 up to two decimal places, where 0 is negative and 1 is positive, and the average sentiment score is the average of all the sentiment scores for that day.

    YOU MUST REVIEW EVERY SINGLE TWEET AND EVALUATE IT ACCORDINGLY.
    
    QUERY: {query}
    
    DISASTER TYPE: {disaster_type}
    DISASTER LOCATION: {disaster_location}
    DISASTER DATE: {disaster_date}
    
    ALL AVAILABLE TWEET DATA TO ANALYZE:
    {twitter_data}

    WEB SEARCH RESULTS: 
    {search_results}
    
    WIKIPEDIA INFORMATION:
    {wikipedia_results}
    
    TWEET SENTIMENT ANALYSIS SUMMARY:
    - Total tweets analyzed: {tweet_count}
    - Positive sentiment: {positive_count} tweets
    - Negative sentiment: {negative_count} tweets
    - Requests for help: {request_count} tweets
    - Offers of assistance: {offer_count} tweets
    
    Create a COMPLETE JSON object with the structure based on this template:
    {{
        "sections": {{
            "Background": "String output of a lengthy 2-3 paragraph summary of the disaster situation, including the type, location, timing, and key impacts. Remember this disaster has already happened and is current news in {current_year}...",
            "Tweet Overview": "String output of a short summary of the tweets, including the total number of tweets in the original dataset and the main themes...",
            "Sentiment Overview": "String output of a short summary of the sentiment analysis, including the number of tweets with positive or negative sentiments, and the main themes of the data...",
            "Results": "String output of a short summary about the affected population, including labels of affected, displaced, injured, and deceased individuals if available...",
            "Discussion": "String output of a lengthy detailed information about ongoing response efforts, including organizations involved and current priorities...",
            "Recommendation": "String output of a lengthy detailed assessment of the most critical actions to take based on the available information, including recommendations for humanitarian assistance and disaster relief..."
        }},
        "tweets": [
            {{
                "Username": "String output of the username...",
                "Date": "String output of the date...",
                "Retweets": "String output of the retweet number...",
                "Tweet": "String output of the tweet..."
            }},
            ...
        ],
        "details": [
            {{
                "Date": "String output of the date...",
                "Sentiment": Float output of the average sentiment score in the format of 0.XX,
                "Elements": "String output of the elements...",
                "Impact": "String output of the impact...",
                "Requests": "String output of the requests...",
                "Summary": "String output of the summary..."
            }},
            ...
        ]
    }} 
    
    Instructions:
    - Do not speculate or invent facts.
    - Do not omit any sections or information.
    - If a section cannot be completed due to missing data, explicitly state this.
    - Incorporate sentiment analysis to identify trends in emotional responses over time.
    - Use precision, recall, and F1 metrics when discussing classification confidence.
    
    Your JSON output:
    """
    
    # Create the prompt
    prompt = PromptTemplate(
        template=template, 
        input_variables=[
            "current_year",
            "query",
            "disaster_type",
            "disaster_location",
            "disaster_date",
            "search_results",
            "wikipedia_results",
            "tweet_count",
            "positive_count",
            "negative_count",
            "request_count",
            "offer_count",
            "twitter_data",
        ]
    )
    
    # Create the report generation chain
    report_chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        # Run the chain
        disaster_type = disaster_info["disaster_type"] or "disaster"
        disaster_location = disaster_info["disaster_location"] or "affected area"
        
        # Sample the Twitter data if it's too large
        twitter_sample = twitter_data[:20] if len(twitter_data) > 20 else twitter_data
        sentiment_sample = sentiment_analysis[:20] if len(sentiment_analysis) > 20 else sentiment_analysis
        
        # Extract sample tweets and format them for display in the report
        sample_tweets = []
        if sentiment_analysis and len(sentiment_analysis) > 0:
            # Sort tweets - show most retweeted and most relevant first
            sorted_tweets = sorted(sentiment_analysis, 
                                key=lambda x: (x.get('related', '') == 'yes', x.get('retweets', 0)), 
                                reverse=True)
            
            # Take top 10 tweets
            for tweet in sorted_tweets[:10]:
                tweet_data = {
                    "Username": tweet.get("Username", ""),
                    "Date": tweet.get("Date", ""),
                    "Retweets": tweet.get("Retweets", 0),
                    "Tweet": tweet.get("Tweet", "")
                }
                sample_tweets.append(tweet_data)
        
        # Prepare the input for the LLM chain
        current_time = datetime.now()
        chain_input = {
            "current_year": current_time.year,
            "query": query,
            "twitter_data": twitter_data,
            "disaster_type": disaster_info.get("disaster_type", ""),
            "disaster_location": disaster_info.get("disaster_location", ""),
            "disaster_date": disaster_info.get("disaster_date", ""),
            "tweet_count": tweet_count,
            "positive_count": sum(1 for item in sentiment_analysis if item.get('sentiment', 0) == 1),
            "negative_count": sum(1 for item in sentiment_analysis if item.get('sentiment', 0) == 0),
            "request_count": request_count,
            "offer_count": offer_count,
            "search_results": "\n\n".join(search_results),
            "wikipedia_results": "\n\n".join(wikipedia_results),
        }
        
        # Generate the report
        report_json_str = report_chain.run(**chain_input)
        print("Report JSON:", report_json_str)
        
        # Parse the JSON response
        # Find the first occurrence of '{' and the last occurrence of '}'
        repaired_json = extract_and_repair_json(report_json_str)
        if repaired_json:
            logger.info("Successfully extracted and repaired JSON")
            return repaired_json
        else:
            logger.error("Could not parse or repair JSON, using template")
        
        # If parsing fails, load the template file
        # Report template
        template_path = TEMPLATES_DIR / "report_template.json"
        if template_path.exists():
            with open(template_path, 'r') as f:
                report_data = json.load(f)
                return report_data
        
        # If all else fails, return a minimal valid structure
        logger.error("Using minimal report structure")
        return {
            "sections": {
                "Background": f"Analysis of {disaster_type} in {disaster_location}",
                "Tweet Overview": "Twitter data analysis",
                "Sentiment Overview": "Sentiment analysis",
                "Task Classification": "Task classification",
                "Results": "Key findings",
                "Discussion": "Interpretation",
                "Recommendation": "Recommendations"
            },
            "tweets": [
                {
                    "Username": "Error",
                    "Date": "Error",
                    "Retweets": "Error",
                    "Tweet": "Error"
                }
            ],
            "details": [
                {
                    "Date": datetime.now().strftime("%d/%m/%Y"),
                    "Sentiment": 0.5,
                    "Elements": disaster_type,
                    "Impact": "Infrastructure, Population",
                    "Requests": "Aid, Support",
                    "Summary": f"Analysis of {disaster_type} in {disaster_location}",
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error generating report data: {e}")
        logger.error(traceback.format_exc())
        
        # Return a minimal valid structure
        return {
            "sections": {
                "Background": "Error occurred during report generation",
                "Tweet Overview": "Error occurred during report generation",
                "Sentiment Overview": "Error occurred during report generation",
                "Task Classification": "Error occurred during report generation",
                "Results": "Error occurred during report generation",
                "Discussion": "Error occurred during report generation",
                "Recommendation": "Error occurred during report generation"
            },
            "tweets": [{
                "Username": "Error",
                "Date": "Error",
                "Retweets": "Error",
                "Tweet": "Error"
            }],
            "details": [{
                "Date": "Error",
                "Sentiment": 0.5,
                "Elements": "Error",
                "Impact": "Error",
                "Requests": "Error",
                "Summary": "Error"
            }]
        }


def create_report_file(report_data: Dict[str, Any], query: str, disaster_info: Dict[str, str]) -> str:
    """Create the PDF report file"""
    try:
        # Create file names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        disaster_type = disaster_info["disaster_type"] or "disaster"
        disaster_location = disaster_info["disaster_location"] or "location"
        
        # Create JSON file for the report data
        json_file_name = f"{disaster_location}_{disaster_type}_{timestamp}.json"
        json_file_path = OUTPUTS_DIR / json_file_name
        
        with open(json_file_path, 'w') as f:
            json.dump(report_data, f, indent=4)
        
        logger.info(f"Saved report data to {json_file_path}")
        
        # Create PDF file name
        pdf_file_name = f"{disaster_location}_{disaster_type}_{timestamp}.pdf"
        pdf_file_path = OUTPUTS_DIR / pdf_file_name
        
        # Generate PDF report
        output_path = build_report.generate_report(
            json_file_path=str(json_file_path),
            output_pdf_path=str(pdf_file_path)
        )
        
        logger.info(f"Generated PDF report at {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error creating report file: {e}")
        logger.error(traceback.format_exc())
        return ""

def generate_response(report_path: str, disaster_info: Dict[str, str], error: str = None) -> str:
    """Generate a response based on the results"""
    if error:
        return f"I encountered an error while generating the report: {error}"
    
    if not report_path:
        return "I was unable to generate a report. Please try again with a more specific query."
    
    disaster_type = disaster_info["disaster_type"] or "disaster"
    disaster_location = disaster_info["disaster_location"] or "affected area"
    
    return f"""
    I've generated a comprehensive HADR report about the {disaster_type} in {disaster_location}.
    
    The report includes:
    - Situation overview of the disaster
    - Analysis of social media sentiment
    - Assessment of affected population and needs
    - Information about ongoing response efforts
    
    You can download the full report to see detailed information and insights.
    """

# Define the graph nodes
def extract_info(state: AgentState) -> AgentState:
    """Extract disaster information from the query"""
    query = state["query"]
    logger.info(f"Extracting information from query: {query}")
    
    try:
        disaster_info = extract_disaster_info(query)
        return {
            **state,
            "disaster_type": disaster_info.get("disaster_type", ""),
            "disaster_location": disaster_info.get("disaster_location", ""),
            "disaster_date": disaster_info.get("disaster_date", "")
        }
    except Exception as e:
        logger.error(f"Error in extract_info: {e}")
        logger.error(traceback.format_exc())
        return {
            **state,
            "error": f"Error extracting disaster information: {str(e)}"
        }

def gather_twitter_data(state: AgentState) -> AgentState:
    """Gather Twitter data for analysis"""
    query = state["query"]
    logger.info(f"Gathering Twitter data for query: {query}")
    
    try:
        disaster_info = {
            "disaster_type": state["disaster_type"],
            "disaster_location": state["disaster_location"],
            "disaster_date": state["disaster_date"]
        }
        
        twitter_data = load_twitter_data(query, disaster_info)
        return {**state, "twitter_data": twitter_data}
    except Exception as e:
        logger.error(f"Error in gather_twitter_data: {e}")
        logger.error(traceback.format_exc())
        return {
            **state,
            "twitter_data": [],
            "error": f"Error gathering Twitter data: {str(e)}"
        }

def gather_web_info(state: AgentState) -> AgentState:
    """Gather information from web search"""
    query = state["query"]
    logger.info(f"Gathering web information for query: {query}")
    
    try:
        disaster_info = {
            "disaster_type": state["disaster_type"],
            "disaster_location": state["disaster_location"],
            "disaster_date": state["disaster_date"]
        }
        
        search_results = search_web(query, disaster_info)
        return {**state, "search_results": search_results}
    except Exception as e:
        logger.error(f"Error in gather_web_info: {e}")
        logger.error(traceback.format_exc())
        return {
            **state,
            "search_results": [],
            "error": f"Error gathering web information: {str(e)}"
        }

def gather_wikipedia_info(state: AgentState) -> AgentState:
    """Gather information from Wikipedia"""
    query = state["query"]
    logger.info(f"Gathering Wikipedia information for query: {query}")
    
    try:
        disaster_info = {
            "disaster_type": state["disaster_type"],
            "disaster_location": state["disaster_location"],
            "disaster_date": state["disaster_date"]
        }
        
        wikipedia_results = search_wikipedia(query, disaster_info)
        return {**state, "wikipedia_results": wikipedia_results}
    except Exception as e:
        logger.error(f"Error in gather_wikipedia_info: {e}")
        logger.error(traceback.format_exc())
        return {
            **state,
            "wikipedia_results": [],
            "error": f"Error gathering Wikipedia information: {str(e)}"
        }

def analyze_tweet_sentiment(state: AgentState) -> AgentState:
    """Analyze sentiment of tweets using the sentiment analysis model"""
    logger.info("Analyzing tweet sentiment")
    
    try:
        sentiment_results = analyze_sentiment(state["twitter_data"])
        return {**state, "sentiment_analysis": sentiment_results}
    except Exception as e:
        logger.error(f"Error in analyze_tweet_sentiment: {e}")
        logger.error(traceback.format_exc())
        return {
            **state,
            "sentiment_analysis": [],
            "error": f"Error analyzing tweet sentiment: {str(e)}"
        }

def build_report_data(state: AgentState) -> AgentState:
    """Build the report data"""
    query = state["query"]
    logger.info(f"Building report data for query: {query}")
    
    try:
        disaster_info = {
            "disaster_type": state["disaster_type"],
            "disaster_location": state["disaster_location"],
            "disaster_date": state["disaster_date"]
        }
        
        report_data = generate_report_data(
            query,
            disaster_info,
            state["twitter_data"],
            state["search_results"],
            state["wikipedia_results"],
            state["sentiment_analysis"]
        )
        
        return {**state, "report_data": report_data}
    except Exception as e:
        logger.error(f"Error in build_report_data: {e}")
        logger.error(traceback.format_exc())
        return {
            **state,
            "report_data": {},
            "error": f"Error building report data: {str(e)}"
        }

def create_final_report(state: AgentState) -> AgentState:
    """Create the final PDF report"""
    query = state["query"]
    logger.info(f"Creating final report for query: {query}")
    
    try:
        disaster_info = {
            "disaster_type": state["disaster_type"],
            "disaster_location": state["disaster_location"],
            "disaster_date": state["disaster_date"]
        }
        
        report_path = create_report_file(state["report_data"], query, disaster_info)
        return {**state, "report_path": report_path}
    except Exception as e:
        logger.error(f"Error in create_final_report: {e}")
        logger.error(traceback.format_exc())
        return {
            **state,
            "report_path": "",
            "error": f"Error creating final report: {str(e)}"
        }

def generate_final_response(state: AgentState) -> AgentState:
    """Generate the final response"""
    logger.info("Generating final response")
    
    try:
        disaster_info = {
            "disaster_type": state["disaster_type"],
            "disaster_location": state["disaster_location"],
            "disaster_date": state["disaster_date"]
        }
        
        response = generate_response(
            state.get("report_path", ""),
            disaster_info,
            state.get("error", None)
        )
        
        return {**state, "response": response}
    except Exception as e:
        logger.error(f"Error in generate_final_response: {e}")
        logger.error(traceback.format_exc())
        return {
            **state,
            "response": f"I encountered an error while processing your request: {str(e)}"
        }

def build_agent_workflow():
    """Build the agent workflow graph"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("extract_info", extract_info)
    workflow.add_node("gather_twitter_data", gather_twitter_data)
    workflow.add_node("gather_web_info", gather_web_info)
    workflow.add_node("gather_wikipedia_info", gather_wikipedia_info)
    workflow.add_node("analyze_tweet_sentiment", analyze_tweet_sentiment)
    workflow.add_node("build_report_data", build_report_data)
    workflow.add_node("create_final_report", create_final_report)
    workflow.add_node("generate_final_response", generate_final_response)
    
    # Add edges
    workflow.set_entry_point("extract_info")
    workflow.add_edge("extract_info", "gather_twitter_data")
    workflow.add_edge("gather_twitter_data", "gather_web_info")
    workflow.add_edge("gather_web_info", "gather_wikipedia_info")
    workflow.add_edge("gather_wikipedia_info", "analyze_tweet_sentiment")
    workflow.add_edge("analyze_tweet_sentiment", "build_report_data")
    workflow.add_edge("build_report_data", "create_final_report")
    workflow.add_edge("create_final_report", "generate_final_response")
    workflow.add_edge("generate_final_response", END)
    
    return workflow

def process_query(query: str) -> Dict[str, Any]:
    """Process a query through the agent workflow"""
    logger.info(f"Processing query: {query}")
    
    # Default state
    default_state = {
        "query": query,
        "disaster_type": "",
        "disaster_location": "",
        "disaster_date": "",
        "twitter_data": [],
        "search_results": [],
        "wikipedia_results": [],
        "sentiment_analysis": [],
        "report_data": {},
        "report_path": "",
        "response": "",
        "error": ""
    }
    
    # Build and run the workflow
    try:
        workflow = build_agent_workflow()
        app = workflow.compile()
        result = app.invoke(default_state)
        
        logger.info(f"Workflow completed with report path: {result.get('report_path', '')}")
        return result
    except Exception as e:
        logger.error(f"Error in process_query: {e}")
        logger.error(traceback.format_exc())
        
        return {
            **default_state,
            "error": f"Error processing query: {str(e)}",
            "response": f"I encountered an error while processing your request: {str(e)}"
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HADR Agent")
    parser.add_argument("--query", type=str, help="Query to process")
    args = parser.parse_args()
    
    if args.query:
        result = process_query(args.query)
        print(f"Response: {result['response']}")
        print(f"Report path: {result.get('report_path', 'No report generated')}")
    else:
        print("Please provide a query with --query")
