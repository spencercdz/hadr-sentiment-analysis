# src/ai_agent/agents/hadr_agent.py

import os
import sys
import json
import pandas as pd
import logging
import gc
from pathlib import Path
from typing import Dict, List, Any, TypedDict
import traceback
from datetime import datetime, timedelta
from langchain_community.llms import Ollama
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from langchain.chains import LLMChain
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from tqdm import tqdm
import re
import csv
import yaml
from typing import Any, Dict, List, Optional, TypedDict

# Configure paths
current_dir = Path(__file__).parent
tools_dir = current_dir / "tools"
sys.path.append(str(tools_dir))

# Configure root project path to allow importing models
project_root = current_dir.parent.parent.parent
models_dir = project_root / "src" / "models"
sys.path.append(str(project_root))
sys.path.append(str(models_dir))

# Import Tuned LLM class
llm_dir = project_root / "src" / "models" / "llm"
sys.path.append(str(llm_dir))

# Import directly with full path to avoid custom class registry issues
from huggingface_hub import snapshot_download
from src.models.llm.tuned import TunedLLM
from src.models.llm.labels import get_all_labels
from src.ai_agent.agents.tools.build_report import generate_report

# Import sentiment analysis packages for Twitter sentiment analysis
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    HAS_SENTIMENT_MODEL = True
except ImportError:
    HAS_SENTIMENT_MODEL = False
    print("Warning: transformers or torch package not available for sentiment analysis")

# Import the build_report module
from .tools.build_report import generate_report

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

# Elements, Impact, Requests mapping
elements_map = ["weather_related", "floods", "storm", "fire", "earthquake", "cold", "other_weather"
                ]
impact_map = ['infrastructure_related', 'transport', 'buildings', 'electricity', 'hospitals', 'shops',
                'aid_centers', 'other_infrastructure', 'death', 'child_alone', 'search_and_rescue'
                ]
requests_map = ['medical_help', 'medical_products', 'water', 'food', 'shelter', 'clothing',
                 'money', 'tools', 'other_aid'
                 ]

def init_llm() -> Ollama:
    """Initializes and returns the Ollama LLM for agent operations."""
    try:
        return Ollama(model="qwen2.5-coder:14b", temperature = 0)
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        logger.info("Falling back to default LLM")
        return Ollama(model="qwen2.5-coder:14b", temperature = 0)

def init_sentiment_model() -> Optional[TunedLLM]:
    """Initializes and returns the custom TunedLLM for multi-label classification."""
    # First, try to download from Hugging Face Hub
    try:
        repo_id = "spencercdz/xlm-roberta-sentiment-requests"
        token = os.environ.get("HF_TOKEN")
        logger.info(f"Attempting to download model repository '{repo_id}' from the Hub...")
        model_path = snapshot_download(
            repo_id=repo_id, repo_type="model", token=token, allow_patterns="final_model/**"
        )
        logger.info(f"Model downloaded to cache: {model_path}")
        final_model_path = Path(model_path) / "final_model"
        if not final_model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found in cache: {final_model_path}.")
        model = TunedLLM.load_from_disk(final_model_path)
        logger.info(f"Successfully initialized TunedLLM with model from Hub: {repo_id}")
        return model
    except Exception as e:
        logger.warning(f"Failed to download or load model from Hugging Face Hub: {e}")
        logger.info("Falling back to loading a local model.")

    # If downloading fails, try to load from a local path
    try:
        local_model_path = project_root / "models" / "tuned" / "cardiffnlp_twitter-xlm-roberta-base-sentiment" / "final_model"
        logger.info(f"Attempting to load model from local path: {local_model_path}")
        if not local_model_path.is_dir():
            raise FileNotFoundError(f"Local model directory not found: {local_model_path}.")
        model = TunedLLM.load_from_disk(local_model_path)
        logger.info("Successfully initialized TunedLLM with local model.")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize TunedLLM from both Hub and local path: {e}", exc_info=True)
        return None
 
def analyze_sentiment(twitter_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyzes tweets using the TunedLLM and calculates a 0-100 weighted sentiment score.
    """
    logger.info("-----ANALYZING SENTIMENT WITH TUNED LLM-----")
    if not twitter_data:
        return []

    sentiment_model = init_sentiment_model()
    if not sentiment_model:
        return []

    tweet_texts = [str(tweet.get('tweet', tweet.get('text', ''))) for tweet in twitter_data]
    
    try:
        logger.info(f"Running predictions on {len(tweet_texts)} tweets...")
        predictions = sentiment_model.predict(tweet_texts)
        results_for_report = []
        sentiment_weights = [0.0, 50.0, 100.0]  # [negative, neutral, positive]

        for i, original_tweet in enumerate(tqdm(twitter_data, desc="Processing Predictions")):
            if i >= len(predictions):
                continue
            
            pred = predictions[i]
            processed_tweet = original_tweet.copy()

            for task_name, task_output in pred.items():
                if task_name == "multilabel_scores":
                    continue
                if isinstance(task_output, dict) and "prediction" in task_output:
                    processed_tweet[task_name] = task_output["prediction"]
                    processed_tweet[f"{task_name}_confidence"] = task_output.get("confidence", 0.0)

            sentiment_output = pred.get('sentiment', {})
            sentiment_scores = sentiment_output.get('scores', [])
            
            weighted_sentiment_score = 50.0 
            if len(sentiment_scores) == len(sentiment_weights):
                weighted_sentiment_score = sum(p * w for p, w in zip(sentiment_scores, sentiment_weights))
            
            processed_tweet['sentiment'] = weighted_sentiment_score
            results_for_report.append(processed_tweet)

        logger.info(f"Successfully analyzed and processed {len(results_for_report)} tweets.")
        
        del sentiment_model
        if 'torch' in sys.modules and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return results_for_report

    except Exception as e:
        logger.error(f"An error occurred during sentiment analysis: {e}", exc_info=True)
        return []

def init_tools() -> List[Tool]:
    """Initializes and returns a list of tools for the agent."""
    tools = []
    try:
        search = DuckDuckGoSearchAPIWrapper()
        tools.append(Tool(
            name="web_search", func=search.run,
            description="Search the web for current information about disasters and events"
        ))
        logger.info("DuckDuckGo search tool initialized")
    except Exception as e:
        logger.error(f"Error initializing DuckDuckGo search tool: {e}")
    try:
        wikipedia = WikipediaAPIWrapper()
        tools.append(Tool(
            name="wikipedia_search", func=wikipedia.run,
            description="Search Wikipedia for background information on locations and types of disasters"
        ))
        logger.info("Wikipedia search tool initialized")
    except Exception as e:
        logger.error(f"Error initializing Wikipedia search tool: {e}")
    return tools

def extract_disaster_info(query: str) -> Dict[str, str]:
    """Extracts disaster type, location, and date from a user query using an LLM."""
    llm = init_llm()
    current_year = datetime.now().year
    template = """
    Extract the disaster type, location, and date from the following query.
    IMPORTANT: The current year is {current_year} and any events referenced for this year have already occurred.
    Query: {query}
    Provide the output in a clean JSON format:
    {{
        "disaster_type": "type of disaster",
        "disaster_location": "location of disaster",
        "disaster_date": "date in YYYY-MM-DD format"
    }}
    """
    prompt = PromptTemplate(template=template, input_variables=["query", "current_year"])
    extraction_chain = LLMChain(llm=llm, prompt=prompt)
    try:
        response = extraction_chain.run(query=query, current_year=current_year)
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No JSON object found in LLM response.")
    except Exception as e:
        logger.error(f"Error extracting disaster info: {e}. Falling back.")
        # Fallback to simple extraction if JSON parsing fails
        disaster_types = ["earthquake", "flood", "hurricane", "tsunami", "wildfire", "tornado"]
        disaster_type = ""
        for dtype in disaster_types:
            if dtype in query.lower():
                disaster_type = dtype
                break
        words = query.split()
        disaster_location = words[0] if len(words) > 0 else ""
        disaster_date = words[-1] if len(words) > 1 else ""
        return {"disaster_type": disaster_type, "disaster_location": disaster_location, "disaster_date": disaster_date}

def load_twitter_data(query: str, disaster_info: Dict[str, str]) -> List[Dict[str, Any]]:
    """Loads and filters Twitter data from a CSV file based on disaster information."""
    location = disaster_info.get("disaster_location", "").lower().replace(" ", "_")
    disaster_type = disaster_info.get("disaster_type", "").lower().replace(" ", "_")
    date_str = disaster_info.get("disaster_date", "")
    year = date_str.split('-')[0] if date_str else ""

    if not all([location, disaster_type, year]):
        logger.error("Could not determine disaster directory. Missing location, type, or year.")
        return []

    dir_name = f"{location}_{disaster_type}_{year}"
    disaster_dir = TWITTER_DATA_DIR / dir_name
    logger.info(f"Looking for Twitter data in: {disaster_dir}")

    if not disaster_dir.is_dir():
        logger.warning(f"Directory not found: {disaster_dir}")
        return []

    csv_files = list(disaster_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {disaster_dir}")
        return []
    
    try:
        df = pd.read_csv(csv_files[0])
        df.columns = [col.lower().strip() for col in df.columns]
        logger.info(f"Loaded {len(df)} tweets from {csv_files[0]}")
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Error loading Twitter data from {csv_files[0]}: {e}")
        return []

def _perform_search(search_tool, query: str, log_message: str) -> List[str]:
    """Helper function to perform a search and handle errors."""
    try:
        results = search_tool.run(query)
        logger.info(f"{log_message} for query: '{query}'")
        return [results]
    except Exception as e:
        logger.error(f"Error during {log_message} for query '{query}': {e}")
        return []

def search_web(query: str, disaster_info: Dict[str, str]) -> List[str]:
    """Searches the web for information related to the disaster."""
    search_tool = DuckDuckGoSearchAPIWrapper()
    search_results = []
    q = f"{disaster_info['disaster_type']} {disaster_info['disaster_location']} {disaster_info['disaster_date']}"
    search_results.extend(_perform_search(search_tool, f"{q} disaster impact statistics", "web search"))
    search_results.extend(_perform_search(search_tool, f"{q} humanitarian assistance", "web search"))
    logger.info(f"Collected {len(search_results)} web search results.")
    return search_results

def search_wikipedia(query: str, disaster_info: Dict[str, str]) -> List[str]:
    """Searches Wikipedia for background information on the disaster."""
    wikipedia_tool = WikipediaAPIWrapper()
    wikipedia_results = []
    if disaster_info["disaster_type"]:
        wikipedia_results.extend(_perform_search(wikipedia_tool, f"{disaster_info['disaster_type']} disaster", "wikipedia search"))
    if disaster_info["disaster_location"]:
        wikipedia_results.extend(_perform_search(wikipedia_tool, disaster_info["disaster_location"], "wikipedia search"))
    logger.info(f"Collected {len(wikipedia_results)} Wikipedia results.")
    return wikipedia_results

def extract_and_repair_json(text: str) -> Optional[Dict[str, Any]]:
    """Extracts and repairs a JSON object from a string."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed: {e}, attempting repair")
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.error("JSON repair failed")
                return None
    return None

def _generate_all_daily_summaries(daily_stats: List[Dict[str, Any]]) -> Dict[str, str]:
    """Generates a narrative summary for each day in a SINGLE, efficient LLM call."""
    logger.info(f"Generating narrative summaries for {len(daily_stats)} days in a single LLM call...")
    if not daily_stats:
        return {}

    llm = init_llm()
    stats_for_prompt = []
    for day in daily_stats:
        top_labels = sorted(day.get('label_counts', {}).items(), key=lambda item: item[1], reverse=True)[:5]
        stats_for_prompt.append({
            "date": day['date'],
            "tweet_count": day['tweet_count'],
            "average_sentiment": round(day['sentiment_score'], 2),
            "top_themes": {label: count for label, count in top_labels}
        })

    prompt_template = """
    You are a data analyst summarizing daily social media activity during a crisis.
    Based on the following list of daily statistics, generate a comprehensive, multi-sentence narrative summary (at least 5-6 sentences) for EACH day.
    For each day, your summary MUST explicitly mention:
    - The overall sentiment (with score and trend if possible)
    - The most prominent elements (themes/topics)
    - The main impacts observed
    - The types and volume of requests
    Use all available data (sentiment, label counts, tweet count, etc.) to provide a thorough, context-rich summary for each day.
    The sentiment score is on a 0-100 scale (0=very negative, 50=neutral, 100=very positive).
    
    DATA:
    {daily_statistics}
    
    Your output MUST be a valid JSON object, where each key is the date string and the value is the generated summary string for that day.
    
    Example Output Format:
    {{
      "2025-03-28": "On this day, social media activity was high with 500 tweets, reflecting a predominantly negative sentiment (avg score: 25.5). The main elements discussed included building collapses and requests for shelter. The impact was significant, with many users reporting infrastructure damage and urgent needs. Requests for shelter and medical help were especially frequent. The sentiment remained low throughout the day, indicating ongoing distress. Overall, the data suggests a community in urgent need of assistance, with clear calls for help and reports of severe impact.",
      "2025-03-29": "Activity decreased to 300 tweets. Sentiment slightly improved to 35.2, with discussions shifting towards aid distribution and medical help. The main elements were food and water shortages. Impact reports lessened, but requests for aid remained high. The community's mood showed slight improvement, but needs persisted."
    }}

    JSON Response:
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    summary_chain = LLMChain(llm=llm, prompt=prompt)

    try:
        response = summary_chain.run(daily_statistics=json.dumps(stats_for_prompt, indent=2))
        summaries = extract_and_repair_json(response)
        if isinstance(summaries, dict):
            logger.info("Successfully generated daily summaries from LLM.")
            return summaries
        else:
            logger.error("LLM did not return a valid JSON object for daily summaries.")
            return {}
    except Exception as e:
        logger.error(f"Error generating daily summaries: {e}", exc_info=True)
        return {}

def _aggregate_results_by_date(results: List[Dict[str, Any]], all_labels: dict = None) -> List[Dict[str, Any]]:
    """
    Aggregates statistics by date and then calls an efficient LLM summarizer.
    """
    if not results:
        return []
    
    if all_labels is None: all_labels = get_all_labels()
    
    tweets_by_date = {}
    
    for tweet in results:
        tweet_norm = {k.lower(): v for k, v in tweet.items()}
        date_str_raw = (tweet_norm.get('date', '') or tweet_norm.get('time', ''))
        date_str = date_str_raw.split(' ')[0].split('_')[0]
        if not date_str:
            continue
            
        if date_str not in tweets_by_date:
            tweets_by_date[date_str] = {
                'sentiment_sum': 0.0,
                'count': 0,
                'label_counts': {label: 0 for label in all_labels if label != 'sentiment'},
                'sentiment_counts': {'negative': 0, 'neutral': 0, 'positive': 0}
            }

        sentiment_value = float(tweet_norm.get('sentiment', 50.0))
        tweets_by_date[date_str]['sentiment_sum'] += sentiment_value
        tweets_by_date[date_str]['count'] += 1
        
        if sentiment_value > 60: tweets_by_date[date_str]['sentiment_counts']['positive'] += 1
        elif sentiment_value < 40: tweets_by_date[date_str]['sentiment_counts']['negative'] += 1
        else: tweets_by_date[date_str]['sentiment_counts']['neutral'] += 1
        
        for key, value in tweet_norm.items():
            if key in tweets_by_date[date_str]['label_counts']:
                if str(value).lower() not in ['no', 'unknown']:
                    tweets_by_date[date_str]['label_counts'][key] += 1

    daily_stats_for_summary = []
    for date, data in sorted(tweets_by_date.items()):
        avg_sentiment = data['sentiment_sum'] / data['count'] if data['count'] > 0 else 50.0
        daily_stats_for_summary.append({
            'date': date,
            'sentiment_score': avg_sentiment,
            'tweet_count': data['count'],
            'sentiment_counts': data['sentiment_counts'],
            'label_counts': data['label_counts']
        })

    llm_generated_summaries = _generate_all_daily_summaries(daily_stats_for_summary)

    final_aggregated_results = []
    for day_data in daily_stats_for_summary:
        day_data['Summary'] = llm_generated_summaries.get(day_data['date'], "Summary could not be generated.")
        final_aggregated_results.append(day_data)
        
    logger.info(f"Aggregated statistics and generated summaries for {len(final_aggregated_results)} dates.")
    return final_aggregated_results

def generate_report_data(
    query: str, 
    disaster_info: Dict[str, str], 
    twitter_data: List[Dict[str, Any]], 
    search_results: List[str], 
    wikipedia_results: List[str],
    sentiment_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates complete structured report data with sections, tweets, and details."""
    llm = init_llm()
    tweet_count = len(sentiment_analysis)
    
    positive_count = sum(1 for item in sentiment_analysis if float(item.get('sentiment', 50.0)) > 60)
    negative_count = sum(1 for item in sentiment_analysis if float(item.get('sentiment', 50.0)) < 40)
    neutral_count = tweet_count - positive_count - negative_count
    
    avg_sentiment = sum(float(item.get('sentiment', 50.0)) for item in sentiment_analysis) / max(1, len(sentiment_analysis))
    
    request_count = sum(1 for item in sentiment_analysis if str(item.get('request', 'no')).lower() == 'yes')
    medical_help_count = sum(1 for item in sentiment_analysis if str(item.get('medical_help', 'no')).lower() == 'yes')
    
    template = """
    You are an expert humanitarian assistance and disaster relief (HADR) analyst. You need to create a structured report about a real world disaster, while maintaining high contextual knowledge with the information provided to you.

    IMPORTANT: The current year is {current_year}. This is not a hypothetical or future scenario.

    CRITICAL: Your output MUST be valid JSON without any explanation text before or after. ONLY return the exact JSON object with ONLY the "sections" key.

    CONTEXT:
    - User Query: {query}
    - Disaster Info: {disaster_info}
    - Web Search Results: {search_results}
    - Wikipedia Information: {wikipedia_results}

    DATA FOR YOUR ANALYSIS:
    - Day-by-Day Analysis (sentiment is 0-100 scale, with pre-generated summaries): {day_by_day_data}
    - Sample of Top Tweets by Retweet Count: {top_tweets}
    - Overall Statistics: Total Tweets: {tweet_count}, Avg Sentiment: {avg_sentiment:.2f}, Positive: {positive_count}, Neutral: {neutral_count}, Negative: {negative_count}, Total Requests: {request_count}, Medical Requests: {medical_help_count}

    TASK:
    Analyze all the provided context and data to generate the following report sections. Synthesize the information into coherent, insightful narratives. The daily data, including the pre-generated summaries, shows trends; use it to describe how the situation and public mood evolved.

    YOUR JSON OUTPUT STRUCTURE:
    {{
        "sections": {{
            "Background": "A detailed, multi-paragraph (at least 4-5 paragraphs) summary of the disaster situation, using the web/wiki info. Provide as much context, background, and detail as possible.",
            "Tweet Overview": "A comprehensive overview (at least 2-3 paragraphs) of the tweet data, highlighting influential users, recurring themes, and notable tweet content from the top tweets. Include statistics and trends.",
            "Sentiment Overview": "A thorough, multi-paragraph (at least 3-4 paragraphs) summary of the sentiment analysis. Use the daily summaries and stats to describe trends over time. Mention the overall average score, the breakdown of positive/neutral/negative tweets, and discuss changes in sentiment and their possible causes.",
            "Results": "A detailed summary (at least 2-3 paragraphs) of the disaster's impact on the population, inferring from all data including the daily label counts. Discuss the scale and nature of the impact, referencing specific data points.",
            "Discussion": "A multi-paragraph (at least 4-5 paragraphs) discussion of ongoing response efforts and priorities, integrating web search results with needs identified from tweet labels. Analyze the effectiveness and gaps in the response.",
            "Recommendation": "A detailed set of critical actions (at least 4-5 paragraphs) for humanitarian aid based on your synthesis of all available information. Provide actionable, prioritized recommendations."
        }}
    }}
    """
    
    prompt = PromptTemplate.from_template(template)
    report_chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        def clean_tweet_text(raw_text):
            if not raw_text or not isinstance(raw_text, str): return ""
            try:
                text = raw_text.encode('utf-8').decode('unicode_escape', errors='replace')
                text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
                text = re.sub(r'http[s]?://\S+', '', text)
                text = re.sub(r'[\r\n\t]+', ' ', text)
                return re.sub(r'\s+', ' ', text).strip()
            except Exception: return str(raw_text)[:100]

        top_tweets = []
        if sentiment_analysis:
            sorted_tweets = sorted(sentiment_analysis, key=lambda x: int(x.get('retweets', 0)), reverse=True)
            for tweet in sorted_tweets[:10]:
                top_tweets.append({
                    "Username": tweet.get("username", ""), "Date": tweet.get("time", "").split("_")[0],
                    "Retweets": str(tweet.get("retweets", "0")), "Tweet": clean_tweet_text(tweet.get("text", ""))
                })
        
        day_by_day_data = _aggregate_results_by_date(sentiment_analysis)
        
        chain_input = {
            "current_year": datetime.now().year, "query": query,
            "disaster_info": json.dumps(disaster_info),
            "tweet_count": tweet_count, "avg_sentiment": avg_sentiment,
            "positive_count": positive_count, "neutral_count": neutral_count,
            "negative_count": negative_count, "request_count": request_count,
            "medical_help_count": medical_help_count,
            "search_results": "\n\n".join(search_results),
            "wikipedia_results": "\n\n".join(wikipedia_results),
            "top_tweets": json.dumps(top_tweets, indent=2),
            "day_by_day_data": json.dumps(day_by_day_data, indent=2)
        }
        
        report_json_str = report_chain.run(**chain_input)
        logger.info("LLM generated report sections.")
        
        repaired_json = extract_and_repair_json(report_json_str)
        if repaired_json and 'sections' in repaired_json:
            report_data = {"sections": repaired_json["sections"]}
        else:
            logger.error("Could not parse JSON from LLM, using fallback.")
            report_data = {"sections": {"Background": "Report generation failed."}}
        
        final_top_tweets = top_tweets
        while len(final_top_tweets) < 10:
            final_top_tweets.append({"Username": "N/A", "Date": "N/A", "Retweets": "0", "Tweet": "N/A"})

        final_day_by_day_data = []
        for day_data in day_by_day_data:
            elements_labels = [k for k, v in day_data['label_counts'].items() if k in elements_map and v > 0]
            impact_labels = [k for k, v in day_data['label_counts'].items() if k in impact_map and v > 0]
            request_labels = [k for k, v in day_data['label_counts'].items() if k in requests_map and v > 0]
            final_day_by_day_data.append({
                "Date": day_data.get("date", "N/A"),
                "Sentiment": round(day_data.get("sentiment_score", 50.0), 2),
                "Elements": ", ".join([l.replace('_', ' ').title() for l in elements_labels]),
                "Impact": ", ".join([l.replace('_', ' ').title() for l in impact_labels]),
                "Requests": ", ".join([l.replace('_', ' ').title() for l in request_labels]),
                "Summary": day_data.get("Summary", "No summary available.")
            })

        report_data["tweets"] = final_top_tweets[:10]
        report_data["details"] = final_day_by_day_data
        
        logger.info("Generated complete report data.")
        return report_data
        
    except Exception as e:
        logger.error(f"Error generating report data: {e}", exc_info=True)
        return {"sections": {"error": f"Failed to generate report data: {e}"}, "tweets": [], "details": []}

def generate_response(report_path: str, disaster_info: Dict[str, str], error: str = None) -> str:
    """Generate a response based on the results"""
    if error:
        return f"I encountered an error while generating the report: {error}"
    if not report_path:
        return "I was unable to generate a report. Please try again with a more specific query."
    disaster_type = disaster_info.get("disaster_type", "disaster")
    disaster_location = disaster_info.get("disaster_location", "affected area")
    return f"""I've generated a comprehensive HADR report about the {disaster_type} in {disaster_location}.
            The report includes:
            - Situation overview of the disaster
            - Analysis of social media sentiment
            - Assessment of affected population and needs
            - Information about ongoing response efforts
            You can download the full report to see detailed information and insights."""

# Define the graph nodes
def extract_info(state: AgentState) -> AgentState:
    """Extract disaster information from the query"""
    query = state["query"]
    logger.info(f"Extracting information from query: {query}")
    disaster_info = extract_disaster_info(query)
    return {**state, **disaster_info}

def gather_twitter_data(state: AgentState) -> AgentState:
    """Gather Twitter data for analysis"""
    query = state["query"]
    logger.info(f"Gathering Twitter data for query: {query}")
    disaster_info = {k: state[k] for k in ['disaster_type', 'disaster_location', 'disaster_date']}
    twitter_data = load_twitter_data(query, disaster_info)
    return {**state, "twitter_data": twitter_data}

def gather_web_info(state: AgentState) -> AgentState:
    """Gather information from web search"""
    query = state["query"]
    logger.info(f"Gathering web information for query: {query}")
    disaster_info = {k: state[k] for k in ['disaster_type', 'disaster_location', 'disaster_date']}
    search_results = search_web(query, disaster_info)
    return {**state, "search_results": search_results}

def gather_wikipedia_info(state: AgentState) -> AgentState:
    """Gather information from Wikipedia"""
    query = state["query"]
    logger.info(f"Gathering Wikipedia information for query: {query}")
    disaster_info = {k: state[k] for k in ['disaster_type', 'disaster_location', 'disaster_date']}
    wikipedia_results = search_wikipedia(query, disaster_info)
    return {**state, "wikipedia_results": wikipedia_results}

def analyze_tweet_sentiment(state: AgentState) -> AgentState:
    """Analyze sentiment of tweets using the sentiment analysis model"""
    logger.info("Analyzing tweet sentiment")
    twitter_data = state.get('twitter_data', [])
    if not twitter_data:
        logger.warning("No Twitter data to analyze.")
        return {**state, "sentiment_analysis": []}
    
    sentiment_results = analyze_sentiment(twitter_data)
    
    output_directory = OUTPUTS_DIR / "reports"
    output_directory.mkdir(exist_ok=True)
    query_name = state.get('query', 'general').replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_path = output_directory / f"{query_name}_{timestamp}_sentiment_scores.csv"

    pd.DataFrame(sentiment_results).to_csv(csv_file_path, index=False)
    logger.info(f"Sentiment analysis results saved to {csv_file_path}")

    return {**state, "sentiment_analysis": sentiment_results}

def build_report_data(state: AgentState) -> AgentState:
    """Build the report data"""
    query = state["query"]
    logger.info(f"Building report data for query: {query}")
    try:
        complete_report_data = generate_report_data(
            query,
            {"disaster_type": state["disaster_type"], "disaster_location": state["disaster_location"], "disaster_date": state["disaster_date"]},
            state["twitter_data"], state["search_results"], state["wikipedia_results"], state["sentiment_analysis"]
        )
        return {**state, "report_data": complete_report_data}
    except Exception as e:
        logger.error(f"Error in build_report_data: {e}", exc_info=True)
        return {**state, "report_data": {}, "error": f"Error building report data: {str(e)}"}

def create_final_report(state: AgentState) -> AgentState:
    """Create the final PDF report"""
    query = state["query"]
    logger.info(f"Creating final report for query: {query}")
    if not state.get("report_data") or "error" in state.get("report_data", {}).get("sections", {}):
        logger.error("Report data is missing or contains errors, skipping PDF generation.")
        return {**state, "report_path": "", "error": state.get("error", "Report data could not be generated.")}

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        disaster_type = state["disaster_type"] or "disaster"
        disaster_location = state["disaster_location"] or "location"
        base_name = f"{disaster_location}_{disaster_type}_{timestamp}".replace(" ", "_")
        
        json_file_path = OUTPUTS_DIR / f"{base_name}.json"
        with open(json_file_path, 'w') as f:
            json.dump(state["report_data"], f, indent=4)
        logger.info(f"Saved report data to {json_file_path}")
        
        pdf_file_path = OUTPUTS_DIR / f"{base_name}.pdf"
        report_path = generate_report(json_file_path=str(json_file_path), output_pdf_path=str(pdf_file_path))
        return {**state, "report_path": report_path}
    except Exception as e:
        logger.error(f"Error in create_final_report: {e}", exc_info=True)
        return {**state, "report_path": "", "error": f"Error creating final report: {str(e)}"}

def generate_final_response(state: AgentState) -> AgentState:
    """Generate the final response"""
    logger.info("Generating final response")
    response = generate_response(
        state.get("report_path", ""),
        {"disaster_type": state["disaster_type"], "disaster_location": state["disaster_location"]},
        state.get("error", None)
    )
    return {**state, "response": response}

def build_agent_workflow():
    """Build the agent workflow graph"""
    workflow = StateGraph(AgentState)
    workflow.add_node("extract_info", extract_info)
    workflow.add_node("gather_twitter_data", gather_twitter_data)
    workflow.add_node("gather_web_info", gather_web_info)
    workflow.add_node("gather_wikipedia_info", gather_wikipedia_info)
    workflow.add_node("analyze_tweet_sentiment", analyze_tweet_sentiment)
    workflow.add_node("build_report_data", build_report_data)
    workflow.add_node("create_final_report", create_final_report)
    workflow.add_node("generate_final_response", generate_final_response)
    
    workflow.set_entry_point("extract_info")
    workflow.add_edge("extract_info", "gather_twitter_data")
    workflow.add_edge("gather_twitter_data", "gather_web_info")
    workflow.add_edge("gather_web_info", "gather_wikipedia_info")
    workflow.add_edge("gather_wikipedia_info", "analyze_tweet_sentiment")
    workflow.add_edge("analyze_tweet_sentiment", "build_report_data")
    workflow.add_edge("build_report_data", "create_final_report")
    workflow.add_edge("create_final_report", "generate_final_response")
    workflow.add_edge("generate_final_response", END)
    
    return workflow.compile()

def process_query(query: str) -> Dict[str, Any]:
    logger.info(f"Processing query: {query}")
    default_state = {
        "query": query, "disaster_type": "", "disaster_location": "", "disaster_date": "",
        "twitter_data": [], "search_results": [], "wikipedia_results": [],
        "sentiment_analysis": [], "report_data": {}, "report_path": "",
        "response": "", "error": ""
    }
    try:
        app = build_agent_workflow()
        result = app.invoke(default_state)
        logger.info(f"Workflow completed with report path: {result.get('report_path', 'N/A')}")
        return result
    except Exception as e:
        logger.error(f"Error in process_query: {e}", exc_info=True)
        return {**default_state, "error": str(e), "response": "A critical error occurred."}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HADR Agent")
    parser.add_argument("--query", type=str, help="Query to process")
    args = parser.parse_args()
    if args.query:
        result = process_query(args.query)
        print(f"\n---AGENT FINAL RESPONSE---\n{result['response']}")
        if result.get('report_path'):
            print(f"Report Path: {result.get('report_path')}")
    else:
        print("Please provide a query with --query")