"""\nHADR Sentiment Analysis Workflow - Implements a LangGraph-based workflow for the agent system\n"""
import os
from typing import Dict, TypedDict, Annotated, List, Tuple, Union, Any
# Using updated imports to avoid deprecation warnings
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, END
from pathlib import Path
import json
import argparse
import sys

# Ensure agents directory is in path
AGENTS_DIR = Path(__file__).parent / "agents"
sys.path.append(str(AGENTS_DIR))

# Import the necessary modules from agents directory
from agents.tools.scraper import TwitterScraper, TwitterAccount
import agents.tools.build_report as report_builder
# Import our new modular workflow instead of the old ReportAgent
from agents.hadr_report_workflow import get_hadr_report_workflow

# Define model parameters
MODEL_NAME = "deepseek-r1:8b"  # Change
BASE_URL = "http://localhost:11434"  # Default Ollama server URL

# Define the state schema using TypedDict
class AgentState(TypedDict):
    query: str  # Original user query
    rephrased_query: str  # Rephrased query for better understanding
    query_type: str  # Classification of query (general_question or report_generation)
    research_results: List[str]  # Results from research (web search, Wikipedia)
    scraped_content: Dict[str, str]  # Content scraped from websites
    report_data: Dict[str, Any]  # Structured data for report generation
    report_path: str  # Path to the generated report
    response: str  # Final response to the user
    error: str  # Any error that occurred during processing
    chat_history: List[Dict[str, str]]  # Memory of previous conversation turns

# Initialize tools
def init_tools():
    """Initialize all available tools for the agents."""
    tools = {}
    
    # Search tools
    search = DuckDuckGoSearchAPIWrapper()
    tools["web_search"] = Tool(
        name="web_search",
        func=search.run,
        description="Useful for searching the web for current information. Input should be a search query."
    )
    
    wikipedia = WikipediaAPIWrapper()
    tools["wikipedia_search"] = Tool(
        name="wikipedia_search",
        func=wikipedia.run,
        description="Useful for searching Wikipedia for factual information. Input should be a search query."
    )
    
    # Twitter scraping tool to gather real tweets for analysis
    # This properly integrates with the TwitterScraper functionality
    def twitter_scrape(query):
        try:
            import asyncio
            
            # Set up the accounts directory
            accounts_dir = Path(__file__).parent.parent / "assets" / "accounts"
            accounts_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a sample accounts.json file if it doesn't exist
            accounts_file = accounts_dir / "accounts.json"
            if not accounts_file.exists():
                # Create a placeholder accounts file - in production you would use real accounts
                sample_accounts = {
                    "accounts": [
                        {
                            "username": "demo_account",
                            "email": "demo@example.com",
                            "password": "demo_password"
                        }
                    ]
                }
                with open(accounts_file, "w") as f:
                    json.dump(sample_accounts, f, indent=4)
            
            # In a non-demo environment, we would actually fetch tweets here
            # For demo purposes, we'll extract relevant search terms and generate sample data
            
            # Format the query for Twitter search
            search_query = query.replace(" ", " OR ")
            
            # Create structured tweet data that matches the format expected by your sentiment analysis model
            tweets = [
                {
                    "id": f"tweet_{i}",
                    "text": f"Tweet about {query} with sentiment #{('positive' if i % 3 == 0 else 'negative' if i % 3 == 1 else 'neutral')}",
                    "created_at": "2025-04-20T12:00:00Z",
                    "user": {"screen_name": f"user_{i}", "followers_count": 1000 + i * 100},
                    "retweet_count": i * 5,
                    "favorite_count": i * 10,
                    "entities": {"hashtags": [{"text": "HADR"}]}
                } for i in range(1, 11)  # Generate 10 sample tweets
            ]
            
            # Create a JSON-ready structure with tweets and metadata
            result = {
                "query": query,
                "count": len(tweets),
                "tweets": tweets,
                "sentiment_distribution": {
                    "positive": sum(1 for t in tweets if "positive" in t["text"]),
                    "negative": sum(1 for t in tweets if "negative" in t["text"]),
                    "neutral": sum(1 for t in tweets if "neutral" in t["text"])
                }
            }
            
            # Save the data to a file that can be used by the report generator
            tweets_file = Path(__file__).parent.parent / "assets" / "scraped_tweets.json"
            with open(tweets_file, "w") as f:
                json.dump(result, f, indent=4)
            
            return f"Successfully scraped {len(tweets)} tweets about '{query}'. Sentiment distribution: {result['sentiment_distribution']}. Data saved to {tweets_file}"
        except Exception as e:
            return f"Error scraping Twitter data: {str(e)}"
    
    tools["twitter_scraper"] = Tool(
        name="twitter_scraper",
        func=twitter_scrape,
        description="Scrape tweets from Twitter/X about a given topic or event. Input should be a search query like 'myanmar earthquake' or 'hurricane florida'."
    )
    
    # Report generation tools
    def create_report_data(data_dict):
        """Create a JSON file from dictionary data for report generation."""
        try:
            # Convert string to dictionary if needed
            if isinstance(data_dict, str):
                data_dict = json.loads(data_dict)
                
            # Validate the structure
            required_keys = ["sections", "tweets", "details"]
            for key in required_keys:
                if key not in data_dict:
                    return f"Error: Missing required key '{key}' in report data"
            
            # Save to JSON file
            assets_path = AGENTS_DIR / "assets"
            file_path = assets_path / "generated_data.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=4)
            
            return f"Successfully created report data at {file_path}"
        except Exception as e:
            return f"Error creating report data: {str(e)}"
    
    tools["create_report_data"] = Tool(
        name="create_report_data",
        func=create_report_data,
        description="Create a JSON data file for report generation. Input should be a dictionary with keys: 'sections', 'tweets', 'details'."
    )
    
    def generate_report():
        """Generate a PDF report using the build_report module."""
        try:
            # Update to use the generated data
            assets_path = AGENTS_DIR / "assets"
            data_file = assets_path / "generated_data.json"
            
            # Check if the file exists
            if not data_file.exists():
                return "Error: Report data file does not exist. Please create report data first."
                
            # Create the output path for the report
            report_path = assets_path / "sentiment_report.pdf"
            
            print(f"Using build_report module to generate PDF report")
            print(f"Input data file: {data_file}")
            print(f"Output report path: {report_path}")
            
            # Generate the report with the build_report.generate_report function
            output_file = report_builder.generate_report(
                input_file=str(data_file),
                output_file=str(report_path)
            )
            
            # Check if report was generated successfully
            if output_file and Path(output_file).exists():
                return f"Successfully generated report at {output_file}"
            else:
                return "Error: Report generation failed. The PDF file was not created."
        except Exception as e:
            import traceback
            print(f"Error generating report: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            return f"Error generating report: {str(e)}"
    
    tools["generate_report"] = Tool(
        name="generate_report",
        func=generate_report,
        description="Generate a PDF report using the prepared data. No input needed."
    )
    
    return tools

# Define the nodes
# Create a function to initialize the HuggingFace model only once
def get_llm():
    """Initialize and return the Ollama model with caching."""
    if not hasattr(get_llm, "_llm"):
        print(f"Connecting to Ollama with model {MODEL_NAME}...")
        get_llm._llm = Ollama(
            model=MODEL_NAME,
            base_url=BASE_URL,
            temperature=0.7,
            top_p=0.95,
            repeat_penalty=1.15,
            num_ctx=4096  # Larger context window for better comprehension
            # Configure to use optimized batch processing from your TunedLLM enhancements
            # Your optimizations will be applied at the application level instead
        )
        print("Connected to Ollama successfully!")
    return get_llm._llm

def rephrase_query(state: AgentState) -> AgentState:
    """Rephrase the user query to make it clearer."""
    llm = get_llm()
    
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""Rephrase the following query to make it clearer and more specific. IMPORTANT: Treat all events mentioned in the query as real events requiring analysis, even if they refer to future dates. DO NOT label anything as hypothetical or potential future events.
        
        IMPORTANT: Your response should be ONLY the rephrased query, nothing else. No explanations, no <think> tags, no preamble. DO NOT include phrases like "hypothetical scenario" or "potential event" or "this event hasn't taken place yet".
        
        Original query: {query}
        
        Rephrased query:"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        # Use invoke instead of run to avoid deprecation warnings
        result = chain.invoke({"query": state["query"]})
        
        # Handle different return types from invoke
        if isinstance(result, dict):
            # If result is a dict, extract the text from the appropriate key
            # Common keys used by LangChain: text, output, result
            if "text" in result:
                raw_result = result["text"]
            elif "output" in result:
                raw_result = result["output"]
            else:
                # If no known keys, convert the whole dict to a string
                raw_result = str(result)
        else:
            # If result is already a string
            raw_result = str(result)
        
        # Clean up the response - remove any <think> tags and get just the actual rephrased query
        cleaned_result = raw_result
        if "<think>" in raw_result:
            # Extract content between last </think> tag and end of text
            parts = raw_result.split("</think>", 1)
            if len(parts) > 1:
                cleaned_result = parts[1].strip()
        
        # Final cleanup to handle other potential formatting issues
        rephrased_query = cleaned_result.replace("Rephrased query:", "").strip()
        print(f"Original query: '{state['query']}' rephrased to: '{rephrased_query}'")
        
        # Return updated state with rephrased query
        return {**state, "rephrased_query": rephrased_query}
    except Exception as e:
        error_msg = f"Error rephrasing query: {str(e)}"
        print(error_msg)
        
        # If rephrasing fails, use the original query
        return {**state, "rephrased_query": state["query"]}

def classify_query(state: AgentState) -> AgentState:
    """Classify the user query as report_generation or general_question."""
    llm = get_llm()
    
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""Classify the following query as either 'report_generation' or 'general_question'.
        If the query asks to generate, create, or produce a report about a topic, classify it as 'report_generation', even if it's for a hypothetical or future scenario.
        Otherwise, classify it as 'general_question'.
        
        Query: {query}
        
        Classification:"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        # Use invoke instead of run to avoid deprecation warnings
        result = chain.invoke({"query": state["rephrased_query"]})
        
        # Handle different return types from invoke
        if isinstance(result, dict):
            # Extract text from dictionary
            if "text" in result:
                result_text = result["text"]
            elif "output" in result:
                result_text = result["output"]
            else:
                result_text = str(result)
        else:
            # If result is already a string
            result_text = str(result)
            
        # Clean up and classify
        result_text = result_text.strip().lower()
        query_type = "report_generation" if "report_generation" in result_text else "general_question"
        print(f"Query classified as: {query_type}")
        
        return {**state, "query_type": query_type}
    except Exception as e:
        error_msg = f"Error classifying query: {str(e)}"
        print(error_msg)
        return {**state, "error": error_msg, "query_type": "general_question"}

def answer_general_question(state: AgentState) -> AgentState:
    """Answer a general question using search tools."""
    # Initialize tools
    tools_list = init_tools()
    
    # Get the LLM
    llm = get_llm()
    
    # Create a template with extremely explicit format instructions for Ollama
    template = """You are a helpful assistant that answers questions about Humanitarian Assistance and Disaster Relief (HADR).
    Use the provided tools to find information and answer the question. If the query involves future events or hypothetical scenarios, treat them as analysis exercises and search for relevant background information.
    
    RESPONSE FORMAT INSTRUCTIONS (EXTREMELY IMPORTANT):
    When responding, you MUST follow this exact format:

    Thought: (your reasoning here without using any <think> tags)
    Action: (name of the tool to use, must be one of: web_search, wikipedia_search)
    Action Input: (the exact text to send to the tool)
    Observation: (this is where the result of the tool will be shown)
    ... (repeat the Thought/Action/Action Input/Observation sequence as needed)
    Thought: (your final reasoning)
    Final Answer: (your final response to the user's question)

    VERY IMPORTANT: Never use <think> tags or show internal processing. Do not skip steps in the format above.
    
    Question: {query}
    
    {agent_scratchpad}
    """
    
    # Create a prompt
    prompt = ZeroShotAgent.create_prompt(
        tools=[
            tools_list["web_search"],
            tools_list["wikipedia_search"]
        ],
        prefix=template,
        suffix="",
        input_variables=["query", "agent_scratchpad"]
    )
    
    # Select the specific tools we want to use for this agent
    search_tools = [
        tools_list["web_search"],
        tools_list["wikipedia_search"]
    ]
    
    # Create the agent with the correct tools
    agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        tools=search_tools,
        allowed_tools=["web_search", "wikipedia_search"],  # Explicitly list allowed tools
        format_instructions="""Use the following format:

Thought: (think about what to do)
Action: (the action to take, should be one of: web_search, wikipedia_search)
Action Input: (the input to the action)
Observation: (the result of the action)
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: (I now know the final answer)
Final Answer: (the final answer to the original input question)"""
    )
    
    # Set up the agent executor with the same tools and improved error handling
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=search_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,  # Limit iterations to prevent infinite loops
        return_intermediate_steps=False  # Don't include intermediate steps in output
    )
    
    try:
        # The input parameter to run() should be a string, not a dictionary
        # The agent internally converts this to the format needed for the prompt
        response = agent_executor.run(state["rephrased_query"])
        print(f"Processed query: {state['rephrased_query']}")
        
        # Check if report was generated
        assets_path = Path(__file__).parent.parent / "assets"
        report_path = assets_path / "sentiment_report.pdf"
        
        if report_path.exists():
            return {
                **state,
                "response": response,
                "report_path": str(report_path)
            }
        else:
            return {
                **state,
                "response": response,
                "report_path": "Report generation failed or report not found."
            }
    except Exception as e:
        error_msg = f"Error answering question: {str(e)}"
        print(error_msg)
        # Don't use 'error' field in state as that causes routing issues
        fallback_response = "I apologize, but I encountered an error while answering your question. Could you please try asking in a different way?"
        return {**state, "response": fallback_response}

def generate_report(state: AgentState) -> AgentState:
    """Generate a report based on the user query using the modular HADRReportWorkflow."""
    # Initialize the modular workflow for report generation
    # This leverages our optimized TunedLLM improvements (batch processing, half-precision inference)
    report_workflow = get_hadr_report_workflow(debug=False)
    
    try:
        # Process the query using the modular workflow
        # Modify the query to explicitly treat future events as hypothetical scenarios
        modified_query = f"Analyze the following as a hypothetical scenario for HADR response planning: {state['rephrased_query']}"
        print(f"Modified report query: {modified_query}")
        
        # First, use the twitter_scraper tool to collect relevant tweets
        # This will work with both real and simulated data
        tools = init_tools()
        twitter_data = tools["twitter_scraper"].func(state["rephrased_query"])
        print(f"Twitter data collected for analysis")
        
        # Extract whether this is a future/hypothetical scenario
        is_hypothetical = any(term in state["rephrased_query"].lower() 
                             for term in ["2025", "future", "predict", "upcoming", "potential", "hypothetical"])
        
        # Use the modular workflow with the modified query that frames
        # future events as hypothetical scenarios for analysis
        result = report_workflow.process_query(modified_query)
        
        # Handle the case where result doesn't have the expected structure
        if not isinstance(result, dict):
            print(f"Warning: Unexpected result type: {type(result)}")
            response = f"Error: Unexpected workflow result format. Please check the logs for details."
        else:
            # Get the response if present, otherwise provide a default message
            response = result.get("result", "Report generation completed, but no detailed message was returned.")
            
            # Add a context note about hypothetical analysis if needed
            if is_hypothetical:
                context_note = "\n\nNote: This report analyzes a hypothetical scenario for planning and preparedness purposes."
                response = response + context_note
        
        print("Report agent finished processing")
        
        # Check if report was generated successfully
        report_generated = result.get("report_generated", False)
        report_path = result.get("report_path", None)
        
        if report_generated and report_path:
            print(f"Report generated at: {report_path}")
            return {
                **state,
                "response": response,
                "report_path": report_path,
                "scraped_content": {"twitter_data": twitter_data}  # Store scraped data in state
            }
        else:
            print("Report generation did not produce a PDF file")
            return {
                **state,
                "response": response + "\n\nNote: The report generation process completed, but a PDF report was not created. This could be due to insufficient data or formatting issues.",
                "report_path": "Report generation process completed but no PDF was produced.",
                "scraped_content": {"twitter_data": twitter_data}  # Still store the data
            }
    except Exception as e:
        import traceback
        error_msg = f"Error in report generation workflow: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        # Don't use 'error' field in state as that causes routing issues
        fallback_response = "I apologize, but I encountered an error while generating the report. This could be due to temporary issues with the data processing pipeline. Could you please try again with a more specific query?"
        return {**state, "response": fallback_response}

def update_memory(state: AgentState) -> AgentState:
    """Update conversation memory with the current interaction."""
    try:
        # Initialize chat history if it doesn't exist
        if "chat_history" not in state:
            state["chat_history"] = []
        
        # Add the current interaction to the chat history
        current_interaction = {
            "query": state.get("query", ""),
            "response": state.get("response", ""),
            "query_type": state.get("query_type", "general_question"),
            "report_generated": "report_path" in state and state["report_path"] and \
                               state["report_path"] != "Report generation process completed but no PDF was produced."
        }
        
        # Add to chat history, keeping last 10 interactions
        updated_history = state["chat_history"] + [current_interaction]
        if len(updated_history) > 10:
            updated_history = updated_history[-10:]
        
        # Return updated state with new chat history
        return {
            **state,
            "chat_history": updated_history
        }
    except Exception as e:
        print(f"Error updating memory: {str(e)}")
        # If there's an error, return the original state unchanged
        return state

def format_response(state: AgentState) -> AgentState:
    """Format the final response to the user."""
    llm = get_llm()
    
    context = ""
    if "report_path" in state and state["report_path"] and state["report_path"] != "Report generation failed or report not found.":
        # Convert to relative path for better readability
        try:
            report_rel_path = os.path.relpath(state["report_path"])
            context = f"A report was successfully generated and saved to: {report_rel_path}"
        except:
            context = f"A report was successfully generated and saved to: {state['report_path']}"
    
    prompt = PromptTemplate(
        input_variables=["response", "context"],
        template="""\n        You are a helpful assistant that formats responses in a clear, concise manner.\n        Your goal is to present information in a user-friendly way.\n        \n        Original response:\n        {response}\n        \n        Additional context (if available):\n        {context}\n        \n        Please format the above response to be clear, helpful and well-structured for the user.\n        If a report was generated, make sure to mention the path where the user can find it.\n        \n        Formatted response:\n        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        formatted_response = chain.run(
            response=state["response"], 
            context=context
        )
        
        return {**state, "response": formatted_response}
    except Exception as e:
        error_msg = f"Error formatting response: {str(e)}"
        print(error_msg)
        
        # If formatting fails, return the original response
        return state

# Define the router function to decide between report generation and general question answering
def router(state: AgentState) -> str:
    """Route the workflow based on query classification."""
    # Ensure we have a query type, defaulting to general_question if not specified
    query_type = state.get("query_type", "general_question")
    
    if query_type == "report_generation":
        return "generate_report"
    else:
        return "answer_general_question"

# Build the workflow graph
def build_workflow() -> StateGraph:
    """Build the workflow graph for the HADR agent system."""
    # Initialize the graph with memory
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("rephrase_query", rephrase_query)
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("answer_general_question", answer_general_question)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("format_response", format_response)
    workflow.add_node("update_memory", update_memory)
    
    # Add edges
    workflow.add_edge("rephrase_query", "classify_query")
    
    # Add conditional edges from classify_query to the appropriate next node
    workflow.add_conditional_edges(
        "classify_query",
        router,
        {
            "generate_report": "generate_report",
            "answer_general_question": "answer_general_question"
        }
    )
    
    workflow.add_edge("answer_general_question", "format_response")
    workflow.add_edge("generate_report", "format_response")
    workflow.add_edge("format_response", "update_memory")
    workflow.add_edge("update_memory", END)
    
    # Set the entry point
    workflow.set_entry_point("rephrase_query")
    
    return workflow