"""
HADR Sentiment Analysis - Modular Report Generation Workflow

This module provides a clean interface to the modular HADR report generation components:
- DataRetrievalAgent: Fetches raw data from Twitter CSV/JSON files
- DataSummarizationAgent: Summarizes data into structured JSON format
- ReportBuilderAgent: Builds the final PDF report
"""

import argparse
import sys
from pathlib import Path
import json

# Import the modular components
from .base import ReportWorkflowState, BASE_DIR, ASSETS_DIR, TWITTER_DATA_DIR
from .data_retrieval_agent import DataRetrievalAgent
from .data_summarization_agent import DataSummarizationAgent
from .report_builder_agent import ReportBuilderAgent

# Using updated imports to avoid deprecation warnings
from langchain_community.llms import Ollama


class HADRReportWorkflow:
    """Main coordinator for HADR report generation using modular components."""
    
    def __init__(self, debug=False):
        """Initialize the workflow with all required components."""
        print("\n===== INITIALIZING HADR REPORT WORKFLOW =====")
        
        # Initialize the LLM first
        self.llm = self._init_ollama_model()
        
        # Set up consistent paths
        self.base_path = BASE_DIR
        self.assets_path = ASSETS_DIR
        self.raw_data_path = TWITTER_DATA_DIR
        
        # Print the paths for verification
        print(f"Base path: {self.base_path}")
        print(f"Assets path: {self.assets_path}")
        print(f"Raw data path: {self.raw_data_path}")
        
        # Flag for debug mode
        self.debug = debug
        if self.debug:
            print("Debug mode is enabled")
        
        # Initialize workflow state
        self.workflow_state = ReportWorkflowState()
        
        # Initialize sub-agents
        self._init_agents()
    
    def _init_ollama_model(self):
        """Initialize an Ollama model for local inference.
        
        This connects to a locally running Ollama server for more efficient deployment.
        Incorporates optimized batch processing from TunedLLM improvements.
        """
        print("Connecting to Ollama with model deepseek-r1:8b...")
        try:
            # Create Ollama LLM instance with optimized parameters
            # Applying the optimizations from the TunedLLM class
            ollama_llm = Ollama(
                model="deepseek-r1:8b",
                temperature=0.3,  # Lower temperature for more consistent outputs
                num_ctx=4096,     # Larger context window
                batch_size=16,    # Optimized batch size for better performance
                repeat_penalty=1.1,  # Prevent repetition
                callback_manager=None
            )
            return ollama_llm
        except Exception as e:
            print(f"Error initializing Ollama: {str(e)}")
            print("Falling back to default LLM configuration")
            return Ollama(model="deepseek-r1:8b")
    
    def _init_agents(self):
        """Initialize the specialized sub-agents for the report workflow."""
        # Initialize the data retrieval agent
        self.data_retrieval_agent = DataRetrievalAgent(
            llm=self.llm,
            workflow_state=self.workflow_state,
            raw_data_path=self.raw_data_path
        )
        
        # Initialize the data summarization agent
        self.data_summarization_agent = DataSummarizationAgent(
            llm=self.llm,
            workflow_state=self.workflow_state,
            json_structure_template=""  # Will use default template
        )
        
        # Initialize the report builder agent
        self.report_builder_agent = ReportBuilderAgent(
            llm=self.llm,
            workflow_state=self.workflow_state,
            assets_path=self.assets_path
        )
    
    def process_query(self, query):
        """Process a report generation request through the modular workflow."""
        try:
            print(f"\nProcessing query: {query}")
            
            # Store the original query
            self.workflow_state.query = query
            
            # Reset workflow state for new query
            self.workflow_state.data_loaded = False
            self.workflow_state.data_summarized = False
            self.workflow_state.report_generated = False
            
            # Parse disaster information from query
            self._parse_disaster_info(query)
            
            # 1. Retrieve data
            print("\n=== Step 1: Data Retrieval ===")
            data_result = self.data_retrieval_agent.retrieve_data()
            
            if not data_result["success"]:
                return {"success": False, "message": f"Failed to retrieve data: {data_result['message']}"}
            
            # 2. Summarize data
            print("\n=== Step 2: Data Summarization ===")
            summary_result = self.data_summarization_agent.summarize_data()
            
            if not summary_result["success"]:
                return {"success": False, "message": f"Failed to summarize data: {summary_result['message']}"}
            
            # 3. Build report
            print("\n=== Step 3: Report Generation ===")
            report_result = self.report_builder_agent.build_report()
            
            if report_result["success"]:
                return {
                    "success": True, 
                    "message": "Report generated successfully",
                    "report_path": str(self.workflow_state.report_path),
                    "json_path": str(self.workflow_state.output_path),
                    "report_generated": True,
                    "result": f"Generated report about {self.workflow_state.disaster_type} in {self.workflow_state.disaster_location}. PDF available at {self.workflow_state.report_path}."
                }
            else:
                return report_result
        except Exception as e:
            import traceback
            print(f"Error in process_query: {str(e)}")
            print(traceback.format_exc())
            return {"success": False, "message": f"Error processing query: {str(e)}"}
    
    def _parse_disaster_info(self, query):
        """Parse the disaster information from the query."""
        # Extract disaster type
        disaster_types = ["earthquake", "flood", "hurricane", "typhoon", "tsunami", 
                         "wildfire", "tornado", "drought", "cyclone", "landslide"]
        query_lower = query.lower()
        
        for disaster_type in disaster_types:
            if disaster_type in query_lower:
                self.workflow_state.disaster_type = disaster_type
                break
        
        # Extract year if present
        import re
        year_match = re.search(r"20[2-3][0-9]", query)
        if year_match:
            self.workflow_state.disaster_year = year_match.group(0)
        
        # Extract location (simplistic approach)
        # Try to find the location before the disaster type
        if self.workflow_state.disaster_type:
            parts = query_lower.split(self.workflow_state.disaster_type)
            if len(parts) > 1 and parts[0].strip():
                words = parts[0].strip().split()
                if words:
                    # Use the last word before the disaster type as location
                    self.workflow_state.disaster_location = words[-1].capitalize()
        
        # If no location found yet, try to extract any capitalized words
        if not self.workflow_state.disaster_location:
            capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
            if capitalized_words:
                self.workflow_state.disaster_location = capitalized_words[0]


# Create a function that can be called from workflows.py
def get_hadr_report_workflow(debug=False):
    """Get or create a HADRReportWorkflow instance."""
    # Could be extended to use a singleton pattern if needed
    return HADRReportWorkflow(debug=debug)


# Main execution point for standalone use
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="HADR Report Generator")
    parser.add_argument("--query", type=str, help="Disaster query to process", default="")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Create the workflow
    workflow = HADRReportWorkflow(debug=args.debug)
    
    # If query provided via command line, process it
    if args.query:
        result = workflow.process_query(args.query)
        
        if result["success"]:
            print(f"\nSuccessfully generated report: {result['report_path']}")
            print(f"JSON data saved to: {result['json_path']}")
        else:
            print(f"\nFailed to generate report: {result['message']}")
    else:
        # Interactive mode
        print("\nHADR Report Generator Interactive Mode")
        print("Enter 'quit' or 'exit' to end the program")
        
        while True:
            query = input("\nEnter disaster query (e.g., 'Turkey earthquake 2023'): ")
            
            if query.lower() in ["quit", "exit"]:
                print("Exiting...")
                break
                
            if not query.strip():
                print("Please enter a valid query.")
                continue
                
            result = workflow.process_query(query)
            
            if result["success"]:
                print(f"\nSuccessfully generated report: {result['report_path']}")
                print(f"JSON data saved to: {result['json_path']}")
            else:
                print(f"\nFailed to generate report: {result['message']}")
