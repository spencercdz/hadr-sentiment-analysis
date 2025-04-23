"""
Tool Registry - Manages and registers all available tools for agent use
"""
from langchain.agents import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
import json
import os
from pathlib import Path

# Tool imports
import sys
sys.path.append(str(Path(__file__).parent))
from tools.scraper import WebScraper  # Import your scraper.py as a module
import tools.build_report as report_builder

class ToolRegistry:
    def __init__(self):
        """Initialize the tool registry with all available tools."""
        self.tools = {}
        self.register_all_tools()
        
    def register_all_tools(self):
        """Register all available tools."""
        self._register_search_tools()
        self._register_report_tools()
        
    def _register_search_tools(self):
        """Register search related tools."""
        # DuckDuckGo Search
        search = DuckDuckGoSearchAPIWrapper()
        
        self.tools["web_search"] = Tool(
            name="web_search",
            func=search.run,
            description="Useful for searching the web for current information or pages. Input should be a search query."
        )
        
        # Wikipedia Search
        wikipedia = WikipediaAPIWrapper()
        
        self.tools["wikipedia_search"] = Tool(
            name="wikipedia_search",
            func=wikipedia.run,
            description="Useful for searching Wikipedia for factual information. Input should be a search query."
        )
    
    def _register_report_tools(self):
        """Register report generation related tools."""
        # Web Scraper
        scraper = WebScraper()
        
        self.tools["web_scraper"] = Tool(
            name="web_scraper",
            func=scraper.scrape_url,
            description="Scrape content from a webpage. Input should be a URL to scrape."
        )
        
        # JSON Data Creator
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
                assets_path = Path(__file__).parent / "assets"
                file_path = assets_path / "generated_data.json"
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data_dict, f, indent=4)
                
                return f"Successfully created report data at {file_path}"
            except Exception as e:
                return f"Error creating report data: {str(e)}"
        
        self.tools["create_report_data"] = Tool(
            name="create_report_data",
            func=create_report_data,
            description="Create a JSON data file for report generation. Input should be a dictionary with keys: 'sections', 'tweets', 'details'."
        )
        
        # Report Generator
        def generate_report():
            """Generate a PDF report using the build_report module."""
            try:
                # Update to use the generated data
                assets_path = Path(__file__).parent / "assets"
                data_file = assets_path / "generated_data.json"
                
                # Check if the file exists
                if not data_file.exists():
                    return "Error: Report data file does not exist. Please create report data first."
                
                # Generate the report
                report = report_builder.SentimentReport()
                report.add_data()  # Uses the JSON data by default
                report.generate_report()
                
                report_path = assets_path / "sentiment_report.pdf"
                return f"Successfully generated report at {report_path}"
            except Exception as e:
                return f"Error generating report: {str(e)}"
        
        self.tools["generate_report"] = Tool(
            name="generate_report",
            func=generate_report,
            description="Generate a PDF report using the prepared data. No input needed."
        )
    
    def get_all_tools(self):
        """Get all registered tools."""
        return list(self.tools.values())
    
    def get_search_tools(self):
        """Get search related tools."""
        return [self.tools["web_search"], self.tools["wikipedia_search"]]
    
    def get_report_tools(self):
        """Get report generation related tools."""
        return [
            self.tools["web_search"],
            self.tools["wikipedia_search"],
            self.tools["web_scraper"],
            self.tools["create_report_data"],
            self.tools["generate_report"]
        ]
