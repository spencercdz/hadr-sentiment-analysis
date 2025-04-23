"""
Base classes and common utilities for HADR sentiment analysis agents
"""
import threading
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import sys

# Define core paths
BASE_DIR = Path(__file__).parent.parent  # ai_agent directory
ASSETS_DIR = BASE_DIR / "assets"  # Create consistent assets path
ASSETS_DIR.mkdir(exist_ok=True)  # Ensure assets directory exists

# Ensure standard asset subdirectories exist
TWITTER_DATA_DIR = ASSETS_DIR / "twitter_data"
TWITTER_DATA_DIR.mkdir(exist_ok=True)

TEMPLATES_DIR = ASSETS_DIR / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

# Ensure we can import from tools directory
TOOLS_DIR = Path(__file__).parent / "tools"
sys.path.append(str(TOOLS_DIR))

class AgentType(Enum):
    """Enum for different types of agents in the report generation workflow."""
    DATA_RETRIEVAL = "data_retrieval"
    DATA_SUMMARIZATION = "data_summarization"
    REPORT_BUILDER = "report_builder"
    MAIN = "main"

class ReportWorkflowState:
    """Shared state between agents in the report generation workflow."""
    def __init__(self):
        # Data paths
        self.raw_data_path: Optional[Path] = None
        self.output_path: Optional[Path] = None
        self.report_path: Optional[Path] = None
        
        # Processing state
        self.query: str = ""
        self.disaster_location: str = ""
        self.disaster_type: str = ""
        self.disaster_year: str = ""
        self.data_loaded: bool = False
        self.data_summarized: bool = False
        self.report_generated: bool = False
        
        # Data containers
        self.raw_data: Dict[str, Any] = {}
        self.summarized_data: Dict[str, Any] = {}
        self.report_data: Dict[str, Any] = {}
        
        # Agent references
        self.agents: Dict[AgentType, Any] = {}
        
        # Locks for thread safety
        self.locks: Dict[str, threading.Lock] = {
            "data": threading.Lock(),
            "report": threading.Lock()
        }
        
        # Template structure
        self.json_structure_template: str = ""
