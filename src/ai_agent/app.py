"""
HADR Sentiment Analysis System - Streamlit Chat Interface

This Streamlit app provides a user-friendly interface for interacting with
the HADR Sentiment Analysis system based on LangGraph workflow.
"""
import streamlit as st
import os
import time
import sys
import logging
from pathlib import Path

# Ensure the module path is correct
SRC_DIR = Path(__file__).parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Import the HADR agent
from ai_agent.agents.hadr_agent import process_query

# Set up logging for the app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hadr_app")

# Set page configuration
st.set_page_config(
    page_title="HADR Sentiment Analysis",
    page_icon="🌍",
    layout="wide",
)

# Define paths
ASSETS_PATH = Path(__file__).parent / "assets"
OUTPUTS_PATH = ASSETS_PATH / "outputs"
OUTPUTS_PATH.mkdir(exist_ok=True, parents=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

if "reports" not in st.session_state:
    st.session_state.reports = []

# Helper function to check for report files
def check_for_reports():
    """Check the outputs folder for PDF reports."""
    reports = list(OUTPUTS_PATH.glob("*.pdf"))
    return [report for report in reports if report.is_file()]

# Helper function to get relative path
def get_relative_path(path):
    """Convert an absolute path to a relative path for display."""
    try:
        return path.relative_to(Path.cwd())
    except ValueError:
        return path

# Function to handle user queries
def handle_query(query):
    """Process the user query through the HADR agent."""
    st.session_state.processing = True
    logger.info(f"Processing query: {query}")

    # Create a placeholder for the "thinking" animation
    thinking_placeholder = st.empty()
    
    # Show "thinking" animation
    for i in range(3):
        thinking_placeholder.markdown(f"Processing{'.' * (i + 1)}")
        time.sleep(0.5)
    
    # Process the query
    try:
        # Before processing, check existing reports
        before_reports = check_for_reports()
        
        # Use the HADR agent to process the query
        result = process_query(query)
        
        # After processing, check for new reports
        after_reports = check_for_reports()
        new_reports = [r for r in after_reports if r not in before_reports]
        
        # Add new reports to session state
        for report in new_reports:
            st.session_state.reports.append(report)
            logger.info(f"New report generated: {report}")
        
        # Also check if there's a report path in the result
        if "report_path" in result and result["report_path"]:
            report_path = Path(result["report_path"])
            if report_path.exists() and report_path not in st.session_state.reports:
                st.session_state.reports.append(report_path)
                logger.info(f"Added report from result: {report_path}")
        
        thinking_placeholder.empty()
        return result.get("response", "I couldn't generate a response. Please try again.")
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        thinking_placeholder.empty()
        return f"Error processing query: {str(e)}"
    finally:
        st.session_state.processing = False

# Custom CSS for UI styling
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
    flex-direction: row;
    align-items: flex-start;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.assistant {
    background-color: #475063;
}
.chat-message .avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    object-fit: cover;
    margin-right: 1rem;
}
.chat-message .message {
    flex: 1;
}
.report-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #3a4049;
    margin-bottom: 1rem;
}
.stButton>button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# App title
st.title("🌍 HADR Sentiment Analysis System")
st.markdown("Ask questions or request reports about humanitarian assistance and disaster relief situations, including projected future events.")
st.markdown("*Example query: 'Generate a report about the Myanmar earthquake 2025'*")

# Sidebar for reports
with st.sidebar:
    st.header("📊 Generated Reports")
    
    # System Status Section
    with st.expander("System Information", expanded=False):
        st.markdown("### About This System")
        st.markdown("""
        This HADR (Humanitarian Assistance and Disaster Relief) Sentiment Analysis system uses:
        
        - **LangGraph** for agent workflow orchestration
        - **Twitter data analysis** for real-time information
        - **Search and Wikipedia integration** for comprehensive context
        - **Automated report generation** with actionable insights
        """)
        
        # Show available toolsets
        st.markdown("### Available Toolsets")
        tools = [
            "Twitter Data Analysis", 
            "DuckDuckGo Search", 
            "Wikipedia Research",
            "PDF Report Generation"
        ]
        for tool in tools:
            st.markdown(f"🔧 {tool}")
    
    if st.session_state.reports:
        for i, report_path in enumerate(st.session_state.reports):
            with st.expander(f"Report {i+1}: {report_path.name}"):
                st.markdown(f"📄 **Path**: `{get_relative_path(report_path)}`")
                st.markdown(f"📅 **Created**: {report_path.stat().st_mtime}")
                
                # Create a download button for the report
                with open(report_path, "rb") as file:
                    st.download_button(
                        label="Download Report",
                        data=file,
                        file_name=report_path.name,
                        mime="application/pdf"
                    )
    else:
        st.info("No reports generated yet. Ask for a report to see it here.")
    
    # Add a clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("What would you like to know about HADR situations?", disabled=st.session_state.processing):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)
    
    # Process the query and get response
    response = handle_query(user_input)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)
        
        # Check if new reports were generated during processing
        current_reports = check_for_reports()
        new_reports = [r for r in current_reports if r not in st.session_state.reports]
        
        if new_reports:
            st.markdown("---")
            st.markdown("**📊 New Report Generated:**")
            for report in new_reports:
                st.markdown(f"📄 [{report.name}]({get_relative_path(report)})")
                st.session_state.reports.append(report)

# Footer
st.markdown("---")
st.markdown("Powered by LangGraph. Built for HADR Sentiment Analysis.")
