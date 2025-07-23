"""
HADR Sentiment Analysis System - Streamlit Chat Interface

This Streamlit app provides a user-friendly interface for interacting with
the HADR Sentiment Analysis system based on LangGraph workflow.
"""
import streamlit as st
import time
import sys
import logging
from pathlib import Path
import time

# Ensure the module path is correct
SRC_DIR = Path(__file__).parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from ai_agent.agents.hadr_agent import process_query_stream, warmup_ollama

# Import the HADR agent

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
    # Load all existing PDF reports from outputs folder
    reports = list(OUTPUTS_PATH.glob("*.pdf"))
    st.session_state.reports = [r for r in reports if r.is_file()]

if "csvs" not in st.session_state:
    # Load all existing CSVs from outputs/reports folder
    csv_dir = OUTPUTS_PATH / "reports"
    if csv_dir.exists():
        csvs = list(csv_dir.glob("*.csv"))
        st.session_state.csvs = [c for c in csvs if c.is_file()]
    else:
        st.session_state.csvs = []

# Helper function to check for report files
def check_for_reports():
    """Check the outputs folder for PDF reports."""
    reports = list(OUTPUTS_PATH.glob("*.pdf"))
    return [report for report in reports if report.is_file()]

# Helper function to check for CSV files
def check_for_csvs():
    """Check the outputs/reports folder for CSV prediction files."""
    csv_dir = OUTPUTS_PATH / "reports"
    if csv_dir.exists():
        csvs = list(csv_dir.glob("*.csv"))
        return [c for c in csvs if c.is_file()]
    return []

# Helper function to get relative path
def get_relative_path(path):
    """Convert an absolute path to a relative path for display."""
    try:
        return path.relative_to(Path.cwd())
    except ValueError:
        return path

# Modern animated spinner SVG (fallback to ⏳ if not supported)
SPINNER_SVG = '''<svg width="18" height="18" viewBox="0 0 38 38" xmlns="http://www.w3.org/2000/svg" stroke="#2196f3"><g fill="none" fill-rule="evenodd"><g transform="translate(1 1)" stroke-width="4"><circle stroke-opacity=".3" cx="18" cy="18" r="18"/><path d="M36 18c0-9.94-8.06-18-18-18"><animateTransform attributeName="transform" type="rotate" from="0 18 18" to="360 18 18" dur="0.9s" repeatCount="indefinite"/></path></g></g></svg>'''

# Stepper/progress UI for the agent workflow

# Warm up Ollama model at app startup (only once per session)
@st.cache_resource
def do_warmup():
    warmup_ollama()

with st.spinner('Warming up Ollama model. This may take awhile.'):
    do_warmup()

def handle_query_stream(query):
    steps = [
        ("extract_info", "Extracting disaster information"),
        ("gather_twitter_data", "Gathering Twitter data"),
        ("gather_web_info", "Searching the web"),
        ("gather_wikipedia_info", "Retrieving Wikipedia info"),
        ("analyze_tweet_sentiment", "Analyzing tweet sentiment"),
        ("build_report_data", "Building report data"),
        ("create_final_report", "Generating PDF report"),
        ("generate_final_response", "Finalizing response")
    ]
    step_idx = {name: i for i, (name, _) in enumerate(steps)}
    progress_card = st.empty()
    timer_placeholder = st.empty()
    logs = []
    final_response = None
    final_state = None
    step_start_times = [None] * len(steps)
    step_durations = [None] * len(steps)
    start_time = time.time()
    idx = 0

    # Get the generator
    gen = process_query_stream(query)
    current_result = next(gen)
    step = current_result["step"]
    log_msg = current_result["logs"].split("\n")[-1]
    logs.append(log_msg)
    idx = step_idx[step]
    step_start_times[idx] = time.time()

    while True:
        now = time.time()
        elapsed = now - start_time

        def step_icon(i, name):
            if i < idx:
                return '<span style="color:#4caf50;font-size:1.2em;">✅</span>'
            elif i == idx:
                return f'<span style="display:inline-block;vertical-align:middle;">{SPINNER_SVG}</span>'
            else:
                return '<span style="color:#bbb;font-size:1.2em;">⬜</span>'

        progress_card.markdown(f"""
        <div style='background:#23272e;padding:1.5rem 1.5rem 1rem 1.5rem;border-radius:1rem;margin-bottom:2rem;box-shadow:0 2px 8px #0002;'>
            <div style='font-size:1.1rem;margin-bottom:0.7rem;color:#fff;'><b>⏱️ Elapsed time:</b> {elapsed:.1f} seconds</div>
            <ol style='list-style-type:none;padding-left:0;margin:0;'>
                {''.join([
                    f'<li style="display:flex;align-items:center;margin-bottom:0.5rem;'
                    + (f'background:#1a1d22;border-radius:0.5rem;padding:0.4rem 0.8rem;' if i == idx else '')
                    + '">' + step_icon(i, name)
                    + f'<span style="margin-left:0.7em;font-weight:{'bold' if i == idx else 'normal'};color:{'#4caf50' if i < idx else ('#2196f3' if i == idx else '#bbb')};">{desc}</span>'
                    + (f'<span style="font-size:0.9em;color:#bbb;margin-left:0.7em;">({step_durations[i]:.1f}s)</span>' if step_durations[i] is not None and i < idx else '')
                    + '</li>'
                    for i, (name, desc) in enumerate(steps)
                ])}
            </ol>
            <hr style='border:none;border-top:1px solid #333;margin:1.2rem 0 0.7rem 0;' />
            <div style='color:#fff;font-size:1.05rem;'><b>Latest step:</b> {log_msg}</div>
        </div>
        """, unsafe_allow_html=True)

        # Try to get the next step if available
        try:
            next_result = next(gen)
            # Mark duration for the previous step
            prev_idx = idx
            idx = step_idx[next_result["step"]]
            now = time.time()
            if step_start_times[idx] is None:
                step_start_times[idx] = now
                if prev_idx is not None and step_start_times[prev_idx] is not None and step_durations[prev_idx] is None:
                    step_durations[prev_idx] = now - step_start_times[prev_idx]
            log_msg = next_result["logs"].split("\n")[-1]
            logs.append(log_msg)
            current_result = next_result
            if next_result["step"] == "generate_final_response":
                # Mark the last step's duration
                if step_start_times[idx] is not None and step_durations[idx] is None:
                    step_durations[idx] = time.time() - step_start_times[idx]
                final_response = next_result["state"].get("response", "")
                final_state = next_result["state"]
                break
        except StopIteration:
            break

        time.sleep(0.1)  # Update UI every 0.1s

    progress_card.empty()
    timer_placeholder.empty()
    return final_response, "\n".join(logs), final_state

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
st.title("HADR Sentiment Analysis System")
st.markdown("Ask questions or request reports about humanitarian assistance and disaster relief situations.")
st.markdown("*Example query: 'myanmar earthquake 2025'*")

# Sidebar for reports
with st.sidebar:
    # System Status Section
    st.header("System Overview")
    with st.expander("Detailed Information", expanded=False):
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

    # Reports section
    st.header("Generated Reports")    
    if st.session_state.reports:
        for i, report_path in enumerate(st.session_state.reports):
            with st.expander(f"Report {i+1}: {report_path.name}"):
                st.markdown(f"**Path**: `{get_relative_path(report_path)}`")
                st.markdown(f"**Generated**: {report_path.stat().st_mtime}")
                
                # Create a download button for the report
                with open(report_path, "rb") as file:
                    st.download_button(
                        label="Download Report",
                        data=file,
                        file_name=report_path.name,
                        mime="application/pdf" if report_path.suffix == ".pdf" else "application/json"
                    )
    else:
        st.info("No reports generated yet. Ask for a report to see it here.")
    
    # CSVs section
    st.subheader("Generated Predictions")
    if st.session_state.csvs:
        for i, csv_path in enumerate(st.session_state.csvs):
            with st.expander(f"CSV {i+1}: {csv_path.name}"):
                st.markdown(f"**Path**: `{get_relative_path(csv_path)}`")
                st.markdown(f"**Generated**: {csv_path.stat().st_mtime}")
                st.markdown(f"**Size**: {csv_path.stat().st_size/1024:.1f} KB")
                with open(csv_path, "rb") as file:
                    st.download_button(
                        label="Download CSV",
                        data=file,
                        file_name=csv_path.name,
                        mime="text/csv"
                    )
    else:
        st.info("No prediction CSVs found yet. Generate a report to create one.")
    
    # Add a clear chat button
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("What would you like to know about HADR situations?", disabled=st.session_state.processing):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    
    # Run the streaming handler with Gemini-style thinking UI
    response, logs, final_state = handle_query_stream(user_input)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)
        with st.expander("📝 Agent Thinking Log", expanded=False):
            st.code(logs, language="text")
        
        # Check if new reports were generated during processing
        current_reports = check_for_reports()
        new_reports = [r for r in current_reports if r not in st.session_state.reports]
        
        if new_reports:
            st.markdown("---")
            st.markdown("**📊 New Report Generated:**")
            for report in new_reports:
                st.markdown(f"📄 [{report.name}]({get_relative_path(report)})")
                st.session_state.reports.append(report)