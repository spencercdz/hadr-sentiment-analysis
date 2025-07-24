# HADR Sentiment Analysis Agent

## Overview

This application uses LangGraph to create an intelligent agent that analyzes humanitarian assistance and disaster relief (HADR) situations. The agent processes queries like "myanmar earthquake 2025", retrieves Twitter data, performs sentiment analysis, gathers additional information from web searches and Wikipedia, and generates comprehensive PDF reports.

## Features

- **LangGraph Agent Workflow**: Sequential processing of information through specialized nodes
- **Twitter Data Analysis**: Extracts and analyzes tweets about disaster events
- **Sentiment Analysis**: Uses advanced multi-task classification model to analyze tweet sentiment
- **Web Search Integration**: Gathers real-time information from DuckDuckGo
- **Wikipedia Integration**: Collects background information on disasters and locations
- **PDF Report Generation**: Creates detailed reports with visualizations and analysis
- **Streamlit Interface**: User-friendly chat interface for interacting with the agent

## System Requirements

- Python 3.8+
- Required Python packages (listed in requirements.txt)
- Local LLM server running (e.g., Ollama with Mistral or similar model)

## Installation

1. Clone the repository
2. Install dependencies:

```bash
cd hadr-sentiment-analysis/src/ai_agent
pip install -r requirements.txt
```

3. Ensure you have Ollama installed and running with the Mistral model:

```bash
ollama pull mistral
```

## Usage

### Starting the Application

Run the Streamlit application:

```bash
python run_app.py
```

Or directly with Streamlit:

```bash
cd hadr-sentiment-analysis/src/ai_agent
streamlit run app.py
```

### Using the Agent

1. Open the application in your browser (typically at http://localhost:8501)
2. Enter a query like "Generate a report about the Myanmar earthquake 2025"
3. Wait for the agent to process the query and generate a report
4. Download the PDF report from the sidebar

### **Accessing All Reports and Prediction CSVs**

- **All previously generated PDF and JSON reports** are automatically loaded from `assets/outputs/` and shown in the sidebar, even from previous sessions.
- **All prediction CSV files** (containing tweet-level sentiment and label predictions) are loaded from `assets/outputs/reports/` and are also available in the sidebar.
- You can download any report or CSV at any time, including those from earlier sessions.
- Both reports and CSVs are persistent and always available in the sidebar for analysis.

### About the Prediction CSV Files

- Each time a report is generated, a corresponding CSV file is created in `assets/outputs/reports/`.
- These CSVs contain detailed tweet-level predictions, including sentiment scores and label assignments for each tweet analyzed.
- You can use these CSVs for further analysis, visualization, or integration with other tools.
- The sidebar provides download buttons and basic info (filename, size, date) for each CSV.

### Running the Agent from Command Line

You can also run the agent directly from the command line:

```bash
cd hadr-sentiment-analysis/src/ai_agent
python -c "from agents.hadr_agent import process_query; result = process_query('Myanmar earthquake 2025'); print(result['response'])"
```

## Customization

### Twitter Data

The system looks for Twitter data in CSV format in the `assets/twitter_data/` directory. Ensure your CSV files have the following columns:
- query
- tweet_id
- time
- language
- username
- verified
- followers
- location
- retweets
- favorites
- replies
- text

### Report Template

You can customize the report template by modifying the JSON file at `assets/templates/report_template.json`.

## Advanced Features

The sentiment analysis model (TunedLLM) includes several advanced features:
- Multi-task classification (genre, related, request, offer, aid_related)
- Optimized batch processing with dynamic sizing
- Half-precision inference on GPU
- Memory optimization
- Parallel processing

## Troubleshooting

If you encounter issues:

1. Ensure Ollama is running with the Mistral model
2. Check that all dependencies are installed
3. Verify Twitter data is available in the correct format
4. Check log files for detailed error messages

## License

This project is licensed under the MIT License - see the LICENSE file for details.
