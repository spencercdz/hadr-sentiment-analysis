# HADR Sentiment Analysis

This project provides a comprehensive solution for Humanitarian Assistance and Disaster Relief (HADR) sentiment analysis. It includes a sophisticated AI agent that can process natural language queries about disasters, analyze related social media data (specifically from Twitter), and generate detailed reports.

## Features

- **AI Agent**: A LangGraph-based agent that orchestrates the entire workflow, from data gathering to report generation.
- **Multi-Label Sentiment Analysis**: Utilizes a fine-tuned `cardiffnlp/twitter-xlm-roberta-base-sentiment` model for detailed sentiment and category analysis of tweets.
- **Data Pipeline**: Includes scripts for downloading, processing, and preparing data for model training and evaluation.
- **Model Training and Evaluation**: A complete pipeline for training and evaluating the sentiment analysis model.
- **Web Interface**: A Streamlit application provides a user-friendly chat interface to interact with the AI agent.
- **Automated Reporting**: Generates detailed PDF reports summarizing the analysis, including visualizations.

## Project Structure

```
├── config/                    # Configuration files
│   └── model_config.yaml      # Model and training hyperparameters
├── data/
│   ├── metadata/              # Metadata about the datasets
│   ├── processed/             # Processed and cleaned data
│   └── raw/                   # Raw data from various sources
├── evaluation/                # Evaluation results and reports
├── notebooks/                 # Jupyter notebooks for exploration and analysis
├── src/
│   ├── ai_agent/              # The AI agent application
│   │   ├── agents/            # Core agent logic
│   │   ├── app.py             # Streamlit application
│   │   └── run_app.py         # Script to run the Streamlit app
│   ├── data/                  # Data loading and preprocessing scripts
│   │   └── data_loader.py     # Script to download the dataset
│   └── models/                # Model-related scripts
│       ├── llm/               # Language model definitions
│       ├── train.py           # Model training script
│       └── test.py            # Model evaluation script
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- An environment with the packages listed in `requirements.txt` installed.
- Ollama running with a suitable model (e.g., `qwen2.5-coder:14b`) for the AI agent.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd hadr-sentiment-analysis
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the dataset:**

    Run the `data_loader.py` script to download the necessary dataset from Hugging Face Hub.

    ```bash
    python src/data/data_loader.py
    ```

### Running the AI Agent

To start the AI agent and its web interface, run the following command:

```bash
streamlit run src/ai_agent/app.py
```

This will launch a Streamlit application in your browser where you can interact with the agent.

### Training the Model

To train the sentiment analysis model, you can use the `train.py` script. The training process is configured via `config/model_config.yaml`.

```bash
python src/models/train.py
```

## Configuration

The main configuration for the project can be found in `config/model_config.yaml`. This file controls model selection, training hyperparameters, and other settings.

## Usage

Once the AI agent is running, you can provide it with queries such as "Generate a report about the Myanmar earthquake 2025". The agent will then:

1.  Extract key information from the query.
2.  Gather relevant tweets.
3.  Perform sentiment and category analysis on the tweets.
4.  Search the web and Wikipedia for additional context.
5.  Generate a comprehensive PDF report with the findings.

Generated reports and prediction data (in CSV format) are saved in the `src/ai_agent/assets/outputs/` directory and are accessible through the Streamlit interface.