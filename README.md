# HADR Sentiment Analysis Platform

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation Guide](#installation-guide)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Model Training](#model-training)
- [Output Artifacts](#output-artifacts)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The HADR (Humanitarian Assistance and Disaster Relief) Sentiment Analysis Platform is an advanced AI-powered system designed to process and analyze social media data during disaster events. The platform provides real-time sentiment analysis, topic classification, and automated report generation to assist humanitarian organizations in making data-driven decisions during crisis situations.

This solution combines state-of-the-art natural language processing techniques with a user-friendly interface, enabling both technical and non-technical users to gain valuable insights from social media data during humanitarian crises.

## Key Features

### 🎯 Core Functionality
- **Multi-task Learning Model**: Advanced transformer-based model fine-tuned for humanitarian context, handling multiple classification tasks simultaneously
- **Real-time Processing**: Optimized pipeline for low-latency analysis of streaming social media data
- **Multi-language Support**: Built on XLM-RoBERTa for robust cross-lingual understanding
- **Automated Report Generation**: Dynamic PDF reports with interactive visualizations and executive summaries

### 🤖 AI/ML Capabilities
- **Model Architecture**: Fine-tuned `cardiffnlp/twitter-xlm-roberta-base-sentiment` with custom classification heads
- **Training Features**:
  - Mixed-precision training (FP16) for faster convergence
  - Early stopping with configurable patience
  - Custom learning rate scheduling
  - Weight decay for regularization
- **Data Augmentation**: Dynamic text augmentation with configurable strength and application rate
- **Efficient Processing**: Optimized batch processing and memory management

### 📊 Data Processing
- **Input Formats**: Supports JSON, CSV, and raw text inputs
- **Text Processing**:
  - Advanced tokenization with subword units
  - Dynamic sequence length adjustment
  - Language-agnostic text normalization
- **Quality Control**:
  - Automated data validation
  - Outlier detection
  - Duplicate removal

### 🖥️ User Interface & Integration
- **Web Application**: Interactive Streamlit-based interface
- **API Endpoints**: RESTful API for programmatic access
- **Visualization**: Interactive dashboards with Plotly
- **Deployment**: Containerized with Docker for easy scaling

## System Architecture

### High-Level Architecture
```
humanitarian-sentiment-analysis/
├── config/                    # Configuration files
│   ├── model_config.yaml      # Model training and inference parameters
│   └── app_config.yaml        # Application runtime configuration
│
├── data/                      # Data storage
│   ├── raw/                   # Raw data from sources
│   ├── processed/             # Cleaned and processed data
│   ├── models/                # Trained model checkpoints
│   └── metadata/              # Dataset statistics and documentation
│
├── evaluation/                # Model evaluation results
│   ├── metrics/               # Performance metrics
│   ├── confusion_matrices/    # Classification visualizations
│   └── reports/               # Detailed evaluation reports
│
├── logs/                      # Application and training logs
│   ├── app/                   # Web application logs
│   ├── training/              # Model training logs
│   └── inference/             # Prediction service logs
│
├── notebooks/                 # Jupyter notebooks
│   ├── 1.0-data-exploration/  # Initial data analysis
│   ├── 2.0-model-development/ # Model prototyping
│   └── 3.0-evaluation/        # Results analysis
│
└── src/                       # Source code
    ├── ai_agent/              # AI agent implementation
    │   ├── agents/            # Agent logic and workflows
    │   ├── assets/            # Static files and templates
    │   ├── app.py             # Streamlit application
    │   └── run_app.py         # Application entry point
    │
    ├── data/                  # Data processing
    │   ├── loaders/          # Data loading utilities
    │   ├── processors/       # Data transformation logic
    │   └── validators/       # Data quality checks
    │
    └── models/                # Model implementation
        ├── llm/              # Language model components
        ├── train.py          # Training pipeline
        └── test.py           # Evaluation pipeline
```

### Technical Stack
- **Backend**: Python 3.8+
- **Machine Learning**: PyTorch, Transformers, Scikit-learn
- **NLP**: SpaCy, NLTK, Hugging Face Transformers
- **Web Interface**: Streamlit
- **Data Processing**: Pandas, NumPy, Dask
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Model Serving**: FastAPI
- **MLOps**: MLflow, Weights & Biases
- **Containerization**: Docker

## Installation Guide

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM (32GB recommended for large datasets)
- 50GB+ free disk space (for models and datasets)

### Setup Instructions

1. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Download required language models**
   ```bash
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root with the following variables:
   ```
   # Model configuration
   MODEL_NAME=cardiffnlp/twitter-xlm-roberta-base-sentiment
   MAX_SEQUENCE_LENGTH=128
   BATCH_SIZE=32
   
   # Application settings
   PORT=8501
   DEBUG=False
   LOG_LEVEL=INFO
   ```

## Configuration

The system is highly configurable through YAML configuration files:

### Model Configuration (`config/model_config.yaml`)

```yaml
# Base model configuration
model_name: "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Training hyperparameters
training:
  num_epochs: 1000                   # Maximum number of training epochs
  learning_rate: 2e-5                # Initial learning rate
  batch_size: 32                     # Training batch size
  max_length: 256                    # Maximum sequence length
  weight_decay: 0.01                 # L2 regularization
  early_stopping_patience: 50        # Epochs to wait before early stopping
  fp16: true                         # Enable mixed-precision training

  # Data augmentation settings
  augmentation:
    enabled: true                   # Toggle for data augmentation
    rate: 0.3                       # Percentage of samples to augment
    strength: 0.15                  # Intensity of text transformations

# Model Hub configuration (optional)
hub:
  enabled: true                     # Enable model upload to hub
  repo_id: "spencercdz/xlm-roberta-sentiment-requests"  # Target repository
  private: true                     # Keep model private
  token: null                       # Authentication token (set in environment)
```

## Usage Examples

### Running the Web Interface
```bash
# Start the Streamlit application
streamlit run src/ai_agent/app.py
```

### Using the Python API
```python
from src.models.llm.tuned import TunedLLM

# Initialize the model
model = TunedLLM(model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# Make predictions
predictions = model.predict([
    "Help needed after earthquake in region X. Many injured people need medical attention.",
    "Donations of food and water being collected at the community center.",
    "Just felt a strong tremor. Is everyone okay?"
])

# Process predictions
for text, prediction in zip(texts, predictions):
    print(f"Text: {text}")
    print(f"Sentiment: {prediction['sentiment']}")
    print(f"Urgency: {prediction['urgency']}")
    print(f"Categories: {', '.join(prediction['categories'])}")
    print("-" * 50)
```

## Model Training

### Training a New Model

```bash
# Basic training command
python src/models/train.py \
    --config config/model_config.yaml \
    --train_file data/processed/train1.csv \
    --validation_file data/processed/validation1.csv \
    --output_dir models/hadr_sentiment_v1

# Advanced options (override config)
python src/models/train.py \
    --config config/model_config.yaml \
    --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --train_file data/processed/train1.csv \
    --validation_file data/processed/validation1.csv \
    --output_dir models/hadr_sentiment_v1 \
    --num_epochs 1000 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --fp16 true \
    --early_stopping_patience 50
```

### Evaluation Metrics

During training and evaluation, the following metrics are tracked:

**Classification Metrics**:
- **Weighted F1 Score**: Primary metric for model selection
- **Precision/Recall**: Per-class and weighted averages
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Detailed error analysis

**Training Metrics**:
- **Learning Rate**: Current learning rate value
- **Loss**: Training and validation loss curves
- **Gradient Norm**: For gradient flow monitoring
- **Memory Usage**: GPU/CPU utilization

**Model Performance**:
- **Inference Speed**: Predictions per second
- **Memory Footprint**: Model size and VRAM usage
- **Batch Processing**: Throughput optimization

## Output Artifacts

### Model Outputs
- **Trained Models**: Saved in `models/` directory
- **Evaluation Reports**: Generated in `evaluation/reports/`
- **Visualizations**: Confusion matrices and other plots in `evaluation/visualizations/`

### Application Outputs
- **Reports**: Generated PDF reports in `outputs/reports/`
- **Logs**: Application and model logs in `logs/`
- **Temporary Files**: Cached data in `temp/`

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Enable gradient accumulation
   - Use mixed precision training

2. **Installation Problems**
   - Ensure all system dependencies are installed
   - Use the exact package versions from requirements.txt
   - Check CUDA/cuDNN compatibility with PyTorch version

3. **Model Performance Issues**
   - Try different learning rates
   - Increase training data size
   - Adjust model architecture parameters

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with ❤️ for humanitarian causes
- Uses open-source libraries from the PyData and PyTorch ecosystems
- Special thanks to the Hugging Face community for pre-trained models