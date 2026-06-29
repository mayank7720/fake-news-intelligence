# Fake News Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fake-news-intelligence-ecvawhkgxfilrbw6pnentc.streamlit.app/)

Fake News Intelligence is a Python-based news analysis project that classifies article text as REAL or FAKE and provides supporting analysis signals such as sentiment, clickbait likelihood, and credibility scoring.

The project includes:
- A modular NLP and machine learning pipeline.
- A Streamlit web application for interactive analysis.
- Synthetic data generation for local experimentation.
- Tests for core pipeline components.

## Table of Contents

- Overview
- Key Features
- Tech Stack
- Project Structure
- Getting Started
- Running the Application
- Training the Model
- Testing
- Documentation
- License

## Overview

The system combines text preprocessing, feature engineering, and an ensemble classification strategy:

- Preprocessing via NLTK-based cleaning, tokenization, stopword removal, and lemmatization.
- TF-IDF feature extraction with linguistic/statistical signals.
- Ensemble prediction using Passive Aggressive and Logistic Regression models.
- Auxiliary analysis modules for sentiment, clickbait, explainability, and credibility scoring.

## Key Features

- Binary fake-news classification (REAL/FAKE).
- Confidence-aware predictions.
- Explainability module for feature-level interpretation.
- Sentiment analysis using VADER.
- Rule-based clickbait detection.
- Composite credibility scoring (0-100 with grade mapping).
- Streamlit dashboard for single-text and dataset workflows.
- HTML report generation support.

## Tech Stack

- Python 3.8+
- Streamlit
- scikit-learn
- NLTK
- pandas
- NumPy
- Plotly
- Matplotlib
- WordCloud
- joblib

## Project Structure

```text
fake-news-intelligence/
  app.py
  train_model.py
  setup.py
  requirements.txt
  README.md
  data/
    generate_sample_data.py
    sample/
  src/
    __init__.py
    preprocessor.py
    feature_engineer.py
    model.py
    explainer.py
    sentiment.py
    clickbait.py
    credibility.py
    report_generator.py
  tests/
    test_pipeline.py
  docs/
    architecture.md
    deployment_guide.md
    project_report.md
```

## Getting Started

### Prerequisites

- Python 3.8 or newer
- pip
- Git

### Installation

```bash
git clone https://github.com/mayank7720/fake-news-intelligence.git
cd fake-news-intelligence
python -m venv venv
```

Activate the virtual environment:

- Windows (PowerShell)

```powershell
venv\Scripts\Activate.ps1
```

- macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Generate sample data:

```bash
python data/generate_sample_data.py
```

## Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Default local URL:

```text
http://localhost:8501
```

## Training the Model

Train and persist the model from generated sample data:

```bash
python train_model.py
```

This writes the trained model to the `models/` directory.

## Testing

Run the pipeline test suite:

```bash
pytest tests/test_pipeline.py -v
```

## Documentation

Additional documentation is available in the `docs/` directory:

- `docs/architecture.md`
- `docs/deployment_guide.md`
- `docs/project_report.md`

## License

This project is licensed under the MIT License.