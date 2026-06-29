# Fake News Intelligence System — Deployment Guide

This guide details the procedures for setting up, running, testing, and deploying the **Fake News Intelligence System** in local development, containerized environments, and cloud hosting platforms.

---

## 1. Local Development Setup

### 1.1. Prerequisites
- **Python**: Version 3.8 to 3.11 (tested on Python 3.9/3.10)
- **Git**: For checking out files (optional if downloaded directly)

### 1.2. Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd fake_news_intelligence
   ```

2. **Set Up a Virtual Environment**
   Using `venv` (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Verify installation and download NLTK data**
   The application will automatically attempt to download NLTK datasets at startup. However, you can pre-download them manually using the following command:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon'); nltk.download('punkt_tab'); nltk.download('omw-1.4')"
   ```

5. **Generate Sample Data**
   Run the synthetic data generator to populate sample databases for modeling:
   ```bash
   python data/generate_sample_data.py
   ```

6. **Run Unit Tests**
   Ensure all pipeline modules are operating correctly:
   ```bash
   pytest tests/test_pipeline.py -v
   ```

7. **Run the Streamlit Dashboard**
   Launch the interactive web portal locally:
   ```bash
   streamlit run app.py
   ```
   Open your browser and navigate to `http://localhost:8501`.

---

## 2. Docker Deployment

For clean isolation and cross-platform deployment, a Docker configuration is provided.

### 2.1. Dockerfile
Create a `Dockerfile` in the root folder with the following contents:

```dockerfile
# Use a slim Python base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8501

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK resources
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon'); nltk.download('punkt_tab'); nltk.download('omw-1.4')"

# Copy application files
COPY . .

# Generate initial sample data and pre-train model
RUN python data/generate_sample_data.py

# Expose Streamlit's default port
EXPOSE 8501

# Run the app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2.2. docker-compose.yml
Create a `docker-compose.yml` to simplify container lifecycle management:

```yaml
version: '3.8'

services:
  fake-news-system:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_HEADLESS=true
    restart: always
```

### 2.3. Run Containerized
To build and start the application via Compose:
```bash
docker-compose up --build -d
```
The dashboard will be active at `http://localhost:8501`.

---

## 3. Streamlit Community Cloud Deployment

Streamlit Community Cloud is the fastest way to share this portfolio project for free.

### 3.1. Repository Setup
1. Push your code, including `requirements.txt`, `setup.py`, `app.py`, `src/`, `data/`, and `.streamlit/` folders, to a public GitHub repository.
2. Verify that `data/generate_sample_data.py` is included so the app can compile its training sample files automatically at deployment time.

### 3.2. Cloud Console Configuration
1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **New app**.
3. Select your repository, branch, and specify `app.py` as the main entry point.
4. Click **Deploy**. Streamlit Cloud will parse `requirements.txt`, spin up a secure container, execute NLTK downloads on import, and host your application publicly.

The official live version of the application is hosted at:
👉 **[Fake News Intelligence System](https://fake-news-intelligence-ecvawhkgxfilrbw6pnentc.streamlit.app/)**

---

## 4. Troubleshooting & Production Considerations

- **NLTK Download Timeout**: If deployment containers lack outbound network access during runtime, download the files locally and copy them into your Docker image inside the `~/.nltk_data` directory.
- **Memory Consumption**: TF-IDF matrices and text arrays can use significant RAM if batch-processing large CSVs (>100,000 rows). Adjust parameters like `max_features` in `FeatureEngineer` or chunk size during batch inference to control memory usage.
- **Model Storage**: Make sure the folder `models/` is writable by the user process, as trained models are serialized to this directory using `joblib`.
