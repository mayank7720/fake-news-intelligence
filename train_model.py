"""
Train Model Utility Script
==========================

Loads the generated sample news dataset, trains the ensemble Fake News Classifier,
and serialises the trained model to disk so it is immediately available for the
Streamlit dashboard.

Usage:
    python train_model.py
"""

import os
import sys
import logging
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("train_model")

# Make sure src package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import FakeNewsClassifier

def train_and_persist():
    # Paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_dir, "data", "sample", "sample_data.csv")
    model_dir = os.path.join(project_dir, "models")
    model_path = os.path.join(model_dir, "fake_news_model.pkl")

    # Ensure models directory exists
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(data_path):
        logger.error("Sample dataset not found at %s. Please run python data/generate_sample_data.py first.", data_path)
        sys.exit(1)

    logger.info("Loading sample news dataset from %s...", data_path)
    df = pd.read_csv(data_path)

    # Validate columns
    required_cols = ["text", "label"]
    for col in required_cols:
        if col not in df.columns:
            logger.error("Missing required column '%s' in dataset.", col)
            sys.exit(1)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    logger.info("Dataset loaded. Total records: %d", len(texts))
    logger.info("Label distribution: \n%s", df["label"].value_counts())

    logger.info("Initializing FakeNewsClassifier...")
    clf = FakeNewsClassifier()

    logger.info("Starting model training (Preprocessing + Feature Engineering + Classifier)...")
    metrics = clf.train(texts, labels)

    logger.info("Model training complete!")
    logger.info("=" * 60)
    logger.info("Model Evaluation Metrics:")
    logger.info("Accuracy  : %.4f", metrics["accuracy"])
    logger.info("Precision : %.4f", metrics["precision"])
    logger.info("Recall    : %.4f", metrics["recall"])
    logger.info("F1-Score  : %.4f", metrics["f1"])
    logger.info("ROC-AUC   : %.4f", metrics["roc_auc"])
    logger.info("=" * 60)
    
    logger.info("Classification Report:\n%s", metrics["classification_report"])

    logger.info("Saving trained model to %s...", model_path)
    clf.save(model_path)
    logger.info("Model saved successfully. The dashboard will now automatically load this pre-trained model!")

if __name__ == "__main__":
    train_and_persist()
