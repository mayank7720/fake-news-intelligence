"""
Fake News Classifier Module
=============================

Ensemble classifier combining a Passive-Aggressive classifier with
Logistic Regression for robust fake-news detection. The Logistic
Regression model also supplies calibrated probability estimates used
for confidence scoring and explanation.

Usage:
    >>> clf = FakeNewsClassifier()
    >>> metrics = clf.train(texts, labels)
    >>> result = clf.predict("Some article text …")
    >>> clf.save("model.joblib")
"""

import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .preprocessor import TextPreprocessor
from .feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class FakeNewsClassifier:
    """Two-model ensemble for fake-news classification.

    Models
    ------
    * **PassiveAggressiveClassifier** – fast, online-capable linear model
      that is effective for text classification.
    * **LogisticRegression** – provides calibrated probability estimates
      and interpretable feature coefficients.

    The final prediction is determined by majority vote: if both models
    agree, that label is used; otherwise the Logistic Regression
    prediction (which includes calibrated probabilities) takes
    precedence.

    Attributes
    ----------
    pa_model : PassiveAggressiveClassifier
    lr_model : LogisticRegression
    preprocessor : TextPreprocessor
    feature_engineer : FeatureEngineer
    is_trained : bool

    Examples
    --------
    >>> clf = FakeNewsClassifier()
    >>> metrics = clf.train(train_texts, train_labels)
    >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    >>> pred = clf.predict("Breaking news article here …")
    >>> print(pred['label'], pred['confidence'])
    """

    _LABEL_MAP = {0: "FAKE", 1: "REAL"}

    def __init__(self) -> None:
        self.pa_model = PassiveAggressiveClassifier(
            max_iter=1000,
            random_state=42,
            tol=1e-3,
        )
        self.lr_model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
            solver="lbfgs",
        )
        self.preprocessor = TextPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.is_trained: bool = False
        logger.debug("FakeNewsClassifier initialised.")

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #

    def train(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """Train both models and return evaluation metrics.

        Parameters
        ----------
        texts : list[str]
            Raw article texts.
        labels : list[int]
            Binary labels (``0`` = fake, ``1`` = real).

        Returns
        -------
        dict
            Dictionary containing:

            * ``accuracy``, ``precision``, ``recall``, ``f1`` – floats
            * ``roc_auc`` – float (ROC-AUC from LR probabilities)
            * ``confusion_matrix`` – ndarray
            * ``classification_report`` – str
            * ``X_test``, ``y_test``, ``y_pred``, ``y_proba`` – arrays
              kept for downstream analysis / plotting.

        Raises
        ------
        ValueError
            If *texts* and *labels* have different lengths or are empty.
        """
        if not texts or not labels:
            raise ValueError("texts and labels must be non-empty.")
        if len(texts) != len(labels):
            raise ValueError(
                f"Length mismatch: {len(texts)} texts vs {len(labels)} labels."
            )

        logger.info("Starting training on %d samples …", len(texts))

        # 1. Preprocess ---------------------------------------------------
        logger.info("Preprocessing texts …")
        processed_texts = self.preprocessor.preprocess_batch(texts)

        # 2. Train / test split -------------------------------------------
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            processed_texts,
            labels,
            test_size=0.20,
            stratify=labels,
            random_state=42,
        )
        logger.info(
            "Split: %d train / %d test.", len(X_train_text), len(X_test_text)
        )

        # 3. Feature extraction --------------------------------------------
        logger.info("Fitting feature engineer on training data …")
        X_train = self.feature_engineer.fit_transform(X_train_text)
        X_test = self.feature_engineer.transform(X_test_text)

        # 4. Model training ------------------------------------------------
        logger.info("Training PassiveAggressiveClassifier …")
        self.pa_model.fit(X_train, y_train)

        logger.info("Training LogisticRegression …")
        self.lr_model.fit(X_train, y_train)

        self.is_trained = True
        logger.info("Both models trained successfully.")

        # 5. Evaluation ----------------------------------------------------
        y_pred_pa = self.pa_model.predict(X_test)
        y_pred_lr = self.lr_model.predict(X_test)
        y_proba = self.lr_model.predict_proba(X_test)

        # Ensemble: agree → use that; disagree → use LR.
        y_pred = np.where(y_pred_pa == y_pred_lr, y_pred_pa, y_pred_lr)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        except (ValueError, IndexError):
            roc_auc = 0.0

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["FAKE", "REAL"])

        logger.info(
            "Evaluation — Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | ROC-AUC: %.4f",
            acc, prec, rec, f1, roc_auc,
        )

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "classification_report": report,
            "X_test": X_test,
            "y_test": np.asarray(y_test),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    # ------------------------------------------------------------------ #
    #  Prediction                                                         #
    # ------------------------------------------------------------------ #

    def predict(self, text: str) -> Dict[str, Any]:
        """Classify a single article as REAL or FAKE.

        Parameters
        ----------
        text : str
            Raw article text.

        Returns
        -------
        dict
            * ``label`` – ``'REAL'`` or ``'FAKE'``
            * ``confidence`` – float in [0, 1]
            * ``probabilities`` – ``{'REAL': float, 'FAKE': float}``

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if not self.is_trained:
            raise ValueError(
                "Model is not trained. Call train() or load a saved model first."
            )

        processed = self.preprocessor.preprocess(text)
        features = self.feature_engineer.transform([processed])

        pred_pa = int(self.pa_model.predict(features)[0])
        pred_lr = int(self.lr_model.predict(features)[0])
        proba = self.lr_model.predict_proba(features)[0]

        # Ensemble decision.
        if pred_pa == pred_lr:
            final_label_idx = pred_pa
        else:
            final_label_idx = pred_lr

        label = self._LABEL_MAP.get(final_label_idx, "UNKNOWN")
        confidence = float(np.max(proba))
        probabilities = {
            "REAL": float(proba[1]) if len(proba) > 1 else 0.0,
            "FAKE": float(proba[0]) if len(proba) > 0 else 0.0,
        }

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Persist the entire classifier to disk.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``"model.joblib"``).
        """
        joblib.dump(self, path)
        logger.info("Model saved to %s.", path)

    @classmethod
    def load(cls, path: str) -> "FakeNewsClassifier":
        """Load a previously saved classifier from disk.

        Parameters
        ----------
        path : str
            Path to a ``.joblib`` file created by :meth:`save`.

        Returns
        -------
        FakeNewsClassifier
            The deserialised classifier instance.
        """
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(
                f"Loaded object is {type(model).__name__}, expected FakeNewsClassifier."
            )
        logger.info("Model loaded from %s.", path)
        return model

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return f"FakeNewsClassifier(status={status})"
