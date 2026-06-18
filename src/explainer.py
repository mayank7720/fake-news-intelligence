"""
Prediction Explainer Module
=============================

Provides human-readable explanations for fake-news predictions by
analysing Logistic Regression feature coefficients and mapping them
back to words and linguistic cues present in the input text.

Usage:
    >>> explainer = PredictionExplainer(trained_model)
    >>> explanation = explainer.explain("Some suspicious article text …")
    >>> print(explanation['explanation_text'])
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from .model import FakeNewsClassifier

logger = logging.getLogger(__name__)


class PredictionExplainer:
    """Explain individual predictions of a :class:`FakeNewsClassifier`.

    The explainer uses the Logistic Regression coefficients stored in the
    trained classifier to determine which features (words, bigrams,
    linguistic cues) contributed most to each prediction.

    Parameters
    ----------
    model : FakeNewsClassifier
        A **trained** classifier instance.

    Raises
    ------
    ValueError
        If the supplied model has not been trained.

    Examples
    --------
    >>> explainer = PredictionExplainer(clf)
    >>> result = explainer.explain("BREAKING: Shocking discovery!")
    >>> for word, weight in result['top_fake_indicators']:
    ...     print(f"  {word}: {weight:.4f}")
    """

    def __init__(self, model: FakeNewsClassifier) -> None:
        if not model.is_trained:
            raise ValueError(
                "The supplied FakeNewsClassifier has not been trained. "
                "Train the model before creating an explainer."
            )
        self.model = model
        self._feature_names: List[str] = model.feature_engineer.get_feature_names()
        self._coefs: np.ndarray = model.lr_model.coef_.ravel()
        logger.debug(
            "PredictionExplainer initialised with %d features.", len(self._feature_names)
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def explain(self, text: str) -> Dict[str, Any]:
        """Generate a full explanation for the model's prediction on *text*.

        Parameters
        ----------
        text : str
            Raw article text.

        Returns
        -------
        dict
            * ``prediction`` – dict from :meth:`FakeNewsClassifier.predict`
            * ``top_fake_indicators`` – list of (feature, weight) tuples
              (top 10 features pushing toward FAKE)
            * ``top_real_indicators`` – list of (feature, weight) tuples
              (top 10 features pushing toward REAL)
            * ``explanation_text`` – human-readable explanation string
            * ``highlighted_words`` – dict mapping words present in *text*
              to their signed importance scores
            * ``linguistic_analysis`` – dict of linguistic features
        """
        prediction = self.model.predict(text)

        # --- Feature importance (global) ---
        top_fake, top_real = self._global_indicators()

        # --- Features present in this text ---
        highlighted = self._highlight_text(text)

        # --- Linguistic features ---
        linguistic = self.model.feature_engineer.get_linguistic_features(text)

        # --- Text-specific top indicators ---
        text_fake, text_real = self._text_specific_indicators(text)

        # Use text-specific indicators if available; fall back to global.
        final_fake = text_fake if text_fake else top_fake
        final_real = text_real if text_real else top_real

        explanation_text = self._build_explanation_text(
            prediction, final_fake, final_real, linguistic
        )

        return {
            "prediction": prediction,
            "top_fake_indicators": final_fake,
            "top_real_indicators": final_real,
            "explanation_text": explanation_text,
            "highlighted_words": highlighted,
            "linguistic_analysis": linguistic,
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _global_indicators(
        self, top_n: int = 10
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Return the top *n* global features for FAKE and REAL.

        Positive coefficients push toward class 1 (REAL); negative
        coefficients push toward class 0 (FAKE).
        """
        indices_sorted = np.argsort(self._coefs)

        # Most negative → strongest FAKE indicators.
        fake_idx = indices_sorted[:top_n]
        fake_indicators = [
            (self._feature_names[i], float(self._coefs[i]))
            for i in fake_idx
            if i < len(self._feature_names)
        ]

        # Most positive → strongest REAL indicators.
        real_idx = indices_sorted[-top_n:][::-1]
        real_indicators = [
            (self._feature_names[i], float(self._coefs[i]))
            for i in real_idx
            if i < len(self._feature_names)
        ]

        return fake_indicators, real_indicators

    def _text_specific_indicators(
        self, text: str, top_n: int = 10
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Return importance scores for features actually *present* in *text*.

        Only TF-IDF features that have a non-zero value in the text's
        transformed representation are considered.
        """
        processed = self.model.preprocessor.preprocess(text)
        features = self.model.feature_engineer.transform([processed])

        # Get non-zero feature indices.
        nonzero_indices = features.nonzero()[1]

        present_features: List[Tuple[str, float]] = []
        for idx in nonzero_indices:
            if idx < len(self._feature_names):
                name = self._feature_names[idx]
                weight = float(self._coefs[idx]) * float(features[0, idx])
                present_features.append((name, weight))

        # Sort by weight.
        present_features.sort(key=lambda x: x[1])

        fake_indicators = present_features[:top_n]  # most negative
        real_indicators = present_features[-top_n:][::-1]  # most positive

        return fake_indicators, real_indicators

    def _highlight_text(self, text: str) -> Dict[str, float]:
        """Map words in *text* to their importance scores.

        Only words that appear in the model vocabulary are included.
        """
        processed = self.model.preprocessor.preprocess(text)
        words = set(processed.split())

        vocab = self.model.feature_engineer.tfidf.vocabulary_
        highlighted: Dict[str, float] = {}
        for word in words:
            if word in vocab:
                idx = vocab[word]
                highlighted[word] = float(self._coefs[idx])

        return highlighted

    @staticmethod
    def _build_explanation_text(
        prediction: Dict[str, Any],
        fake_indicators: List[Tuple[str, float]],
        real_indicators: List[Tuple[str, float]],
        linguistic: Dict[str, float],
    ) -> str:
        """Compose a human-readable explanation paragraph."""
        label = prediction["label"]
        confidence = prediction["confidence"] * 100

        lines = [
            f"This article was classified as {label} with {confidence:.1f}% confidence.",
            "",
        ]

        if label == "FAKE":
            lines.append("Key indicators suggesting this article may be FAKE:")
            for word, weight in fake_indicators[:5]:
                lines.append(f"  • '{word}' (weight: {weight:.4f})")
        else:
            lines.append("Key indicators suggesting this article is REAL:")
            for word, weight in real_indicators[:5]:
                lines.append(f"  • '{word}' (weight: {weight:.4f})")

        # Linguistic flags.
        flags: List[str] = []
        if linguistic.get("exclamation_count", 0) > 3:
            flags.append("excessive exclamation marks")
        if linguistic.get("caps_ratio", 0) > 0.3:
            flags.append("high proportion of capital letters")
        if linguistic.get("readability_score", 100) < 30:
            flags.append("low readability score")
        if linguistic.get("unique_word_ratio", 1) < 0.4:
            flags.append("low vocabulary diversity")
        if linguistic.get("question_mark_count", 0) > 3:
            flags.append("many question marks")

        if flags:
            lines.append("")
            lines.append("Stylistic flags detected: " + ", ".join(flags) + ".")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"PredictionExplainer(features={len(self._feature_names)})"
