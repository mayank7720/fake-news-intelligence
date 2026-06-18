"""
Sentiment Analysis Module for the Fake News Intelligence System.

This module provides a production-ready ``SentimentAnalyzer`` class that
leverages NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner)
to compute fine-grained sentiment metrics on news text.  In addition to
standard polarity scores it derives:

* A human-readable sentiment label (Positive / Negative / Neutral).
* An emotional-intensity tier (Low / Medium / High).
* An approximate subjectivity score based on the ratio of emotionally
  charged tokens identified by VADER.
* A ``fake_news_sentiment_flag`` that fires when the overall compound
  score is unusually extreme—a common trait of fabricated headlines.

Usage
-----
>>> from src.sentiment import SentimentAnalyzer
>>> analyzer = SentimentAnalyzer()
>>> result = analyzer.analyze("Breaking: shocking scandal rocks the nation!")
>>> print(result["sentiment_label"])
Negative
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful NLTK VADER bootstrap
# ---------------------------------------------------------------------------
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as _VaderSIA
    # Attempt to instantiate—this will fail if the lexicon is missing.
    _VaderSIA()
except LookupError:
    # The VADER lexicon hasn't been downloaded yet; fetch it silently.
    import nltk

    logger.info("VADER lexicon not found. Downloading now …")
    try:
        nltk.download("vader_lexicon", quiet=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to download VADER lexicon automatically: %s. "
            "Please run `nltk.download('vader_lexicon')` manually.",
            exc,
        )
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as _VaderSIA
except ImportError as exc:
    raise ImportError(
        "NLTK is required for sentiment analysis.  "
        "Install it with `pip install nltk`."
    ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_COMPOUND_POS_THRESHOLD: float = 0.05
"""Compound score ≥ this value is labelled *Positive*."""

_COMPOUND_NEG_THRESHOLD: float = -0.05
"""Compound score ≤ this value is labelled *Negative*."""

_INTENSITY_LOW_CEIL: float = 0.3
"""Absolute compound below this → *Low* intensity."""

_INTENSITY_MED_CEIL: float = 0.6
"""Absolute compound below this (but ≥ low ceiling) → *Medium* intensity."""

_FAKE_NEWS_COMPOUND_THRESHOLD: float = 0.7
"""Absolute compound exceeding this triggers the fake-news flag."""

_WORD_TOKENISE_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z']+")
"""Lightweight regex tokeniser used for subjectivity approximation."""


# ---------------------------------------------------------------------------
# Default (neutral) result template
# ---------------------------------------------------------------------------

def _default_result() -> Dict[str, Any]:
    """Return a default neutral sentiment result dictionary.

    This is returned whenever the input text is ``None``, empty, or
    otherwise unsuitable for analysis.

    Returns
    -------
    Dict[str, Any]
        A dictionary with all expected keys set to neutral / zero values.
    """
    return {
        "compound": 0.0,
        "positive": 0.0,
        "negative": 0.0,
        "neutral": 1.0,
        "sentiment_label": "Neutral",
        "emotional_intensity": "Low",
        "subjectivity": 0.0,
        "fake_news_sentiment_flag": False,
    }


# ---------------------------------------------------------------------------
# SentimentAnalyzer
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """High-level sentiment analyser built on NLTK VADER.

    The class wraps VADER's ``SentimentIntensityAnalyzer`` and enriches its
    output with derived metrics that are particularly useful for fake-news
    detection pipelines.

    Attributes
    ----------
    _vader : SentimentIntensityAnalyzer
        The underlying VADER analyser instance.

    Examples
    --------
    >>> analyzer = SentimentAnalyzer()
    >>> result = analyzer.analyze("This is a great day!")
    >>> result["sentiment_label"]
    'Positive'
    >>> result["fake_news_sentiment_flag"]
    False
    """

    def __init__(self) -> None:
        """Initialise the SentimentAnalyzer.

        Raises
        ------
        RuntimeError
            If the VADER analyser cannot be instantiated (e.g. missing
            lexicon after a failed download).
        """
        try:
            self._vader: _VaderSIA = _VaderSIA()
        except Exception as exc:
            raise RuntimeError(
                "Could not initialise the VADER sentiment analyser. "
                "Ensure the vader_lexicon is downloaded: "
                "`nltk.download('vader_lexicon')`."
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyse the sentiment of *text* and return enriched metrics.

        Parameters
        ----------
        text : str
            The input text to analyse.  ``None`` and empty / whitespace-only
            strings are handled gracefully and return a default neutral result.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:

            - **compound** (*float*) – Overall sentiment score in [-1, 1].
            - **positive** (*float*) – Positive sentiment proportion.
            - **negative** (*float*) – Negative sentiment proportion.
            - **neutral** (*float*) – Neutral sentiment proportion.
            - **sentiment_label** (*str*) – ``'Positive'``, ``'Negative'``, or
              ``'Neutral'`` derived from *compound* using ±0.05 thresholds.
            - **emotional_intensity** (*str*) – ``'Low'``, ``'Medium'``, or
              ``'High'`` derived from ``abs(compound)``.
            - **subjectivity** (*float*) – Approximate subjectivity score in
              [0, 1] based on the ratio of emotionally charged words.
            - **fake_news_sentiment_flag** (*bool*) – ``True`` when the
              absolute compound score exceeds 0.7, indicating an unusually
              extreme sentiment often associated with fabricated content.

        Examples
        --------
        >>> analyzer = SentimentAnalyzer()
        >>> result = analyzer.analyze("Absolutely terrible and outrageous!")
        >>> result["sentiment_label"]
        'Negative'
        >>> result["emotional_intensity"]
        'High'
        """
        # ---- Guard: None / empty input --------------------------------
        if text is None or not isinstance(text, str) or not text.strip():
            logger.debug("Empty or None input received; returning default neutral result.")
            return _default_result()

        # ---- Core VADER scores ----------------------------------------
        try:
            scores: Dict[str, float] = self._vader.polarity_scores(text)
        except Exception:
            logger.exception("VADER polarity_scores raised an unexpected error.")
            return _default_result()

        compound: float = scores["compound"]
        positive: float = scores["pos"]
        negative: float = scores["neg"]
        neutral: float = scores["neu"]

        # ---- Derived metrics ------------------------------------------
        sentiment_label: str = self._label_sentiment(compound)
        emotional_intensity: str = self._classify_intensity(compound)
        subjectivity: float = self._approximate_subjectivity(text)
        fake_flag: bool = abs(compound) > _FAKE_NEWS_COMPOUND_THRESHOLD

        return {
            "compound": round(compound, 4),
            "positive": round(positive, 4),
            "negative": round(negative, 4),
            "neutral": round(neutral, 4),
            "sentiment_label": sentiment_label,
            "emotional_intensity": emotional_intensity,
            "subjectivity": round(subjectivity, 4),
            "fake_news_sentiment_flag": fake_flag,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _label_sentiment(compound: float) -> str:
        """Map a compound score to a human-readable label.

        Parameters
        ----------
        compound : float
            The VADER compound score in [-1, 1].

        Returns
        -------
        str
            ``'Positive'``, ``'Negative'``, or ``'Neutral'``.
        """
        if compound >= _COMPOUND_POS_THRESHOLD:
            return "Positive"
        if compound <= _COMPOUND_NEG_THRESHOLD:
            return "Negative"
        return "Neutral"

    @staticmethod
    def _classify_intensity(compound: float) -> str:
        """Map the absolute compound score to an intensity tier.

        Parameters
        ----------
        compound : float
            The VADER compound score in [-1, 1].

        Returns
        -------
        str
            ``'Low'``, ``'Medium'``, or ``'High'``.
        """
        abs_compound: float = abs(compound)
        if abs_compound < _INTENSITY_LOW_CEIL:
            return "Low"
        if abs_compound < _INTENSITY_MED_CEIL:
            return "Medium"
        return "High"

    def _approximate_subjectivity(self, text: str) -> float:
        """Estimate subjectivity as the ratio of emotionally charged words.

        VADER assigns a non-zero valence to words it considers emotional.
        This method tokenises *text*, evaluates each token individually, and
        returns the fraction of tokens with a non-zero compound score as a
        rough proxy for subjectivity.

        Parameters
        ----------
        text : str
            The input text (assumed non-empty at this point).

        Returns
        -------
        float
            A value in [0.0, 1.0] where higher means more subjective.
        """
        tokens: List[str] = _WORD_TOKENISE_PATTERN.findall(text)
        if not tokens:
            return 0.0

        emotional_count: int = 0
        for token in tokens:
            try:
                token_score: float = self._vader.polarity_scores(token)["compound"]
                if token_score != 0.0:
                    emotional_count += 1
            except Exception:  # noqa: BLE001
                # Skip tokens that cause unexpected errors.
                continue

        return emotional_count / len(tokens)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(vader={self._vader!r})"
