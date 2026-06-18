"""
Credibility Scoring Module for the Fake News Intelligence System.

This module provides the ``CredibilityScorer`` class, which fuses multiple
analytic signals into a single credibility assessment for a piece of news
content.  The scorer combines:

* **ML model confidence** (40 % weight) – how confident the classifier is.
* **Sentiment analysis** (15 %) – extreme sentiment lowers credibility.
* **Clickbait detection** (15 %) – clickbait patterns lower credibility.
* **Linguistic quality** (15 %) – readability and vocabulary richness.
* **Source credibility** (15 %) – attribution, quotes, dates, named entities.

Usage
-----
>>> from src.credibility import CredibilityScorer
>>> scorer = CredibilityScorer()
>>> result = scorer.score("According to Reuters, the summit concluded on Friday.")
>>> print(result["grade"])
B
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for sibling modules — avoids circular-import issues when
# components are instantiated inside __init__.
# ---------------------------------------------------------------------------

_SentimentAnalyzer = None
_ClickbaitDetector = None


def _get_sentiment_analyzer_cls():
    """Lazy-import SentimentAnalyzer to avoid circular imports."""
    global _SentimentAnalyzer
    if _SentimentAnalyzer is None:
        from src.sentiment import SentimentAnalyzer as _SA
        _SentimentAnalyzer = _SA
    return _SentimentAnalyzer


def _get_clickbait_detector_cls():
    """Lazy-import ClickbaitDetector to avoid circular imports."""
    global _ClickbaitDetector
    if _ClickbaitDetector is None:
        from src.clickbait import ClickbaitDetector as _CD
        _ClickbaitDetector = _CD
    return _ClickbaitDetector


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WEIGHTS: Dict[str, float] = {
    "ml_confidence": 0.40,
    "sentiment": 0.15,
    "clickbait": 0.15,
    "linguistic_quality": 0.15,
    "source_credibility": 0.15,
}
"""Signal weights – must sum to 1.0."""

_GRADE_MAP: List[tuple] = [
    (90, "A", "#00ff88"),
    (80, "B", "#88cc00"),
    (70, "C", "#ffaa00"),
    (60, "D", "#ff6600"),
    (0,  "F", "#ff4444"),
]
"""(min_score, grade, hex_colour) – evaluated top-down; first match wins."""

_ATTRIBUTION_PHRASES: List[str] = [
    "according to",
    "reported by",
    "sources say",
    "officials said",
    "a spokesperson",
    "in a statement",
    "press release",
    "cited by",
    "as reported by",
    "confirmed by",
]
"""Phrases that indicate proper source attribution."""

_DATE_PATTERN: re.Pattern[str] = re.compile(
    r"""
    (?:\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)          # DD/MM/YYYY or similar
    |(?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*
       \s+\d{1,2},?\s+\d{4}\b)                      # January 1, 2024
    |(?:\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b)            # ISO date
    |(?:\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b)
    """,
    re.IGNORECASE | re.VERBOSE,
)
"""Regex for recognising concrete dates or day names in text."""

_SENTENCE_SPLIT: re.Pattern[str] = re.compile(r"[.!?]+")
"""Simple sentence-boundary splitter."""

_DEFAULT_ML_CONFIDENCE: float = 50.0
"""Fallback ML confidence when no model prediction is provided."""


# ---------------------------------------------------------------------------
# CredibilityScorer
# ---------------------------------------------------------------------------

class CredibilityScorer:
    """Multi-signal credibility scorer for news articles.

    Combines machine-learning model confidence with rule-based linguistic
    and source-credibility heuristics to produce a single 0–100 score
    together with a letter grade and actionable risk / positive signals.

    Parameters
    ----------
    model : object, optional
        A trained classifier with a ``predict_proba`` method.  If *None*,
        a default ML confidence of 50 is assumed during scoring.
    sentiment_analyzer : SentimentAnalyzer, optional
        An instance of ``SentimentAnalyzer``.  Created automatically if
        not supplied.
    clickbait_detector : ClickbaitDetector, optional
        An instance of ``ClickbaitDetector``.  Created automatically if
        not supplied.

    Examples
    --------
    >>> scorer = CredibilityScorer()
    >>> result = scorer.score("Scientists at MIT published new findings today.")
    >>> result["grade"] in ("A", "B", "C", "D", "F")
    True
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        sentiment_analyzer: Optional[Any] = None,
        clickbait_detector: Optional[Any] = None,
    ) -> None:
        """Initialise the CredibilityScorer with optional components."""
        self.model = model

        if sentiment_analyzer is not None:
            self.sentiment_analyzer = sentiment_analyzer
        else:
            try:
                sa_cls = _get_sentiment_analyzer_cls()
                self.sentiment_analyzer = sa_cls()
            except Exception:
                logger.warning("Could not create default SentimentAnalyzer; sentiment signal disabled.")
                self.sentiment_analyzer = None

        if clickbait_detector is not None:
            self.clickbait_detector = clickbait_detector
        else:
            try:
                cd_cls = _get_clickbait_detector_cls()
                self.clickbait_detector = cd_cls()
            except Exception:
                logger.warning("Could not create default ClickbaitDetector; clickbait signal disabled.")
                self.clickbait_detector = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        text: str,
        headline: Optional[str] = None,
        prediction_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute a multi-signal credibility score for *text*.

        Parameters
        ----------
        text : str
            The article body text to evaluate.
        headline : str, optional
            The article headline (used for clickbait analysis).
        prediction_result : dict, optional
            Output from the ML classifier.  Expected keys:

            - ``label`` (*str*) – ``"REAL"`` or ``"FAKE"``.
            - ``confidence`` (*float*) – Model confidence in [0, 1].

            If not provided a lightweight text-only analysis is performed
            with a default ML confidence of 50.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the following keys:

            - **overall_score** (*int*) – Weighted credibility score 0–100.
            - **grade** (*str*) – Letter grade A / B / C / D / F.
            - **grade_color** (*str*) – Hex colour string for UI display.
            - **breakdown** (*dict*) – Per-signal scores (all 0–100).
            - **risk_factors** (*list[str]*) – Identified risk warnings.
            - **positive_signals** (*list[str]*) – Identified positive indicators.
        """
        # Guard: None / empty input
        if text is None or not isinstance(text, str) or not text.strip():
            return self._empty_result()

        risk_factors: List[str] = []
        positive_signals: List[str] = []

        # --- 1. ML confidence score ---
        ml_score = self._compute_ml_confidence(prediction_result, risk_factors, positive_signals)

        # --- 2. Sentiment score ---
        sentiment_score = self._compute_sentiment_score(text, risk_factors, positive_signals)

        # --- 3. Clickbait score ---
        clickbait_score = self._compute_clickbait_score(
            headline if headline else text[:200], risk_factors, positive_signals,
        )

        # --- 4. Linguistic quality score ---
        linguistic_score = self._compute_linguistic_quality(text, risk_factors, positive_signals)

        # --- 5. Source credibility score ---
        source_score = self._compute_source_credibility(text, risk_factors, positive_signals)

        # --- Weighted overall ---
        overall = (
            ml_score * _WEIGHTS["ml_confidence"]
            + sentiment_score * _WEIGHTS["sentiment"]
            + clickbait_score * _WEIGHTS["clickbait"]
            + linguistic_score * _WEIGHTS["linguistic_quality"]
            + source_score * _WEIGHTS["source_credibility"]
        )
        overall = int(min(100, max(0, round(overall))))

        grade, grade_color = self._grade_from_score(overall)

        return {
            "overall_score": overall,
            "grade": grade,
            "grade_color": grade_color,
            "breakdown": {
                "ml_confidence_score": int(round(ml_score)),
                "sentiment_score": int(round(sentiment_score)),
                "clickbait_score": int(round(clickbait_score)),
                "linguistic_quality_score": int(round(linguistic_score)),
                "source_credibility_score": int(round(source_score)),
            },
            "risk_factors": risk_factors,
            "positive_signals": positive_signals,
        }

    # ------------------------------------------------------------------
    # Signal computation helpers
    # ------------------------------------------------------------------

    def _compute_ml_confidence(
        self,
        prediction_result: Optional[Dict[str, Any]],
        risk_factors: List[str],
        positive_signals: List[str],
    ) -> float:
        """Derive a 0–100 ML confidence score from the prediction result.

        Parameters
        ----------
        prediction_result : dict or None
            The classifier output dict.
        risk_factors : list
            Mutable list to append risk warnings to.
        positive_signals : list
            Mutable list to append positive indicators to.

        Returns
        -------
        float
            Score in [0, 100].
        """
        if prediction_result is None:
            return _DEFAULT_ML_CONFIDENCE

        confidence = prediction_result.get("confidence", 0.5)
        label = prediction_result.get("label", "UNKNOWN")

        # If model says REAL with high confidence → high ML score
        if label == "REAL":
            ml_score = confidence * 100
            if confidence >= 0.85:
                positive_signals.append(f"ML model is highly confident this is REAL ({confidence:.0%})")
            elif confidence < 0.6:
                risk_factors.append(f"ML model has low confidence in REAL prediction ({confidence:.0%})")
        else:
            # Model says FAKE → invert: high confidence in FAKE = low credibility
            ml_score = (1 - confidence) * 100
            if confidence >= 0.85:
                risk_factors.append(f"ML model is highly confident this is FAKE ({confidence:.0%})")
            elif confidence < 0.6:
                positive_signals.append("ML model has low confidence in FAKE prediction")

        return min(100.0, max(0.0, ml_score))

    def _compute_sentiment_score(
        self,
        text: str,
        risk_factors: List[str],
        positive_signals: List[str],
    ) -> float:
        """Compute credibility-adjusted sentiment score (0–100).

        Extreme sentiment (very positive or very negative) reduces the
        score because fabricated news frequently uses emotionally charged
        language.

        Parameters
        ----------
        text : str
            Article body text.
        risk_factors : list
            Mutable list for risk warnings.
        positive_signals : list
            Mutable list for positive indicators.

        Returns
        -------
        float
            Score in [0, 100] – 100 means neutral / balanced sentiment.
        """
        if self.sentiment_analyzer is None:
            return 70.0  # neutral fallback

        try:
            sentiment = self.sentiment_analyzer.analyze(text)
        except Exception:
            logger.exception("Sentiment analysis failed; using fallback score.")
            return 70.0

        compound = abs(sentiment.get("compound", 0.0))

        # Linear mapping: |compound| 0 → 100, |compound| 1 → 0
        score = max(0.0, 100.0 * (1.0 - compound))

        if compound > 0.7:
            risk_factors.append(
                f"Extremely {sentiment.get('sentiment_label', 'biased')} sentiment detected "
                f"(compound={sentiment.get('compound', 0):.2f})"
            )
        elif compound < 0.2:
            positive_signals.append("Balanced, neutral sentiment indicates objective reporting")

        if sentiment.get("fake_news_sentiment_flag", False):
            risk_factors.append("Sentiment intensity flagged as potential fake-news indicator")

        return score

    def _compute_clickbait_score(
        self,
        headline: str,
        risk_factors: List[str],
        positive_signals: List[str],
    ) -> float:
        """Compute credibility-adjusted clickbait score (0–100).

        The raw clickbait score (0–100, where 100 = definitely clickbait)
        is inverted so that 100 = not clickbait = high credibility.

        Parameters
        ----------
        headline : str
            The headline or first 200 chars of text.
        risk_factors : list
            Mutable list for risk warnings.
        positive_signals : list
            Mutable list for positive indicators.

        Returns
        -------
        float
            Inverted score in [0, 100].
        """
        if self.clickbait_detector is None:
            return 80.0  # optimistic fallback

        try:
            cb = self.clickbait_detector.detect(headline)
        except Exception:
            logger.exception("Clickbait detection failed; using fallback score.")
            return 80.0

        raw_score = cb.get("clickbait_score", 0)
        inverted = 100 - raw_score

        if cb.get("is_clickbait", False):
            indicators = cb.get("indicators", [])
            summary = ", ".join(indicators[:3]) if indicators else "multiple patterns"
            risk_factors.append(f"Clickbait patterns detected: {summary}")
        else:
            positive_signals.append("No significant clickbait patterns found in headline")

        return max(0.0, min(100.0, float(inverted)))

    def _compute_linguistic_quality(
        self,
        text: str,
        risk_factors: List[str],
        positive_signals: List[str],
    ) -> float:
        """Evaluate linguistic quality as a 0–100 score.

        Checks:
        * Average sentence length (ideal: 15–25 words).
        * Vocabulary richness (unique-word ratio).
        * Overall word count (very short articles are penalised).

        Parameters
        ----------
        text : str
            Article body text.
        risk_factors : list
            Mutable list for risk warnings.
        positive_signals : list
            Mutable list for positive indicators.

        Returns
        -------
        float
            Linguistic quality score 0–100.
        """
        words = text.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0

        # --- Sentence length ---
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        num_sentences = max(len(sentences), 1)
        avg_sentence_len = word_count / num_sentences

        # Ideal range: 15–25 words per sentence
        if 15 <= avg_sentence_len <= 25:
            sentence_score = 100.0
            positive_signals.append("Well-structured sentences with appropriate length")
        elif avg_sentence_len < 8:
            sentence_score = 40.0
            risk_factors.append(f"Very short average sentence length ({avg_sentence_len:.1f} words)")
        elif avg_sentence_len > 40:
            sentence_score = 40.0
            risk_factors.append(f"Excessively long sentences ({avg_sentence_len:.1f} words avg)")
        else:
            # Gradual penalty outside the sweet spot
            distance = min(abs(avg_sentence_len - 15), abs(avg_sentence_len - 25))
            sentence_score = max(40.0, 100.0 - distance * 4)

        # --- Vocabulary richness ---
        unique_words = set(w.lower() for w in words)
        vocab_ratio = len(unique_words) / word_count if word_count else 0.0

        if vocab_ratio >= 0.6:
            vocab_score = 100.0
            positive_signals.append(f"Rich vocabulary (unique-word ratio: {vocab_ratio:.0%})")
        elif vocab_ratio >= 0.4:
            vocab_score = 70.0
        else:
            vocab_score = max(30.0, vocab_ratio * 150)
            risk_factors.append(f"Low vocabulary diversity (unique-word ratio: {vocab_ratio:.0%})")

        # --- Word count ---
        if word_count >= 100:
            length_score = 100.0
        elif word_count >= 50:
            length_score = 80.0
        elif word_count >= 20:
            length_score = 60.0
        else:
            length_score = 35.0
            risk_factors.append(f"Very short article ({word_count} words)")

        # Combined (equal sub-weights)
        return (sentence_score + vocab_score + length_score) / 3.0

    def _compute_source_credibility(
        self,
        text: str,
        risk_factors: List[str],
        positive_signals: List[str],
    ) -> float:
        """Evaluate source-credibility signals as a 0–100 score.

        Checks for:
        * Attribution phrases ("according to", "officials said", etc.).
        * Quoted sources (text in quotation marks).
        * Specific dates or day references.
        * Named entities (capitalised multi-word proper nouns).

        Parameters
        ----------
        text : str
            Article body text.
        risk_factors : list
            Mutable list for risk warnings.
        positive_signals : list
            Mutable list for positive indicators.

        Returns
        -------
        float
            Source credibility score 0–100.
        """
        score = 50.0  # Start at neutral baseline

        text_lower = text.lower()

        # --- Attribution phrases ---
        attribution_count = sum(1 for phrase in _ATTRIBUTION_PHRASES if phrase in text_lower)
        if attribution_count >= 2:
            score += 20
            positive_signals.append(f"Multiple source attributions found ({attribution_count})")
        elif attribution_count == 1:
            score += 10
            positive_signals.append("Source attribution detected")
        else:
            risk_factors.append("No source attribution phrases found")
            score -= 10

        # --- Quoted sources ---
        quotes = re.findall(r'["\u201c](.*?)["\u201d]', text)
        meaningful_quotes = [q for q in quotes if len(q) > 10]
        if meaningful_quotes:
            score += min(15, len(meaningful_quotes) * 5)
            positive_signals.append(f"Contains {len(meaningful_quotes)} quoted source(s)")
        else:
            risk_factors.append("No direct quotes from sources")

        # --- Specific dates ---
        date_matches = _DATE_PATTERN.findall(text)
        if date_matches:
            score += 10
            positive_signals.append("Contains specific dates/time references")
        else:
            risk_factors.append("No specific dates or temporal references found")
            score -= 5

        # --- Named entities (capitalised sequences) ---
        # Simple heuristic: find sequences of 2+ capitalised words not at
        # sentence start.
        named_entities = re.findall(r'(?<=[.!?]\s)[A-Z][a-z]+(?:\s[A-Z][a-z]+)+', text)
        # Also check for organisations / proper nouns mid-sentence
        mid_sentence_caps = re.findall(r'(?<=\s)[A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})+', text)
        entity_count = len(set(named_entities + mid_sentence_caps))
        if entity_count >= 3:
            score += 10
            positive_signals.append(f"Multiple named entities referenced ({entity_count})")
        elif entity_count >= 1:
            score += 5

        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _grade_from_score(score: int) -> tuple:
        """Map a 0–100 score to a letter grade and hex colour.

        Parameters
        ----------
        score : int
            Overall credibility score.

        Returns
        -------
        tuple[str, str]
            ``(grade, hex_color)`` e.g. ``("A", "#00ff88")``.
        """
        for threshold, grade, color in _GRADE_MAP:
            if score >= threshold:
                return grade, color
        return "F", "#ff4444"

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """Return a default empty/error result.

        Returns
        -------
        Dict[str, Any]
            A result dict with zero scores and no signals.
        """
        return {
            "overall_score": 0,
            "grade": "F",
            "grade_color": "#ff4444",
            "breakdown": {
                "ml_confidence_score": 0,
                "sentiment_score": 0,
                "clickbait_score": 0,
                "linguistic_quality_score": 0,
                "source_credibility_score": 0,
            },
            "risk_factors": ["Unable to analyse: no text provided"],
            "positive_signals": [],
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"model={'present' if self.model else 'None'}, "
            f"sentiment={'ready' if self.sentiment_analyzer else 'None'}, "
            f"clickbait={'ready' if self.clickbait_detector else 'None'})"
        )
