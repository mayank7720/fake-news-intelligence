"""
Clickbait Detection Module
===========================

Provides heuristic-based clickbait detection for news headlines by analyzing
linguistic patterns commonly associated with clickbait content.

Classes:
    ClickbaitDetector: Scores headlines on a 0–100 clickbait scale and
                       returns granular indicator details.

Usage::

    >>> detector = ClickbaitDetector()
    >>> result = detector.detect("You Won't BELIEVE What Happened Next!!")
    >>> result["is_clickbait"]
    True
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class _PatternRule:
    """Internal descriptor for a single clickbait pattern rule.

    Attributes:
        name: Short, machine-friendly identifier for the pattern.
        description: Human-readable explanation shown in results.
        severity: Weight added to the clickbait score when matched (0–100).
        checker: Callable that accepts the headline text and returns
                 ``True`` if the pattern is detected.
    """

    name: str
    description: str
    severity: int
    # We store the checker as a plain attribute; frozen dataclass still
    # allows callable fields.
    checker: object  # Callable[[str], bool] — using object to avoid runtime overhead


class ClickbaitDetector:
    """Heuristic clickbait detector for news headlines.

    The detector evaluates a headline against a configurable set of
    linguistic pattern rules.  Each matched rule contributes a severity
    score; the total is capped at 100.  A headline is classified as
    clickbait when the aggregate score meets or exceeds the
    ``threshold``.

    Parameters:
        threshold: Minimum score (inclusive) to flag a headline as
                   clickbait.  Defaults to ``40``.

    Examples:
        >>> detector = ClickbaitDetector()
        >>> result = detector.detect("10 SHOCKING Secrets They Don't Want You to Know!!")
        >>> result["is_clickbait"]
        True
        >>> 0 <= result["clickbait_score"] <= 100
        True

        >>> result = detector.detect("Federal Reserve raises interest rates by 0.25%")
        >>> result["is_clickbait"]
        False
    """

    # ── Sensationalist / emotional vocabulary ──────────────────────────
    _SENSATIONAL_WORDS: List[str] = [
        "shocking",
        "unbelievable",
        "you won't believe",
        "breaking",
        "urgent",
        "exposed",
        "secret",
        "terrifying",
    ]

    # ── Superlative phrases ────────────────────────────────────────────
    _SUPERLATIVE_PHRASES: List[str] = [
        "best ever",
        "worst",
        "most amazing",
    ]

    # ── Vague pronoun / curiosity-gap phrases ──────────────────────────
    _VAGUE_PRONOUN_PHRASES: List[str] = [
        "this is why",
        "here's what",
        "here is what",
        "what happened next",
    ]

    # ── Emotional manipulation phrases ─────────────────────────────────
    _EMOTIONAL_MANIPULATION_PHRASES: List[str] = [
        "will make you cry",
        "will blow your mind",
    ]

    # ── Urgency phrases ───────────────────────────────────────────────
    _URGENCY_PHRASES: List[str] = [
        "act now",
        "before it's too late",
        "before its too late",
    ]

    # Pre-compiled regex patterns (class-level for performance)
    _RE_NUMBERED_LIST = re.compile(
        r"\b\d{1,3}\s+(?:things?|reasons?|ways?|facts?|tips?|signs?|steps?|secrets?|tricks?)\b",
        re.IGNORECASE,
    )
    _RE_QUESTION_HEADLINE = re.compile(
        r"(?:^|\s)(?:did you know|what if|who else|can you|have you|is this|are you|why do|how come)\b",
        re.IGNORECASE,
    )
    _RE_EXCESSIVE_PUNCTUATION = re.compile(r"[!]{2,}|[?]{2,}")

    def __init__(self, threshold: int = 40) -> None:
        if not isinstance(threshold, int) or not (0 <= threshold <= 100):
            raise ValueError(
                f"threshold must be an integer between 0 and 100, got {threshold!r}"
            )
        self._threshold: int = threshold
        self._rules: List[_PatternRule] = self._build_rules()

    # ── Public API ─────────────────────────────────────────────────────

    def detect(self, headline: str) -> Dict[str, object]:
        """Analyse *headline* for clickbait patterns.

        Parameters:
            headline: The news headline to evaluate.

        Returns:
            A dictionary with the following keys:

            * **is_clickbait** (*bool*) – ``True`` when the aggregate
              score meets or exceeds the configured threshold.
            * **clickbait_score** (*int*) – Aggregate score in the
              range 0–100.
            * **indicators** (*List[str]*) – Human-readable description
              of every matched pattern.
            * **indicator_details** (*List[Dict[str, object]]*) – List
              of dicts, each containing ``pattern``, ``description``,
              and ``severity`` keys.

        Raises:
            TypeError: If *headline* is not a string or ``None``.

        Examples:
            >>> ClickbaitDetector().detect("")
            {'is_clickbait': False, 'clickbait_score': 0, 'indicators': [], 'indicator_details': []}

            >>> ClickbaitDetector().detect(None)
            {'is_clickbait': False, 'clickbait_score': 0, 'indicators': [], 'indicator_details': []}
        """
        # ── Guard: None / empty / wrong type ──────────────────────────
        if headline is None:
            return self._empty_result()

        if not isinstance(headline, str):
            raise TypeError(
                f"headline must be a string or None, got {type(headline).__name__}"
            )

        stripped = headline.strip()
        if not stripped:
            return self._empty_result()

        # ── Evaluate every rule ───────────────────────────────────────
        matched_indicators: List[str] = []
        matched_details: List[Dict[str, object]] = []
        raw_score: int = 0

        for rule in self._rules:
            try:
                if rule.checker(stripped):  # type: ignore[operator]
                    matched_indicators.append(rule.description)
                    matched_details.append(
                        {
                            "pattern": rule.name,
                            "description": rule.description,
                            "severity": rule.severity,
                        }
                    )
                    raw_score += rule.severity
            except Exception:
                # Individual rule failures must not break the pipeline.
                continue

        capped_score: int = min(raw_score, 100)

        return {
            "is_clickbait": capped_score >= self._threshold,
            "clickbait_score": capped_score,
            "indicators": matched_indicators,
            "indicator_details": matched_details,
        }

    # ── Private helpers ────────────────────────────────────────────────

    @staticmethod
    def _empty_result() -> Dict[str, object]:
        """Return a neutral result dict for empty / ``None`` input."""
        return {
            "is_clickbait": False,
            "clickbait_score": 0,
            "indicators": [],
            "indicator_details": [],
        }

    # ── Rule builders ──────────────────────────────────────────────────

    def _build_rules(self) -> List[_PatternRule]:
        """Construct the ordered list of pattern rules.

        Each rule encapsulates its own detection logic so that new
        patterns can be added with minimal coupling.
        """
        return [
            _PatternRule(
                name="excessive_capitalization",
                description="Excessive capitalization (>30% uppercase letters)",
                severity=15,
                checker=self._check_excessive_caps,
            ),
            _PatternRule(
                name="sensationalist_words",
                description="Sensationalist or emotional language detected",
                severity=20,
                checker=self._check_sensationalist_words,
            ),
            _PatternRule(
                name="numbered_list",
                description="Numbered list pattern (e.g. '10 things', '5 reasons')",
                severity=10,
                checker=self._check_numbered_list,
            ),
            _PatternRule(
                name="question_headline",
                description="Question headline pattern (e.g. 'Did you know?', 'What if?')",
                severity=10,
                checker=self._check_question_headline,
            ),
            _PatternRule(
                name="superlatives",
                description="Superlative language detected (e.g. 'best ever', 'most amazing')",
                severity=10,
                checker=self._check_superlatives,
            ),
            _PatternRule(
                name="vague_pronouns",
                description="Vague pronoun / curiosity-gap phrase (e.g. 'This is why', 'What happened next')",
                severity=15,
                checker=self._check_vague_pronouns,
            ),
            _PatternRule(
                name="excessive_punctuation",
                description="Excessive punctuation (repeated '!!' or '??')",
                severity=10,
                checker=self._check_excessive_punctuation,
            ),
            _PatternRule(
                name="emotional_manipulation",
                description="Emotional manipulation phrase (e.g. 'will make you cry', 'will blow your mind')",
                severity=15,
                checker=self._check_emotional_manipulation,
            ),
            _PatternRule(
                name="urgency_words",
                description="Urgency language detected (e.g. 'act now', 'before it's too late')",
                severity=10,
                checker=self._check_urgency,
            ),
        ]

    # ── Individual pattern checkers ────────────────────────────────────

    @staticmethod
    def _check_excessive_caps(text: str) -> bool:
        """Return ``True`` when >30 % of alphabetic characters are uppercase."""
        alpha_chars = [ch for ch in text if ch.isalpha()]
        if not alpha_chars:
            return False
        upper_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
        return upper_ratio > 0.30

    @classmethod
    def _check_sensationalist_words(cls, text: str) -> bool:
        """Return ``True`` if any sensationalist keyword or phrase appears."""
        lowered = text.lower()
        return any(word in lowered for word in cls._SENSATIONAL_WORDS)

    @classmethod
    def _check_numbered_list(cls, text: str) -> bool:
        """Return ``True`` for numbered-list headline patterns."""
        return cls._RE_NUMBERED_LIST.search(text) is not None

    @classmethod
    def _check_question_headline(cls, text: str) -> bool:
        """Return ``True`` for question-style headlines."""
        return cls._RE_QUESTION_HEADLINE.search(text) is not None

    @classmethod
    def _check_superlatives(cls, text: str) -> bool:
        """Return ``True`` if superlative phrases are found."""
        lowered = text.lower()
        return any(phrase in lowered for phrase in cls._SUPERLATIVE_PHRASES)

    @classmethod
    def _check_vague_pronouns(cls, text: str) -> bool:
        """Return ``True`` for curiosity-gap / vague pronoun phrases."""
        lowered = text.lower()
        return any(phrase in lowered for phrase in cls._VAGUE_PRONOUN_PHRASES)

    @classmethod
    def _check_excessive_punctuation(cls, text: str) -> bool:
        """Return ``True`` when repeated ``!!`` or ``??`` are present."""
        return cls._RE_EXCESSIVE_PUNCTUATION.search(text) is not None

    @classmethod
    def _check_emotional_manipulation(cls, text: str) -> bool:
        """Return ``True`` for emotional manipulation phrases."""
        lowered = text.lower()
        return any(phrase in lowered for phrase in cls._EMOTIONAL_MANIPULATION_PHRASES)

    @classmethod
    def _check_urgency(cls, text: str) -> bool:
        """Return ``True`` for urgency-inducing phrases."""
        lowered = text.lower()
        return any(phrase in lowered for phrase in cls._URGENCY_PHRASES)
