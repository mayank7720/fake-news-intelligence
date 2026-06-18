"""
Feature Engineering Module
===========================

Combines TF-IDF text features with hand-crafted linguistic features to
produce a rich feature matrix for fake-news classification.

Usage:
    >>> fe = FeatureEngineer()
    >>> X = fe.fit_transform(["This is a sample article.", "Another one."])
    >>> print(X.shape)
"""

import re
import logging
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Extract TF-IDF *and* linguistic features from raw text.

    The TF-IDF vectorizer is configured with bigrams and sensible
    document-frequency thresholds. Linguistic features capture stylistic
    cues that are strong indicators of fake news (e.g. exclamation marks,
    caps ratio, readability).

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size for TF-IDF.
    ngram_range : tuple[int, int]
        N-gram range for TF-IDF.
    max_df : float
        Ignore terms with document frequency above this threshold.
    min_df : int
        Ignore terms with document frequency below this threshold.

    Examples
    --------
    >>> fe = FeatureEngineer()
    >>> X = fe.fit_transform(["Breaking: experts agree!", "Officials reported today."])
    >>> fe.get_feature_names()[:5]
    """

    # Names of the linguistic features (order matters).
    _LING_FEATURE_NAMES: List[str] = [
        "word_count",
        "avg_word_length",
        "sentence_count",
        "avg_sentence_length",
        "exclamation_count",
        "question_mark_count",
        "caps_ratio",
        "quote_count",
        "unique_word_ratio",
        "readability_score",
    ]

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        max_df: float = 0.95,
        min_df: int = 2,
    ) -> None:
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self._is_fitted = False
        logger.debug(
            "FeatureEngineer created (max_features=%d, ngram_range=%s).",
            max_features,
            ngram_range,
        )

    # ------------------------------------------------------------------ #
    #  Linguistic feature extraction                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_linguistic_features(text: str) -> Dict[str, float]:
        """Compute hand-crafted linguistic / stylistic features.

        Parameters
        ----------
        text : str
            The raw (uncleaned) text of the article.

        Returns
        -------
        dict[str, float]
            Dictionary with the following keys:

            * ``word_count`` – total number of whitespace-delimited tokens.
            * ``avg_word_length`` – mean character length of tokens.
            * ``sentence_count`` – approximate number of sentences.
            * ``avg_sentence_length`` – mean words per sentence.
            * ``exclamation_count`` – number of ``!`` characters.
            * ``question_mark_count`` – number of ``?`` characters.
            * ``caps_ratio`` – fraction of uppercase letters.
            * ``quote_count`` – number of quotation-mark characters.
            * ``unique_word_ratio`` – ratio of unique tokens to total tokens.
            * ``readability_score`` – Flesch-Kincaid readability approximation.
        """
        if not text:
            return {name: 0.0 for name in FeatureEngineer._LING_FEATURE_NAMES}

        words = text.split()
        word_count = len(words)
        avg_word_length = (
            np.mean([len(w) for w in words]) if word_count > 0 else 0.0
        )

        # Sentence splitting (simple heuristic).
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)
        avg_sentence_length = word_count / sentence_count

        exclamation_count = text.count("!")
        question_mark_count = text.count("?")

        alpha_chars = [c for c in text if c.isalpha()]
        caps_ratio = (
            sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if alpha_chars
            else 0.0
        )

        quote_count = text.count('"') + text.count("'") + text.count("\u201c") + text.count("\u201d")

        unique_words = set(w.lower() for w in words)
        unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0.0

        # Flesch-Kincaid approximation.
        total_syllables = sum(FeatureEngineer._count_syllables(w) for w in words)
        if word_count > 0 and sentence_count > 0:
            readability_score = (
                206.835
                - 1.015 * (word_count / sentence_count)
                - 84.6 * (total_syllables / word_count)
            )
        else:
            readability_score = 0.0

        return {
            "word_count": float(word_count),
            "avg_word_length": float(avg_word_length),
            "sentence_count": float(sentence_count),
            "avg_sentence_length": float(avg_sentence_length),
            "exclamation_count": float(exclamation_count),
            "question_mark_count": float(question_mark_count),
            "caps_ratio": float(caps_ratio),
            "quote_count": float(quote_count),
            "unique_word_ratio": float(unique_word_ratio),
            "readability_score": float(readability_score),
        }

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Approximate syllable count for a single English word.

        Uses a vowel-group heuristic; not perfect but sufficient for
        readability scoring.
        """
        word = word.lower().strip()
        if not word:
            return 0
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        # Adjust for silent 'e'.
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    # ------------------------------------------------------------------ #
    #  Fit / Transform interface                                          #
    # ------------------------------------------------------------------ #

    def fit(self, texts: List[str]) -> "FeatureEngineer":
        """Fit the TF-IDF vectorizer on *texts*.

        Parameters
        ----------
        texts : list[str]
            Training corpus (ideally preprocessed).

        Returns
        -------
        FeatureEngineer
            ``self``, for method chaining.
        """
        logger.info("Fitting TF-IDF on %d documents …", len(texts))
        self.tfidf.fit(texts)
        self._is_fitted = True
        logger.info("TF-IDF vectorizer fitted.")
        return self

    def transform(self, texts: List[str]) -> sparse.csr_matrix:
        """Transform *texts* into a TF-IDF feature matrix.

        Parameters
        ----------
        texts : list[str]
            Input texts (preprocessed for TF-IDF).

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse feature matrix of shape ``(n_texts, n_tfidf)``.

        Raises
        ------
        RuntimeError
            If the vectorizer has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureEngineer has not been fitted. Call fit() first."
            )

        tfidf_matrix = self.tfidf.transform(texts)
        logger.debug("Transformed %d texts → feature matrix %s.", len(texts), tfidf_matrix.shape)
        return tfidf_matrix

    def fit_transform(self, texts: List[str]) -> sparse.csr_matrix:
        """Fit on *texts* and return the transformed feature matrix.

        Convenience wrapper around :meth:`fit` + :meth:`transform`.

        Parameters
        ----------
        texts : list[str]
            Training corpus.

        Returns
        -------
        scipy.sparse.csr_matrix
            TF-IDF feature matrix.
        """
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of all feature names.

        Returns
        -------
        list[str]
            TF-IDF feature names.

        Raises
        ------
        RuntimeError
            If the vectorizer has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureEngineer has not been fitted. Call fit() first."
            )
        return list(self.tfidf.get_feature_names_out())

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"FeatureEngineer(status={status})"
