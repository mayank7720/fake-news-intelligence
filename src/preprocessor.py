"""
Text Preprocessor Module
=========================

Provides comprehensive text cleaning, tokenization, stopword removal,
and lemmatization for NLP-based fake news detection.

Usage:
    >>> preprocessor = TextPreprocessor()
    >>> clean = preprocessor.preprocess("Some <b>HTML</b> text with URLs http://example.com")
    >>> print(clean)
"""

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def _ensure_nltk_data() -> None:
    """Download required NLTK data packages if not already present."""
    import nltk

    packages = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, package_name in packages:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info("Downloading NLTK package: %s", package_name)
            try:
                nltk.download(package_name, quiet=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to download NLTK package '%s': %s", package_name, exc
                )


# Run once at import time so downstream code can rely on the data.
_ensure_nltk_data()

import nltk  # noqa: E402  (must come after _ensure_nltk_data)
from nltk.tokenize import word_tokenize  # noqa: E402
from nltk.corpus import stopwords  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402


class TextPreprocessor:
    """End-to-end text preprocessing pipeline for news articles.

    The pipeline performs the following steps in order:
    1. HTML tag removal
    2. URL removal
    3. Email address removal
    4. Special character removal (keeps alphanumeric and spaces)
    5. Whitespace normalisation
    6. Lowercasing
    7. Tokenization (NLTK ``word_tokenize``)
    8. Stopword removal (English)
    9. Lemmatization (WordNet)

    Parameters
    ----------
    extra_stopwords : list[str] | None
        Additional stopwords to remove beyond the default NLTK English set.

    Examples
    --------
    >>> tp = TextPreprocessor()
    >>> tp.preprocess("Visit http://example.com for <b>details</b>!!!")
    'visit examplecom detail'
    """

    # ---- pre-compiled regex patterns ----
    _RE_HTML = re.compile(r"<[^>]+>")
    _RE_URL = re.compile(r"https?://\S+|www\.\S+")
    _RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
    _RE_SPECIAL = re.compile(r"[^a-zA-Z0-9\s]")
    _RE_WHITESPACE = re.compile(r"\s+")

    def __init__(self, extra_stopwords: Optional[List[str]] = None) -> None:
        self._stop_words = set(stopwords.words("english"))
        if extra_stopwords:
            self._stop_words.update(extra_stopwords)
        self._lemmatizer = WordNetLemmatizer()
        logger.debug(
            "TextPreprocessor initialised with %d stopwords.", len(self._stop_words)
        )

    # --------------------------------------------------------------------- #
    #  Individual pipeline steps                                              #
    # --------------------------------------------------------------------- #

    def clean_text(self, text: str) -> str:
        """Remove HTML tags, URLs, emails, special chars; normalise whitespace and lowercase.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        str
            Cleaned text string.
        """
        if not text:
            return ""
        text = self._RE_HTML.sub(" ", text)
        text = self._RE_URL.sub(" ", text)
        text = self._RE_EMAIL.sub(" ", text)
        text = self._RE_SPECIAL.sub(" ", text)
        text = self._RE_WHITESPACE.sub(" ", text).strip()
        text = text.lower()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize *text* using NLTK ``word_tokenize``.

        Parameters
        ----------
        text : str
            Input text (ideally already cleaned).

        Returns
        -------
        list[str]
            List of token strings.
        """
        if not text:
            return []
        try:
            return word_tokenize(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("word_tokenize failed, falling back to split(): %s", exc)
            return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove English stopwords from a token list.

        Parameters
        ----------
        tokens : list[str]
            List of word tokens.

        Returns
        -------
        list[str]
            Filtered token list.
        """
        if not tokens:
            return []
        return [t for t in tokens if t.lower() not in self._stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens using the WordNet lemmatizer.

        Parameters
        ----------
        tokens : list[str]
            List of word tokens.

        Returns
        -------
        list[str]
            Lemmatized token list.
        """
        if not tokens:
            return []
        return [self._lemmatizer.lemmatize(t) for t in tokens]

    # --------------------------------------------------------------------- #
    #  Full pipeline                                                          #
    # --------------------------------------------------------------------- #

    def preprocess(self, text: str) -> str:
        """Run the full preprocessing pipeline and return a single string.

        Pipeline: clean → tokenize → remove stopwords → lemmatize → join.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        str
            Fully preprocessed text as a space-joined string.
        """
        if not text:
            return ""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return " ".join(tokens)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts.

        Parameters
        ----------
        texts : list[str]
            List of raw text strings.

        Returns
        -------
        list[str]
            List of preprocessed text strings.
        """
        return [self.preprocess(t) for t in texts]

    def __repr__(self) -> str:
        return f"TextPreprocessor(stopwords={len(self._stop_words)})"
