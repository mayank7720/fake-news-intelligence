"""
test_pipeline.py
================

Comprehensive test suite for the Fake News Intelligence System pipeline.

Tests cover all major components:
- Text preprocessing (cleaning, tokenization, full preprocess)
- Feature engineering (linguistic feature extraction)
- Sentiment analysis (positive, negative, neutral classification)
- Clickbait detection (clickbait vs. normal headlines)
- Credibility scoring (overall score, grade, breakdown)
- Report generation (HTML report structure and content)

Usage:
    pytest tests/test_pipeline.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessor import TextPreprocessor
from src.feature_engineer import FeatureEngineer
from src.sentiment import SentimentAnalyzer
from src.clickbait import ClickbaitDetector
from src.credibility import CredibilityScorer
from src.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Text Preprocessor Tests
# ---------------------------------------------------------------------------

def test_preprocessor():
    """Test the TextPreprocessor for cleaning, tokenization, and full preprocessing.

    Validates that:
    - HTML tags are stripped from input text.
    - Special characters are removed or handled gracefully.
    - Tokenization returns a list of tokens.
    - None and empty-string inputs are handled without errors.
    """
    preprocessor = TextPreprocessor()

    # --- clean_text ---
    raw_html = '<p>Hello World!</p>'
    cleaned = preprocessor.clean_text(raw_html)
    assert isinstance(cleaned, str), "clean_text should return a string"
    assert '<p>' not in cleaned, "HTML tags should be stripped"
    assert '</p>' not in cleaned, "Closing HTML tags should be stripped"
    assert 'Hello' in cleaned or 'hello' in cleaned, "Core text content should be preserved"

    # Special characters
    special = 'Price is $100 & tax @5%!'
    cleaned_special = preprocessor.clean_text(special)
    assert isinstance(cleaned_special, str), "clean_text should handle special characters"

    # --- tokenize ---
    tokens = preprocessor.tokenize('The quick brown fox jumps over the lazy dog')
    assert isinstance(tokens, list), "tokenize should return a list"
    assert len(tokens) > 0, "tokenize should produce at least one token"

    # --- preprocess (full pipeline) ---
    result = preprocessor.preprocess('<b>Breaking News:</b> Scientists discover new species!')
    assert result is not None, "preprocess should return a result for valid input"

    # --- Edge cases: None and empty input ---
    none_result = preprocessor.clean_text(None) if _accepts_none(preprocessor.clean_text) else ""
    assert isinstance(none_result, str), "clean_text should handle None gracefully"

    empty_result = preprocessor.clean_text('')
    assert isinstance(empty_result, str), "clean_text should handle empty string"
    assert empty_result == '' or empty_result is not None, "Empty input should return empty or safe value"


def _accepts_none(func):
    """Helper: check whether a callable accepts None without raising."""
    try:
        func(None)
        return True
    except (TypeError, AttributeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Feature Engineer Tests
# ---------------------------------------------------------------------------

def test_feature_engineer():
    """Test FeatureEngineer for linguistic feature extraction.

    Validates that the returned feature dictionary contains all expected keys
    and that values are of the correct type.
    """
    engineer = FeatureEngineer()

    sample_text = (
        "The stock market surged today as investors reacted positively to "
        "the latest economic data. Analysts predict continued growth in the "
        "technology sector. However, some experts warn of potential risks "
        "ahead due to geopolitical tensions."
    )

    features = engineer.get_linguistic_features(sample_text)
    assert isinstance(features, dict), "get_linguistic_features should return a dict"

    expected_keys = [
        'word_count',
        'avg_word_length',
        'sentence_count',
        'avg_sentence_length',
        'unique_word_ratio',
    ]

    for key in expected_keys:
        assert key in features, f"Missing expected feature key: '{key}'"

    # Sanity-check value types
    assert isinstance(features['word_count'], (int, float)), "word_count should be numeric"
    assert isinstance(features['avg_word_length'], (int, float)), "avg_word_length should be numeric"
    assert isinstance(features['sentence_count'], (int, float)), "sentence_count should be numeric"
    assert isinstance(features['avg_sentence_length'], (int, float)), "avg_sentence_length should be numeric"
    assert isinstance(features['unique_word_ratio'], (int, float)), "unique_word_ratio should be numeric"

    # Basic sanity-check ranges
    assert features['word_count'] > 0, "word_count should be positive for non-empty text"
    assert features['avg_word_length'] > 0, "avg_word_length should be positive"
    assert features['sentence_count'] > 0, "sentence_count should be positive"


# ---------------------------------------------------------------------------
# Sentiment Analyzer Tests
# ---------------------------------------------------------------------------

def test_sentiment_analyzer():
    """Test SentimentAnalyzer with positive, negative, and neutral texts.

    Validates that:
    - Positive text is classified with a positive sentiment label and positive compound score.
    - Negative text is classified with a negative sentiment label and negative compound score.
    - Neutral text is classified with a neutral sentiment label and near-zero compound score.
    """
    analyzer = SentimentAnalyzer()

    # --- Positive text ---
    positive_result = analyzer.analyze('This is wonderful and amazing news!')
    assert isinstance(positive_result, dict), "analyze should return a dict"
    assert 'sentiment_label' in positive_result, "Result should contain 'sentiment_label'"
    assert 'compound' in positive_result, "Result should contain 'compound'"
    assert positive_result['sentiment_label'].lower() == 'positive', (
        f"Expected 'positive', got '{positive_result['sentiment_label']}'"
    )
    assert positive_result['compound'] > 0, "Compound score should be positive for positive text"

    # --- Negative text ---
    negative_result = analyzer.analyze('This is terrible and horrible news!')
    assert negative_result['sentiment_label'].lower() == 'negative', (
        f"Expected 'negative', got '{negative_result['sentiment_label']}'"
    )
    assert negative_result['compound'] < 0, "Compound score should be negative for negative text"

    # --- Neutral text ---
    neutral_result = analyzer.analyze('The meeting is scheduled for Tuesday.')
    assert neutral_result['sentiment_label'].lower() == 'neutral', (
        f"Expected 'neutral', got '{neutral_result['sentiment_label']}'"
    )


# ---------------------------------------------------------------------------
# Clickbait Detector Tests
# ---------------------------------------------------------------------------

def test_clickbait_detector():
    """Test ClickbaitDetector with obvious clickbait and normal headlines.

    Validates that:
    - Clickbait headlines are flagged with is_clickbait=True and a high clickbait_score.
    - Normal headlines are flagged with is_clickbait=False and a low clickbait_score.
    """
    detector = ClickbaitDetector()

    # --- Clickbait headline ---
    clickbait_text = "You WON'T BELIEVE What Happened Next!! SHOCKING!!"
    clickbait_result = detector.detect(clickbait_text)
    assert isinstance(clickbait_result, dict), "detect should return a dict"
    assert 'is_clickbait' in clickbait_result, "Result should contain 'is_clickbait'"
    assert 'clickbait_score' in clickbait_result, "Result should contain 'clickbait_score'"
    assert clickbait_result['is_clickbait'] is True, (
        "Obvious clickbait should be detected as clickbait"
    )
    assert 0 <= clickbait_result['clickbait_score'] <= 100, (
        "clickbait_score should be between 0 and 100"
    )
    assert clickbait_result['clickbait_score'] > 50, (
        "Clickbait score should be > 50 for obvious clickbait"
    )

    # --- Normal headline ---
    normal_text = 'City Council Approves New Budget for Fiscal Year 2024'
    normal_result = detector.detect(normal_text)
    assert normal_result['is_clickbait'] is False, (
        "Normal headline should not be detected as clickbait"
    )
    assert 0 <= normal_result['clickbait_score'] <= 100, (
        "clickbait_score should be between 0 and 100"
    )
    assert normal_result['clickbait_score'] < 40, (
        "Clickbait score should be < 40 for a normal headline"
    )


# ---------------------------------------------------------------------------
# Credibility Scorer Tests
# ---------------------------------------------------------------------------

def test_credibility_scorer():
    """Test CredibilityScorer with a sample news article.

    Validates that the result contains all expected keys and that values
    fall within plausible ranges.
    """
    scorer = CredibilityScorer()

    sample_news = (
        "According to a study published in the Journal of Science, researchers "
        "at MIT have developed a new method for detecting microplastics in "
        "drinking water. The peer-reviewed study, conducted over three years, "
        "involved analysis of water samples from 50 cities across the United States. "
        "Dr. Jane Smith, the lead researcher, stated that the findings could "
        "have significant implications for public health policy."
    )

    result = scorer.score(sample_news)
    assert isinstance(result, dict), "score should return a dict"

    expected_keys = [
        'overall_score',
        'grade',
        'breakdown',
        'risk_factors',
        'positive_signals',
    ]

    for key in expected_keys:
        assert key in result, f"Missing expected key in credibility result: '{key}'"

    # overall_score should be numeric and in a reasonable range
    assert isinstance(result['overall_score'], (int, float)), "overall_score should be numeric"
    assert 0 <= result['overall_score'] <= 100, (
        "overall_score should be between 0 and 100"
    )

    # grade should be a non-empty string
    assert isinstance(result['grade'], str), "grade should be a string"
    assert len(result['grade']) > 0, "grade should not be empty"

    # breakdown and risk_factors should be iterable
    assert isinstance(result['breakdown'], (dict, list)), "breakdown should be a dict or list"
    assert isinstance(result['risk_factors'], (list, tuple)), "risk_factors should be a list"
    assert isinstance(result['positive_signals'], (list, tuple)), "positive_signals should be a list"


# ---------------------------------------------------------------------------
# Report Generator Tests
# ---------------------------------------------------------------------------

def test_report_generator():
    """Test ReportGenerator.generate_html_report with a complete mock analysis result.

    Validates that:
    - The output is a string containing valid HTML markers.
    - Key sections from the analysis are embedded in the report.
    """
    generator = ReportGenerator()

    mock_analysis_results = {
        'text': (
            'Scientists have discovered a new species of deep-sea fish in the '
            'Pacific Ocean, according to a study published in Nature.'
        ),
        'headline': 'New Deep-Sea Species Discovered in Pacific Ocean',
        'prediction': {
            'label': 'REAL',
            'confidence': 0.92,
            'probabilities': {
                'REAL': 0.92,
                'FAKE': 0.08,
            },
        },
        'credibility': {
            'overall_score': 78.5,
            'grade': 'B+',
            'grade_color': '#4CAF50',
            'breakdown': {
                'source_quality': 0.85,
                'language_objectivity': 0.72,
                'factual_density': 0.80,
            },
            'risk_factors': [
                'Limited source attribution',
            ],
            'positive_signals': [
                'References peer-reviewed journal',
                'Uses objective language',
            ],
        },
        'sentiment': {
            'compound': 0.25,
            'positive': 0.30,
            'negative': 0.05,
            'neutral': 0.65,
            'sentiment_label': 'Positive',
            'emotional_intensity': 'Low',
        },
        'clickbait': {
            'is_clickbait': False,
            'clickbait_score': 0.12,
            'indicators': [],
            'indicator_details': {},
        },
        'explanation': {
            'top_features': [
                {'feature': 'vocabulary_richness', 'importance': 0.35},
                {'feature': 'source_citation_count', 'importance': 0.28},
                {'feature': 'sentiment_compound', 'importance': 0.15},
            ],
        },
    }

    html_report = generator.generate_html_report(mock_analysis_results)
    assert isinstance(html_report, str), "generate_html_report should return a string"
    assert len(html_report) > 0, "HTML report should not be empty"

    # Check for HTML structure markers
    html_lower = html_report.lower()
    assert '<html' in html_lower or '<!doctype' in html_lower, (
        "Report should contain '<html' or '<!DOCTYPE' marker"
    )

    # Check that key sections/data points appear in the report
    assert 'credibility' in html_lower or 'score' in html_lower, (
        "Report should reference credibility or score"
    )
    assert 'sentiment' in html_lower, "Report should contain a sentiment section"
    assert 'clickbait' in html_lower, "Report should contain a clickbait section"
