"""
Report Generator Module
========================

Generates stand-alone, premium dark-themed HTML analysis reports from the
results of the Fake News Intelligence System analysis pipeline.
"""

from datetime import datetime
from typing import Any, Dict


class ReportGenerator:
    """Generates downloadable, beautifully styled HTML reports from analysis results."""

    def __init__(self) -> None:
        pass

    def generate_html_report(self, results: Dict[str, Any]) -> str:
        """Create a premium, dark-themed HTML report representing the full news analysis.

        Parameters
        ----------
        results : dict
            A dictionary containing the analysis results. Expected keys include:
            - text: str
            - headline: str (optional)
            - prediction: dict with 'label' and 'confidence'
            - credibility: dict with 'overall_score', 'grade', 'grade_color', 'breakdown',
                           'risk_factors', and 'positive_signals'
            - sentiment: dict with 'sentiment_label', 'compound', and 'emotional_intensity'
            - clickbait: dict with 'is_clickbait', 'clickbait_score', and 'indicators'
            - explanation: dict with 'explanation_text' or 'top_features'

        Returns
        -------
        str
            A standalone HTML document as a string.
        """
        headline = results.get("headline") or "Untitled News Article"
        text = results.get("text") or "No text content analyzed."
        prediction = results.get("prediction") or {}
        credibility = results.get("credibility") or {}
        sentiment = results.get("sentiment") or {}
        clickbait = results.get("clickbait") or {}
        explanation = results.get("explanation") or {}

        # Safe values
        verdict = prediction.get("label") or "UNKNOWN"
        confidence = prediction.get("confidence", 0.0)
        # Convert confidence to percentage if float in 0-1
        if 0.0 < confidence <= 1.0:
            confidence_pct = f"{confidence * 100:.1f}%"
        else:
            confidence_pct = f"{confidence:.1f}%"

        cred_score = credibility.get("overall_score") or 0
        if isinstance(cred_score, float) and cred_score <= 1.0:
            cred_score = int(cred_score * 100)
        else:
            cred_score = int(cred_score)

        grade = credibility.get("grade") or "N/A"
        grade_color = credibility.get("grade_color") or "#888"

        sentiment_label = sentiment.get("sentiment_label") or "Neutral"
        sentiment_compound = sentiment.get("compound", 0.0)
        sentiment_intensity = sentiment.get("emotional_intensity") or "Low"

        cb_score = clickbait.get("clickbait_score") or 0
        if isinstance(cb_score, float) and cb_score <= 1.0:
            cb_score = int(cb_score * 100)
        else:
            cb_score = int(cb_score)
        is_cb = clickbait.get("is_clickbait", False)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # HTML generation
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake News Intelligence Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {{
            --bg: #0a0a1a;
            --card-bg: #1a1a2e;
            --border: rgba(255, 255, 255, 0.08);
            --text-main: #e0e0e0;
            --text-muted: #8888aa;
            --cyan: #00d4ff;
            --purple: #7c3aed;
            --real-color: #00ff88;
            --fake-color: #ff6b6b;
            --warning-color: #ffaa00;
        }}

        body {{
            background-color: var(--bg);
            color: var(--text-main);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }}

        .container {{
            max-width: 900px;
            width: 100%;
        }}

        header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            position: relative;
        }}

        .gradient-title {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--cyan), var(--purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 10px 0;
        }}

        .subtitle {{
            color: var(--text-muted);
            font-size: 1rem;
            margin: 0;
        }}

        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }}

        .card-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 20px;
            color: var(--cyan);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 8px;
        }}

        /* Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }}

        .metric-card {{
            background-color: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}

        .verdict-badge {{
            display: inline-block;
            padding: 10px 24px;
            border-radius: 50px;
            font-weight: 700;
            font-size: 1.5rem;
            margin-top: 10px;
        }}

        .verdict-real {{
            background-color: rgba(0, 255, 136, 0.1);
            color: var(--real-color);
            border: 2px solid var(--real-color);
            box-shadow: 0 0 15px rgba(0, 255, 136, 0.2);
        }}

        .verdict-fake {{
            background-color: rgba(255, 107, 107, 0.1);
            color: var(--fake-color);
            border: 2px solid var(--fake-color);
            box-shadow: 0 0 15px rgba(255, 107, 107, 0.2);
        }}

        .credibility-ring-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}

        .credibility-ring {{
            position: relative;
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: conic-gradient({grade_color} {cred_score * 3.6}deg, rgba(255, 255, 255, 0.05) 0deg);
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 10px;
        }}

        .credibility-ring::before {{
            content: "";
            position: absolute;
            width: 84px;
            height: 84px;
            border-radius: 50%;
            background-color: var(--card-bg);
        }}

        .credibility-val {{
            position: relative;
            font-size: 1.8rem;
            font-weight: 700;
            color: {grade_color};
        }}

        .grade-badge {{
            font-size: 0.9rem;
            margin-top: 4px;
            font-weight: 600;
            color: var(--text-muted);
        }}

        .article-headline {{
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 12px;
            line-height: 1.4;
            color: #ffffff;
        }}

        .article-text {{
            font-size: 0.95rem;
            line-height: 1.6;
            color: var(--text-muted);
            max-height: 200px;
            overflow-y: auto;
            background-color: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 6px;
            border: 1px solid var(--border);
        }}

        /* Signal details */
        .signals-split {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 768px) {{
            .signals-split {{
                grid-template-columns: 1fr;
            }}
        }}

        .signal-list {{
            list-style-type: none;
            padding-left: 0;
            margin: 0;
        }}

        .signal-item {{
            padding: 10px 12px;
            border-radius: 6px;
            margin-bottom: 10px;
            font-size: 0.9rem;
            display: flex;
            align-items: flex-start;
        }}

        .signal-item::before {{
            margin-right: 10px;
            font-weight: bold;
        }}

        .risk-item {{
            background-color: rgba(255, 107, 107, 0.05);
            border-left: 3px solid var(--fake-color);
            color: #ffd0d0;
        }}
        .risk-item::before {{
            content: "⚠️";
        }}

        .positive-item {{
            background-color: rgba(0, 255, 136, 0.05);
            border-left: 3px solid var(--real-color);
            color: #d0ffd0;
        }}
        .positive-item::before {{
            content: "✅";
        }}

        .score-bar-bg {{
            background-color: rgba(255,255,255,0.05);
            height: 8px;
            border-radius: 10px;
            width: 100%;
            margin-top: 8px;
            overflow: hidden;
        }}
        .score-bar-fill {{
            height: 100%;
            border-radius: 10px;
        }}

        .explanation-text {{
            font-style: italic;
            line-height: 1.5;
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.01);
            padding: 15px;
            border-radius: 6px;
            border-left: 3px solid var(--cyan);
        }}

        .features-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
        }}

        .feature-tag {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
        }}

        footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            font-size: 0.8rem;
            color: var(--text-muted);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="gradient-title">Fake News Intelligence Report</div>
            <div class="subtitle">Generated on {timestamp} &bull; Powered by NLP & ML Classification</div>
        </header>

        <!-- VERDICT & CREDIBILITY -->
        <div class="metrics-grid">
            <div class="metric-card" style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <div style="font-size: 0.9rem; color: var(--text-muted); font-weight: 500;">AI CLASSIFICATION</div>
                <div class="verdict-badge verdict-{'real' if verdict == 'REAL' else 'fake'}">{verdict}</div>
                <div style="font-size: 0.85rem; margin-top: 10px; color: var(--text-muted);">Confidence: {confidence_pct}</div>
            </div>

            <div class="metric-card credibility-ring-container">
                <div style="font-size: 0.9rem; color: var(--text-muted); font-weight: 500;">CREDIBILITY SCORE</div>
                <div class="credibility-ring">
                    <span class="credibility-val">{cred_score}</span>
                </div>
                <div class="grade-badge">Grade: {grade}</div>
            </div>
            
            <div class="metric-card" style="display: flex; flex-direction: column; justify-content: center; text-align: left;">
                <div style="font-size: 0.9rem; color: var(--text-muted); font-weight: 500; margin-bottom: 10px;">ADDITIONAL SIGNALS</div>
                <div>
                    <span style="font-size: 0.85rem; color: var(--text-muted);">Sentiment:</span>
                    <strong style="color: var(--cyan); float: right;">{sentiment_label} ({sentiment_intensity})</strong>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="background-color: var(--cyan); width: {min(100, max(0, int((sentiment_compound + 1) * 50)))}%;"></div>
                    </div>
                </div>
                <div style="margin-top: 12px;">
                    <span style="font-size: 0.85rem; color: var(--text-muted);">Clickbait Score:</span>
                    <strong style="color: {'var(--fake-color)' if is_cb else 'var(--real-color)'}; float: right;">{cb_score}/100</strong>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="background-color: {'var(--fake-color)' if is_cb else 'var(--cyan)'}; width: {cb_score}%;"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ARTICLE CONTENT -->
        <div class="card">
            <div class="card-title">Analyzed Article</div>
            <div class="article-headline">{headline}</div>
            <div class="article-text">{text}</div>
        </div>

        <!-- DETAILED CREDIBILITY BREAKDOWN -->
        <div class="card">
            <div class="card-title">Credibility Signal Breakdown</div>
            <div class="metrics-grid" style="grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 15px; margin-bottom: 0;">
                """

        breakdown = credibility.get("breakdown") or {}
        for key, val in breakdown.items():
            name = key.replace("_score", "").replace("_", " ").title()
            val_int = int(val * 100) if isinstance(val, float) and val <= 1.0 else int(val)
            # Pick a color
            color = "var(--real-color)" if val_int >= 75 else ("var(--warning-color)" if val_int >= 50 else "var(--fake-color)")
            html += f"""
                <div style="background-color: rgba(255, 255, 255, 0.01); border: 1px solid var(--border); padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 6px;">{name}</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: {color};">{val_int}</div>
                    <div class="score-bar-bg" style="height: 4px; margin-top: 6px;">
                        <div class="score-bar-fill" style="background-color: {color}; width: {val_int}%;"></div>
                    </div>
                </div>"""

        html += """
            </div>
        </div>

        <!-- EXPLANATION -->
        <div class="card">
            <div class="card-title">Prediction Explanation</div>
            """

        expl_text = explanation.get("explanation_text")
        if not expl_text and "top_features" in explanation:
            features = explanation.get("top_features") or []
            feats_str = ", ".join([f"{f.get('feature')} ({f.get('importance'):.2f})" for f in features[:5]])
            expl_text = f"This article was classified with {confidence_pct} confidence. Key influencing features include: {feats_str}."

        if not expl_text:
            expl_text = "Detailed linguistic and model feature coefficients indicate patterns correlating with typical historical predictions."

        html += f"""
            <div class="explanation-text">{expl_text}</div>
            """

        top_features = explanation.get("top_features") or []
        if top_features:
            html += """
            <div style="margin-top: 15px;">
                <span style="font-size: 0.85rem; color: var(--text-muted); font-weight: 500;">Top Features / Indicators:</span>
                <div class="features-list">
            """
            for item in top_features[:10]:
                feat_name = item.get("feature") or item.get("word") or "unknown"
                importance = item.get("importance") or item.get("weight") or 0.0
                sign = "+" if importance >= 0 else ""
                html += f"""<span class="feature-tag">{feat_name}: {sign}{importance:.3f}</span>"""
            html += """
                </div>
            </div>
            """

        html += """
        </div>

        <!-- RISK AND POSITIVE SIGNALS -->
        <div class="signals-split">
            <div class="card" style="margin-bottom: 0;">
                <div class="card-title" style="color: var(--fake-color);">Risk Factors</div>
                <ul class="signal-list">
        """

        risks = credibility.get("risk_factors") or []
        if not risks:
            html += """<li class="signal-item positive-item" style="border-left-color: var(--real-color); color: #d0ffd0; background: rgba(0, 255, 136, 0.02);">No significant risk factors detected.</li>"""
        else:
            for r in risks:
                html += f"""<li class="signal-item risk-item">{r}</li>"""

        html += """
                </ul>
            </div>

            <div class="card" style="margin-bottom: 0;">
                <div class="card-title" style="color: var(--real-color);">Positive Signals</div>
                <ul class="signal-list">
        """

        positives = credibility.get("positive_signals") or []
        if not positives:
            html += """<li class="signal-item risk-item" style="border-left-color: var(--warning-color); color: #ffe6aa; background: rgba(255, 170, 0, 0.02);">No strong positive credibility signals detected.</li>"""
        else:
            for p in positives:
                html += f"""<li class="signal-item positive-item">{p}</li>"""

        html += f"""
                </ul>
            </div>
        </div>

        <footer>
            <p>Fake News Intelligence System &bull; Stand-alone Analysis Report &bull; Generated dynamically</p>
            <p style="font-size: 0.7rem; color: rgba(255, 255, 255, 0.2); margin-top: 5px;">Disclaimer: This report is generated by an AI classifier and rule-based scoring system. It is designed to assist in evaluation and does not guarantee absolute factual truth.</p>
        </footer>
    </div>
</body>
</html>
"""
        return html
