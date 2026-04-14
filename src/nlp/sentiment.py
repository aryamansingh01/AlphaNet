"""Financial sentiment analysis using FinBERT and VADER."""

import pandas as pd


class SentimentAnalyzer:
    """Score financial text for sentiment (bullish/bearish/neutral)."""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Lazy load FinBERT model."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def score_text(self, text: str) -> dict:
        """Score a single text. Returns {label, score, probabilities}."""
        if self.model is None:
            self.load_model()

        import torch

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        labels = ["positive", "negative", "neutral"]
        scores = {label: round(prob.item(), 4) for label, prob in zip(labels, probs)}
        top_label = max(scores, key=scores.get)

        return {"label": top_label, "score": scores[top_label], "probabilities": scores}

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
        """Score a dataframe of texts. Adds sentiment columns."""
        results = df[text_col].apply(self.score_text)
        df["sentiment_label"] = results.apply(lambda x: x["label"])
        df["sentiment_score"] = results.apply(lambda x: x["score"])
        return df
