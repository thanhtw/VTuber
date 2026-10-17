"""Documented pretrained social-media sentiment; VADER is an explicit sensitivity backend."""
import re
import numpy as np


class Sentiment:
    def __init__(self, study):
        self.backend = study.get("sentiment_backend", "transformer")
        self.identifier = study.get("sentiment_model", "cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.revision = study.get("sentiment_revision", "main")
        self.batch_size = study.get("inference_batch_size", 16)
        if self.backend == "vader":
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            try:
                self.vader = SentimentIntensityAnalyzer()
            except LookupError:
                raise ValueError("Run: python -m nltk.downloader vader_lexicon") from None
            self.identifier, self.revision = "nltk.sentiment.vader", "installed-nltk"
        else:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
            except ImportError:
                raise ValueError('Install research dependencies: python -m pip install -e ".[research]" and install PyTorch for your machine.') from None
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.identifier, revision=self.revision)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.identifier, revision=self.revision)
            self.model.eval()
            self.revision = getattr(self.model.config, "_commit_hash", None) or self.revision
            labels = [str(self.model.config.id2label[i]).lower() for i in range(3)]
            if set(labels) != {"negative", "neutral", "positive"}:
                raise ValueError("Sentiment model must declare negative, neutral, and positive labels.")
            self.labels = labels

    def score(self, texts):
        if self.backend == "vader":
            result = []
            for text in texts:
                value = self.vader.polarity_scores(text)
                compound = value['compound']
                label = "positive" if compound >= .05 else "negative" if compound <= -.05 else "neutral"
                result.append({"polarity": label, "positive": value['pos'], "neutral": value['neu'],
                               "negative": value['neg'], "compound": compound})
            return result
        chunks, owners, weights = [], [], []
        limit = min(512, self.tokenizer.model_max_length) - self.tokenizer.num_special_tokens_to_add()
        for owner, text in enumerate(texts):
            cleaned = re.sub(r"https?://\S+", "http", re.sub(r"@\S+", "@user", text))
            tokens = self.tokenizer.encode(cleaned, add_special_tokens=False)
            for start in range(0, max(1, len(tokens)), limit):
                part = tokens[start:start+limit]
                chunks.append(self.tokenizer.prepare_for_model(part, add_special_tokens=True))
                owners.append(owner)
                weights.append(max(1, len(part)))
        aggregates = np.zeros((len(texts), 3))
        counts = np.zeros(len(texts))
        for start in range(0, len(chunks), self.batch_size):
            batch = self.tokenizer.pad(chunks[start:start+self.batch_size], padding=True, return_tensors="pt")
            with self.torch.no_grad():
                scores = self.torch.softmax(self.model(**batch).logits, dim=-1).cpu().numpy()
            for offset, score in enumerate(scores):
                index = start+offset
                aggregates[owners[index]] += score * weights[index]
                counts[owners[index]] += weights[index]
        result = []
        for total, count in zip(aggregates, counts):
            probs = total/count
            result.append({"polarity": self.labels[int(probs.argmax())],
                           **{label: float(probs[i]) for i, label in enumerate(self.labels)},
                           "compound": float(probs[self.labels.index('positive')] - probs[self.labels.index('negative')])})
        return result
