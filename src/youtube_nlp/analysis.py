"""LDA on video metadata and VADER on English or original comment text."""
import csv
import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from .translation import translate_comments
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


log = logging.getLogger(__name__)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def model_topics(videos, num_topics, top_words, random_state, topic_names=None):
    documents = [" ".join([v.get("title", ""), v.get("description", ""),
                           " ".join(v.get("tags", []))]) for v in videos]
    if not documents:
        return [], []
    vectorizer = CountVectorizer(stop_words="english", strip_accents="unicode", max_features=10000)
    try:
        counts = vectorizer.fit_transform(documents)
    except ValueError as exc:
        if "empty vocabulary" in str(exc):
            return [], []
        raise
    n_topics = min(num_topics, counts.shape[0], counts.shape[1])
    model = LatentDirichletAllocation(n_components=n_topics, random_state=random_state,
                                     learning_method="batch", max_iter=20)
    probabilities = model.fit_transform(counts)
    vocabulary = vectorizer.get_feature_names_out()
    topics = [{"topic_id": i + 1,
               "words": [{"word": str(vocabulary[j]), "weight": float(weights[j])}
                         for j in weights.argsort()[::-1][:top_words]]}
              for i, weights in enumerate(model.components_)]
    for topic in topics:
        topic["name"] = (topic_names or {}).get(str(topic["topic_id"])) or " / ".join(
            word["word"] for word in topic["words"][:3])
    assignments = [{"video_id": video["video_id"], "title": video.get("title", ""),
                    "dominant_topic": int(scores.argmax()) + 1,
                    "topic_name": topics[int(scores.argmax())]["name"],
                    **{f"topic_{i + 1}": float(score) for i, score in enumerate(scores)}}
                   for video, scores in zip(videos, probabilities)]
    return topics, assignments


def sentiment_comments(comments):
    if not comments:
        return []
    try:
        analyzer = SentimentIntensityAnalyzer()
    except LookupError:
        raise ValueError("VADER lexicon is missing. Run: python -m nltk.downloader vader_lexicon") from None
    rows = []
    for comment in comments:
        scores = analyzer.polarity_scores(comment.get("text_en", comment["text"]))
        label = "positive" if scores["compound"] >= 0.05 else "negative" if scores["compound"] <= -0.05 else "neutral"
        rows.append({**comment, **scores, "sentiment": label})
    return rows


def analyze(videos, comments, config):
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    topics, assignments = model_topics(videos, config.num_topics, config.top_words, config.random_state, config.topic_names)
    if comments and not config.translate_comments:
        raise ValueError("Comment analysis requires English translation. Set TRANSLATE_COMMENTS=true and OPENAI_API_KEY in .env.")
    if comments:
        sentiment_comments([{"text": "lexicon availability check"}])
    if comments:
        comments = translate_comments(comments, config)
        if any("text_en" not in comment or not isinstance(comment["text_en"], str)
               or (comment["text"].strip() and not comment["text_en"].strip()) for comment in comments):
            raise ValueError("English translation is missing for a comment. Analysis stopped before sentiment scoring.")
    sentiments = sentiment_comments(comments)
    write_json(output / "analyzed_dataset.json", {
        "keywords": list(config.search_keywords),
        "translated_to_english": config.translate_comments,
        "translation_model": config.translation_model if config.translate_comments else None,
        "videos": videos,
        "comments": sentiments,
    })
    log.info("Saved comments with original text, English text, and sentiment to %s",
             output / "analyzed_dataset.json")
    write_json(output / "topics.json", topics)
    write_csv(output / "video_topics.csv", assignments,
              ["video_id", "title", "dominant_topic", "topic_name"] + [f"topic_{i + 1}" for i in range(len(topics))])
    write_csv(output / "comment_sentiment.csv", sentiments,
              ["comment_id", "video_id", "text", "text_en", "source_language", "like_count", "published_at", "search_keywords", "neg", "neu", "pos", "compound", "sentiment"])
    totals = {label: sum(row["sentiment"] == label for row in sentiments)
              for label in ("positive", "neutral", "negative")}
    write_json(output / "summary.json", {"keyword": config.keyword, "keywords": list(config.search_keywords), "translated_to_english": config.translate_comments, "videos": len(videos),
               "comments": len(comments), "topics": len(topics), "sentiment_counts": totals})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(list(totals), list(totals.values()), color=["seagreen", "gray", "indianred"])
    axes[0].set(title="Comment sentiment", ylabel="Comments")
    axes[1].hist([row["compound"] for row in sentiments], bins=20, range=(-1, 1), color="steelblue")
    axes[1].set(title="VADER compound scores", xlabel="Compound score", ylabel="Comments")
    if not sentiments:
        axes[1].text(0.5, 0.5, "No comments available", transform=axes[1].transAxes, ha="center")
    fig.tight_layout()
    fig.savefig(output / "sentiment.png", dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar([f"{t['topic_id']}: {t['name']}" for t in topics],
             [sum(row["dominant_topic"] == t["topic_id"] for row in assignments) for t in topics])
    axis.set(title="Video metadata topics", ylabel="Videos")
    axis.tick_params(axis="x", labelrotation=25)
    if not topics:
        axis.text(0.5, 0.5, "No usable metadata for topic modeling", transform=axis.transAxes, ha="center")
    fig.tight_layout()
    fig.savefig(output / "topics.png", dpi=150)
    plt.close(fig)
