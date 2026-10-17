from dataclasses import replace
from unittest.mock import patch
import pytest
from youtube_nlp.config import Config
from youtube_nlp.youtube import YouTubeClient, YouTubeError
from youtube_nlp.analysis import analyze, model_topics, sentiment_comments


@pytest.fixture
def config(tmp_path):
    return Config("fake", "test", 2, 2, "en", 5, 4, 42, tmp_path)


def test_pagination_and_disabled_comments(config):
    client = YouTubeClient("fake")
    calls = []
    def fake_get(resource, **params):
        calls.append((resource, params))
        if resource == "search":
            if "pageToken" not in params:
                return {"items": [{"id": {"videoId": "a"}}], "nextPageToken": "next"}
            return {"items": [{"id": {"videoId": "b"}}]}
        if resource == "videos":
            return {"items": [{"id": i, "snippet": {"title": i}} for i in ("a", "b")]}
        if params["videoId"] == "b":
            raise YouTubeError(403, "commentsDisabled")
        page = 2 if "pageToken" in params else 1
        return {"items": [{"snippet": {"topLevelComment": {
            "id": str(page), "snippet": {"textOriginal": "Great!"}}}}],
            **({"nextPageToken": "more"} if page == 1 else {})}
    with patch.object(client, "_get", side_effect=fake_get):
        videos, comments = client.collect(config)
    client.close()
    assert len(videos) == len(comments) == 2
    assert [c["comment_id"] for c in comments] == ["1", "2"]
    assert calls[1][1]["pageToken"] == "next"


def test_lda_probabilities(config):
    videos = [{"video_id": "a", "title": "music singing concert"},
              {"video_id": "b", "title": "python programming software"}]
    topics, rows = model_topics(videos, 5, 3, 42)
    assert len(topics) == 2
    assert all(sum(r[f"topic_{i}"] for i in (1, 2)) == pytest.approx(1) for r in rows)
    assert model_topics([{"video_id": "a", "title": "the and"}], 5, 3, 42) == ([], [])


def test_sentiment_thresholds():
    with patch("youtube_nlp.analysis.SentimentIntensityAnalyzer") as analyzer:
        analyzer.return_value.polarity_scores.side_effect = [
            {"compound": v, "neg": 0, "neu": 1, "pos": 0} for v in [0.05, -0.05, 0]]
        rows = sentiment_comments([{"text": "test"} for _ in range(3)])
    assert [row["sentiment"] for row in rows] == ["positive", "negative", "neutral"]


def test_empty_outputs(config):
    analyze([], [], config)
    for name in ["analyzed_dataset.json", "topics.json", "video_topics.csv", "comment_sentiment.csv", "summary.json", "topics.png", "sentiment.png"]:
        assert (config.output_dir / name).stat().st_size > 0


def test_config_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("MAX_VIDEOS", "0")
    with pytest.raises(ValueError, match="MAX_VIDEOS"):
        Config.load(str(tmp_path / "absent"), require_key=False)
