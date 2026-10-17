from dataclasses import replace
from unittest.mock import Mock, patch
import json
import pytest
from youtube_nlp.config import Config
from youtube_nlp.youtube import YouTubeClient
from youtube_nlp.translation import translate_comments
from youtube_nlp.analysis import model_topics, sentiment_comments


@pytest.fixture
def config(tmp_path):
    return Config("key", "one", 2, 2, "en", 2, 3, 42, tmp_path,
                  keywords=("one", "two"), translate_comments=True, openai_api_key="private")


def test_multi_keyword_deduplicates_and_attributes(config):
    client = YouTubeClient("key")
    def get(resource, **params):
        if resource == "search":
            return {"items": [{"id": {"videoId": "shared"}}]}
        if resource == "videos":
            return {"items": [{"id": "shared", "snippet": {"title": "Music"}}]}
        return {"items": [{"snippet": {"topLevelComment": {"id": "c", "snippet": {"textOriginal": "Nice"}}}}]}
    with patch.object(client, "_get", side_effect=get) as request:
        videos, comments = client.collect(config)
    assert len(videos) == len(comments) == 1
    assert videos[0]["search_keywords"] == ["one", "two"]
    assert comments[0]["search_keywords"] == ["one", "two"]
    assert sum(c.args[0] == "commentThreads" for c in request.call_args_list) == 1
    client.close()


def test_translation_cache_and_sentiment(config):
    response = Mock(ok=True, status_code=200)
    response.json.return_value = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps({"text_en": "I love it!", "source_language": "zh-TW"})}]}]}
    comments = [{"text": "我喜歡", "comment_id": str(i)} for i in range(2)]
    with patch("youtube_nlp.translation.requests.Session") as session:
        post = session.return_value.__enter__.return_value.post
        post.return_value = response
        result = translate_comments(comments, config)
        assert post.call_count == 1
        assert post.call_args.kwargs["json"]["model"] == "gpt-4.1-mini"
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer private"
        assert translate_comments(comments, config) == result
        assert post.call_count == 1
    assert result[0]["text"] == "我喜歡"
    assert result[0]["text_en"] == "I love it!"
    with patch("youtube_nlp.analysis.SentimentIntensityAnalyzer") as analyzer:
        analyzer.return_value.polarity_scores.return_value = {"compound": 0.8}
        assert sentiment_comments(result)[0]["sentiment"] == "positive"
        analyzer.return_value.polarity_scores.assert_called_with("I love it!")


def test_translation_failure_does_not_score_original(config):
    with patch("youtube_nlp.translation.requests.Session") as session:
        session.return_value.__enter__.return_value.post.return_value = Mock(ok=False, status_code=403)
        with pytest.raises(ValueError, match="billing"):
            translate_comments([{"text": "你好"}], config)
    assert not (config.output_dir / "translation_cache.json").exists()


def test_topic_names():
    topics, rows = model_topics([{"video_id": "v", "title": "music singing concert"}], 1, 3, 42, {"1": "Music"})
    assert topics[0]["name"] == rows[0]["topic_name"] == "Music"
    topics, _ = model_topics([{"video_id": "v", "title": "music singing concert"}], 1, 3, 42)
    assert " / " in topics[0]["name"]


def test_config_new_options(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_KEYWORDS", '["music", "gaming", "music"]')
    monkeypatch.setenv("TOPIC_NAMES", '{"1": "Music"}')
    monkeypatch.setenv("TRANSLATE_COMMENTS", "true")
    config = Config.load(str(tmp_path / "missing"), require_key=False)
    assert config.search_keywords == ("music", "gaming")
    assert config.topic_names == {"1": "Music"}
    assert config.translate_comments
    monkeypatch.setenv("SEARCH_KEYWORDS", "music,gaming")
    with pytest.raises(ValueError, match="JSON array"):
        Config.load(str(tmp_path / "missing"), require_key=False)


def test_translated_analysis_exports_names_and_english(config):
    import csv
    from youtube_nlp.analysis import analyze
    comment = {"comment_id": "c", "video_id": "v", "text": "喜歡", "like_count": 0, "published_at": "", "search_keywords": ["one"]}
    translated = {**comment, "text_en": "I love it!", "source_language": "zh-TW"}
    with patch("youtube_nlp.analysis.translate_comments", return_value=[translated]), \
         patch("youtube_nlp.analysis.SentimentIntensityAnalyzer") as analyzer:
        analyzer.return_value.polarity_scores.return_value = {"compound": 0.8, "pos": 1, "neu": 0, "neg": 0}
        analyze([{"video_id": "v", "title": "music concert singing"}], [comment], replace(config, topic_names={"1": "Music"}))
    with (config.output_dir / "comment_sentiment.csv").open() as file:
        row = next(csv.DictReader(file))
        assert row["text"] == "喜歡"
        assert row["text_en"] == "I love it!"
    assert json.loads((config.output_dir / "topics.json").read_text())[0]["name"] == "Music"
    dataset = json.loads((config.output_dir / "analyzed_dataset.json").read_text(encoding="utf-8"))
    assert dataset["comments"][0]["text"] == "喜歡"
    assert dataset["comments"][0]["text_en"] == "I love it!"
    assert dataset["comments"][0]["source_language"] == "zh-TW"
    assert dataset["comments"][0]["sentiment"] == "positive"
    assert dataset["translation_model"] == config.translation_model
    assert dataset["videos"][0]["video_id"] == "v"


@pytest.mark.parametrize("payload,expected", [
    ({"status": "incomplete", "output": []}, "incomplete"),
    ({"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]}, "declined"),
    ({"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "invalid json"}]}]}, "invalid"),
])
def test_gpt_failed_outputs_are_not_cached(config, payload, expected):
    with patch("youtube_nlp.translation.requests.Session") as session:
        response = session.return_value.__enter__.return_value.post.return_value
        response.ok = True
        response.status_code = 200
        response.json.return_value = payload
        with pytest.raises(ValueError, match=expected):
            translate_comments([{"text": "你好"}], config)
    assert not (config.output_dir / "translation_cache.json").exists()


def test_model_change_invalidates_translation_cache(config):
    with patch("youtube_nlp.translation.requests.Session") as session:
        post = session.return_value.__enter__.return_value.post
        post.return_value = Mock(ok=True, status_code=200)
        post.return_value.json.return_value = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps({"text_en": "Hello", "source_language": "zh"})}]}]}
        translate_comments([{"text": "你好"}], config)
        translate_comments([{"text": "你好"}], replace(config, translation_model="other-model"))
        assert post.call_count == 2


def test_pipeline_rejects_disabled_translation_before_scoring(config):
    from youtube_nlp.analysis import analyze
    with patch("youtube_nlp.analysis.SentimentIntensityAnalyzer") as analyzer:
        with pytest.raises(ValueError, match="requires English translation"):
            analyze([], [{"text": "你好"}], replace(config, translate_comments=False))
        analyzer.assert_not_called()


def test_pipeline_rejects_missing_english_before_scoring(config):
    from youtube_nlp.analysis import analyze
    with patch("youtube_nlp.analysis.translate_comments", return_value=[{"text": "你好"}]), \
         patch("youtube_nlp.analysis.sentiment_comments") as sentiment:
        with pytest.raises(ValueError, match="English translation is missing"):
            analyze([], [{"text": "你好"}], config)
        # The sole call checks local lexicon availability, never real comment sentiment.
        sentiment.assert_called_once_with([{"text": "lexicon availability check"}])
