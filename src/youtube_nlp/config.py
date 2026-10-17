"""Environment configuration with early validation."""
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class Config:
    api_key: str
    keyword: str
    max_videos: int
    max_comments: int
    language: str
    num_topics: int
    top_words: int
    random_state: int
    output_dir: Path

    keywords: tuple[str, ...] = ()
    translate_comments: bool = False
    openai_api_key: str = ""
    topic_names: dict[str, str] = field(default_factory=dict)
    translation_model: str = "gpt-4.1-mini"

    @property
    def search_keywords(self):
        return self.keywords or (self.keyword,)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ENV, require_key: bool = True):
        env_path = Path(path).expanduser().resolve()
        # Read without modifying process state: repeated runs can use different env files.
        values = {**dotenv_values(env_path), **os.environ}
        def get(name, default=""):
            value = values.get(name, default)
            return default if value is None else value
        key = get("YOUTUBE_API_KEY", "").strip()
        keyword = get("SEARCH_KEYWORD", "VTuber").strip()
        raw_keywords = get("SEARCH_KEYWORDS", "").strip()
        try:
            keywords = json.loads(raw_keywords) if raw_keywords else [keyword]
        except json.JSONDecodeError:
            raise ValueError('SEARCH_KEYWORDS must be a JSON array, e.g. ["VTuber", "gaming"].') from None
        if not isinstance(keywords, list) or not keywords or any(not isinstance(k, str) or not k.strip() for k in keywords):
            raise ValueError("SEARCH_KEYWORDS must contain nonempty strings.")
        keywords = tuple(dict.fromkeys(k.strip() for k in keywords))
        try:
            names = json.loads(get("TOPIC_NAMES", "{}"))
        except json.JSONDecodeError:
            raise ValueError('TOPIC_NAMES must be a JSON object, e.g. {"1": "Music"}.') from None
        if not isinstance(names, dict) or any(not k.isdigit() or int(k) < 1 or not isinstance(v, str) or not v.strip() for k, v in names.items()):
            raise ValueError("TOPIC_NAMES must map positive topic IDs to nonempty names.")
        translate = get("TRANSLATE_COMMENTS", "true").strip().lower()
        if translate not in {"true", "false"}:
            raise ValueError("TRANSLATE_COMMENTS must be true or false.")
        if require_key and (not key or key == "replace_with_your_api_key"):
            raise ValueError(f"Set YOUTUBE_API_KEY in {env_path} before collecting data.")
        if not keyword and not raw_keywords:
            raise ValueError("SEARCH_KEYWORD cannot be empty.")
        def integer(name, default, minimum=1):
            try:
                value = int(get(name, str(default)))
            except ValueError:
                raise ValueError(f"{name} must be an integer.") from None
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}.")
            return value
        output_dir = Path(get("OUTPUT_DIR", "output")).expanduser()
        if not output_dir.is_absolute():
            output_dir = env_path.parent / output_dir
        return cls(key, keywords[0], integer("MAX_VIDEOS", 20),
                   integer("MAX_COMMENTS_PER_VIDEO", 100, 0),
                   get("SEARCH_LANGUAGE", "en").strip(),
                   integer("NUM_TOPICS", 5), integer("TOP_WORDS", 10),
                   integer("RANDOM_STATE", 42, 0),
                   output_dir, keywords,
                   translate == "true", get("OPENAI_API_KEY", "").strip(), names,
                   get("TRANSLATION_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini")
