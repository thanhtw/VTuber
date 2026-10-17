# YouTube NLP

Collect YouTube video metadata and top-level comments for a configurable search keyword, discover metadata topics with scikit-learn Latent Dirichlet Allocation (LDA), score comments using NLTK VADER, and generate Matplotlib charts.

## Run as a Python file (Anaconda)

```bash
conda activate VTuber
cd /home/selab/Desktop/Thomas/VTuber
python -m pip install -e "."
python main.py check
python main.py
```

`python main.py` collects and analyzes. Set your own `YOUTUBE_API_KEY` and `OPENAI_API_KEY` in the project's `.env` first. GPT translation requires an OpenAI API key; the program cannot create one for you. `check` verifies local configuration and the VADER lexicon without making API calls. It does not verify credentials online. If the lexicon is missing, run `python -m nltk.downloader vader_lexicon`.

```bash
python main.py collect   # Collect without needing the OpenAI key
python main.py analyze   # Translate/analyze a previously saved dataset
python main.py --env research.env
```

If both `.venv` and Conda appear in your prompt, activate only the Conda environment you intend to use. Check the interpreter with `python -c "import sys; print(sys.executable)"`.

## Setup

Python 3.10+ is required.

Using your existing Anaconda environment:

```bash
conda activate VTuber
python -m pip install -e '.[dev]'
python -m nltk.downloader vader_lexicon
cp .env.example .env  # First setup only; preserve an existing .env
```

Alternatively, create a virtual environment with `python -m venv .venv` and activate it with `source .venv/bin/activate` before installing.

If `youtube-nlp` is not found, run `python -m youtube_nlp run`. If Python cannot find the module, run `python -m pip install -e .` in the activated environment from this project directory.

Create a Google Cloud project, enable **YouTube Data API v3**, and create an API key. Add it to `.env`. Keep this file private; restrict the key to the YouTube Data API in Google Cloud.

```dotenv
YOUTUBE_API_KEY=your_key
SEARCH_KEYWORD=artificial intelligence
# Optional: overrides SEARCH_KEYWORD; JSON array inside single quotes
SEARCH_KEYWORDS='["VTuber", "virtual idol", "gaming"]'
TRANSLATE_COMMENTS=true
OPENAI_API_KEY=your_openai_api_key
TRANSLATION_MODEL=gpt-4.1-mini
TOPIC_NAMES='{}'
MAX_VIDEOS=20
MAX_COMMENTS_PER_VIDEO=100
SEARCH_LANGUAGE=en
NUM_TOPICS=5
TOP_WORDS=10
RANDOM_STATE=42
OUTPUT_DIR=output
```

Change `SEARCH_KEYWORD` to change the search. Environment variables already exported in your shell take precedence over `.env`. With `SEARCH_KEYWORDS`, `MAX_VIDEOS` applies to each keyword. Videos and comments are deduplicated across searches, and `search_keywords` records matching searches in the saved data. Collection still retrieves top-level comments only, up to the configured cap; translating all collected comments does not remove that cap. Limits are per run; set comments to `0` for metadata only. `SEARCH_LANGUAGE` biases search relevance and does not strictly filter languages. Set it empty to omit the language hint. By default the program reads `.env` beside `main.py`, regardless of your current working directory. `--env` selects another file. Relative output paths are resolved relative to the selected environment file, including when `OUTPUT_DIR` is supplied through the shell.

## Run

```bash
youtube-nlp run                  # collect, then analyze
youtube-nlp collect              # API collection only
youtube-nlp analyze              # analyze saved data without an API key
youtube-nlp run --env research.env
# Equivalent: python -m youtube_nlp run
```

Outputs in `OUTPUT_DIR`:

- `dataset.json`: raw metadata, original top-level comments, search keyword, and UTC collection time.
- `analyzed_dataset.json`: video metadata and analyzed comments, including original `text`, English translation `text_en`, detected `source_language`, VADER scores, and sentiment. Translation model and keyword settings are included. This file is written after successful analysis; `dataset.json` stays available for reanalysis.
- `topics.json`: topic IDs, names, and highest-weight words.
- `video_topics.csv`: each video's dominant topic and topic probabilities.
- `comment_sentiment.csv`: original `text`, English `text_en`, detected `source_language`, matching keywords, VADER scores, and sentiment labels.
- `translation_cache.json`: English translations reused across analysis runs in the same output directory.
- `summary.json`: sample counts and sentiment totals.
- `topics.png`: number of videos assigned to each topic.
- `sentiment.png`: sentiment counts and compound-score histogram.

Files are overwritten on subsequent runs in the same output directory. Use separate output directories to preserve experiments. Collection is saved before analysis so it can be reanalyzed if a lexicon is missing. Disabled comments and deleted videos are skipped with a warning; authentication, quota, and other API errors stop collection. Transient HTTP errors, connection failures, and timeouts make up to three attempts, with progress logged for each request. Each attempt uses a 5-second connection timeout and a 20-second read timeout. Ctrl+C exits cleanly; incomplete collection is not saved. Search and comment requests paginate up to the configured limits; API quota may prevent completion.

## Interpretation and limitations

Each video's title, description, and tags form one LDA document. CountVectorizer lowercases text and removes English stop words. Topic count is capped by the number of videos and vocabulary size. Empty or stop-word-only metadata produces empty topic outputs. Small samples and repeated descriptions can produce weak topics; collect a larger, relevant corpus for meaningful results.

By default, VADER scores English translations. English translation is required before comment sentiment analysis. Setting `TRANSLATE_COMMENTS=false` stops analysis with a setup message; collection-only mode remains available. Compound scores >= 0.05 are positive, <= -0.05 negative, otherwise neutral. The lexicon and stop-word setup target English; translation can change sentiment cues, and sarcasm or domain slang may still be inaccurate. Video metadata is not translated, so LDA remains limited by multilingual metadata and its English stop words. Language hints do not guarantee English comments. Topic IDs are statistical clusters; names are descriptive labels rather than model predictions.

Collection includes public **top-level comments only**, without replies. Comment order is newest first, so the sample is not representative of all viewers. No commenter names are collected, but comment text may contain personal data. Charts work without a graphical desktop.

## Tests

```bash
pytest
```

Tests use fake API responses and local fixtures; no API key is required.

References: [YouTube API](https://developers.google.com/youtube/v3/docs), [commentThreads.list](https://developers.google.com/youtube/v3/docs/commentThreads/list), [NLTK VADER](https://www.nltk.org/api/nltk.sentiment.vader.html).

## English comment translation with GPT

Set `OPENAI_API_KEY` in `.env`. The project calls the OpenAI Responses API directly using `requests`; no additional Python package is needed. `TRANSLATION_MODEL` defaults to `gpt-4.1-mini` and can be changed to a model supporting Responses and structured JSON outputs. Google Translation credentials are no longer used. Your YouTube API key remains required for collection.

```dotenv
TRANSLATE_COMMENTS=true
OPENAI_API_KEY=your_openai_api_key
TRANSLATION_MODEL=gpt-4.1-mini
```

Every nonempty collected comment is sent to GPT for English translation and source-language identification. GPT is instructed to retain meaning, sentiment, negation, slang, punctuation, and emoji, and to treat comment text as data rather than instructions. English text is requested unchanged. Original text remains in `dataset.json`, `analyzed_dataset.json`, and the sentiment CSV. English translations are included in `analyzed_dataset.json` and `comment_sentiment.csv` as `text_en`; VADER scores `text_en`. Translation can still alter sentiment cues.

Duplicate texts reuse cached results. The cache includes the provider, model, target language, and prompt version in its key so Google translations and results from other GPT models are not reused. Successfully completed comments are cached immediately; rerunning resumes from that cache. Long comments are split without truncation. API errors, refusals, and invalid or incomplete responses stop analysis without silently using original text. Translation requests use a 5-second connection timeout and 60-second read timeout with up to three attempts for transient failures.

Translation sends text to the OpenAI API and incurs API usage charges. Keep the key in the ignored `.env` file. Live tests need your credentials; automated tests mock responses.

Translate existing collected comments without collecting again:

```bash
conda activate VTuber
python -m youtube_nlp analyze
```

See [official OpenAI documentation on structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses).

## Naming topics

LDA discovers word distributions and does not generate semantic category names. The project initially labels each topic with its three strongest words. Inspect `topics.json` and the corresponding titles in `video_topics.csv`, then set human-readable names in `.env`:

```dotenv
TOPIC_NAMES='{"1": "Music and singing", "2": "Gaming", "3": "Community discussion"}'
```

Run `python -m youtube_nlp analyze` again. Names appear in `topics.json`, `video_topics.csv`, and `topics.png`. Unspecified IDs keep their automatic word labels. Names do not affect clustering. Topic IDs can change when the dataset or model settings change, so review the names after new collection. Reanalyzing the same dataset with the same settings uses the fixed random seed and cached translations.
