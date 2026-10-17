"""Command line entry point."""
import argparse
import json
import logging
from datetime import datetime, timezone
import requests
from .analysis import analyze, write_json, sentiment_comments
from .config import Config, DEFAULT_ENV
from .youtube import YouTubeClient, YouTubeError


def main():
    parser = argparse.ArgumentParser(description="YouTube metadata LDA and comment VADER analysis")
    parser.add_argument("command", choices=["collect", "analyze", "run", "check"], nargs="?", default="run")
    parser.add_argument("--env", default=str(DEFAULT_ENV), help="Configuration file (default: project .env)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = Config.load(args.env, require_key=args.command not in {"analyze", "check"})
        from pathlib import Path
        logging.info("Configuration: %s", Path(args.env).expanduser().resolve())
        if args.command == "check":
            missing = []
            if not config.api_key or config.api_key == "replace_with_your_api_key":
                missing.append("YOUTUBE_API_KEY")
            if config.translate_comments and not config.openai_api_key:
                missing.append("OPENAI_API_KEY")
            if missing:
                raise ValueError(f"Missing {', '.join(missing)} in {Path(args.env).resolve()}. "
                                 "Add your API credentials to that file. No network requests were made.")
            sentiment_comments([{"text": "setup check"}])
            logging.info("Local configuration and VADER are ready. API credentials have not been verified online.")
            return
        if args.command in {"run", "analyze"} and not config.translate_comments:
            raise ValueError("Comment analysis requires TRANSLATE_COMMENTS=true and OPENAI_API_KEY in .env. "
                             "Use python main.py collect for collection only.")
        if args.command in {"run", "analyze"} and config.translate_comments and not config.openai_api_key:
            raise ValueError(f"OPENAI_API_KEY is missing in {Path(args.env).resolve()}. "
                             "Add your own OpenAI API key to enable GPT translation. "
                             "To collect without translation first, run: python main.py collect")
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if args.command in {"collect", "run"}:
            client = YouTubeClient(config.api_key)
            try:
                videos, comments = client.collect(config)
            finally:
                client.close()
            write_json(config.output_dir / "dataset.json", {
                "keyword": config.keyword, "keywords": list(config.search_keywords), "collected_at": datetime.now(timezone.utc).isoformat(),
                "videos": videos, "comments": comments})
        else:
            dataset_path = config.output_dir / "dataset.json"
            if not dataset_path.exists():
                raise ValueError(f"No saved dataset at {dataset_path}. Run: python main.py collect")
            data = json.loads(dataset_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("videos"), list) or not isinstance(data.get("comments"), list):
                raise ValueError("dataset.json must contain videos and comments arrays.")
            videos, comments = data["videos"], data["comments"]
            # Label re-analysis with the actual collection keyword.
            from dataclasses import replace
            config = replace(config, keyword=data.get("keyword", config.keyword),
                             keywords=tuple(data.get("keywords", [data.get("keyword", config.keyword)])))
        if args.command in {"analyze", "run"}:
            analyze(videos, comments, config)
        logging.info("Done. Results: %s", config.output_dir.resolve())
    except KeyboardInterrupt:
        parser.exit(130, "\nCancelled by user. Collection is saved only after it completes.\n")
    except requests.Timeout:
        parser.exit(1, "API request timed out after retries. Check connectivity to YouTube and OpenAI, VPN/proxy settings, and try again.\n")
    except requests.ConnectionError:
        parser.exit(1, "Cannot connect to API services after retries. Check your internet connection, DNS, and VPN/proxy settings.\n")
    except requests.RequestException:
        # Request exceptions may contain URLs with the API key; never print them.
        parser.exit(1, "Network request failed. Check connectivity and retry.\n")
    except (ValueError, OSError, KeyError, YouTubeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
