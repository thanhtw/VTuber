"""Staged manuscript workflow; no automated replacement for human screening."""
import argparse
import logging
from pathlib import Path
import requests
from ..config import DEFAULT_ENV, PROJECT_ROOT
from ..youtube import YouTubeError
from .settings import Study
from .collection import RegisteredClient, discover, collect
from .screening import report, update_template
from .storage import save, read_json


def main():
    parser = argparse.ArgumentParser(description='Educational VTuber manuscript RQ1–RQ3 research workflow')
    parser.add_argument('command', choices=['check', 'discover', 'screening-report', 'collect', 'analyze'])
    parser.add_argument('--env', default=str(DEFAULT_ENV))
    parser.add_argument('--study', default=str(PROJECT_ROOT / 'research/study.json'))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        study = Study.load(args.env, args.study)
        study.prepare()
        if args.command == 'check':
            import importlib.util
            missing = [name for name in ['gensim', 'networkx', 'transformers', 'torch'] if importlib.util.find_spec(name) is None]
            print('Study output:', study.root)
            print('YouTube API key:', 'configured' if study.api.api_key and study.api.api_key != 'replace_with_your_api_key' else 'missing')
            print('OpenAI API key:', 'configured' if study.api.openai_api_key else 'missing')
            print('Missing research packages:', ', '.join(missing) or 'none')
            print('No network requests made. Human screening and translation/aspect audits remain required.')
            if missing or not study.api.openai_api_key or not study.api.api_key or study.api.api_key == 'replace_with_your_api_key':
                parser.exit(1, 'Complete setup before live research collection and analysis. See research/README.md.\n')
        elif args.command == 'screening-report':
            update_template(study.root / 'screening.csv', read_json(study.root / 'candidates.json', []))
            update_template(study.root / 'channel_screening.csv', read_json(study.root / 'channel_candidates.json', []))
            save(study.root / 'screening_summary.json', {'videos': report(study.root / 'screening.csv'), 'channels': report(study.root / 'channel_screening.csv')})
            print('Saved screening_summary.json')
        elif args.command in {'discover', 'collect'}:
            if not study.api.api_key or study.api.api_key == 'replace_with_your_api_key':
                raise ValueError('Set YOUTUBE_API_KEY in your .env.')
            client = RegisteredClient(study.api.api_key, study.root)
            try:
                (discover if args.command == 'discover' else collect)(study, client)
            finally:
                client.close()
        else:
            from .pipeline import run_analysis
            run_analysis(study)
    except KeyboardInterrupt:
        parser.exit(130, '\nCancelled. API registry and completed checkpoints remain available.\n')
    except ImportError:
        parser.exit(1, 'Research dependencies are missing. Run: python -m pip install -e ".[research]"\n')
    except requests.RequestException:
        parser.exit(1, 'API request failed. Check network access and retry; credentials were not printed.\n')
    except (ValueError, KeyError, OSError, YouTubeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
