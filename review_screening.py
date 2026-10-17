"""Resume cached AI metadata screening: python review_screening.py."""
from pathlib import Path
import sys
import logging
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from youtube_nlp.research.settings import Study
from youtube_nlp.research.review import run_review

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
    try:
        print(run_review(Study.load()))
    except Exception as exc:
        raise SystemExit(f'Review stopped: {type(exc).__name__}. Completed batch caches remain available.') from None
