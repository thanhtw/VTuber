"""Manuscript workflow: python research_main.py --help."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
if __name__ == "__main__":
    from youtube_nlp.research.cli import main
    main()
