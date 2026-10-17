"""Run from a checkout: python main.py [run|collect|analyze|check]."""
from pathlib import Path
import sys


def main():
    # Support direct execution without requiring an editable package installation.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    try:
        from youtube_nlp.cli import main as run
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}.\n"
            'Activate your Conda environment, then run: python -m pip install -e "."'
        ) from None
    run()


if __name__ == "__main__":
    main()
