import os
from pathlib import Path
import subprocess
import sys

from youtube_nlp.config import Config

ROOT = Path(__file__).resolve().parents[1]


def test_python_file_runs_from_other_folder(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "main.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert "check" in result.stdout


def test_check_reports_missing_key_without_traceback(tmp_path):
    env_file = tmp_path / "research.env"
    env_file.write_text("YOUTUBE_API_KEY=fake\nOPENAI_API_KEY=\nTRANSLATE_COMMENTS=true\n")
    env = {k: v for k, v in os.environ.items() if k not in {"YOUTUBE_API_KEY", "OPENAI_API_KEY"}}
    result = subprocess.run([sys.executable, str(ROOT / "main.py"), "check", "--env", str(env_file)],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert "OPENAI_API_KEY" in result.stderr
    assert str(env_file) in result.stderr
    assert "No network requests" in result.stderr
    assert "Traceback" not in result.stderr


def test_config_files_are_independent_and_paths_relative_to_env(tmp_path, monkeypatch):
    for key in ["SEARCH_KEYWORDS", "SEARCH_KEYWORD", "OUTPUT_DIR"]:
        monkeypatch.delenv(key, raising=False)
    first, second = tmp_path / "one.env", tmp_path / "two.env"
    first.write_text("SEARCH_KEYWORD=music\nOUTPUT_DIR=results\n")
    second.write_text("SEARCH_KEYWORD=gaming\n")
    a = Config.load(first, require_key=False)
    b = Config.load(second, require_key=False)
    assert a.search_keywords == ("music",)
    assert b.search_keywords == ("gaming",)
    assert a.output_dir == tmp_path / "results"
    assert b.output_dir == tmp_path / "output"


def test_shell_environment_has_documented_precedence(tmp_path, monkeypatch):
    path = tmp_path / "test.env"
    path.write_text("SEARCH_KEYWORD=music\n")
    monkeypatch.delenv("SEARCH_KEYWORDS", raising=False)
    monkeypatch.setenv("SEARCH_KEYWORD", "gaming")
    assert Config.load(path, require_key=False).search_keywords == ("gaming",)
