from unittest.mock import Mock
import pytest
import requests
from youtube_nlp.youtube import YouTubeClient, YouTubeError


def test_api_errors_do_not_expose_key():
    client = YouTubeClient("secret-key")
    response = Mock(ok=False, status_code=403)
    response.json.return_value = {"error": {"errors": [{"reason": "quotaExceeded"}]}}
    client.session.get = Mock(return_value=response)
    with pytest.raises(YouTubeError, match="quotaExceeded") as caught:
        client._get("search", q="test")
    assert "secret-key" not in str(caught.value)
    client.close()


def test_timeout_retries_are_limited_and_logged(caplog):
    from unittest.mock import patch
    client = YouTubeClient("secret-key")
    client.session.get = Mock(side_effect=requests.Timeout("secret-key"))
    with patch("youtube_nlp.youtube.time.sleep"), pytest.raises(requests.Timeout):
        client._get("search", q="test")
    assert client.session.get.call_count == 3
    assert client.session.get.call_args.kwargs["timeout"] == (5, 20)
    assert "retrying" in caplog.text
    assert "secret-key" not in caplog.text
    client.close()


@pytest.mark.parametrize("failure,code,message", [
    (KeyboardInterrupt(), 130, "Cancelled by user"),
    (requests.Timeout("secret-key"), 1, "timed out after retries"),
])
def test_cli_handles_interruption_and_timeout(failure, code, message, tmp_path, capsys):
    from unittest.mock import patch
    from youtube_nlp.cli import main
    from youtube_nlp.config import Config
    config = Config("secret-key", "test", 1, 1, "en", 1, 3, 42, tmp_path, translate_comments=True, openai_api_key="fake-openai")
    with patch("sys.argv", ["youtube-nlp", "run"]), \
         patch("youtube_nlp.cli.Config.load", return_value=config), \
         patch("youtube_nlp.cli.YouTubeClient") as client:
        client.return_value.collect.side_effect = failure
        with pytest.raises(SystemExit) as caught:
            main()
        client.return_value.close.assert_called_once()
    assert caught.value.code == code
    stderr = capsys.readouterr().err
    assert message in stderr
    assert "secret-key" not in stderr
    assert "Traceback" not in stderr
