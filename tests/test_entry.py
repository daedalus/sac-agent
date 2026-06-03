from __future__ import annotations

from unittest.mock import patch

from sac.__main__ import _main


class TestMain:
    def test_main_with_args(self):
        with patch("sac.__main__.main") as mock_main:
            with patch("sac.__main__.interactive") as mock_interactive:
                with patch("sys.argv", ["sac", "task"]):
                    _main()
                    mock_main.assert_called_once()
                    mock_interactive.assert_not_called()

    def test_main_without_args(self):
        with patch("sac.__main__.main") as mock_main:
            with patch("sac.__main__.interactive") as mock_interactive:
                with patch("sys.argv", ["sac"]):
                    _main()
                    mock_main.assert_not_called()
                    mock_interactive.assert_called_once()

    def test_main_returns_zero(self):
        with patch("sac.__main__.main"):
            with patch("sys.argv", ["sac", "task"]):
                assert _main() == 0
