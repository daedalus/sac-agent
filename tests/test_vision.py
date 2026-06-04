from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from sac.core import image_to_data_uri
from sac.vision import VisionSDK


def _fake_llm_client() -> MagicMock:
    client = MagicMock()
    client.call_with_images.return_value = "analysis result"
    return client


class TestImageToDataUri:
    def test_png(self, tmp_path: Path) -> None:
        p = tmp_path / "img.png"
        p.write_bytes(b"fake_png_bytes")
        uri = image_to_data_uri(p)
        assert uri.startswith("data:image/png;base64,")
        encoded = uri.split(",", 1)[1]
        assert base64.b64decode(encoded) == b"fake_png_bytes"

    def test_jpg(self, tmp_path: Path) -> None:
        p = tmp_path / "img.jpg"
        p.write_bytes(b"fake_jpg_bytes")
        uri = image_to_data_uri(p)
        assert uri.startswith("data:image/jpeg;base64,")

    def test_svg_raises_value_error(self, tmp_path: Path) -> None:
        p = tmp_path / "img.svg"
        p.write_bytes(b"<svg></svg>")
        with pytest.raises(ValueError, match="SVG images are not supported"):
            image_to_data_uri(p)

    def test_tiff(self, tmp_path: Path) -> None:
        p = tmp_path / "img.tiff"
        p.write_bytes(b"fake_tiff_bytes")
        uri = image_to_data_uri(p)
        assert uri.startswith("data:image/tiff;base64,")

    def test_unknown_extension_falls_back_to_png(self, tmp_path: Path) -> None:
        p = tmp_path / "img.foo"
        p.write_bytes(b"fake_bytes")
        uri = image_to_data_uri(p)
        assert uri.startswith("data:image/png;base64,")

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        p = tmp_path / "img.png"
        p.write_bytes(b"data")
        uri = image_to_data_uri(Path(p))
        assert uri.startswith("data:image/png;base64,")


class TestVisionSDK:
    def test_analyze_with_file_path(self, tmp_path: Path) -> None:
        llm = _fake_llm_client()
        vs = VisionSDK(llm=llm)
        p = tmp_path / "img.png"
        p.write_bytes(b"img_bytes")
        result = vs.analyze(str(p), "What is this?")
        assert result == "analysis result"
        _, images = llm.call_with_images.call_args[0]
        assert len(images) == 1
        assert images[0].startswith("data:image/png;base64,")

    def test_analyze_with_bytes(self) -> None:
        llm = _fake_llm_client()
        vs = VisionSDK(llm=llm)
        result = vs.analyze(b"raw_bytes", "Describe it")
        assert result == "analysis result"
        _, images = llm.call_with_images.call_args[0]
        assert len(images) == 1
        assert images[0].startswith("data:image/png;base64,")

    def test_analyze_with_data_uri(self) -> None:
        llm = _fake_llm_client()
        vs = VisionSDK(llm=llm)
        data_uri = "data:image/png;base64,abc123"
        result = vs.analyze(data_uri, "Prompt")
        assert result == "analysis result"
        _, images = llm.call_with_images.call_args[0]
        assert images[0] == data_uri

    def test_analyze_url(self) -> None:
        llm = _fake_llm_client()
        vs = VisionSDK(llm=llm)
        fake_image_bytes = b"fake_http_image"
        encoded = base64.b64encode(fake_image_bytes).decode()

        with patch.object(httpx, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.content = fake_image_bytes
            mock_response.headers = {"content-type": "image/webp"}
            mock_get.return_value = mock_response

            result = vs.analyze_url("https://example.com/img.webp", "Analyze")
            assert result == "analysis result"

        _, images = llm.call_with_images.call_args[0]
        assert images[0] == f"data:image/webp;base64,{encoded}"

    def test_analyze_url_http_error(self) -> None:
        llm = _fake_llm_client()
        vs = VisionSDK(llm=llm)
        with patch.object(httpx, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404", request=MagicMock(), response=MagicMock()
            )
            mock_get.return_value = mock_response
            with pytest.raises(httpx.HTTPStatusError):
                vs.analyze_url("https://example.com/404", "Analyze")
