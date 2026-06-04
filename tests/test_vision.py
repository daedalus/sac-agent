from __future__ import annotations

import base64
import os
import tempfile
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
    def test_png(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake_png_bytes")
            p = f.name
        try:
            uri = image_to_data_uri(p)
            assert uri.startswith("data:image/png;base64,")
            encoded = uri.split(",", 1)[1]
            assert base64.b64decode(encoded) == b"fake_png_bytes"
        finally:
            os.unlink(p)

    def test_jpg(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake_jpg_bytes")
            p = f.name
        try:
            uri = image_to_data_uri(p)
            assert uri.startswith("data:image/jpeg;base64,")
        finally:
            os.unlink(p)

    def test_svg(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            f.write(b"<svg></svg>")
            p = f.name
        try:
            uri = image_to_data_uri(p)
            assert uri.startswith("data:image/svg+xml;base64,")
        finally:
            os.unlink(p)

    def test_tiff(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as f:
            f.write(b"fake_tiff_bytes")
            p = f.name
        try:
            uri = image_to_data_uri(p)
            assert uri.startswith("data:image/tiff;base64,")
        finally:
            os.unlink(p)

    def test_unknown_extension_falls_back_to_png(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".foo", delete=False) as f:
            f.write(b"fake_bytes")
            p = f.name
        try:
            uri = image_to_data_uri(p)
            assert uri.startswith("data:image/png;base64,")
        finally:
            os.unlink(p)

    def test_accepts_path_object(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"data")
            p = f.name
        try:
            uri = image_to_data_uri(Path(p))
            assert uri.startswith("data:image/png;base64,")
        finally:
            os.unlink(p)


class TestVisionSDK:
    def test_analyze_with_file_path(self) -> None:
        llm = _fake_llm_client()
        vs = VisionSDK(llm=llm)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"img_bytes")
            p = f.name
        try:
            result = vs.analyze(p, "What is this?")
            assert result == "analysis result"
            _, images = llm.call_with_images.call_args[0]
            assert len(images) == 1
            assert images[0].startswith("data:image/png;base64,")
        finally:
            os.unlink(p)

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
