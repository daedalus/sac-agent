from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import httpx

from sac.core import image_to_data_uri

if TYPE_CHECKING:
    from sac.llm import LLMSDKClient


class VisionSDK:
    def __init__(self, llm: LLMSDKClient) -> None:
        self._llm = llm

    def analyze(self, image: str | bytes, prompt: str) -> str:
        if isinstance(image, str):
            if image.startswith("data:"):
                data_uri = image
            else:
                data_uri = image_to_data_uri(image)
        else:
            b64 = base64.b64encode(image).decode()
            data_uri = f"data:image/png;base64,{b64}"
        return self._llm.call_with_images(prompt, [data_uri])

    def analyze_url(self, url: str, prompt: str) -> str:
        resp = httpx.get(url, follow_redirects=True, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "image/png")
        b64 = base64.b64encode(resp.content).decode()
        data_uri = f"data:{content_type};base64,{b64}"
        return self._llm.call_with_images(prompt, [data_uri])
