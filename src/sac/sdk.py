from __future__ import annotations

from typing import TYPE_CHECKING

from sac.llm import DEFAULT_MODEL, LLMSDKClient
from sac.search import SearchSDK
from sac.storage import FilesystemSDK
from sac.utils import UtilsSDK


class AgenticSearchSDK:
    def __init__(
        self,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str = DEFAULT_MODEL,
        fs_dir: str | Path | None = None,
        brave_key: str | None = None,
    ) -> None:
        self.search = SearchSDK(brave_key=brave_key)
        self.llm = LLMSDKClient(
            base_url=llm_base_url,
            api_key=llm_api_key,
            model=llm_model,
        )
        self.fs = FilesystemSDK(fs_dir=fs_dir)
        self.utils = UtilsSDK()


if TYPE_CHECKING:
    from pathlib import Path
