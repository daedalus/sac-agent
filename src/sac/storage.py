from __future__ import annotations

import pickle
import re
import tempfile
from pathlib import Path
from typing import Any


class FilesystemSDK:
    def __init__(self, fs_dir: str | Path | None = None) -> None:
        self._dir = Path(fs_dir) if fs_dir else Path(tempfile.mkdtemp(prefix="sac_fs_"))
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def dir(self) -> Path:
        return self._dir

    def write(self, key: str, data: Any) -> None:  # noqa: ANN401
        safe = self._safe_key(key)
        path = self._dir / safe
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def read(self, key: str) -> Any:  # noqa: ANN401
        safe = self._safe_key(key)
        path = self._dir / safe
        if not path.exists():
            raise FileNotFoundError(f"Key '{key}' not found")
        with open(path, "rb") as f:
            return pickle.load(f)

    def list(self) -> list[str]:
        return sorted(p.stem for p in self._dir.iterdir() if p.suffix == ".pkl")

    def exists(self, key: str) -> bool:
        return (self._dir / self._safe_key(key)).exists()

    @staticmethod
    def _safe_key(key: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
        return f"{safe}.pkl"
