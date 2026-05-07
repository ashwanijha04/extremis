from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..types import LogEntry


class FileLogStore:
    """
    Append-only daily JSONL log. One file per day at {log_dir}/YYYY-MM-DD.jsonl.
    fsync on every write for durability.
    """

    def __init__(self, log_dir: str, namespace: str = "default") -> None:
        base = Path(log_dir).expanduser()
        # Each namespace gets its own subdirectory: log_dir/{namespace}/
        self._dir = base / namespace if namespace != "default" else base
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, date: datetime) -> Path:
        return self._dir / f"{date.strftime('%Y-%m-%d')}.jsonl"

    def _checkpoint_path(self) -> Path:
        return self._dir / ".checkpoint"

    def append(self, entry: LogEntry) -> None:
        path = self._path_for(entry.timestamp)
        line = entry.model_dump_json() + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def read_since(self, checkpoint: Optional[str]) -> list[LogEntry]:
        """
        checkpoint format: "YYYY-MM-DD.jsonl:N" where N is the byte offset.
        Returns all entries written after that offset across all files.
        """
        entries: list[LogEntry] = []
        checkpoint_file, checkpoint_offset = self._parse_checkpoint(checkpoint)

        for log_file in sorted(self._dir.glob("*.jsonl")):
            filename = log_file.name
            start_offset = 0
            if checkpoint_file and filename < checkpoint_file:
                continue
            if checkpoint_file and filename == checkpoint_file:
                start_offset = checkpoint_offset

            with open(log_file, "r", encoding="utf-8") as f:
                f.seek(start_offset)
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(LogEntry.model_validate_json(line))

        return entries

    def get_checkpoint(self) -> Optional[str]:
        if not self._checkpoint_path().exists():
            return None
        return self._checkpoint_path().read_text(encoding="utf-8").strip()

    def set_checkpoint(self, checkpoint: str) -> None:
        with open(self._checkpoint_path(), "w", encoding="utf-8") as f:
            f.write(checkpoint)
            f.flush()
            os.fsync(f.fileno())

    def current_checkpoint(self) -> str:
        """Returns a checkpoint string pointing to the end of the current log."""
        today = datetime.now(tz=timezone.utc)
        path = self._path_for(today)
        offset = path.stat().st_size if path.exists() else 0
        return f"{path.name}:{offset}"

    @staticmethod
    def _parse_checkpoint(checkpoint: Optional[str]) -> tuple[Optional[str], int]:
        if not checkpoint:
            return None, 0
        parts = checkpoint.rsplit(":", 1)
        if len(parts) == 2:
            return parts[0], int(parts[1])
        return checkpoint, 0
