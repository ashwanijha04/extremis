import sys as _sys

if _sys.version_info < (3, 11):
    raise RuntimeError(
        f"\n\nextremis requires Python 3.11 or later.\n"
        f"You have Python {_sys.version_info.major}.{_sys.version_info.minor}.\n\n"
        f"Fix it:\n"
        f"  macOS:   brew install python@3.11\n"
        f"  Then:    /opt/homebrew/bin/pip3.11 install extremis\n"
        f"  Linux:   sudo apt install python3.11\n"
        f"  Windows: https://python.org/downloads\n"
    )

from . import wrap  # noqa: F401  — makes `from extremis.wrap import Anthropic` work
from .api import Extremis
from .async_api import AsyncExtremis
from .client import HostedClient
from .config import Config
from .types import FeedbackSignal, LogEntry, Memory, MemoryLayer, RecallResult

__all__ = [
    "Extremis",
    "AsyncExtremis",
    "HostedClient",
    "Config",
    "Memory",
    "MemoryLayer",
    "LogEntry",
    "RecallResult",
    "FeedbackSignal",
    "wrap",
]
