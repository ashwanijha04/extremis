"""
extremis.wrap — drop-in LLM client wrappers with automatic memory.

Replace one import, get persistent memory for free.

    from extremis.wrap import Anthropic, OpenAI
"""

from .anthropic import Anthropic
from .openai import OpenAI

__all__ = ["Anthropic", "OpenAI"]
