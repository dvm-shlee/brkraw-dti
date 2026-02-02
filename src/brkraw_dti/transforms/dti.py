"""Transform helpers for brkraw-mrs metadata and info mapping."""

from __future__ import annotations

import os
import re
from typing import Any, Optional, cast
from importlib import metadata


def strip_bruker_string(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    return text.strip()


def strip_jcamp_string(value: Optional[str]) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    text = re.sub(r"\^+", " ", text)
    return " ".join(text.split())
