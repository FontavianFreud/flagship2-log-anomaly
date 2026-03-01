from __future__ import annotations

import re

_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bblk_-?\d+\b"), "<BLOCK>"),
    (re.compile(r"(/\d+\.\d+\.\d+\.\d+)(:\d+)?"), "<IP>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (re.compile(r"\b\d+\b"), "<NUM>"),
]

def to_template(content: str) -> str:
    s = content
    for pat, repl in _REPLACEMENTS:
        s = pat.sub(repl, s)
    return s
