from __future__ import annotations

import re

# Lightweight, deterministic normalization.
# Goal: map many similar log lines to the same template.
_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    # HDFS block ids
    (re.compile(r"\bblk_-?\d+\b"), "<BLOCK>"),
    # IP:port patterns (your logs look like /10.250.19.102:54106)
    (re.compile(r"(/\d+\.\d+\.\d+\.\d+)(:\d+)?"), "<IP>"),
    # Hex
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    # Plain integers
    (re.compile(r"\b\d+\b"), "<NUM>"),
]

def to_template(content: str) -> str:
    s = content
    for pat, repl in _REPLACEMENTS:
        s = pat.sub(repl, s)
    return s
