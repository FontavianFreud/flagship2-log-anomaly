from __future__ import annotations

from datetime import datetime

def floor_to_window(ts: datetime, window_seconds: int) -> datetime:
    epoch = int(ts.timestamp())
    floored = epoch - (epoch % window_seconds)
    return datetime.fromtimestamp(floored)
