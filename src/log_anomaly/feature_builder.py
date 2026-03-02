from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from log_anomaly.parsing import LogEvent, parse_hdfs_line
from log_anomaly.templating import to_template
from log_anomaly.windowing import floor_to_window


@dataclass
class WindowAgg:
    total: int = 0
    non_info: int = 0
    template_counts: Dict[str, int] = None
    unique_templates: set = None
    other_templates: int = 0

    def __post_init__(self) -> None:
        if self.template_counts is None:
            self.template_counts = defaultdict(int)
        if self.unique_templates is None:
            self.unique_templates = set()


def iter_events(path: Path, max_lines: Optional[int]) -> Iterable[LogEvent]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            evt = parse_hdfs_line(line)
            if evt is not None:
                yield evt
            if max_lines is not None and i >= max_lines:
                break


def build_template_vocab(path: Path, top_k: int, max_lines: Optional[int]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for evt in tqdm(iter_events(path, max_lines=max_lines), desc="Pass1 vocab"):
        counts[to_template(evt.content)] += 1
    most_common = counts.most_common(top_k)
    return {tmpl: idx for idx, (tmpl, _) in enumerate(most_common)}


def write_vocab(vocab: Dict[str, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(vocab, indent=2, sort_keys=True), encoding="utf-8")


def flush_window_rows(
    window_start: datetime,
    aggs: Dict[Tuple[str, datetime], WindowAgg],
    vocab: Dict[str, int],
    running_means: Dict[str, float],
    alpha: float,
    rows: list,
) -> None:
    for (group_key, ws), agg in list(aggs.items()):
        if ws != window_start:
            continue

        total = agg.total
        err_ratio = (agg.non_info / total) if total > 0 else 0.0

        prev_mean = running_means.get(group_key, 0.0)
        mean = (alpha * total) + ((1 - alpha) * prev_mean)
        running_means[group_key] = mean

        burstiness = (total / (mean + 1e-9)) if mean > 0 else 1.0

        row = {
            "group": group_key,
            "window_start": ws,
            "total_events": total,
            "unique_templates": len(agg.unique_templates),
            "error_ratio": err_ratio,
            "burstiness": burstiness,
            "other_template_count": agg.other_templates,
        }

        for tmpl, idx in vocab.items():
            row[f"tmpl_{idx}"] = agg.template_counts.get(tmpl, 0)

        rows.append(row)
        del aggs[(group_key, ws)]


def build_features(
    input_path: Path,
    out_features: Path,
    out_vocab: Path,
    window_seconds: int = 300,
    top_k: int = 200,
    max_lines: Optional[int] = None,
    alpha: float = 0.2,
) -> None:
    print("Pass 1: building template vocab...")
    vocab = build_template_vocab(input_path, top_k=top_k, max_lines=max_lines)
    write_vocab(vocab, out_vocab)
    print(f"Vocab size: {len(vocab)} (top_k={top_k})")

    print("Pass 2: building window features...")
    aggs: Dict[Tuple[str, datetime], WindowAgg] = {}
    running_means: Dict[str, float] = {}
    rows: list = []

    current_window: Optional[datetime] = None

    for evt in tqdm(iter_events(input_path, max_lines=max_lines), desc="Pass2 windows"):
        ws = floor_to_window(evt.ts, window_seconds)
        group_key = evt.component or "global"

        if current_window is None:
            current_window = ws

        if ws != current_window:
            flush_window_rows(
                window_start=current_window,
                aggs=aggs,
                vocab=vocab,
                running_means=running_means,
                alpha=alpha,
                rows=rows,
            )
            current_window = ws

        key = (group_key, ws)
        if key not in aggs:
            aggs[key] = WindowAgg()

        agg = aggs[key]
        agg.total += 1
        if evt.level != "INFO":
            agg.non_info += 1

        tmpl = to_template(evt.content)
        agg.unique_templates.add(tmpl)

        if tmpl in vocab:
            agg.template_counts[tmpl] += 1
        else:
            agg.other_templates += 1

    if current_window is not None:
        flush_window_rows(
            window_start=current_window,
            aggs=aggs,
            vocab=vocab,
            running_means=running_means,
            alpha=alpha,
            rows=rows,
        )

    df = pd.DataFrame(rows)
    df.sort_values(["window_start", "group"], inplace=True)
    out_features.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_features, index=False)
    print(f"Wrote features: {out_features} rows={len(df)} cols={len(df.columns)}")
