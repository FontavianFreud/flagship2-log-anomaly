from __future__ import annotations

import argparse
from pathlib import Path

from log_anomaly.feature_builder import build_features


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--out-vocab", type=str, default="data/processed/template_vocab.json")
    p.add_argument("--window-seconds", type=int, default=300)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--max-lines", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.2)
    args = p.parse_args()

    build_features(
        input_path=Path(args.input),
        out_features=Path(args.out),
        out_vocab=Path(args.out_vocab),
        window_seconds=args.window_seconds,
        top_k=args.top_k,
        max_lines=args.max_lines,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
