import json
import time
from pathlib import Path

import requests

API_URL = "http://127.0.0.1:8000"
TIMEOUT_S = 20.0


def wait_for_health() -> dict:
    deadline = time.time() + TIMEOUT_S
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{API_URL}/health", timeout=2.0)
            if r.status_code == 200:
                return r.json()
            last_err = f"health status={r.status_code} body={r.text[:200]}"
        except Exception as e:
            last_err = repr(e)
        time.sleep(0.5)
    raise RuntimeError(f"API did not become healthy within {TIMEOUT_S}s. Last error: {last_err}")


def load_feature_cols() -> list[str]:
    schema_path = Path("artifacts") / "feature_schema.json"
    if not schema_path.exists():
        raise RuntimeError("artifacts/feature_schema.json not found. Build or generate artifacts first.")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    cols = schema.get("feature_columns")
    if not isinstance(cols, list) or not cols:
        raise RuntimeError("feature_schema.json missing non-empty 'feature_columns' list.")
    return cols


def score_once(model: str, feature_cols: list[str]) -> dict:
    # Deterministic toy features: mostly zeros, a few small non-zero values.
    feats = {c: 0.0 for c in feature_cols}
    for i, c in enumerate(feature_cols[:3]):
        feats[c] = float(i + 1)

    payload = {
        "model": model,
        "rows": [
            {"group": "blk_0001", "window_start": "2000-01-01T00:00:00Z", "features": feats},
            {"group": "blk_0002", "window_start": "2000-01-01T00:05:00Z", "features": feats},
        ],
    }

    r = requests.post(f"{API_URL}/score_batch", json=payload, timeout=5.0)
    if r.status_code != 200:
        raise RuntimeError(f"/score_batch failed for model={model}: status={r.status_code} body={r.text[:400]}")
    out = r.json()

    assert out["model"] == model
    assert isinstance(out["threshold"], (int, float))
    assert len(out["rows"]) == 2
    for row in out["rows"]:
        assert "anomaly_score" in row
        assert "is_anomaly" in row
    return out


def main() -> None:
    health = wait_for_health()
    print("HEALTH:", health)

    feature_cols = load_feature_cols()
    print("FEATURE_COLS:", len(feature_cols))

    # Smoke both supported models.
    out_if = score_once("iforest", feature_cols)
    print("SCORE iforest OK. threshold=", out_if["threshold"])

    out_oc = score_once("ocsvm", feature_cols)
    print("SCORE ocsvm OK. threshold=", out_oc["threshold"])

    m = requests.get(f"{API_URL}/metrics", timeout=5.0)
    if m.status_code != 200:
        raise RuntimeError(f"/metrics failed: status={m.status_code} body={m.text[:200]}")
    print("METRICS:", m.json())

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()