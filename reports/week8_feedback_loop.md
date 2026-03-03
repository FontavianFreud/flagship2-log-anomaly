# Week 8: Feedback Loop and Evaluation

## Goal
Add a minimal human-in-the-loop workflow for reviewing flagged anomaly windows, storing labels, and evaluating alert quality using practical proxy metrics:
- **precision@k** (triage quality of the top ranked anomalies)
- **daily anomaly-rate stability** (how spiky alert volume is day-to-day)
- **score drift** between train and validation (distribution shift proxy)

## What was built
### Review workflow (human in the loop)
1) Export a review batch from `data/processed/scored_windows.parquet` to CSV (top anomalies for a model/split).
2) Manually fill:
   - `label`: 1 = worth investigating, 0 = not worth investigating
   - `notes`: short justification (optional)
3) Ingest labeled rows into SQLite: `data/feedback/feedback.sqlite`.
4) Evaluate precision@10 and monitoring proxies using the labeled subset.

### Storage
- SQLite DB: `data/feedback/feedback.sqlite` (local, ignored in git)
- Review CSVs: `data/review/*.csv` (local, ignored in git)

## Labeling definition
A label of **1 (worth investigating)** means: the window is abnormal enough that it should be surfaced to a human reviewer in an on-call / monitoring setting.  
A label of **0** means: statistically unusual but not worth the human’s time given a constrained review queue.

Labeling was done using context metrics included in the review CSV:
- `total_events`, `burstiness`, `unique_templates`, `other_template_count`, `error_ratio`
- plus within-group percentile ranks (`p_*`) to judge severity relative to the same component’s baseline.

## Metrics (k=10, validation split)

### Isolation Forest (iforest)
- **precision@10 on reviewed top-k/day (val): 0.80** (n_reviewed=10)
- val daily anomaly-rate mean: **0.007389**
- val daily anomaly-rate CV: **1.414**
- score drift KS(train vs val): **0.1229**

Interpretation:
- The top-ranked anomalies are mostly worth investigating, but not all (2/10 labeled as not worth investigating).
- Alert volume varies substantially day-to-day (high CV), which is expected for bursty behavior and suggests possible value in stability-oriented thresholding or per-group calibration later.
- Score distribution differs modestly between train and val (KS ~0.12), a drift proxy indicating non-stationarity over time.

### One-Class SVM (ocsvm)
- **precision@10 on reviewed top-k/day (val): 0.8571** (n_reviewed=7)
- val daily anomaly-rate mean: **0.004310**
- val daily anomaly-rate CV: **1.414**
- score drift KS(train vs val): **0.1259**

Interpretation:
- OCSVM’s reviewed top anomalies show strong triage quality on the labeled subset.
- It flags fewer anomalies on average than Isolation Forest (lower mean daily anomaly rate), but still exhibits high day-to-day variability (high CV).
- Drift proxy is similar to Isolation Forest (KS ~0.13).

## Takeaways
- A feedback loop now exists: export → label → ingest → evaluate.
- precision@10 is a practical “ops-style” metric for ranked anomaly review queues.
- Both models show high variability in daily anomaly rates, motivating future work on stabilizing alert volume (stability-based thresholds, per-group thresholds, or rolling recalibration).
- Score distribution drift is nonzero, suggesting behavior changes over time (a realistic monitoring concern).

## Artifacts and scripts
- Export review batch: `scripts/export_review_batch.py`
- Ingest labels: `scripts/ingest_review_batch.py`
- Evaluate metrics: `scripts/evaluate_feedback.py`
- Feedback storage helper: `src/log_anomaly/feedback.py`

## Next (Week 9 preview)
- Add lightweight monitoring outputs (time series summaries of anomaly rate and score drift).
- Expand labeling coverage (more days, more models) to make precision@k estimates more stable.
- Explore stability-based thresholding or per-group thresholds to reduce day-to-day alert variability.