# Intelliswitch — Phase 2: Simulated Online Serving & Continuous Retraining

This file is the working plan for **Phase 2**. Phase 1 (the MATLAB→Python refactor:
feature extraction, geometry labels, model, serving entry point) is **done** and
lives in `src/intelliswitch/`. Phase 2 builds a streaming MLOps system **on top of
that package** — do not reinvent extraction, QC, or the model.

## 0. Objective (read this first)

The goal is to **learn and exercise deployment / ML system design / streaming
MLOps** — NOT to improve model accuracy. When choosing between options, optimize
for *learning value and realism*, not for what is easiest. LightGBM hyperparameters
are **fixed/known**; there is no tuning in the loop.

We simulate an **online serving system** by replaying the historical trackside log
files through a real streaming stack, logging predictions and (initially unlabeled)
features, and **retraining whenever a track-geometry inspection campaign arrives**
(sparse, delayed labels).

## 1. What already exists (Phase 1 — reuse these)

Install once: `pip install -e ".[dev]"`. Key building blocks:

- `intelliswitch.paths` — centralized data/model locations (medallion tiers:
  `data/raw` → `data/interim/raw_cache` → `data/processed/feature_store`, `models/`).
  Override with `INTELLISWITCH_DATA_DIR` / `INTELLISWITCH_MODELS_DIR`.
- `intelliswitch.Config` — all parameters/hyperparameters (freq grid, QC thresholds,
  `cp_decimate`, feature transform `sqrt(PSD)/speed`, LightGBM params, train/test day
  ranges, aggregation methods, S&C selection `min_speed_kmh`/`n_groups_required`).
- `io.decode_log(path)->RawRecord`, `io.raw_cache` (training cache only; **not** used
  at serving).
- `features.extract_features(source)->FeatureResult` — decode→filter→wheel-detect→
  PSD→**QC**. QC lives **only here** (single source of truth). Returns `.accepted`,
  `.reason`, `.features` (one row per accepted sensor×wheel-set).
- `serving.predict_recording(source, model)->InferenceResult` — the streaming unit of
  work: runs extract+QC, then per-(sensor×wheel-set) prediction aggregated
  **wheel-set → location → S&C**; or `status="rejected"` with a reason. Accepts a
  `.log` path or a `RawRecord`.
- `model.train(df, cfg, kind)`, `model.QualityModel` (artifact = fitted estimator +
  cfg + feature columns + training speed range; `.save()/.load()` via joblib),
  `model.evaluate`, `model.leave_one_location_out`. Backends: `"plsr"`, `"lgbm"`.
- `geometry.align_campaigns`, `geometry.build_deg_model`, `geometry.deg_model.attach_labels`
  — produce the quality-index labels (training-time only; **not** needed at inference).
- `pipeline.build_feature_store`, `pipeline.load_feature_store`, `pipeline.select_trains`.

**Invariant:** features stored = RAW PSD; the `sqrt(PSD)/speed` transform is applied
at model time (`build_X`). QC is computed once in `extract_features`; serving and
training both consume its result — never re-implement it.

## 2. Scenario & parameters

- **Data span:** 2016-07-13 … 2019-02-14. Log filenames are `yymmdd_HHMMSS.log`
  (lexical sort == chronological).
- **Simulation start `t0` = 2018-06-01.** All data **before t0** trains the initial
  model. From t0 forward we stream/serve/monitor/retrain.
- **Stream one log at a time**, in timestamp order (≈80/day is just context; do NOT
  batch a day at once). A paced producer + a triggered consumer => Kafka holds a
  **queue**; observe backlog/lag building and draining. Delete each staged log after
  it is processed.
- **Labels are sparse & delayed:** they only exist when a **geometry campaign** is
  measured. On each campaign arrival, (re)label all stored recordings up to that date
  and **retrain**; promote the new model to `champion`; subsequent predictions use it.

## 3. Architecture (locked decisions)

```
 LOCAL producer            OBJECT STORAGE         CONFLUENT CLOUD      DATABRICKS (Spark)
 stream logs 1-by-1 ─────▶ (Unity Catalog volume)       │
 (timestamp order)               ▲                       │
        └── emit POINTER event ──────────────────────▶ Kafka topic ──▶ Structured Streaming
            {path, event_time}        (the QUEUE)                       (Trigger.AvailableNow)
                                                                        foreachBatch:
                                                                         read blob → predict_recording
                                                                         (decode/extract/QC/predict)
                                                                         ├ accepted → Delta features+preds
                                                                         └ rejected → Delta dead_letter
                                                                        then delete staged log
                              campaign arrives → Retrain Job: geometry→labels → MERGE into Delta
                                                  → train LightGBM → MLflow register → promote champion
                              Monitoring: Databricks SQL over Delta (lag, accept/reject, drift, accuracy)
```

**Locked choices (and why):**

- **Infra = Databricks (Spark Structured Streaming), NOT Flink.** Flink does not run
  *on* Databricks (it only connects as an external Unity Catalog client); Databricks'
  engine is Spark. Free Edition is $0.
- **Extraction/QC run IN THE STREAM** (not at the edge) — so the stream does real
  work (distributed scipy via `mapInPandas`/`foreachBatch`, library-on-cluster,
  QC-as-branching) and you reuse `predict_recording` whole. Also more realistic (edge
  ships files; cloud processes).
- **Claim-check pattern:** Kafka must not carry 12 MB logs. The producer stages each
  log to **object storage** and emits a small **pointer** event; Spark reads the blob.
- **Kafka = Confluent Cloud** (Basic cluster: scales to zero, $0 idle, $400/30-day
  credits). Required because Databricks (cloud) cannot reach a laptop-local broker.
- **Object storage = Databricks Unity Catalog Volume** (default — the cluster reads it
  natively, one platform, free). *Alternative: S3 (AWS free tier) if you want to
  exercise cloud storage + IAM.*
- **Feature store = Delta tables; model registry = MLflow** (both native to Databricks).
- **Data layout = medallion** (raw→interim/silver→processed/gold), matching
  `intelliswitch.paths`.

## 4. Constraints & gotchas (Free Edition reality)

- Free Edition is **serverless + quota-limited**; streaming supports only
  `Trigger.AvailableNow` / `Trigger.Once` (**no continuous** stream). So run the
  consumer as **repeated `Trigger.AvailableNow`** (scheduled job or loop) that drains
  whatever the producer has queued — this is exactly what surfaces the queue/lag.
- If you exceed quota, compute is shut off for the day — keep volume tiny (1-by-1 +
  cleanup) and runs bounded.
- The `intelliswitch` package must be **installed on the cluster** (build a wheel or
  `%pip install` from the repo). scipy/lightgbm needed.
- Use **checkpointing + Kafka offsets** for exactly-once / resumable consumption.
- Secrets (Confluent API keys) via **Databricks secrets**, never hard-coded.
- Cost guardrail: Confluent Basic scales to zero; Databricks Free is $0; if switching
  to S3/paid Databricks, watch continuously-running compute.

## 5. Build plan — step by step (each step has an acceptance gate)

**S0 — Prerequisites & connectivity.** Create Databricks Free Edition workspace +
Confluent Cloud (Basic cluster, one topic e.g. `intelliswitch.recordings`, API keys);
create a Unity Catalog **volume** for staged logs; store Confluent keys in Databricks
secrets; install the `intelliswitch` wheel on the cluster.
*Gate:* from a Databricks notebook, produce+consume a test Kafka message and read/write
the volume.

**S1 — Phase-2 config & scaffolding.** Add a small `phase2/` config (Kafka topic/bootstrap,
volume URI, Delta catalog/schema, MLflow model name) + connectivity smoke tests. Keep
all names in one config object.
*Gate:* config loads; smoke tests pass.

**S2 — Pointer event schema.** Define the Kafka message:
`{schema_version, log_path (volume URI), source_name, event_time (from filename)}`.
JSON to start (Avro + Schema Registry is a stretch goal).
*Gate:* serialize/deserialize round-trip + documented schema.

**S3 — Replay producer (local).** Iterate logs in timestamp order starting at `t0`;
for each: upload the `.log` to the volume, emit a pointer event, optional pacing sleep
to build a queue; resumable (track last-sent). 
*Gate:* on a small subset, messages land in Kafka in order and logs appear in the volume.

**S4 — Bootstrap model v1 (offline).** Build the labeled training set from data
**< t0** (extract features for those logs + geometry labels via `build_deg_model`/
`attach_labels`), `select_trains`, train LightGBM (fixed HPs), evaluate (overall +
LOLO + per-location), register to MLflow as `intelliswitch_quality`, set alias
`champion`. The registered artifact is the `QualityModel` (carries cfg + feature
contract).
*Gate:* champion exists; `QualityModel.load(...)` predicts on a sample.

**S5 — Streaming inference job (Databricks).** Structured Streaming reading the Kafka
pointer topic with checkpointing; `foreachBatch`: load `champion` (cache; refresh on new
version) → for each pointer, read the blob from the volume → `predict_recording(path,
model)` → flatten `InferenceResult` to rows. **Branch:** accepted → Delta
`features`+`predictions`; rejected → Delta `dead_letter` (with reason). Delete the
staged log after success. Run via repeated `Trigger.AvailableNow`.
*Gate:* queued messages drain into Delta; rejections in dead-letter; consumer lag
visible before/after a trigger (the queue).

**S6 — Delta feature store schema.** Tables: `features` (raw PSD rows + sensor/group +
speed + serving_ts + model_version + `label` NULL), `predictions` (location/S&C
aggregates + OOD flag), `dead_letter` (source + reason + ts). MERGE-friendly keys
(source + sensor + group). Unlabeled initially.
*Gate:* schemas created; idempotent (re-process is safe); time-travel works.

**S7 — Label arrival + retrain job.** Parameterized by a campaign date. Steps:
run geometry pipeline up to that date → labels → **MERGE** labels into `features` for
all recordings with day ≤ campaign → assemble training set (`select_trains` +
labeled rows) → train LightGBM (fixed HPs) → evaluate → register new MLflow version →
promote `champion`. Next stream batches auto-use it.
*Gate:* previously-unlabeled rows get labels; new model version registered; champion
advances; stream picks it up.

**S8 — Simulation orchestrator.** Drive the sim clock from `t0` forward: pace the
producer (1-by-1), run consumer triggers, and fire the retrain job when the clock
crosses each campaign date (campaigns are known from the geometry CSVs, e.g. the
Bladenr8252 set). Support accelerated replay; persist sim state for resume.
*Gate:* end-to-end run over a window: predictions accumulate, retrains fire at campaign
dates, champion versions advance over time.

**S9 — Monitoring & evaluation.** Databricks SQL dashboard over Delta: throughput,
**consumer lag / queue depth**, accept vs reject rate + reason histogram, OOD rate,
prediction distribution over time, model-version timeline; and **post-label accuracy**
(error per location / over time) + retrain impact.
*Gate:* dashboard renders and updates as the sim runs.

## 6. Non-goals

Hyperparameter tuning; sub-second latency; production hardening (HA, authn/z beyond
secrets); changing Phase-1 modeling. Keep the focus on the streaming + retraining loop.

## 7. Working agreement for Claude Code

- Reuse the `intelliswitch` package; never duplicate QC/extraction/model logic.
- Small steps, each with its acceptance gate met before moving on; add tests.
- Keep secrets in Databricks secrets; keep data/model locations via `intelliswitch.paths`
  or explicit Phase-2 config (no hard-coded paths).
- Record any new design decisions here under a `## Decisions log` section as you go.

## 8. Open items to confirm with the user before/early in S0

- Object storage: **Databricks Volume (default)** vs S3.
- Event format: **JSON (default)** vs Avro + Schema Registry.
- Retrain trigger: **auto on campaign-date crossing (default)** vs manual.
