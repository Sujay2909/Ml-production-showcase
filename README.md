# ML Platform

A production-grade machine learning engineering platform showcasing end-to-end ML workflows: distributed ETL, multi-model training with experiment tracking, NLP automation, real-time serving, and full CI/CD.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ML Platform                                  │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │   ETL    │──▶│  ML Pipeline │──▶│   Serving    │                │
│  │ PySpark  │   │ XGB/LGBM/RF  │   │  FastAPI     │                │
│  │ Delta    │   │ MLflow track │   │  Redis cache │                │
│  └──────────┘   └──────────────┘   └──────────────┘                │
│                                                                     │
│  ┌──────────────────────────┐   ┌──────────────────────────┐       │
│  │   NLP Module             │   │   Monitoring             │       │
│  │   BERT classifier        │   │   Prometheus metrics     │       │
│  │   spaCy entity extract   │   │   Drift detection        │       │
│  └──────────────────────────┘   └──────────────────────────┘       │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │   CI/CD: GitHub Actions + Jenkinsfile                       │   │
│  │   Docker multi-stage build + docker-compose (local stack)   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Modules

### `src/etl/` — PySpark ETL Pipeline
- **`ingestion.py`** — Batch and streaming ingestion for 10M–15M daily records with 99.8%+ reliability via schema enforcement, deduplication, and dead-letter tracking.
- **`transformations.py`** — Customer churn feature aggregation and rolling demand signals using Spark Window functions.
- **`delta_writer.py`** — Delta Lake writer supporting APPEND, OVERWRITE, and MERGE (upsert) modes, plus OPTIMIZE / VACUUM helpers.

### `src/ml_pipeline/` — Model Training & Prediction
- **`features.py`** — Sklearn `Pipeline`-compatible feature engineering with `ColumnSelector`, `OutlierClipper`, imputation, and scaling. Serialisable via `joblib`.
- **`models.py`** — Typed hyperparameter dataclasses and `ModelRegistry` factory for XGBoost, LightGBM, and Random Forest (classifier and regressor variants).
- **`train.py`** — `ModelTrainer` orchestrator: cross-validated training, MLflow param/metric/artifact logging, early stopping for gradient boosting models.
- **`predict.py`** — `ModelPredictor`: unified online and batch inference wrapper used by both the API and scheduled scoring jobs.

### `src/nlp/` — Document Intelligence
- **`preprocessor.py`** — Text cleaning (HTML, PII redaction, whitespace), spaCy sentence splitting, and BERT-length truncation.
- **`classifier.py`** — `DocumentClassifier`: BERT `SequenceClassification` with batched inference. Auto-triage splits documents into high-confidence auto-approved and low-confidence flagged-for-review queues.
- **`entity_extractor.py`** — `EntityExtractor`: spaCy NER + rule-based obligation detection. Extracts parties, dates, monetary values, and contract obligations.

### `src/serving/` — Real-Time API
- **`api.py`** — FastAPI application: `/predict/churn`, `/predict/demand`, `/classify/document`, `/extract/entities`, `/health`, `/metrics`. Request timing middleware, Pydantic validation.
- **`cache.py`** — `RedisCache`: deterministic payload-hash keying, TTL management, hit-rate tracking, graceful degradation on Redis failures.

### `src/monitoring/` — Observability
- **`metrics.py`** — `MetricsCollector`: Prometheus `Counter`, `Gauge`, `Histogram` for prediction throughput, latency, model accuracy, cache stats, and PSI-based feature drift scoring.

---

## Quick Start

### 1. Clone and configure
```bash
git clone https://github.com/YOUR_USERNAME/ml-platform.git
cd ml-platform
cp .env.example .env
# Edit .env with your credentials
```

### 2. Run the full local stack (Docker)
```bash
docker-compose up -d
# API:    http://localhost:8000/docs
# MLflow: http://localhost:5000
```

### 3. Install for local development
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm
```

### 4. Run tests
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 5. Start the API server (dev mode)
```bash
python -m src.serving.api
# → http://localhost:8000/docs
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict/churn` | Customer churn probability |
| `POST` | `/predict/demand` | Demand forecasting |
| `POST` | `/classify/document` | Compliance document classification |
| `POST` | `/extract/entities` | Named entity + obligation extraction |
| `GET` | `/health` | Liveness probe |
| `GET` | `/metrics` | Cache and runtime stats |

### Example: Churn prediction
```bash
curl -X POST http://localhost:8000/predict/churn \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "cust_001",
    "total_events": 150,
    "unique_products": 5,
    "total_revenue": 1200.50,
    "avg_order_value": 45.0,
    "avg_session_sec": 180.0,
    "days_since_last_event": 7,
    "customer_tenure_days": 365,
    "active_days": 30,
    "engagement_score": 0.75
  }'
```

### Example: Document classification
```bash
curl -X POST http://localhost:8000/classify/document \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "policy_2024_001",
    "text": "This agreement shall be governed by the compliance standards set forth..."
  }'
```

---

## CI/CD

### GitHub Actions
- **`ci.yml`** — Runs on every push/PR: lint (Black, isort, Flake8, mypy), multi-Python unit tests with coverage, Docker build smoke test, Trivy security scan.
- **`model-retrain.yml`** — Scheduled weekly retraining pipeline with MLflow tracking and Slack failure notifications.

### Jenkins
- **`Jenkinsfile`** — Declarative pipeline mirroring the GitHub Actions stages: Setup → Lint (parallel) → Test → Docker Build → Model Validation → Deploy. Reduces release cycles from weeks to 48 hours.

---

## Project Structure
```
ml-platform/
├── src/
│   ├── etl/                  # PySpark ingestion, Delta Lake
│   ├── ml_pipeline/          # XGBoost, LightGBM, RF, MLflow
│   ├── nlp/                  # BERT classifier, spaCy NER
│   ├── serving/              # FastAPI + Redis
│   ├── monitoring/           # Prometheus metrics
│   ├── logger.py             # Structured JSON logging (structlog)
│   └── settings.py           # Pydantic settings (env / .env)
├── tests/
│   ├── test_ml_pipeline.py
│   ├── test_nlp.py
│   └── test_serving.py
├── configs/
│   └── config.yaml
├── .github/workflows/
│   ├── ci.yml
│   └── model-retrain.yml
├── Jenkinsfile
├── Dockerfile                # Multi-stage build
├── docker-compose.yml        # API + Postgres + Redis + MLflow
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── .env.example
```

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| ETL | PySpark 3.4, Delta Lake, PyArrow |
| ML | XGBoost, LightGBM, scikit-learn |
| NLP | HuggingFace Transformers (BERT), spaCy |
| Experiment tracking | MLflow |
| Serving | FastAPI, Uvicorn |
| Caching | Redis |
| Storage | PostgreSQL, SQLAlchemy |
| Monitoring | Prometheus, structlog |
| CI/CD | GitHub Actions, Jenkins |
| Containers | Docker, docker-compose |

---

## License

MIT
