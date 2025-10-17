# OAMI — Options-Aware Market Intelligence
**Version:** 0.3.0

OAMI is an ML-ready pipeline for market & options data with robust caching, logging, feature engineering, and tests.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
export POLYGON_API_KEY="YOUR_REAL_KEY"
pytest --cov=src/oami --cov-report=term-missing
python -m src.main
```

## Architecture
```mermaid
graph LR
    A[Polygon API] --> B[Data Layer: src/oami/data_layer.py]
    B --> C[CSV Cache: data/csv/timeframe/ticker.csv]
    C --> D[FeatureBuilder: src/oami/features.py]
    C --> E[OptionsSentiment: src/oami/options_features.py]
    D --> F[DatasetBuilder: src/oami/dataset.py]
    E --> F
    F --> G[Advanced Features: src/oami/features_advanced.py]
    G --> H[Models & Evaluation (notebooks)]
```
