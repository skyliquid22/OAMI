# Repository Guidelines
## Project Structure & Module Organization
Source code lives in `src/oami`, with `data_layer.py` handling Polygon ingestion, `features*.py` building feature sets, and `dataset.py` packaging model-ready data. `src/main.py` wires the pipeline. Configuration defaults sit in `configs/config.yaml` and `logging_config.json`. Cached market artifacts are written under `data/`. Research notebooks belong in `notebooks/`, and unit tests mirror modules inside `tests/`. Python package metadata is managed through `pyproject.toml` and `setup.cfg`.

## Build, Test, and Development Commands
Create a virtual environment before contributing: `python -m venv .venv && source .venv/bin/activate`. Install dependencies with `pip install -r requirements.txt` plus `requirements-dev.txt`, then register the package locally via `pip install -e .`. Run the end-to-end pipeline with `python -m src.main`. Execute the default test suite and coverage gate through `pytest --cov=src/oami --cov-report=term-missing`. Format and lint prior to pushing: `black src tests`, `isort src tests`, and `flake8 src tests`.

## Coding Style & Naming Conventions
Follow Black’s 88-character line length and standard 4-space indentation. Keep modules and functions snake_case, classes PascalCase, and constants UPPER_SNAKE. Prefer explicit type hints, especially across data interfaces. Organize imports with `isort` and ensure lint clean builds with `flake8`. Configuration files are YAML or JSON; commit validated schemas only.

## Testing Guidelines
Pytest discovers files matching `tests/test_*.py`; mirror runtime modules with similarly named test modules. Ensure new features include regression coverage and update fixtures in `tests/`. The coverage threshold is 75% branch coverage—run `pytest -q` for fast iteration and reserve `pytest --cov` before review. Use descriptive test names that explain behavior, e.g., `test_dataset_handles_missing_iv`.

## Commit & Pull Request Guidelines
Commits follow conventional prefixes seen in history (`feat(core): ...`). Keep messages in the imperative, referencing modules touched. Each PR should link relevant issues, summarize data or config changes, note required API keys, and attach output diffs or screenshots when behavior changes. Include test results (`pytest --cov`) in the PR description.

## Security & Configuration Tips
Never commit real API keys; depend on the `POLYGON_API_KEY` environment variable loaded via shell or `.env`. Store experiment-specific overrides in gitignored files and document reusable configs under `configs/`. Verify data directories remain lightweight before pushing to avoid bloating the cache store.
