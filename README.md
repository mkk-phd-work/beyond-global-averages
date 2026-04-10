# Beyond Global Averages

Support code and reproducible research material for the article:

Beyond Global Averages: Per-Series Conditional Permutation Importance for Explaining Heterogeneous Effects in Global Forecasting Models

## Purpose

This repository is intentionally designed as article support material.
It provides a compact, reproducible implementation of a global forecasting pipeline and a per-series conditional feature-importance workflow.

The goal is to help readers:

- understand the implementation choices behind the article
- reproduce core experiments and figures
- adapt the methodology to their own multi-series datasets

## Methodological Focus

The implementation follows a modern global forecasting setup:

- one global model over many time series using skforecast
- LightGBM as the predictive backbone
- per-series conditional permutation feature importance with xeries


This repository emphasizes transparency and reproducibility over production deployment.

<!-- 
## Scope

Included:

- data loading utilities for wide and long formats
- train and split helpers for global multi-series forecasting
- evaluation utilities for per-series metrics
- xeries-based explainability wrapper for conditional permutation importance
- minimal notebooks for training and explainability workflows
- smoke and unit tests for baseline correctness

Not included:

- production serving APIs
- MLOps deployment pipelines
- model registry or online monitoring components

## Quick Start

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ty check src tests
uv run jupyter notebook
```

## Repository Layout

- src/cms_forecasting/data: data loading and split logic
- src/cms_forecasting/models: global model creation and training helpers
- src/cms_forecasting/evaluation: forecasting metrics
- src/cms_forecasting/explainability: per-series conditional importance workflows
- notebooks: article support notebooks
- tests: unit and integration smoke tests

## Reproducibility Notes

- Python 3.11 is the target runtime.
- Dependency resolution is managed with uv and locked in uv.lock.
- Type checking uses ty.
- All examples are intended to run in a local research workflow.

## Citation and Usage

If you use this code as a baseline or reference, cite the associated article and clearly state any modifications to data, model configuration, or explainability settings.

 -->
