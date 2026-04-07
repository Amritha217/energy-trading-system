# ⚡ Energy Trading AI

An end-to-end machine-learning pipeline that forecasts electricity demand, generates algorithmic trading signals, and uses an LLM agent to review and explain every trade decision.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [Resume Bullets](#resume-bullets)

---

## Overview

| Layer | Technology |
|---|---|
| Data | PJM PJME hourly load dataset (Kaggle) |
| Forecasting | XGBoost · LightGBM · Prophet |
| Trading strategy | Threshold-based signal generation + conviction-weighted sizing |
| Backtesting | Walk-forward with periodic retraining |
| LLM agent | Mistral-7B-Instruct via Hugging Face Inference API |
| Dashboard | Streamlit + Plotly |
| REST API | FastAPI + Uvicorn |

---

## Architecture

```
Raw CSV (PJME_hourly.csv)
        │
        ▼
 data_pipeline.py   ← clean, interpolate, derive synthetic price + returns
        │
        ▼
 feature_engineering.py  ← lag features, rolling stats, calendar features
        │
        ▼
 forecasting.py     ← train XGBoost / LightGBM / Prophet
        │
        ▼
 trading_strategy.py ← generate_signals → simulate_pnl → build_trade_log
        │
        ▼
 llm_agent.py       ← Mistral-7B (HF API) or rule-based fallback
        │
   ┌────┴────┐
   ▼         ▼
dashboard  api/main.py
(Streamlit) (FastAPI)
```

---

## Features

- **Automated data pipeline** — loads PJM CSV, fixes DST duplicates, interpolates gaps, derives price and return series
- **Feature engineering** — calendar, 24h/168h lag, rolling mean/std with strict look-ahead prevention
- **Multi-model forecasting** — XGBoost and LightGBM trained in parallel; Prophet for long-horizon trend
- **Walk-forward backtesting** — expanding-window retraining for realistic out-of-sample evaluation
- **Algorithmic trading simulation** — conviction-weighted position sizing, transaction costs, Sharpe / drawdown metrics
- **LLM trading agent** — Mistral-7B analyses each signal and returns a structured APPROVE/REJECT decision with risk commentary; automatic rule-based fallback when no API token is present
- **Interactive Streamlit dashboard** — flip-card metrics, actual vs predicted charts, feature importance (both XGBoost and LightGBM), capital curve, signal chart, data quality, batch agent analysis, CSV export
- **FastAPI REST service** — `/predict`, `/agent/decision`, `/strategy/metrics`, `/data/quality` endpoints

---

## Project Structure

```
energy-trading-ai/
├── api/
│   └── main.py               # FastAPI application
├── config/
│   └── settings.py           # All tuneable constants and env-var loading
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── data/
│   ├── PJME_hourly.csv       # Raw PJM dataset (download from Kaggle)
│   └── models/               # Serialised .joblib model files (git-ignored)
├── src/
│   ├── backtesting.py        # Walk-forward backtesting
│   ├── data_pipeline.py      # Data ingestion and preprocessing
│   ├── feature_engineering.py# Feature creation
│   ├── forecasting.py        # Model training, evaluation, persistence
│   ├── llm_agent.py          # Mistral-7B agent + rule-based fallback
│   └── trading_strategy.py   # Signal generation, PnL simulation, trade log
├── .env                      # Environment variables (git-ignored)
├── .gitignore
├── main.py                   # CLI entry point
├── pyproject.toml            # uv / PEP 517 project metadata
├── requirements.txt          # Pip-compatible dependency list
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) **or** pip
- PJME_hourly.csv placed in `data/` ([download from Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption))

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/energy-trading-ai.git
cd energy-trading-ai
```

### 2. Install dependencies

**With uv (recommended):**
```bash
uv sync
```

**With pip:**
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your Hugging Face token (optional — fallback works without it)
```

`.env` format:
```
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### 4. Run the full pipeline

```bash
python main.py
```

---

## Configuration

All parameters are in `config/settings.py`:

| Constant | Default | Description |
|---|---|---|
| `TRAIN_SPLIT` | `0.8` | Fraction of data used for training |
| `THRESHOLD` | `0.001` | Minimum prediction magnitude to trigger a BUY or SELL signal. Lower = more signals. |
| `TRANSACTION_COST` | `0.0005` | Round-trip trading cost (0.05 %) applied on signal changes |
| `INITIAL_CAPITAL` | `10000` | Starting portfolio value in USD |
| `HF_MODEL_ID` | `mistralai/Mistral-7B-Instruct-v0.2` | Hugging Face model for the LLM agent |

**Tip — too many SELL signals, not enough BUY?**  
Lower `THRESHOLD` in `config/settings.py` from `0.001` to `0.0005`. The energy return distribution is skewed slightly negative in this dataset, which means the positive tail (BUY zone) is hit less often at the default threshold.

---

## Running the Project

### Full ML pipeline (train + evaluate + agent demo)
```bash
python main.py
```

### Streamlit dashboard
```bash
python main.py --dashboard
# or directly:
streamlit run dashboard/app.py
```

### FastAPI REST server
```bash
python main.py --api
# Swagger UI → http://localhost:8000/docs
```


---

## API Reference

Start the server with `python main.py --api`, then visit `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/data/summary` | Row count, columns, date range |
| GET | `/data/quality` | Missing values, anomalies, duplicates, volatility |
| GET | `/strategy/metrics` | Sharpe ratio, max drawdown, win rate, total profit |
| POST | `/predict` | Run inference with XGBoost or LightGBM, return evaluation metrics |
| POST | `/agent/decision` | LLM agent APPROVE/REJECT decision for a given signal index |

### Example — POST /predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "xgboost", "n_samples": 50}'
```

### Example — POST /agent/decision
```bash
curl -X POST http://localhost:8000/agent/decision \
  -H "Content-Type: application/json" \
  -d '{"row_index": 0}'
```

---

## Dashboard

Open `http://localhost:8501` after starting the dashboard.

| Tab | Contents |
|---|---|
| 📈 Forecast | MAE / RMSE / MAPE cards, actual vs predicted chart, feature importance (XGBoost and LightGBM) |
| 💰 Strategy & PnL | Total profit / Sharpe / drawdown / win-rate cards, capital curve, daily signal chart |
| 🔍 Data Quality | Missing / anomaly / duplicate / volatility cards, feature drift bar chart, load distribution histogram |
| 🤖 LLM Agent | Single-signal agent decision, batch analysis table |
| 📋 Trade Log | Full timestamped trade log with CSV download |

---

## Resume Bullets

> Built an end-to-end energy demand forecasting system using **XGBoost** and **Prophet** with a walk-forward backtesting framework for algorithmic trading strategy evaluation.

> Developed an **LLM-powered trading agent** (Mistral-7B via Hugging Face) that generates structured APPROVE/REJECT decisions with risk commentary for each model signal; implemented a deterministic rule-based fallback for offline operation.

> Designed an automated **data pipeline** for PJM hourly load data including DST deduplication, time-weighted interpolation, synthetic price derivation, and feature-drift monitoring.

> Deployed the system as a **Streamlit dashboard** with interactive flip-card metrics, Plotly charts, and a **FastAPI** REST service; containerised with **Docker** for reproducible deployment.

---

