from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import run_pipeline, data_quality_report, detect_drift
from src.feature_engineering import build_features
from src.forecasting import load_model, evaluate
from src.trading_strategy import generate_signals, simulate_pnl
from src.llm_agent import agent_decision
from config.settings import FEATURES, TRAIN_SPLIT, DATE_COL

app = FastAPI(title="Energy Trading AI", version="1.0.0")

_cache: dict = {}

def get_state():
    if not _cache:
        df   = run_pipeline()
        df   = build_features(df)
        n    = int(len(df) * TRAIN_SPLIT)
        _cache.update({"df": df, "train": df.iloc[:n], "test": df.iloc[n:]})
    return _cache


@app.get("/")
def root():
    return {"status": "Energy Trading AI is live ⚡"}

@app.get("/data/summary")
def data_summary():
    s = get_state()
    df = s["df"]
    return {"rows": len(df), "columns": list(df.columns),
            "date_range": [str(df[DATE_COL].min()), str(df[DATE_COL].max())]}

@app.get("/data/quality")
def data_quality():
    return data_quality_report(get_state()["df"])

@app.get("/strategy/metrics")
def strategy_metrics():
    s = get_state()
    try:
        model = load_model("xgboost")
    except Exception:
        raise HTTPException(404, "Run `python main.py` first to train models.")
    preds   = model.predict(s["test"][FEATURES]).tolist()
    signals = generate_signals(preds)
    _, metrics = simulate_pnl(preds, signals, s["test"])
    return metrics

class PredictRequest(BaseModel):
    model_name: str = "xgboost"
    n_samples:  int = 100

@app.post("/predict")
def predict(req: PredictRequest):
    s = get_state()
    try:
        model = load_model(req.model_name)
    except Exception:
        raise HTTPException(404, f"Model '{req.model_name}' not found. Run main.py first.")
    X     = s["test"][FEATURES].iloc[:req.n_samples]
    y     = s["test"]["return"].iloc[:req.n_samples]
    preds = model.predict(X).tolist()
    return {"model": req.model_name, "metrics": evaluate(y.values, preds),
            "predictions_preview": preds[:5], "signals_preview": generate_signals(preds)[:5]}

class AgentRequest(BaseModel):
    row_index: int = 0

@app.post("/agent/decision")
def llm_decision(req: AgentRequest):
    s = get_state()
    try:
        model = load_model("xgboost")
    except Exception:
        raise HTTPException(404, "Run main.py first.")
    preds   = model.predict(s["test"][FEATURES]).tolist()
    signals = generate_signals(preds)
    dq      = data_quality_report(s["df"])
    drift   = detect_drift(s["train"], s["test"], FEATURES)
    row     = {"prediction": preds[req.row_index], "signal": signals[req.row_index],
               "confidence": abs(preds[req.row_index]), "price": float(s["test"]["price"].iloc[req.row_index])}
    market  = {"avg_return": float(s["train"]["return"].tail(50).mean()),
               "volatility": float(s["train"]["return"].tail(50).std())}
    return {"row": req.row_index, "signal": row["signal"],
            "decision": agent_decision(row, market, dq, drift)}