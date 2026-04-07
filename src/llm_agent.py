"""

LLM-powered trading decision agent.

Primary path : Hugging Face Inference API → mistralai/Mistral-7B-Instruct-v0.2
Fallback path: Deterministic rule-based engine (no token / no internet required)

The agent receives:
  - The current ML model signal (BUY / SELL / HOLD + confidence)
  - Market context (recent avg return + volatility)
  - Data-quality metrics (missing values, anomalies)
  - Feature-drift scores (train → test distribution shift)

It returns a structured 3-line response:
  1. Decision - APPROVE or REJECT
  2. Risk     - one-sentence risk summary
  3. Improvement - one concrete model/data suggestion
"""

import re
import requests
from config.settings import HF_API_TOKEN, HF_MODEL_ID



# Prompt construction


def build_prompt(
    row: dict,
    market_stats: dict,
    dq_report: dict,
    drift_report: dict,
) -> str:
    
    # Summarise drift as a single scalar for the prompt
    avg_drift   = sum(drift_report.values()) / max(len(drift_report), 1)
    signal_word = {1: "BUY", -1: "SELL", 0: "HOLD"}.get(row["signal"], "HOLD")

    user_message = f"""You are an institutional quantitative trading risk controller.

Market context:
- Avg 50-period return : {market_stats['avg_return']:.6f}
- 50-period volatility : {market_stats['volatility']:.6f}

Current signal:
- Model prediction : {row['prediction']:.6f}
- Signal           : {signal_word}
- Confidence       : {row['confidence']:.6f}
- Current price    : {row['price']:.2f}

Data health:
- Missing values   : {dq_report['missing_values']}
- Anomalies (>3σ)  : {dq_report['anomalies']}
- Avg feature drift: {avg_drift:.4f}

Your response must have exactly 3 lines:
1. Decision: APPROVE or REJECT
2. Risk: one sentence explaining the main risk
3. Improvement: one concrete suggestion to improve the model or data quality"""

    # Mistral instruct chat template wraps the user turn in [INST] … [/INST]
    return f"<s>[INST] {user_message} [/INST]"



# Hugging Face Inference API

def query_hf_api(
    prompt: str,
    model_id: str = None,
    max_tokens: int = 200,
) -> str:
    
    model_id = model_id or HF_MODEL_ID
    api_url  = f"https://api-inference.huggingface.co/models/{model_id}"

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":   max_tokens,
            "temperature":      0.2,   # Low temperature → more deterministic, risk-averse output
            "top_p":            0.9,
            "return_full_text": False, # Return only the generated portion, not the prompt
            "do_sample":        True,
        },
        "options": {
            "wait_for_model": True,  # Block until model is loaded 
            "use_cache":      False, # Force fresh generation each call
        },
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()

        if isinstance(result, list) and len(result) > 0:
            text = result[0].get("generated_text", "").strip()
            return text if text else "[Empty response from model]"
        elif isinstance(result, dict) and "error" in result:
            return f"[HF API error: {result['error']}]"
        else:
            return str(result)

    except requests.exceptions.Timeout:
        return "[Timeout — model may be loading. Try again in 20s]"
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 503:
            return "[Model loading (cold start) — retry in 20 seconds]"
        return f"[HTTP error {resp.status_code}: {e}]"
    except Exception as e:
        return f"[Request error: {e}]"



# Rule-based fallback engine


def rule_based_decision(
    row: dict,
    market_stats: dict,
    dq_report: dict,
    drift_report: dict,
) -> str:
    
    signal_word = {1: "BUY", -1: "SELL", 0: "HOLD"}.get(row["signal"], "HOLD")
    confidence  = row["confidence"]
    prediction  = row["prediction"]
    anomalies   = dq_report.get("anomalies", 0)
    missing     = dq_report.get("missing_values", 0)
    volatility  = market_stats.get("volatility", 0)
    avg_drift   = sum(drift_report.values()) / max(len(drift_report), 1)

    #  Rejection rule evaluation 
    reject_reasons = []

    if signal_word == "HOLD":
        reject_reasons.append("signal is HOLD — no directional conviction")

    if confidence < 0.001:
        reject_reasons.append(
            f"confidence ({confidence:.6f}) is below minimum threshold"
        )

    if anomalies > 20:
        reject_reasons.append(
            f"high anomaly count ({anomalies}) indicates unreliable data"
        )

    if missing > 0:
        reject_reasons.append(f"{missing} missing values detected in pipeline")

    if avg_drift > 0.5:
        reject_reasons.append(
            f"feature drift ({avg_drift:.4f}) is too high — model may be stale"
        )

    if volatility > 0 and abs(prediction) < volatility * 0.5:
        reject_reasons.append(
            "prediction magnitude is too small relative to market volatility"
        )

    decision = "REJECT" if reject_reasons else "APPROVE"

    #  Risk sentence 
    if reject_reasons:
        # Lead with the primary rejection reason
        risk = reject_reasons[0].capitalize() + "."
    elif avg_drift > 0.3:
        risk = (
            f"Moderate feature drift ({avg_drift:.4f}) may reduce "
            "out-of-sample accuracy."
        )
    elif volatility > 0.01:
        risk = (
            f"Elevated market volatility ({volatility:.6f}) could widen actual "
            "slippage beyond modelled cost."
        )
    else:
        risk = (
            f"Signal confidence ({confidence:.6f}) is acceptable but monitor "
            "drift as test window extends."
        )

    #  Improvement suggestion 
    if avg_drift > 0.3:
        improvement = (
            "Retrain the model on a rolling window to reduce feature drift, "
            "and add lag_48 / lag_336 as additional features."
        )
    elif anomalies > 10:
        improvement = (
            "Apply Winsorisation (clip at 2.5σ) during preprocessing to reduce "
            "the impact of outliers on model training."
        )
    elif confidence < 0.002:
        improvement = (
            "Tune the signal threshold in config/settings.py (THRESHOLD) — "
            "current value may be filtering too many valid signals."
        )
    else:
        improvement = (
            "Incorporate external weather or demand-forecast features to improve "
            "model generalisation across seasons."
        )

    return (
        f"1. Decision: {decision}\n"
        f"2. Risk: {risk}\n"
        f"3. Improvement: {improvement}"
    )



# Public entry points

def agent_decision(
    row: dict,
    market_stats: dict,
    dq_report: dict,
    drift_report: dict,
    verbose: bool = True,
) -> str:
    
    prompt = build_prompt(row, market_stats, dq_report, drift_report)

    if HF_API_TOKEN:
        if verbose:
            print(f"  Querying HF API ({HF_MODEL_ID})...")
        response = query_hf_api(prompt)

        # Error responses from query_hf_api are wrapped in square brackets
        if response.startswith("["):
            if verbose:
                print(f"  HF API issue: {response}")
                print("  Falling back to rule-based decision engine...")
            response = rule_based_decision(row, market_stats, dq_report, drift_report)
    else:
        if verbose:
            print("  No HF_API_TOKEN — using rule-based decision engine")
        response = rule_based_decision(row, market_stats, dq_report, drift_report)

    return response


def batch_agent_analysis(
    trade_log_df,
    market_stats: dict,
    dq_report: dict,
    drift_report: dict,
    n_samples: int = 5,
) -> list:
   
    results = []
    samples = trade_log_df.head(n_samples)

    for _, row in samples.iterrows():
        r = {
            "prediction": row["prediction"],
            "signal":     row["signal"],
            "confidence": row["confidence"],
            "price":      row["price"],
        }
        decision = agent_decision(
            r, market_stats, dq_report, drift_report, verbose=False
        )
        results.append({
            "timestamp":  row.get("timestamp", ""),
            "signal":     row["signal"],
            "prediction": row["prediction"],
            "decision":   decision,
        })

    return results