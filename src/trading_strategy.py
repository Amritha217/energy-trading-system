"""
Convert model return predictions into trading signals and simulate PnL.
 
Signal generation
-----------------
Predictions above +THRESHOLD → BUY  (+1)
Predictions below -THRESHOLD → SELL (-1)
Otherwise                    → HOLD  (0)
 
The THRESHOLD value in config/settings.py controls how selective the strategy
is. A lower value generates more signals (higher activity, higher costs); a
higher value generates fewer but higher-confidence signals.
 
PnL simulation
--------------
Starts with INITIAL_CAPITAL ($10,000 by default).
Position size is scaled by |prediction| (conviction-weighted sizing).
Transaction costs are applied on signal changes (not every period).
Capital is floored at 0 to prevent negative balances.
"""



import pandas as pd
import numpy as np
from config.settings import THRESHOLD, TRANSACTION_COST, INITIAL_CAPITAL, DATE_COL



# Convert raw return predictions into directional trading signals.


def generate_signals(predictions: list) -> list:
    return [1 if p > THRESHOLD else -1 if p < -THRESHOLD else 0 for p in predictions]




# Assemble a structured trade log for display and CSV export.


def build_trade_log(test: pd.DataFrame, predictions: list, signals: list) -> pd.DataFrame:
    date_col = test[DATE_COL].values if DATE_COL in test.columns else test.index
    return pd.DataFrame({
        "timestamp":  date_col,
        "prediction": predictions,
        "signal":     signals,
        "confidence": [abs(p) for p in predictions],
        "price":      test["price"].values,
        "actual_ret": test["return"].values,
    })




def simulate_pnl(predictions: list, signals: list, test: pd.DataFrame) -> tuple:
    capital = INITIAL_CAPITAL
    history = [capital]

    for i in range(1, len(predictions)):
        position     = np.clip(signals[i] * abs(predictions[i]), -1, 1)
        position_val = capital * abs(position)
        pnl          = position_val * test["return"].iloc[i] * np.sign(position)
        if signals[i] != signals[i - 1]:
            pnl -= position_val * TRANSACTION_COST
        capital = max(capital + pnl, 0)
        history.append(capital)

    returns  = pd.Series(history).pct_change().dropna()
    cum_max  = pd.Series(history).cummax()
    metrics  = {
        "total_profit":  round(history[-1] - INITIAL_CAPITAL, 2),
        "final_capital": round(history[-1], 2),
        "sharpe_ratio":  round((returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252 * 24), 4),
        "max_drawdown":  round(((pd.Series(history) - cum_max) / cum_max).min(), 4),
        "win_rate":      round(float((returns > 0).mean()), 4),
    }
    return history, metrics