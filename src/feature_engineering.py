"""
Creates all model input features from the preprocessed load DataFrame.
 
Feature groups
--------------
Calendar   - hour, dayofweek, month, quarter
             Capture intra-day, weekly and seasonal demand cycles.
Lag        - lag_24  (same hour yesterday)
             lag_168 (same hour last week)
             Autoregressive signals — energy demand is strongly self-correlated.
Rolling     - rolling_mean_24  (24-hour trailing mean, shifted to avoid leakage)
             rolling_std_24   (24-hour trailing standard deviation)
Volatility  - already computed in data_pipeline.add_synthetic_price()
 
All rolling/lag operations use .shift(1) or .shift(N) before the window to
ensure no information from the current hour leaks into the features (look-ahead
bias prevention).
"""

import pandas as pd
from config.settings import TARGET_COL, DATE_COL, FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"]            = df[DATE_COL].dt.hour
    df["dayofweek"]       = df[DATE_COL].dt.dayofweek
    df["month"]           = df[DATE_COL].dt.month
    df["quarter"]         = df[DATE_COL].dt.quarter
    df["lag_24"]          = df[TARGET_COL].shift(24)
    df["lag_168"]         = df[TARGET_COL].shift(168)
    df["rolling_mean_24"] = df[TARGET_COL].shift(1).rolling(24).mean()
    df["rolling_std_24"]  = df[TARGET_COL].shift(1).rolling(24).std()
    return df.dropna().reset_index(drop=True)


def leakage_check(train: pd.DataFrame, target: str = "return"):
    corr = train[FEATURES + [target]].corr()[target].abs().sort_values(ascending=False)
    high = corr[corr > 0.7]
    print("⚠️  Leakage risk:" if len(high) else " No leakage detected")
    if len(high):
        print(high)
    return corr