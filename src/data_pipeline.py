"""

End-to-end data ingestion and preprocessing pipeline for the PJM energy dataset.

Pipeline stages
---------------
1. load_raw()            - read CSV, parse timestamps, fix DST duplicates,
                           resample to strict hourly frequency, interpolate gaps
2. add_synthetic_price() - derive a price proxy from normalised load,
                           compute 24-hour returns and rolling volatility
3. run_pipeline()        - orchestrate the above two stages (main entry point)

Auxiliary helpers
-----------------
data_quality_report() -  count missing values, anomalies, duplicates and overall volatility
detect_drift()        - compare train vs test feature distributions (mean shift / σ)
"""

import pandas as pd
import numpy as np
from config.settings import DATA_DIR, TARGET_COL, DATE_COL



# Stage 1 — Raw data loading and cleaning

def load_raw(filename: str = "PJME_hourly.csv") -> pd.DataFrame:
    path = DATA_DIR / filename
    df   = pd.read_csv(path)

    # Normalise column names to project constants
    df.rename(columns={"PJME_MW": TARGET_COL}, inplace=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)

    # Drop duplicate timestamps before reindexing.
    # DST transitions create two rows with the same hour; we keep the first.
    df = df.drop_duplicates(subset=[DATE_COL], keep="first")

    # Set datetime as index and enforce hourly frequency.
    df = df.set_index(DATE_COL).asfreq("h")

    # Time-weighted interpolation: fills short gaps (≤ 3 hours) accurately.
    df[TARGET_COL] = df[TARGET_COL].interpolate(method="time", limit=3)

    return df.reset_index()






# Stage 2 — Synthetic price and return derivation     
# Derive a price proxy from normalised load and compute financial returns.



def add_synthetic_price(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Z-score normalisation keeps the price proxy numerically stable
    df["load_norm"] = (df[TARGET_COL] - df[TARGET_COL].mean()) / df[TARGET_COL].std()

    # Simple linear price model: $50/MWh base ± $10 per σ of demand
    df["price"] = 50 + 10 * df["load_norm"]

    # 24-hour log return (pct_change(24) ≈ log-return for small changes)
    df["return"] = df["price"].pct_change(24)

    # Rolling realised volatility used as a model feature
    df["volatility_24"] = df["return"].rolling(24).std()

    return df.dropna().reset_index(drop=True)






# Quality reporting
# Compute a summary of data-quality indicators for display in the dashboard.


def data_quality_report(df: pd.DataFrame) -> dict:
    df = df.copy()
    # Z-score for anomaly detection
    df["load_z"] = (df[TARGET_COL] - df[TARGET_COL].mean()) / df[TARGET_COL].std()
    return {
        "missing_values": int(df.isnull().sum().sum()),
        "anomalies":      int((df["load_z"].abs() > 3).sum()),
        "duplicates":     int(df[DATE_COL].duplicated().sum()),
        "volatility":     round(float(df[TARGET_COL].std()), 2),
    }





# Compute a simple mean-shift drift score for each feature.

def detect_drift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list,
) -> dict:
    
    return {
        f: round(
            abs(train[f].mean() - test[f].mean()) / (train[f].std() + 1e-6),
            4,
        )
        for f in features
    }




# Main pipeline entry point
# Execute the full data pipeline from raw CSV to model-ready DataFrame.

def run_pipeline(filename: str = "PJME_hourly.csv") -> pd.DataFrame:
    print("Loading raw data...")
    df = load_raw(filename)
    df = add_synthetic_price(df)
    print(f"  Ready: {len(df):,} rows")
    return df