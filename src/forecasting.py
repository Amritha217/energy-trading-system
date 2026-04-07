"""

Model training, evaluation and persistence for all forecasting algorithms.
 
Models
------
XGBoost  - gradient-boosted trees, good on tabular data with lag features
LightGBM - faster gradient boosting, comparable accuracy, lower memory use
Prophet  - Facebook's additive decomposition model for long-horizon trend forecasting
 
Evaluation
----------
MAE    - average absolute error (interpretable, same units as target)
RMSE   - root mean squared error (penalises large misses more than MAE)
MAPE%  - mean absolute percentage error (scale-independent, useful for comparison)
 
Model persistence
-----------------
save_model() / load_model() use joblib for efficient serialisation of sklearn-
compatible estimators (XGBRegressor, LGBMRegressor).
"""


import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config.settings import TARGET_COL, DATE_COL, MODELS_DIR, FEATURES




# Train an XGBoost regression model on the provided feature matrix and target.

def train_xgboost(X_train, y_train) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=6, verbosity=0,
    )
    model.fit(X_train, y_train)
    return model



# Train a LightGBM regression model.
 
def train_lightgbm(X_train, y_train) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05,
        num_leaves=63, verbose=-1,
    )
    model.fit(X_train, y_train)
    return model



#    Fit a Prophet time-series model for long-horizon demand forecasting.
 
def train_prophet(df: pd.DataFrame) -> Prophet:
    pdf = df[[DATE_COL, TARGET_COL]].rename(columns={DATE_COL: "ds", TARGET_COL: "y"})
    m   = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                  daily_seasonality=True, interval_width=0.95)
    m.fit(pdf)
    return m





# Generate a future forecast using a fitted Prophet model.

def prophet_predict(model: Prophet, periods: int = 24) -> pd.DataFrame:
    future   = model.make_future_dataframe(periods=periods, freq="h")
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)


def evaluate(y_true, y_pred, label: str = "") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) /
                          (np.abs(y_true) + 1e-8))) * 100
    result = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE%": round(mape, 2)}
    if label:
        print(f"   {label}: {result}")
    return result


def save_model(model, name: str):
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"  Saved: {path.name}")

def load_model(name: str):
    return joblib.load(MODELS_DIR / f"{name}.joblib")