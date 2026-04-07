import argparse

def run_full_pipeline():
    print("\n" + "="*60)
    print("  ENERGY TRADING AI — Full Pipeline")
    print("="*60)

    from src.data_pipeline import run_pipeline, data_quality_report, detect_drift
    from src.feature_engineering import build_features, leakage_check
    from src.forecasting import (train_xgboost, train_lightgbm, train_prophet,
                                  prophet_predict, evaluate, save_model)
    from src.backtesting import walk_forward_backtest
    from src.trading_strategy import generate_signals, simulate_pnl, build_trade_log
    from src.llm_agent import agent_decision
    from config.settings import FEATURES, TRAIN_SPLIT

    # 1. Data
    df    = run_pipeline()
    df    = build_features(df)
    n     = int(len(df) * TRAIN_SPLIT)
    train = df.iloc[:n]
    test  = df.iloc[n:]
    print(f"\n Data: {len(df):,} rows | Train: {len(train):,} | Test: {len(test):,}")

    # 2. Quality
    dq    = data_quality_report(df)
    drift = detect_drift(train, test, FEATURES)
    leakage_check(train)
    print(f"\n Quality : {dq}")
    print(f" Top drift: {dict(list(sorted(drift.items(), key=lambda x:-x[1]))[:3])}")

    # 3. Train
    print("\n Training XGBoost...")
    xgb_m = train_xgboost(train[FEATURES], train["return"])
    save_model(xgb_m, "xgboost")

    print(" Training LightGBM...")
    lgb_m = train_lightgbm(train[FEATURES], train["return"])
    save_model(lgb_m, "lightgbm")

    print(" Training Prophet...")
    prophet_m = train_prophet(train)
    fc = prophet_predict(prophet_m, periods=24)
    print("  Prophet 24h forecast:\n", fc.to_string(index=False))

    # 4. Evaluate
    print()
    xgb_preds = xgb_m.predict(test[FEATURES]).tolist()
    lgb_preds = lgb_m.predict(test[FEATURES]).tolist()
    evaluate(test["return"].values, xgb_preds, label="XGBoost (test set)")
    evaluate(test["return"].values, lgb_preds,  label="LightGBM (test set)")

    # 5. Backtest
    print("\n Walk-forward backtest...")
    _, _, bt_metrics = walk_forward_backtest(df)

    # 6. Strategy
    signals      = generate_signals(xgb_preds)
    cap_hist, pnl = simulate_pnl(xgb_preds, signals, test)
    print("\n💰 PnL Results:")
    for k, v in pnl.items():
        print(f"   {k}: {v}")

    # 7. LLM Agent
    row = {
        "prediction": xgb_preds[0], "signal": signals[0],
        "confidence": abs(xgb_preds[0]), "price": float(test["price"].iloc[0]),
    }
    market = {"avg_return": float(train["return"].tail(50).mean()),
              "volatility": float(train["return"].tail(50).std())}

    print("\n🤖 LLM Agent — first signal decision:")
    print(agent_decision(row, market, dq, drift))
    print("\n Done. Models saved to data/models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api",       action="store_true")
    parser.add_argument("--dashboard", action="store_true")
    args = parser.parse_args()

    if args.api:
        import uvicorn
        uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
    elif args.dashboard:
        import subprocess
        subprocess.run(["streamlit", "run", "dashboard/app.py"])
    else:
        run_full_pipeline()