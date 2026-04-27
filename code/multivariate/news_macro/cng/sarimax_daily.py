import os, sys, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"GPU found: {gpus[0].name}")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("No GPU found, using CPU")

# ── Path setup ──────────────────────────────────────────────────
this_file     = os.path.abspath(__file__)
code_dir      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(this_file))))
capstone_root = os.path.dirname(code_dir)

if code_dir not in sys.path: sys.path.insert(0, code_dir)
os.chdir(capstone_root)

from dataloader import DataLoader
from models     import Models
from eda        import EDA

# ── Config ───────────────────────────────────────────────────
TARGET = "natural_gas"
SEED   = 42

def label_to_orders(label):
    parts = label.replace("SARIMA","").split("x")
    return eval(parts[0]), eval(parts[1])

def setup():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    rel         = os.path.relpath(os.path.dirname(this_file), code_dir)
    name        = os.path.splitext(os.path.basename(this_file))[0]
    results_dir = os.path.join(capstone_root, "results", rel, name)
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy2(this_file, results_dir)
    print(f"Results → {results_dir}")
    return results_dir

def main():
    results_dir = setup()
    dl  = DataLoader()
    eda = EDA(results_dir, TARGET)
    m   = Models()

    # ── 1. Load ──────────────────────────────────────────────────
    df = dl.combine_daily(exclude=[])
    print(f"Loaded: {df.shape} | {df.index.min()} → {df.index.max()}")
    print(df.head())

    # ── 2. EDA ───────────────────────────────────────────────────
    eda.plot_series(df)
    eda.decompose_plot(df, model="additive", period=12)

    print("\n── Stationarity: Original ──")
    stat_orig = eda.stationarity_check(df)

    if "Non-Stationary" in stat_orig["verdict"] or "Trend" in stat_orig["verdict"]:
        df_diff = eda.difference(df, periods=1)
        print("\n── Stationarity: Differenced ──")
        eda.stationarity_check(df_diff)
    else:
        df_diff = df.copy()

    eda.acf_pacf_plot(df,      lags=40, title_suffix="original")
    eda.acf_pacf_plot(df_diff, lags=40, title_suffix="differenced")

    # ── 3. Split — last 5 as test, rest as train ─────────────────
    train_data, test_data = dl.split_classic(df, tp=5)
    train_series = train_data[TARGET]
    test_series  = test_data[TARGET]
    print(f"\nTrain: {len(train_data)} | Test: {len(test_data)}")
    print(f"Test dates: {test_data.index.tolist()}")

    # ── 4. Suggest SARIMA orders ──────────────────────────────────
    print("\n── Suggesting SARIMA orders ──")
    suggested_orders, suggested_seasonal = eda.suggest_sarima_orders(
        train_series, s=12, max_lags=50, significance=0.05)
    print(f"\nSuggested orders:          {suggested_orders}")
    print(f"Suggested seasonal orders: {suggested_seasonal}")

    # ── 5. Fit all suggested SARIMA orders ────────────────────────
    print("\n── SARIMA ──")
    results, results_df, best_key = m.sarima(
        train_series,
        orders          = suggested_orders,
        seasonal_orders = suggested_seasonal
    )
    results_df.to_csv(os.path.join(results_dir, "sarima_order_comparison.csv"), index=False)

    # ── 6. Forecast last 5 steps for each order ───────────────────
    sarima_preds = m.predict_sarima(results, steps=5)

    # ── 7. Individual metrics ─────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{'Model':<35} {'MSE':>8} {'RMSE':>8} {'MAE':>8} {'MAPE%':>8}")
    print(f"{'='*65}")

    individual_metrics = {}
    for key, pred in sarima_preds.items():
        order, seasonal = key
        label = f"SARIMA{order}x{seasonal}"
        mse, rmse, mae, mape, r2 = eda.metrics(test_series, pred, lbl=label)
        individual_metrics[label] = (mse, rmse, mae, mape, r2)
        print(f"{label:<35} {mse:>8.4f} {rmse:>8.4f} {mae:>8.4f} {mape:>8.4f}")
    print(f"{'='*65}")

    # ── 8. Pick best 1 by RMSE ────────────────────────────────────
    best_label  = min(individual_metrics, key=lambda l: individual_metrics[l][1])
    best_order, best_seasonal = label_to_orders(best_label)
    best_pred   = sarima_preds[(best_order, best_seasonal)]
    best_model  = results[(best_order, best_seasonal)]
    print(f"\nBest model: {best_label}")

    # Save best model
    best_model.save(os.path.join(results_dir, f"sarima_{best_order}_{best_seasonal}.pkl"))
    print(f"Saved → sarima_{best_order}_{best_seasonal}.pkl")

    # ── 9. Metrics ───────────────────────────────────────────────
    print(f"\n── Best Model Metrics ──")
    mse, rmse, mae, mape, r2 = individual_metrics[best_label]

    fitted_vals   = np.array(best_model.fittedvalues).flatten()
    fitted_index  = best_model.fittedvalues.index
    fitted_series = pd.Series(fitted_vals, index=fitted_index)

    tr = eda.metrics(train_series.loc[fitted_index], fitted_series, "Train")
    te = (mse, rmse, mae, mape, r2)
    eda.metrics_table(tr, te, te)

    # ── 10. Forecast plot ─────────────────────────────────────────
    eda.forecast_plot_classic(
        train_series = train_series,
        test_act     = test_series,
        preds_dict   = {best_label: best_pred},
        title_suffix = "SARIMA Best"
    )

    # ── 11. Save metrics CSV ──────────────────────────────────────
    rows = []
    for label, met in individual_metrics.items():
        rows.append({"Model": label, "MSE": met[0], "RMSE": met[1],
                     "MAE": met[2], "MAPE%": met[3], "R2": met[4]})

    metrics_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    metrics_df.to_csv(os.path.join(results_dir, "sarima_metrics.csv"), index=False)
    print(f"\nMetrics summary:")
    print(metrics_df.to_string(index=False))
    print(f"\nAll saved → {results_dir}")

if __name__ == "__main__":
    main()