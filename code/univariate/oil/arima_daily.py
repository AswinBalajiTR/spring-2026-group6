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

# ── Path setup ────────────────────────────────────────────────
this_file     = os.path.abspath(__file__)
code_dir      = os.path.dirname(os.path.dirname(os.path.dirname(this_file)))
capstone_root = os.path.dirname(code_dir)
if code_dir not in sys.path: sys.path.insert(0, code_dir)
os.chdir(capstone_root)

from dataloader import DataLoader
from models     import Models
from eda        import EDA

# ── Config ───────────────────────────────────────────────────
TARGET = "wti_crude_oil"
SEED   = 42

def label_to_order(label):
    return eval(label.replace("ARIMA", ""))

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

    # ── 1. Load & clean ──────────────────────────────────────────
    df = dl.read_data("wti_crude_oil.csv")
    df = df[df.index <= "2024-12-31"]
    df = df.rename(columns={"value": TARGET})
    df = df.asfreq("D").ffill()
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

    # ── 3. Split ─────────────────────────────────────────────────
    train_data, test_data = dl.split_classic(df, tp=5)
    test_act = test_data[TARGET]
    print(f"\nTrain: {len(train_data)} | Test: {len(test_data)}")
    print(f"Test dates: {test_data.index.tolist()}")

    # ── 4. Suggest ARIMA orders from ACF/PACF ────────────────────
    print("\n── Suggesting ARIMA orders ──")
    ORDERS = eda.suggest_arima_orders(
        train_data[TARGET],
        max_lags=40,
        significance=0.05
    )
    print(f"\nUsing orders: {ORDERS}")

    # ── 5. Fit all ARIMA orders ───────────────────────────────────
    print("\n── ARIMA ──")
    arima_results, arima_df, best_order = m.arima(
        train_data[TARGET], orders=ORDERS
    )
    arima_preds = m.predict_arima(arima_results, steps=5)

    # ── 6. Individual metrics ─────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{'Model':<25} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'MAPE%':>10}")
    print(f"{'='*65}")

    individual_metrics = {}
    for order, pred in arima_preds.items():
        label = f"ARIMA{order}"
        mse, rmse, mae, mape, r2 = eda.metrics(test_act, pred, lbl=label)
        individual_metrics[label] = (mse, rmse, mae, mape, r2)
        print(f"{label:<25} {mse:>10.4f} {rmse:>10.4f} {mae:>10.4f} {mape:>10.4f}")
    print(f"{'='*65}")

    # ── 7. Pick best 2 by RMSE & save ────────────────────────────
    sorted_models   = sorted(individual_metrics.items(), key=lambda x: x[1][1])
    best_two        = [label for label, _ in sorted_models[:2]]
    best_two_orders = [label_to_order(l) for l in best_two]
    print(f"\nBest 2 models selected: {best_two}")

    for order in best_two_orders:
        arima_results[order].save(os.path.join(results_dir, f"arima_{order}.pkl"))
        print(f"Saved → arima_{order}.pkl")

    # ── 8. Aggregation (mean of best 2) ──────────────────────────
    best_two_preds  = {label: arima_preds[label_to_order(label)] for label in best_two}
    all_pred_arrays = np.array(list(best_two_preds.values()))  # (2, 5)
    aggregated_pred = all_pred_arrays.mean(axis=0)

    print(f"\n── Aggregated Forecast (mean of {best_two}) ──")
    print(f"Dates:     {test_act.index.tolist()}")
    print(f"Actual:    {np.round(test_act.values, 4).tolist()}")
    print(f"Predicted: {np.round(aggregated_pred, 4).tolist()}")

    # ── 9. Aggregated metrics ────────────────────────────────────
    print(f"\n── Aggregated Model Metrics ──")
    agg_mse, agg_rmse, agg_mae, agg_mape, agg_r2 = eda.metrics(
        test_act, aggregated_pred, lbl="Aggregated (Best 2)"
    )

    fitted_vals   = np.array([
        arima_results[o].fittedvalues.values for o in best_two_orders
    ]).mean(axis=0)
    fitted_index  = arima_results[best_two_orders[0]].fittedvalues.index
    fitted_series = pd.Series(fitted_vals, index=fitted_index)

    tr = eda.metrics(train_data[TARGET].loc[fitted_index], fitted_series, "Train")
    te = (agg_mse, agg_rmse, agg_mae, agg_mape, agg_r2)
    bt = te
    eda.metrics_table(tr, te, bt)

    # ── 10. Forecast plot ─────────────────────────────────────────
    eda.forecast_plot_classic(
        train_series = train_data[TARGET],
        test_act     = test_act,
        preds_dict   = {f"Aggregated ({best_two[0]} + {best_two[1]})": aggregated_pred},
        title_suffix = "Best 2 Aggregated"
    )

    # ── 11. Save metrics CSV ──────────────────────────────────────
    rows = []
    for label, met in individual_metrics.items():
        rows.append({"Model": label,  "MSE": met[0], "RMSE": met[1],
                     "MAE":  met[2], "MAPE%": met[3], "R2":  met[4]})
    rows.append({"Model": f"Aggregated ({best_two[0]}+{best_two[1]})",
                 "MSE": agg_mse,  "RMSE": agg_rmse,
                 "MAE": agg_mae,  "MAPE%": agg_mape, "R2": agg_r2})

    metrics_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    metrics_df.to_csv(os.path.join(results_dir, "metrics_summary.csv"), index=False)
    print(f"\nMetrics summary:")
    print(metrics_df.to_string(index=False))
    print(f"\nAll saved → {results_dir}")

if __name__ == "__main__":
    main()