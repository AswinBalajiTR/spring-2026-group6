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
TARGET        = "wti_crude_oil"
INPUT_WINDOW  = 10    # ← change here
OUTPUT_WINDOW = 1    # ← change here
BATCH_SIZE    = 32
EPOCHS        = 250
OPTUNA_EPOCHS = 50
N_TRIALS      = 30
VAL_SPLIT     = 0.15
PATIENCE      = 20
SEED          = 42

def setup():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
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
    eda = EDA(results_dir, TARGET, INPUT_WINDOW)

    # ── 1. Load & clean ──────────────────────────────────────────
    df = dl.read_data("wti_crude_oil.csv")
    df = df[df.index <= "2025-12-31"]
    df = df.rename(columns={"value": TARGET})
    df = df.asfreq("D").ffill()
    print(f"Loaded: {df.shape} | {df.index.min()} → {df.index.max()}")
    print(df.head())

    # ── 3. Split ─────────────────────────────────────────────────
    backtest_data, train_data, test_data = dl.split_time_series(
        df, INPUT_WINDOW, OUTPUT_WINDOW
    )
    print(f"\nBacktest: {len(backtest_data)} | Train: {len(train_data)} | Test: {len(test_data)}")

    # ── 4. Normalize ─────────────────────────────────────────────
    train_scaled, test_scaled, backtest_scaled = eda.normalize(
        train_data, test_data, backtest_data, scaler_type="minmax"
    )

    # ── 5. Build sequences ────────────────────────────────────────
    train_X, train_Y, test_X, test_Y, btest_X, btest_Y = dl.build_sequences(
        backtest_scaled, train_scaled, test_scaled,
        input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW,
        target=TARGET,
        include_target=True  # ← add this
    )
    print(f"train_X: {train_X.shape} | train_Y: {train_Y.shape}")
    print(f"test_X:  {test_X.shape}  | test_Y:  {test_Y.shape}")
    print(f"btest_X: {btest_X.shape} | btest_Y: {btest_Y.shape}")

    # sanity check
    print(f"\ntrain_Y mean: {train_Y.mean():.4f}  std: {train_Y.std():.4f}")
    print(f"train_X mean: {train_X.mean():.4f}  std: {train_X.std():.4f}")

    # ── 6. Tune & Train CNN-LSTM ──────────────────────────────────
    m = Models(INPUT_WINDOW, OUTPUT_WINDOW, n_features=train_X.shape[2])

    print("\n── Tuning CNN-LSTM ──")
    best_params = m.tune_cnn_lstm(
        train_X, train_Y,
        n_trials=N_TRIALS,
        epochs=OPTUNA_EPOCHS,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        patience=PATIENCE
    )
    print(f"\nBest params: {best_params}")

    print("\n── Training CNN-LSTM ──")
    model = m.cnn_lstm(
        train_X, train_Y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        patience=PATIENCE,
        **best_params
    )
    model.save(os.path.join(results_dir, "cnn_lstm_model.keras"))
    print("Saved → cnn_lstm_model.keras")

    # ── 7. Predict ───────────────────────────────────────────────
    train_pred = eda.inverse(model.predict(train_X))
    test_pred  = eda.inverse(model.predict(test_X))
    bt_pred    = eda.inverse(model.predict(btest_X))

    train_act  = eda.inverse(train_Y)
    test_act   = eda.inverse(test_Y)
    bt_act     = eda.inverse(btest_Y)

    # flat prediction check
    print(f"\ntrain_pred std: {train_pred.std():.4f} | train_act std: {train_act.std():.4f}")
    print(f"test_pred  std: {test_pred.std():.4f}  | test_act  std: {test_act.std():.4f}")

    # ── 8. Dates ─────────────────────────────────────────────────
    test_dates = pd.DatetimeIndex([
        test_data.index[i + INPUT_WINDOW] for i in range(len(test_pred))
    ])
    bt_dates = pd.DatetimeIndex([
        backtest_data.index[i + INPUT_WINDOW] for i in range(len(bt_pred))
    ])

    # ── 9. Metrics ───────────────────────────────────────────────
    tr_met, te_met, bt_met = eda.compute_all_metrics(
        train_act, train_pred,
        test_act,  test_pred,
        bt_act,    bt_pred
    )

    # ── 10. Plots & save ─────────────────────────────────────────
    eda.loss_plot(model)
    eda.forecast_plot(df, test_dates, test_act, test_pred)
    eda.backtest_plot(df, bt_dates,   bt_act,   bt_pred)
    eda.metrics_table(tr_met, te_met, bt_met)

    # ── 11. Save metrics CSV ──────────────────────────────────────
    metrics_df = pd.DataFrame([{
        "Model": "CNN-LSTM",
        "MSE":   te_met[0], "RMSE": te_met[1],
        "MAE":   te_met[2], "MAPE%": te_met[3], "R2": te_met[4]
    }])
    metrics_df.to_csv(os.path.join(results_dir, "metrics_summary.csv"), index=False)
    print(f"\nMetrics summary:")
    print(metrics_df.to_string(index=False))
    print(f"\nAll saved → {results_dir}")

if __name__ == "__main__":
    main()