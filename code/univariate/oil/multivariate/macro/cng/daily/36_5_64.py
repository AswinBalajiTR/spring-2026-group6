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
code_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(this_file)))))
capstone_root = os.path.dirname(code_dir)

if code_dir not in sys.path: sys.path.insert(0, code_dir)
os.chdir(capstone_root)

from dataloader import DataLoader
from models     import Models
from eda        import EDA

# ── Config ──────────────────────────────────────────────────────
TARGET        = "natural_gas"
INPUT_WINDOW  = 36
OUTPUT_WINDOW = 5
BATCH_SIZE    = 64
EPOCHS        = 250
OPTUNA_EPOCHS = 50
N_TRIALS      = 30
VAL_SPLIT     = 0.10
PATIENCE      = 22
SEED          = 42

def setup():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED); tf.random.set_seed(SEED)
    rel       = os.path.relpath(os.path.dirname(this_file), code_dir)
    name      = os.path.splitext(os.path.basename(this_file))[0]
    results_dir = os.path.join(capstone_root, "results", rel, name)
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy2(this_file, results_dir)
    print(f"Results → {results_dir}")
    return results_dir

def main():

    results_dir = setup()

    # ── Load ─────────────────────────────────────────────────────
    dl = DataLoader()
    df = dl.combine_daily(exclude=["news_data.csv"])
    df.drop(['natgas_electric','brent_crude_oil','gasoline_price','usd_eur'],inplace=True,axis=1,errors='ignore')


    eda = EDA(results_dir, TARGET, INPUT_WINDOW)

    # ── Multicollinearity Check ───────────────────────────────────
    eda.check_multicollinearity(df,top_n=20,corr_threshold=0.3,vif_threshold=10)

    print(df.head())

    backtest_data, train_data, test_data = dl.split_time_series(df, INPUT_WINDOW, OUTPUT_WINDOW)


    train_data_scaled, test_data_scaled, backtest_data_scaled = eda.normalize(
        train_data, test_data, backtest_data, scaler_type="minmax")

    train_X, train_Y, test_X, test_Y, btest_X, btest_Y = dl.build_sequences(
        backtest_data_scaled, train_data_scaled, test_data_scaled,
        input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW, target=TARGET)

    print(f"train_X: {train_X.shape} | train_Y: {train_Y.shape}")
    print(f"test_X:  {test_X.shape}  | test_Y:  {test_Y.shape}")
    print(f"btest_X: {btest_X.shape} | btest_Y: {btest_Y.shape}")

    # ── Train ─────────────────────────────────────────────────────
    m           = Models(INPUT_WINDOW, OUTPUT_WINDOW, n_features=train_X.shape[2])
    best_params = m.tune_cnn_lstm(train_X, train_Y, n_trials=N_TRIALS,
                                   epochs=OPTUNA_EPOCHS, batch_size=BATCH_SIZE,
                                   val_split=VAL_SPLIT, patience=PATIENCE)
    model       = m.cnn_lstm(train_X, train_Y, epochs=EPOCHS, batch_size=BATCH_SIZE,
                              val_split=VAL_SPLIT, patience=PATIENCE, **best_params)
    model.save(os.path.join(results_dir, "cnn_lstm_model.keras"))

    # ── Predict ───────────────────────────────────────────────────
    train_pred = eda.inverse(model.predict(train_X))
    test_pred = eda.inverse(model.predict(test_X))
    bt_pred = eda.inverse(model.predict(btest_X))

    train_act = eda.inverse(train_Y)
    test_act = eda.inverse(test_Y)
    bt_act = eda.inverse(btest_Y)

    test_dates = test_data.index[-len(test_pred):]
    bt_dates = backtest_data.index[-len(bt_pred):]


    # ── Evaluate & Save ───────────────────────────────────────────
    tr_met, te_met, bt_met = eda.compute_all_metrics(train_act, train_pred, test_act, test_pred, bt_act, bt_pred)
    eda.loss_plot(model)
    eda.forecast_plot(df, test_dates, test_act, test_pred)
    eda.backtest_plot(df, bt_dates,   bt_act,   bt_pred)
    eda.metrics_table(tr_met, te_met, bt_met)
    print(f"All saved → {results_dir}")

if __name__ == "__main__":
    main()