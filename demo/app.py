import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Paths ─────────────────────────────────────────────────────────────────────
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(DEMO_DIR)
CODE_DIR = os.path.join(ROOT, "code")
DATA_DIR = os.path.join(ROOT, "data")
UNI_DIR   = os.path.join(ROOT, "results", "univariate")
MULTI_DIR = os.path.join(ROOT, "results", "multivariate")

for _p in [CODE_DIR, ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Config ────────────────────────────────────────────────────────────────────
ENERGY = {
    "crude_oil": {"label": "Crude Oil (WTI)", "csv": "wti_crude_oil.csv",
                  "col": "wti_crude_oil", "unit": "$/bbl", "folder": "oil"},
    "cng":       {"label": "CNG (Natural Gas)", "csv": "natural_gas.csv",
                  "col": "natural_gas", "unit": "$/MMBtu", "folder": "cng"},
}

# pkl filenames per model × energy
UNI_PKLS = {
    "ARIMA": {
        "crude_oil": ["arima_(0, 1, 0).pkl", "arima_(0, 1, 1).pkl"],
        "cng":       ["arima_(0, 1, 0).pkl", "arima_(1, 1, 0).pkl"],
    },
    "SARIMA": {
        "crude_oil": ["sarima_(1, 1, 0)x(0, 1, 0, 12).pkl", "sarima_(0, 1, 0)x(0, 1, 0, 12).pkl"],
        "cng":       ["sarima_(0, 1, 0)x(0, 1, 0, 12).pkl", "sarima_(0, 1, 1)x(0, 1, 0, 12).pkl"],
    },
}

# results subfolder per model × energy
UNI_FOLDER = {
    "ARIMA":  {"crude_oil": "arima_monthly",  "cng": "arima_monthly"},
    "SARIMA": {"crude_oil": "sarima_daily",    "cng": "sarima_monthly"},
    "LSTM":   {"crude_oil": "lstm_monthly",    "cng": "lstm_monthly"},
}

# LSTM config per energy — extracted exactly from each training script
LSTM_SPEC = {
    "crude_oil": {
        "iw": 5, "ow": 1,
        "resample": "MS",          # monthly — resample("MS").mean()
        "include_target": True,
        "scaler": "minmax",
        "target": "wti_crude_oil",
    },
    "cng": {
        "iw": 10, "ow": 1,
        "resample": None,          # NO resample — stays daily (asfreq("D").ffill())
        "include_target": True,
        "scaler": "minmax",
        "target": "natural_gas",
    },
}


# ── Multivariate Keras config ─────────────────────────────────────────────────
# Key: (feat_set, ekey, model_name)
# Extracted exactly from each training script
MULTI_KERAS_SPEC = {
    # ── Macro ──────────────────────────────────────────────────────────────────
    ("Macro", "crude_oil", "LSTM"): {
        "iw": 36, "ow": 1,
        "exclude": ["news_data.csv"],
        "drop": [],
        "include_target": False,
        "scaler": "minmax",
        "target": "wti_crude_oil",
        "path": ("macro", "oil", "lstm", "36_1_32", "lstm_model.keras"),
    },
    ("Macro", "cng", "LSTM"): {
        "iw": 36, "ow": 1,
        "exclude": ["news_data.csv"],
        "drop": [],
        "include_target": False,
        "scaler": "minmax",
        "target": "natural_gas",
        "path": ("macro", "cng", "lstm", "36_1_32", "lstm_model.keras"),
    },
    ("Macro", "cng", "CNN-LSTM"): {
        "iw": 36, "ow": 1,
        "exclude": ["news_data.csv"],
        "drop": ["pce","brent_crude_oil","ppi","natgas_electric","usd_index",
                 "usd_eur","cpi","cap_util_oil_gas","ind_production_overall",
                 "gasoline_price","ppidp"],
        "include_target": False,
        "scaler": "minmax",
        "target": "natural_gas",
        "path": ("macro", "cng", "monthly", "36_1_32", "cnn_lstm_model.keras"),
    },
    ("Macro", "crude_oil", "CNN-LSTM"): {
        "iw": 36, "ow": 1,
        "exclude": ["news_data.csv"],
        "drop": ["pce","usd_index","ppi","natgas_electric","vehicle_sales",
                 "ind_production_overall","usd_eur","cap_util_oil_gas",
                 "gasoline_price","ppidp","cpi"],
        "include_target": False,
        "scaler": "minmax",
        "target": "wti_crude_oil",
        "path": ("macro", "oil", "monthly", "36_1_32", "cnn_lstm_model.keras"),
    },    # ── News Only ───────────────────────────────────────────────────────────────
    ("News + Macro", "crude_oil", "LSTM"): {
        "iw": 24, "ow": 5,
        "exclude": [],
        "drop": [],
        "include_target": False,
        "scaler": "minmax",
        "target": "wti_crude_oil",
        "path": ("news_macro", "oil", "lstm", "24_5_32", "lstm_model.keras"),
    },
    ("News + Macro", "crude_oil", "CNN-LSTM"): {
        "iw": 36, "ow": 5,
        "exclude": [],
        "drop": [],
        "include_target": False,
        "scaler": "minmax",
        "target": "wti_crude_oil",
        "path": ("news_macro", "oil", "cnn_lstm", "36_5_32", "cnn_lstm_model.keras"),
    },
    ("News + Macro", "cng", "LSTM"): {
        "iw": 36, "ow": 5,
        "exclude": [],
        "drop": [],
        "include_target": False,
        "scaler": "minmax",
        "target": "natural_gas",
        "path": ("news_macro", "cng", "lstm", "36_5_32", "lstm_model.keras"),
    },
    ("News + Macro", "cng", "CNN-LSTM"): {
        "iw": 36, "ow": 5,
        "exclude": [],
        "drop": [],
        "include_target": False,
        "scaler": "minmax",
        "target": "natural_gas",
        "path": ("news_macro", "cng", "cnn_lstm", "36_5_32", "cnn_lstm_model.keras"),
    },
    ("News Only", "cng", "CNN-LSTM"): {
        "iw": 36, "ow": 1,
        "exclude": [],
        "drop": ["brent_crude_oil","wti_crude_oil","gasoline_price","usd_index",
                 "real_usd_index","usd_eur","volatility_index","ind_production_overall",
                 "ind_prod_drill","cap_util","cap_util_oil_gas","natgas_electric",
                 "cpi","ppi","ppidp","pce"],
        "include_target": False,
        "scaler": "minmax",
        "target": "natural_gas",
        "path": ("news", "cng", "monthly", "cnn_lstm_36_1_32", "cnn_lstm_model.keras"),
    },
    ("News Only", "crude_oil", "CNN-LSTM"): {
        "iw": 24, "ow": 5,
        "exclude": [],
        "drop": ["brent_crude_oil","natural_gas","gasoline_price","usd_index",
                 "real_usd_index","usd_eur","volatility_index","ind_production_overall",
                 "ind_prod_drill","cap_util","cap_util_oil_gas","natgas_electric",
                 "cpi","ppi","ppidp","pce"],
        "include_target": False,
        "scaler": "minmax",
        "target": "wti_crude_oil",
        "path": ("news", "oil", "monthly", "cnn_lstm_24_5_32", "cnn_lstm_model.keras"),
    },
}


# ── Multivariate SARIMAX config ───────────────────────────────────────────────
# Key: (feat_set, ekey)
# All SARIMAX models trained on daily data via dl.combine_daily()
# split_classic(df, tp=5) -> forecast(steps=5) -> best by RMSE
MULTI_SARIMAX_SPEC = {
    ("Macro", "crude_oil"): {
        "exclude": ["news_data.csv"], "daily": True,
        "path": ("macro", "oil", "sarimax_daily", "sarima_(0, 1, 0)_(0, 1, 0, 9).pkl"),
    },
    ("Macro", "cng"): {
        "exclude": ["news_data.csv"], "daily": False,
        "path": ("macro", "cng", "sarimax", "sarima_(0, 1, 0)_(0, 1, 0, 12).pkl"),
    },
    ("Macro", "cng"): {
        "exclude": ["news_data.csv"],
        "path": ("macro", "cng", "sarimax", "sarima_(0, 1, 0)_(0, 1, 0, 12).pkl"),
    },
        ("News + Macro", "crude_oil"): {
        "exclude": [], "daily": True,
        "path": ("news_macro", "oil", "sarimax_daily", "sarima_(0, 1, 1)_(0, 1, 0, 9).pkl"),
    },
    ("News + Macro", "cng"): {
        "exclude": [], "daily": True,
        "path": ("news_macro", "cng", "sarimax_daily", "sarima_(0, 1, 1)_(0, 1, 0, 6).pkl"),
    },
}


@st.cache_resource(show_spinner=False)
def load_pkl(path: str):
    import statsmodels.iolib.smpickle as sm
    return sm.load_pickle(path)


def forecast_multi_sarimax(feat_set: str, ekey: str):
    """
    Exact replica of multivariate SARIMAX training script:
      1. dl.combine_daily(exclude=spec["exclude"])  ← daily data
      2. split_classic(df, tp=5)  -> train=df[:-5], test=df[-5:]
      3. res.forecast(steps=5)
      4. No normalization — SARIMAX trained on raw values
    """
    key = (feat_set, ekey)
    if key not in MULTI_SARIMAX_SPEC:
        return None, None, None, None, f"No SARIMAX spec for {key}"

    spec = MULTI_SARIMAX_SPEC[key]

    try:
        from dataloader import DataLoader
        class SafeDataLoader(DataLoader):
            def _load_all(self, exclude=[]):
                dfs = {}
                for fname in os.listdir(self.data_dir):
                    if not fname.endswith(".csv"): continue
                    name = fname.replace(".csv", "")
                    if name in exclude or fname in exclude: continue
                    try:
                        df = pd.read_csv(
                            os.path.join(self.data_dir, fname),
                            parse_dates=["date"], index_col="date", dayfirst=False)
                    except Exception:
                        continue
                    if df.empty: continue
                    if not pd.api.types.is_datetime64_any_dtype(df.index): continue
                    freq = "MS" if self._is_monthly(df) else "D"
                    df = df.resample(freq).ffill()
                    if "value" in df.columns:
                        df = df[["value"]].rename(columns={"value": name})
                    elif len(df.columns) == 1:
                        df.columns = [name]
                    drop_cols = [c for c in df.columns if c in exclude]
                    if drop_cols: df = df.drop(columns=drop_cols)
                    if df.empty or len(df.columns) == 0: continue
                    dfs[name] = df
                return dfs
        dl = SafeDataLoader.__new__(SafeDataLoader)
        dl.data_dir = DATA_DIR
        if spec.get("daily", True):
            df = dl.combine_daily(exclude=spec["exclude"])
        else:
            df = dl.combine_all_as_monthly(exclude=spec["exclude"])
    except Exception as e:
        return None, None, None, None, f"DataLoader failed: {e}"

    target = ENERGY[ekey]["col"]
    if target not in df.columns:
        return None, None, None, None, f"Target '{target}' not in df."

    train = df[target].iloc[:-5]
    test  = df[target].iloc[-5:]

    fpath = os.path.join(MULTI_DIR, *spec["path"])
    if not os.path.exists(fpath):
        return None, None, None, None, f"Model not found:\n{fpath}"

    try:
        res  = load_pkl(fpath)
        pred = np.array(res.forecast(steps=5)).flatten()
    except Exception as e:
        return None, None, None, None, f"Forecast failed: {e}"

    rmse  = float(np.sqrt(np.mean((test.values - pred) ** 2)))
    label = f"SARIMAX · {feat_set}  (RMSE {rmse:.3f})"
    return train, test, pred, label, None


# Columns that can be tweaked and their typical value ranges
TWEAK_COLS = {
    "real_usd_index":  {"label": "Real USD Index",    "min": 80.0,   "max": 140.0,  "default": 110.0},
    "brent_crude_oil": {"label": "Brent Crude Oil",   "min": 20.0,   "max": 150.0,  "default": 75.0},
    "volatility_index":{"label": "Volatility Index",  "min": 10.0,   "max": 100.0,  "default": 35.0},
    "GoldsteinScale":  {"label": "Goldstein Scale",   "min": -10.0,  "max": 10.0,   "default": 0.0},
    "AvgTone":         {"label": "Avg Tone",           "min": -20.0,  "max": 20.0,   "default": 0.0},
}


def apply_tweaks(test_X: np.ndarray, x_cols: list, tweaks: dict,
                 scaler_mins: np.ndarray, scaler_ranges: np.ndarray,
                 all_cols: list) -> np.ndarray:
    """
    Override ALL timesteps of tweaked features in test_X.
    test_X shape : (n_windows, iw, n_x_features)
    x_cols       : feature column names (same order as test_X axis-2)
    tweaks       : {col_name: raw_slider_value}
    scaler_mins  : scaler.data_min_  (one per all_cols)
    scaler_ranges: scaler.data_range_ (one per all_cols)
    all_cols     : full df.columns (used to look up scaler index)
    """
    X = test_X.copy()
    for col, val in tweaks.items():
        if col not in x_cols:
            continue
        fi      = x_cols.index(col)          # index in test_X feature axis
        si      = all_cols.index(col)         # index in scaler arrays
        s_min   = float(scaler_mins[si])
        s_range = float(scaler_ranges[si]) + 1e-8
        scaled  = (val - s_min) / s_range
        X[:, :, fi] = scaled
    return X


def forecast_multi_keras(feat_set: str, ekey: str, model_name: str, tweaks: dict = {}):
    """
    Exact replica of multivariate training script inference for ow=1:

    1. dl.combine_all_as_monthly(exclude=spec["exclude"])
    2. df.drop(spec["drop"])
    3. split_time_series(df, iw, ow=1)  ->  block = iw+5
       test_data = df.iloc[n-block:]    ->  (iw+5) rows
    4. normalize(train_data, test_data, backtest_data, "minmax")
       MinMaxScaler fitted on train_data (multivariate)
    5. build_sequences(..., target=target, include_target=False)
       ow=1, rolling -> tst_idx=[0,1,2,3,4]
       test_X[i] = test_scaled[i:i+iw, x_cols]   (all cols except target)
       test_Y[i] = test_scaled[i+iw, target_col]
    6. model.predict(test_X)  ->  eda.inverse()
       inv: arr * data_range_[t_idx] + data_min_[t_idx]
    7. test_dates = test_data.index[-len(test_pred):]
       (= test_data.index[iw : iw+5])
    """
    from sklearn.preprocessing import MinMaxScaler

    key  = (feat_set, ekey, model_name)
    if key not in MULTI_KERAS_SPEC:
        return None, None, None, None, f"No spec found for {key}"

    spec   = MULTI_KERAS_SPEC[key]
    iw     = spec["iw"]
    ow     = spec["ow"]
    target = spec["target"]
    cfg    = ENERGY[ekey]

    # ── 1. Load multivariate data ─────────────────────────────────────────
    try:
        from dataloader import DataLoader
        class SafeDataLoader(DataLoader):
            def _load_all(self, exclude=[]):
                # Exact copy of DataLoader._load_all but skips CSVs without date index
                dfs = {}
                for fname in os.listdir(self.data_dir):
                    if not fname.endswith(".csv"): continue
                    name = fname.replace(".csv", "")
                    if name in exclude or fname in exclude: continue
                    try:
                        df = pd.read_csv(
                            os.path.join(self.data_dir, fname),
                            parse_dates=["date"], index_col="date", dayfirst=False)
                    except Exception:
                        continue   # skip CSVs missing 'date' column
                    if df.empty: continue
                    if not pd.api.types.is_datetime64_any_dtype(df.index): continue
                    freq = "MS" if self._is_monthly(df) else "D"
                    df = df.resample(freq).ffill()
                    # rename 'value' col to filename stem (single-series FRED files)
                    if "value" in df.columns:
                        df = df[["value"]].rename(columns={"value": name})
                    elif len(df.columns) == 1:
                        df.columns = [name]
                    # multi-col files (e.g. news_data.csv): keep their own col names
                    drop_cols = [c for c in df.columns if c in exclude]
                    if drop_cols: df = df.drop(columns=drop_cols)
                    if df.empty or len(df.columns) == 0: continue
                    dfs[name] = df
                return dfs
        dl = SafeDataLoader.__new__(SafeDataLoader)
        dl.data_dir = DATA_DIR
        df = dl.combine_all_as_monthly(exclude=spec["exclude"])
    except Exception as e:
        return None, None, None, None, f"DataLoader failed: {e}"

    # ── 2. Drop columns ───────────────────────────────────────────────────
    if spec["drop"]:
        df = df.drop(columns=[c for c in spec["drop"] if c in df.columns], errors="ignore")

    if target not in df.columns:
        return None, None, None, None, f"Target '{target}' not in df columns."

    # ── 3. split_time_series(df, iw, ow=1) ───────────────────────────────
    block = iw + 5
    n     = len(df)
    if n <= 2 * block:
        return None, None, None, None, f"Dataset too short: {n} rows, need > {2*block}"
    backtest_data = df.iloc[:block]
    train_data    = df.iloc[block: n - block]
    test_data     = df.iloc[n - block:]       # (iw+5) rows

    # ── 4. normalize(train, test, backtest) ──────────────────────────────
    scaler = MinMaxScaler()
    cols   = df.columns.tolist()
    train_scaled    = pd.DataFrame(scaler.fit_transform(train_data),
                                   index=train_data.index, columns=cols)
    test_scaled     = pd.DataFrame(scaler.transform(test_data),
                                   index=test_data.index,  columns=cols)

    # eda.inverse(): arr * data_range_[t_idx] + data_min_[t_idx]
    t_idx   = cols.index(target)
    t_min   = float(scaler.data_min_[t_idx])
    t_range = float(scaler.data_range_[t_idx])
    def inv(arr): return np.array(arr).flatten() * t_range + t_min

    # ── 5. build_sequences (mirrors dl.build_sequences exactly) ─────────────
    # include_target=False -> x_cols = all cols EXCEPT target
    # ow=1 rolling  -> tst_idx=[0,1,2,3,4]  test_X:(5,iw,nf)  test_Y:(5,)
    # ow=5 single   -> tst_idx=[0]           test_X:(1,iw,nf)  test_Y:(5,)
    x_cols  = [c for c in cols if c != target]
    xi      = [cols.index(c) for c in x_cols]
    arr     = test_scaled.values                       # (iw+5, n_features)
    rolling = (ow == 1)
    tst_idx = list(range(5)) if rolling else [0]
    test_X  = np.array([arr[i: i+iw, :][:, xi]           for i in tst_idx])
    # ow=1: Y[i] = scalar at i+iw  -> shape (5,)
    # ow=5: Y[0] = rows iw..iw+5   -> shape (5,)  (squeeze applied)
    if rolling:
        test_Y = np.array([arr[i+iw, t_idx] for i in tst_idx])          # (5,)
    else:
        test_Y = arr[iw: iw+5, t_idx]                                    # (5,)

    # ── 6. Load model & predict ───────────────────────────────────────────
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import tensorflow as tf

    fpath = os.path.join(MULTI_DIR, *spec["path"])
    if not os.path.exists(fpath):
        return None, None, None, None, f"Model not found:\n{fpath}"

    model    = tf.keras.models.load_model(fpath)

    # Validate / fix feature count to match model's expected input shape
    n_expected = model.input_shape[2]
    n_built    = test_X.shape[2]
    if n_built != n_expected:
        if n_built < n_expected:
            # pad missing features with column mean (neutral, scaled space ≈ 0.5)
            pad    = np.full((test_X.shape[0], iw, n_expected - n_built), 0.5)
            test_X = np.concatenate([test_X, pad], axis=2)
        else:
            test_X = test_X[:, :, :n_expected]

    test_X_orig = test_X.copy()   # keep original for default slider values

    # Apply any user tweaks (slider overrides on specific feature columns)
    if tweaks:
        test_X = apply_tweaks(test_X, x_cols, tweaks,
                              scaler.data_min_, scaler.data_range_, cols)
    raw_pred = model.predict(test_X, verbose=0)
    # ow=1: raw_pred shape (5,1) -> flatten -> 5 values
    # ow=5: raw_pred shape (1,5) -> flatten -> 5 values
    pred = inv(raw_pred)[:5]
    act  = inv(test_Y)[:5]

    # test_dates = test_data.index[-len(test_pred):]
    # = test_data.index[iw : iw+5] for both ow=1 and ow=5
    test_dates = test_data.index[iw: iw + 5]
    test_act   = pd.Series(act,  index=test_dates, name=target)
    train_ser  = train_data[target]

    label = f"{model_name} · {feat_set}  (iw={iw})"

    # ── Backtest (mirrors training script bt logic) ───────────────────────
    backtest_scaled = pd.DataFrame(scaler.transform(backtest_data),
                                   index=backtest_data.index, columns=cols)
    bt_arr  = backtest_scaled.values
    bt_idx  = list(range(5)) if rolling else [0]
    bt_X    = np.array([bt_arr[i: i+iw, :][:, xi] for i in bt_idx])
    if rolling:
        bt_Y = np.array([bt_arr[i+iw, t_idx] for i in bt_idx])
    else:
        bt_Y = bt_arr[iw: iw+5, t_idx]
    # match feature count
    if bt_X.shape[2] < n_expected:
        pad_bt = np.full((bt_X.shape[0], iw, n_expected - bt_X.shape[2]), 0.5)
        bt_X   = np.concatenate([bt_X, pad_bt], axis=2)
    else:
        bt_X = bt_X[:, :, :n_expected]
    bt_raw  = model.predict(bt_X, verbose=0)
    bt_pred = inv(bt_raw)[:5]
    bt_act  = inv(bt_Y)[:5]
    bt_dates = backtest_data.index[iw: iw + 5]
    bt_actual = pd.Series(bt_act, index=bt_dates, name=target)

    # Tweak defaults
    tweak_defaults = {}
    for col in TWEAK_COLS:
        if col in x_cols:
            fi  = x_cols.index(col)
            si  = cols.index(col)
            raw = float(test_X[0, 0, fi]) * float(scaler.data_range_[si]) + float(scaler.data_min_[si])
            tweak_defaults[col] = raw

    return train_ser, test_act, pred, label, None, tweak_defaults, scaler, cols, x_cols, test_X_orig, bt_actual, bt_pred


def forecast_lstm(ekey: str):
    """
    Exact replica of the training script inference pipeline for ow=1:

    split_time_series(df, iw, ow):
        block = iw + 5
        test_data = df.iloc[n - block:]   ← has (iw+5) rows

    build_sequences with include_target=True, ow=1 (rolling):
        tst_idx = [0, 1, 2, 3, 4]
        test_X[i] = test_scaled[i : i+iw, :]     ← all cols as features
        test_Y[i] = test_scaled[i+iw, target_idx] ← target at step i+iw
        -> test_X shape: (5, iw, nf)
        -> test_Y shape: (5,)

    test_dates = test_data.index[i + iw] for i in 0..4
        -> these are the last 5 rows of the test block (the actual targets)

    eda.inverse(arr):
        arr.flatten() * scaler.data_range_[t_idx] + scaler.data_min_[t_idx]
    """
    from sklearn.preprocessing import MinMaxScaler

    spec   = LSTM_SPEC[ekey]
    iw     = spec["iw"]
    ow     = spec["ow"]           # always 1 for univariate LSTM
    target = spec["target"]
    cfg    = ENERGY[ekey]

    # ── 1. Load — exact mirror of each training script ────────────────────
    df = pd.read_csv(os.path.join(DATA_DIR, cfg["csv"]),
                     parse_dates=["date"], index_col="date")
    df = df[df.index <= "2025-12-31"]
    if "value" in df.columns:
        df = df.rename(columns={"value": target})
    if ekey == "cng":
        df = df.asfreq("D").ffill()          # fills daily gaps
    if spec["resample"]:
        df = df.resample(spec["resample"]).mean()
    df = df[[target]]

    # ── 2. split_time_series(df, iw, ow=1): block = iw + 5 ──────────────
    block = iw + 5
    n     = len(df)
    if n <= 2 * block:
        return None, None, None, None, f"Dataset too short: {n} rows"
    backtest_data = df.iloc[:block]
    train_data    = df.iloc[block: n - block]
    test_data     = df.iloc[n - block:]      # shape: (block=iw+5, 1)

    # ── 3. normalize(train_data, test_data, backtest_data) ───────────────
    # Training script: eda.normalize(train_data, test_data, backtest_data)
    # MinMaxScaler fitted on train_data ONLY
    scaler = MinMaxScaler()
    cols   = df.columns.tolist()
    train_scaled    = pd.DataFrame(scaler.fit_transform(train_data),
                                   index=train_data.index, columns=cols)
    test_scaled     = pd.DataFrame(scaler.transform(test_data),
                                   index=test_data.index,  columns=cols)

    # eda.inverse(): arr.flatten() * data_range_[t_idx] + data_min_[t_idx]
    t_idx   = cols.index(target)
    t_min   = float(scaler.data_min_[t_idx])
    t_range = float(scaler.data_range_[t_idx])
    def inv(arr): return np.array(arr).flatten() * t_range + t_min

    # ── 4. build_sequences(include_target=True, ow=1) ────────────────────
    # include_target=True -> get_cols returns all cols for X (including target)
    # ow=1 -> rolling=True -> tst_idx = list(range(5))
    # make(test_data, tst_idx):
    #   X[i] = test_scaled[i : i+iw, all_cols]
    #   Y[i] = test_scaled[i+iw : i+iw+1, target_col]  -> squeezed to scalar
    arr     = test_scaled.values              # (iw+5, 1)
    tst_idx = list(range(5))                  # rolling windows [0,1,2,3,4]
    test_X  = np.array([arr[i: i+iw, :]      for i in tst_idx])  # (5, iw, 1)
    test_Y  = np.array([arr[i+iw, t_idx]     for i in tst_idx])  # (5,)

    # ── 5. Load model & predict ───────────────────────────────────────────
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import tensorflow as tf

    fpath = os.path.join(UNI_DIR, cfg["folder"],
                         UNI_FOLDER["LSTM"][ekey], "lstm_model.keras")
    if not os.path.exists(fpath):
        return None, None, None, None, f"Model not found:\n{fpath}"

    model = tf.keras.models.load_model(fpath)

    # model.predict(test_X) -> shape (5, 1) for ow=1 -> flatten -> 5 values
    raw_pred = model.predict(test_X, verbose=0)   # (5, 1) or (5,)
    pred     = inv(raw_pred)                        # (5,)
    act      = inv(test_Y)                          # (5,) — inverse of actual

    # ── 6. Dates — test_data.index[i + iw] for i in 0..4 ────────────────
    # These are the target positions: test_data[iw], test_data[iw+1], ...
    test_dates = pd.DatetimeIndex([test_data.index[i + iw] for i in range(5)])
    test_act   = pd.Series(act,  index=test_dates, name=target)
    train_ser  = train_data[target]

    label = f"LSTM  (look-back={iw})"

    # ── Backtest (mirrors training script bt logic) ───────────────────────
    backtest_scaled = pd.DataFrame(scaler.transform(backtest_data),
                                   index=backtest_data.index, columns=cols)
    bt_arr   = backtest_scaled.values
    bt_tst   = list(range(5))
    bt_X     = np.array([bt_arr[i: i+iw, :] for i in bt_tst])   # (5, iw, 1)
    bt_Y     = np.array([bt_arr[i+iw, t_idx] for i in bt_tst])   # (5,)
    bt_raw   = model.predict(bt_X, verbose=0)
    bt_pred  = inv(bt_raw)
    bt_act   = inv(bt_Y)
    bt_dates = pd.DatetimeIndex([backtest_data.index[i + iw] for i in range(5)])
    bt_actual = pd.Series(bt_act, index=bt_dates, name=target)

    input_seqs = [
        test_data[target].iloc[i: i+iw].values
        for i in range(5)
    ]
    input_dates = [
        test_data.index[i: i+iw].tolist()
        for i in range(5)
    ]
    return train_ser, test_act, pred, label, None, input_seqs, input_dates, iw, bt_actual, bt_pred


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_series_monthly(ekey: str) -> pd.Series:
    """Monthly series — used by ARIMA and CNG SARIMA (trained on monthly data)."""
    cfg = ENERGY[ekey]
    df  = pd.read_csv(os.path.join(DATA_DIR, cfg["csv"]),
                      parse_dates=["date"], index_col="date")
    df  = df[df.index <= "2025-12-31"]
    if "value" in df.columns:
        df = df.rename(columns={"value": cfg["col"]})
    if ekey == "cng":                    # natural_gas.csv has daily gaps
        df = df.asfreq("D").ffill()
    return df.resample("MS").mean()[cfg["col"]]


@st.cache_data(show_spinner=False)
def load_series_daily(ekey: str) -> pd.Series:
    """Daily series — used by Oil SARIMA (trained on raw daily data, no resample)."""
    cfg = ENERGY[ekey]
    df  = pd.read_csv(os.path.join(DATA_DIR, cfg["csv"]),
                      parse_dates=["date"], index_col="date")
    df  = df[df.index <= "2025-12-31"]
    if "value" in df.columns:
        df = df.rename(columns={"value": cfg["col"]})
    return df[cfg["col"]]


def get_series(ekey: str, model_name: str) -> pd.Series:
    """Return the correct series for the model × energy combination."""
    if model_name == "SARIMA" and ekey == "crude_oil":
        return load_series_daily(ekey)
    return load_series_monthly(ekey)


@st.cache_resource(show_spinner=False)
def load_pkl(path: str):
    import statsmodels.iolib.smpickle as sm
    return sm.load_pickle(path)


# ── Forecast ──────────────────────────────────────────────────────────────────
def forecast_stat(ekey: str, model_name: str):
    """
    Load both pkl files, predict on the test window, return best by RMSE.
    split_classic: train = df[:-5], test = df[-5:]
    predict: res.forecast(steps=5)  (matches training scripts)
    """
    cfg    = ENERGY[ekey]
    folder = os.path.join(UNI_DIR, cfg["folder"], UNI_FOLDER[model_name][ekey])
    fnames = UNI_PKLS[model_name][ekey]
    series = get_series(ekey, model_name)
    train, test = series.iloc[:-5], series.iloc[-5:]

    preds, labels = [], []
    for fname in fnames:
        fpath = os.path.join(folder, fname)
        if not os.path.exists(fpath):
            return None, None, None, None, f"File not found:\n{fpath}"
        try:
            pred = np.array(load_pkl(fpath).forecast(steps=5)).flatten()
            preds.append(pred)
            labels.append(os.path.splitext(fname)[0])
        except Exception as e:
            return None, None, None, None, f"Forecast failed ({fname}): {e}"

    rmses  = [float(np.sqrt(np.mean((test.values - p) ** 2))) for p in preds]
    best_i = int(np.argmin(rmses))
    label  = f"{model_name} · {labels[best_i]}  (RMSE {rmses[best_i]:.3f})"
    return train, test, preds[best_i], label, None


# ── Chart ─────────────────────────────────────────────────────────────────────
def make_chart(train: pd.Series, test: pd.Series, pred: np.ndarray,
               unit: str, title: str) -> go.Figure:
    dark  = st.session_state.get("theme", "dark") == "dark"

    # Color palette per theme
    ctx_color  = "#4a6fa8"  if dark else "#7090b8"
    act_color  = "#3d7fcc"  if dark else "#1d4ed8"
    fore_color = "#c0392b"  if dark else "#b91c1c"
    div_color  = "#4a6080"  if dark else "#94a3b8"
    txt_color  = "#d8e1ee"  if dark else "#1e293b"
    bg_color   = "#0e1520"  if dark else "#ffffff"
    grid_color = "#1a2438"  if dark else "#e2e8f0"
    mrk_border = "#0e1520"  if dark else "#ffffff"

    ctx   = train.iloc[-30:]
    tvals = test.values.flatten()
    pvals = np.array(pred).flatten()
    fig   = go.Figure()

    # Context line
    fig.add_trace(go.Scatter(
        x=ctx.index, y=ctx.values, mode="lines", name="Context",
        line=dict(color=ctx_color, width=1.6, dash="dash"),
        opacity=0.7,
        hovertemplate="%{x|%Y-%m-%d}  %{y:.2f}<extra></extra>"))

    # Connector
    fig.add_trace(go.Scatter(
        x=[ctx.index[-1], test.index[0]], y=[ctx.values[-1], tvals[0]],
        mode="lines", showlegend=False, opacity=0.2,
        line=dict(color=ctx_color, width=1, dash="dash"), hoverinfo="skip"))

    # Actual
    fig.add_trace(go.Scatter(
        x=test.index, y=tvals, mode="lines+markers+text", name="Actual",
        line=dict(color=act_color, width=2.2),
        marker=dict(symbol="circle", size=8, color=act_color,
                    line=dict(color=mrk_border, width=1.5)),
        text=[f"{v:.2f}" for v in tvals], textposition="bottom center",
        textfont=dict(size=10, color=act_color),
        hovertemplate="%{x|%Y-%m-%d}  Actual %{y:.2f}<extra></extra>"))

    # Forecast
    fig.add_trace(go.Scatter(
        x=test.index, y=pvals, mode="lines+markers+text", name="Forecast",
        line=dict(color=fore_color, width=2.2, dash="dash"),
        marker=dict(symbol="triangle-up", size=9, color=fore_color,
                    line=dict(color=mrk_border, width=1.5)),
        text=[f"{v:.2f}" for v in pvals], textposition="top center",
        textfont=dict(size=10, color=fore_color),
        hovertemplate="%{x|%Y-%m-%d}  Forecast %{y:.2f}<extra></extra>"))

    # Train/test divider
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=train.index[-1].strftime("%Y-%m-%d"),
                  x1=train.index[-1].strftime("%Y-%m-%d"),
                  y0=0, y1=1,
                  line=dict(color=div_color, width=1, dash="dot"))

    fig.update_layout(
        title=dict(text=title, font=dict(size=11, color=txt_color), x=0, y=0.97),
        height=430,
        hovermode="x unified",
        margin=dict(l=12, r=12, t=60, b=48),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=txt_color, family="Inter, sans-serif"),
        legend=dict(
            orientation="h", y=1.10, x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=txt_color),
        ),
        xaxis=dict(
            type="date",
            range=[ctx.index[0].strftime("%Y-%m-%d"),
                   (test.index[-1] + pd.DateOffset(months=2)).strftime("%Y-%m-%d")],
            tickangle=-40,
            tickfont=dict(size=10, color=txt_color),
            gridcolor=grid_color,
            linecolor=grid_color,
            tickformatstops=[
                dict(dtickrange=[None, 1000*60*60*24*365*2], value="%b %Y"),
                dict(dtickrange=[1000*60*60*24*365*2, None],  value="%Y"),
            ],
        ),
        yaxis=dict(
            title=dict(text=unit, font=dict(size=11, color=txt_color)),
            tickfont=dict(size=10, color=txt_color),
            gridcolor=grid_color,
            linecolor=grid_color,
        ),
    )
    return fig


def make_backtest_chart(full_series: pd.Series, bt_actual: pd.Series,
                        bt_pred: np.ndarray, unit: str, title: str) -> go.Figure:
    """Backtest plot: shows context from full series around the backtest window."""
    dark = st.session_state.get("theme", "dark") == "dark"
    ctx_color  = "#4a6fa8" if dark else "#7090b8"
    act_color  = "#3d7fcc" if dark else "#1d4ed8"
    fore_color = "#c0392b" if dark else "#b91c1c"
    div_color  = "#4a6080" if dark else "#94a3b8"
    txt_color  = "#d8e1ee" if dark else "#1e293b"
    bg_color   = "#0e1520" if dark else "#ffffff"
    grid_color = "#1a2438" if dark else "#e2e8f0"
    mrk_border = "#0e1520" if dark else "#ffffff"

    tvals = bt_actual.values.flatten()
    pvals = np.array(bt_pred).flatten()

    # Context: 30 points from full series before the backtest window
    ctx_end = full_series.index.get_indexer([bt_actual.index[0]], method="nearest")[0]
    ctx_start = max(0, ctx_end - 30)
    ctx = full_series.iloc[ctx_start:ctx_end]

    fig = go.Figure()

    if len(ctx):
        fig.add_trace(go.Scatter(
            x=ctx.index, y=ctx.values, mode="lines", name="Context",
            line=dict(color=ctx_color, width=1.6, dash="dash"),
            opacity=0.7,
            hovertemplate="%{x|%Y-%m-%d}  %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[ctx.index[-1], bt_actual.index[0]], y=[ctx.values[-1], tvals[0]],
            mode="lines", showlegend=False, opacity=0.2,
            line=dict(color=ctx_color, width=1, dash="dash"), hoverinfo="skip"))

    fig.add_trace(go.Scatter(
        x=bt_actual.index, y=tvals, mode="lines+markers+text", name="Actual",
        line=dict(color=act_color, width=2.2),
        marker=dict(symbol="circle", size=8, color=act_color,
                    line=dict(color=mrk_border, width=1.5)),
        text=[f"{v:.2f}" for v in tvals], textposition="bottom center",
        textfont=dict(size=10, color=act_color),
        hovertemplate="%{x|%Y-%m-%d}  Actual %{y:.2f}<extra></extra>"))

    fig.add_trace(go.Scatter(
        x=bt_actual.index, y=pvals, mode="lines+markers+text", name="Backtest Forecast",
        line=dict(color=fore_color, width=2.2, dash="dash"),
        marker=dict(symbol="triangle-up", size=9, color=fore_color,
                    line=dict(color=mrk_border, width=1.5)),
        text=[f"{v:.2f}" for v in pvals], textposition="top center",
        textfont=dict(size=10, color=fore_color),
        hovertemplate="%{x|%Y-%m-%d}  Backtest %{y:.2f}<extra></extra>"))

    fig.update_layout(
        title=dict(text=title, font=dict(size=11, color=txt_color), x=0, y=0.97),
        height=380,
        hovermode="x unified",
        margin=dict(l=12, r=12, t=60, b=48),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=txt_color, family="Inter, sans-serif"),
        legend=dict(orientation="h", y=1.10, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color=txt_color)),
        xaxis=dict(
            type="date",
            tickangle=-40, tickfont=dict(size=10, color=txt_color),
            gridcolor=grid_color, linecolor=grid_color,
            tickformatstops=[
                dict(dtickrange=[None, 1000*60*60*24*365*2], value="%b %Y"),
                dict(dtickrange=[1000*60*60*24*365*2, None], value="%Y"),
            ],
        ),
        yaxis=dict(
            title=dict(text=unit, font=dict(size=11, color=txt_color)),
            tickfont=dict(size=10, color=txt_color),
            gridcolor=grid_color, linecolor=grid_color,
        ),
    )
    return fig


def make_backtest_chart(full_series: pd.Series, bt_actual: pd.Series,
                        bt_pred: np.ndarray, unit: str, title: str) -> go.Figure:
    dark = st.session_state.get("theme", "dark") == "dark"
    ctx_color  = "#4a6fa8" if dark else "#7090b8"
    act_color  = "#3d7fcc" if dark else "#1d4ed8"
    fore_color = "#c0392b" if dark else "#b91c1c"
    txt_color  = "#d8e1ee" if dark else "#1e293b"
    bg_color   = "#0e1520" if dark else "#ffffff"
    grid_color = "#1a2438" if dark else "#e2e8f0"
    mrk_border = "#0e1520" if dark else "#ffffff"

    tvals = bt_actual.values.flatten()
    pvals = np.array(bt_pred).flatten()

    # 30 points of context before the backtest window
    try:
        ctx_end = full_series.index.get_indexer([bt_actual.index[0]], method="nearest")[0]
    except Exception:
        ctx_end = 30
    ctx = full_series.iloc[max(0, ctx_end - 30): ctx_end]

    fig = go.Figure()
    if len(ctx):
        fig.add_trace(go.Scatter(
            x=ctx.index, y=ctx.values, mode="lines", name="Context",
            line=dict(color=ctx_color, width=1.6, dash="dash"), opacity=0.7,
            hovertemplate="%{x|%Y-%m-%d}  %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[ctx.index[-1], bt_actual.index[0]], y=[ctx.values[-1], tvals[0]],
            mode="lines", showlegend=False, opacity=0.2,
            line=dict(color=ctx_color, width=1, dash="dash"), hoverinfo="skip"))

    fig.add_trace(go.Scatter(
        x=bt_actual.index, y=tvals, mode="lines+markers+text", name="Actual",
        line=dict(color=act_color, width=2.2),
        marker=dict(symbol="circle", size=8, color=act_color,
                    line=dict(color=mrk_border, width=1.5)),
        text=[f"{v:.2f}" for v in tvals], textposition="bottom center",
        textfont=dict(size=10, color=act_color),
        hovertemplate="%{x|%Y-%m-%d}  Actual %{y:.2f}<extra></extra>"))

    fig.add_trace(go.Scatter(
        x=bt_actual.index, y=pvals, mode="lines+markers+text", name="Backtest Pred",
        line=dict(color=fore_color, width=2.2, dash="dash"),
        marker=dict(symbol="triangle-up", size=9, color=fore_color,
                    line=dict(color=mrk_border, width=1.5)),
        text=[f"{v:.2f}" for v in pvals], textposition="top center",
        textfont=dict(size=10, color=fore_color),
        hovertemplate="%{x|%Y-%m-%d}  Backtest %{y:.2f}<extra></extra>"))

    fig.update_layout(
        title=dict(text=title, font=dict(size=11, color=txt_color), x=0, y=0.97),
        height=380, hovermode="x unified",
        margin=dict(l=12, r=12, t=60, b=48),
        paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        font=dict(color=txt_color, family="Inter, sans-serif"),
        legend=dict(orientation="h", y=1.10, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color=txt_color)),
        xaxis=dict(type="date", tickangle=-40,
                   tickfont=dict(size=10, color=txt_color),
                   gridcolor=grid_color, linecolor=grid_color,
                   tickformatstops=[
                       dict(dtickrange=[None, 1000*60*60*24*365*2], value="%b %Y"),
                       dict(dtickrange=[1000*60*60*24*365*2, None], value="%Y"),
                   ]),
        yaxis=dict(title=dict(text=unit, font=dict(size=11, color=txt_color)),
                   tickfont=dict(size=10, color=txt_color),
                   gridcolor=grid_color, linecolor=grid_color),
    )
    return fig


def show_metrics(actual, pred):
    a, p = np.array(actual, float), np.array(pred, float)
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mae  = float(np.mean(np.abs(a - p)))
    mape = float(np.mean(np.abs((a - p) / (np.abs(a) + 1e-8))) * 100)
    r2   = float(1 - np.sum((a - p) ** 2) / (np.sum((a - a.mean()) ** 2) + 1e-8))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE",  f"{rmse:.3f}")
    c2.metric("MAE",   f"{mae:.3f}")
    c3.metric("MAPE%", f"{mape:.2f}")
    c4.metric("R²",    f"{r2:.3f}")


# ── App ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Energy Price Forecasting", layout="wide")

# ── Theme state ───────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def _css(t):
    dark = t == "dark"
    bg      = "#080d15" if dark else "#f4f6f9"
    s1      = "#0e1520" if dark else "#ffffff"
    s2      = "#141d2b" if dark else "#f0f3f8"
    s3      = "#1a2438" if dark else "#e4e9f2"
    border  = "#1e2d42" if dark else "#d0d8e8"
    borderhi= "#2a4060" if dark else "#a8b8d0"
    blue    = "#4d8fd1" if dark else "#2563eb"
    bluelt  = "#6aaad9" if dark else "#3b82f6"
    bluedk  = "#2b5f9e" if dark else "#1d4ed8"
    violet  = "#5a6ea8" if dark else "#6d28d9"
    mauve   = "#7a5c9a" if dark else "#9333ea"
    teal    = "#3a9080" if dark else "#0d9488"
    text    = "#d8e1ee" if dark else "#1e293b"
    sub     = "#8898ae" if dark else "#475569"
    muted   = "#445568" if dark else "#94a3b8"
    dim     = "#2a3a50" if dark else "#cbd5e1"

    # Metric value colors
    m1 = "#5b9fd4" if dark else "#2563eb"
    m2 = "#7a9ecf" if dark else "#4f46e5"
    m3 = "#8b7dc0" if dark else "#7c3aed"
    m4 = "#4a9d8a" if dark else "#0d9488"

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
    --bg:       {bg};
    --s1:       {s1};
    --s2:       {s2};
    --s3:       {s3};
    --border:   {border};
    --bhi:      {borderhi};
    --blue:     {blue};
    --blue-lt:  {bluelt};
    --blue-dk:  {bluedk};
    --violet:   {violet};
    --mauve:    {mauve};
    --teal:     {teal};
    --text:     {text};
    --sub:      {sub};
    --muted:    {muted};
    --dim:      {dim};
}}

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
}}
.block-container {{ padding: 1rem 2.5rem 3rem; max-width: 100%; }}
[data-testid="stAppViewContainer"] {{ background: var(--bg) !important; }}
[data-testid="stSidebar"] {{ background: var(--s1) !important; }}
section[data-testid="stSidebar"] {{ background: var(--s1) !important; }}

@keyframes slideDown {{
    from {{ opacity:0; transform:translateY(-12px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
@keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(8px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
@keyframes barPulse {{
    0%,100% {{ opacity:0.6; }}
    50%      {{ opacity:1; }}
}}
@keyframes borderBreath {{
    0%,100% {{ border-color: var(--border); }}
    50%      {{ border-color: var(--bhi); }}
}}

/* Header */
.hdr {{
    background: var(--s1);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.8rem 2.2rem 1.6rem;
    margin: 0.6rem 0 1.8rem;
    position: relative;
    overflow: hidden;
    animation: slideDown 0.5s cubic-bezier(0.22,1,0.36,1) both;
}}
.hdr::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--blue-dk), var(--blue), var(--violet), var(--mauve));
    animation: barPulse 4s ease-in-out infinite;
}}
.hdr-inner {{ display: flex; align-items: center; justify-content: space-between; gap: 1rem; }}
.hdr h1 {{
    font-size: 1.38rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.5rem;
    line-height: 1.4;
    letter-spacing: -0.15px;
}}
.hdr p {{
    font-size: 0.7rem;
    color: var(--muted);
    margin: 0;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    font-weight: 500;
}}

/* Panel label */
.plabel {{
    display: block;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--blue);
    margin-bottom: 0.8rem;
}}

/* Buttons */
.stButton > button {{
    background: var(--s2) !important;
    color: var(--sub) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 0.65rem 1.1rem !important;
    width: 100% !important;
    text-align: left !important;
    transition: background 0.18s ease,
                border-color 0.18s ease,
                color 0.18s ease,
                box-shadow 0.18s ease,
                transform 0.15s ease !important;
}}
.stButton > button:hover {{
    background: var(--s3) !important;
    border-color: var(--blue-dk) !important;
    color: var(--blue-lt) !important;
    box-shadow: 0 4px 18px rgba(77,143,209,0.14) !important;
    transform: translateY(-1px) !important;
}}
.stButton > button:active {{ transform: translateY(0) !important; box-shadow: none !important; }}

/* Badge */
.badge {{
    background: var(--s2);
    border: 1px solid var(--blue-dk);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-top: 0.7rem;
    animation: fadeUp 0.3s ease, borderBreath 3.5s ease-in-out infinite;
}}
.badge-label {{
    display: block;
    font-size: 0.55rem;
    font-weight: 700;
    letter-spacing: 2.2px;
    text-transform: uppercase;
    color: var(--blue);
    margin-bottom: 0.2rem;
}}
.badge-value {{
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text);
}}

/* Legend */
.legend {{ font-size: 0.77rem; color: var(--muted); line-height: 2.3; }}
.dot-blue   {{ display:inline-block;width:9px;height:9px;border-radius:50%;
               background:var(--blue);margin-right:8px;vertical-align:middle; }}
.dot-pink   {{ display:inline-block;width:9px;height:9px;
               clip-path:polygon(50% 0%,0% 100%,100% 100%);
               background:var(--mauve);margin-right:8px;vertical-align:middle; }}
.dot-dashed {{ display:inline-block;width:16px;height:0;
               border-top:2px dashed var(--dim);margin-right:8px;vertical-align:middle; }}
.legend-note {{ font-size:0.63rem;color:var(--dim);margin-top:0.3rem; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 1px solid var(--border);
    gap: 0;
    margin-bottom: 1rem;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border-radius: 0 !important;
    color: var(--muted) !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.7rem 1.6rem !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.18s ease !important;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--sub) !important; }}
.stTabs [aria-selected="true"] {{
    color: var(--blue-lt) !important;
    border-bottom: 2px solid var(--blue) !important;
}}

/* Selectbox */
.stSelectbox label {{
    font-size: 0.82rem !important;
    color: var(--sub) !important;
    font-weight: 500 !important;
}}
.stSelectbox > div > div {{
    background: var(--s2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: 0.88rem !important;
    transition: border-color 0.18s !important;
}}
.stSelectbox > div > div:focus-within {{
    border-color: var(--blue-dk) !important;
    box-shadow: 0 0 0 3px rgba(45,95,158,0.15) !important;
}}

/* Metrics */
[data-testid="metric-container"] {{
    background: var(--s1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem 1rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    animation: fadeUp 0.4s ease both;
}}
[data-testid="metric-container"]::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
    background: var(--border);
    transition: background 0.2s ease;
}}
[data-testid="metric-container"]:hover {{
    transform: translateY(-2px);
    border-color: var(--bhi);
    box-shadow: 0 8px 28px rgba(0,0,0,0.2);
}}
[data-testid="metric-container"]:hover::after {{ background: var(--blue); }}
[data-testid="metric-container"] label {{
    font-size: 0.59rem !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--blue-lt) !important;
}}
[data-testid="column"]:nth-child(1) [data-testid="stMetricValue"] {{ color:{m1} !important; }}
[data-testid="column"]:nth-child(2) [data-testid="stMetricValue"] {{ color:{m2} !important; }}
[data-testid="column"]:nth-child(3) [data-testid="stMetricValue"] {{ color:{m3} !important; }}
[data-testid="column"]:nth-child(4) [data-testid="stMetricValue"] {{ color:{m4} !important; }}

/* Slider */
.stSlider label {{ font-size:0.82rem !important; color:var(--sub) !important; font-weight:500 !important; }}
.stSlider > div > div > div {{ background: var(--s3) !important; }}
.stSlider > div > div > div > div {{ background: var(--blue) !important; }}

/* Toggle */
.stToggle > label {{ color:var(--sub) !important; font-size:0.86rem !important; font-weight:500 !important; }}

/* Expander */
.streamlit-expanderHeader {{
    background: var(--s2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--sub) !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    transition: border-color 0.18s ease, color 0.18s ease !important;
}}
.streamlit-expanderHeader:hover {{
    border-color: var(--bhi) !important;
    color: var(--text) !important;
}}

/* Alerts */
.stSuccess, .stInfo, .stError {{ border-radius: 8px !important; font-size: 0.86rem !important; }}

/* Spinner */
.stSpinner > div {{ border-top-color: var(--blue) !important; }}

/* Divider */
hr {{ border: none !important; border-top: 1px solid var(--border) !important; margin: 1.2rem 0 !important; }}

/* Dataframe */
.stDataFrame {{ border: 1px solid var(--border) !important; border-radius: 8px !important; }}

/* Caption */
.stCaption {{ color: var(--muted) !important; font-size: 0.78rem !important; }}
</style>
"""

t = st.session_state.theme
st.markdown(_css(t), unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([6, 1])
with hcol1:
    st.markdown("""
    <div class="hdr">
        <h1>Forecasting of Energy Prices Using Economic Indicators and GDELT Derived News Signals</h1>
        <p>Univariate &nbsp;·&nbsp; Multivariate &nbsp;·&nbsp; Macro Indicators &nbsp;·&nbsp; News Sentiment &nbsp;·&nbsp; Deep Learning &amp; Statistical Models</p>
    </div>
    """, unsafe_allow_html=True)
with hcol2:
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    lbl = "Light Mode" if t == "dark" else "Dark Mode"
    if st.button(lbl, key="theme_toggle", use_container_width=True):
        st.session_state.theme = "light" if t == "dark" else "dark"
        st.rerun()

if "energy" not in st.session_state:
    st.session_state.energy = None

left, right = st.columns([1, 3.2], gap="large")

with left:
    st.markdown("<span class='plabel'>Energy Type</span>", unsafe_allow_html=True)

    if st.button("Crude Oil  —  WTI", use_container_width=True, key="btn_oil"):
        st.session_state.energy = "crude_oil"
    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
    if st.button("CNG  —  Natural Gas", use_container_width=True, key="btn_cng"):
        st.session_state.energy = "cng"

    if st.session_state.energy == "crude_oil":
        st.markdown("""<div class='badge'>
            <span class='badge-label'>Selected</span>
            <span class='badge-value'>Crude Oil — WTI</span>
        </div>""", unsafe_allow_html=True)
    elif st.session_state.energy == "cng":
        st.markdown("""<div class='badge'>
            <span class='badge-label'>Selected</span>
            <span class='badge-value'>CNG — Natural Gas</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""<div class='legend'>
        <div><span class='dot-dashed'></span>Context window</div>
        <div><span class='dot dot-blue'></span>Actual values</div>
        <div><span class='dot dot-pink'></span>Forecast values</div>
        <div class='legend-note'>Test window — last 5 periods</div>
    </div>""", unsafe_allow_html=True)

with right:
    if st.session_state.energy is None:
        st.markdown("""
        <div style='
            background: #0e1520;
            border: 1px solid #1e2d42;
            border-radius: 12px;
            padding: 4.5rem 2rem;
            text-align: center;
            margin-top: 0.5rem;
            animation: fadeUp 0.5s ease;
        '>
            <div style='font-size:1.05rem;font-weight:600;color:#d8e1ee;margin-bottom:0.5rem;'>
                Select an energy type to begin
            </div>
            <div style='font-size:0.82rem;color:#445568;'>
                Choose Crude Oil or CNG from the left panel
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        ekey = st.session_state.energy
        unit = ENERGY[ekey]["unit"]
        tab_uni, tab_multi = st.tabs(["Univariate", "Multivariate"])

        with tab_uni:
            model_name = st.selectbox("Select Model", ["ARIMA", "SARIMA", "LSTM"],
                                      key="uni_model")
            # Forecast + Backtest buttons (side by side for LSTM)
            if model_name == "LSTM":
                btn_c1, btn_c2 = st.columns(2)
                run    = btn_c1.button("Forecast", key="uni_run",  use_container_width=True)
                run_bt = btn_c2.button("Backtest", key="uni_bt",   use_container_width=True)
            else:
                run    = st.button("Forecast", key="uni_run")
                run_bt = False

            # Run forecast and store in session state
            if run:
                if model_name in ("ARIMA", "SARIMA"):
                    with st.spinner(f"Loading {model_name} & forecasting…"):
                        train, test, pred, label, err = forecast_stat(ekey, model_name)
                    if err:
                        st.session_state.uni_result = {"err": err}
                    else:
                        st.session_state.uni_result = {
                            "kind": "stat", "train": train, "test": test,
                            "pred": pred, "label": label,
                        }
                else:
                    with st.spinner("Loading LSTM & forecasting…"):
                        train, test, pred, label, err, input_seqs, input_dates, iw, bt_actual, bt_pred = forecast_lstm(ekey)
                    if err:
                        st.session_state.uni_result = {"err": err}
                    else:
                        st.session_state.uni_result = {
                            "kind": "lstm", "train": train, "test": test,
                            "pred": pred, "label": label,
                            "input_seqs": input_seqs, "input_dates": input_dates, "iw": iw,
                            "bt_actual": bt_actual, "bt_pred": bt_pred,
                            "show_bt": False,
                        }

            if run_bt:
                res = st.session_state.get("uni_result", {})
                if res.get("kind") == "lstm":
                    res["show_bt"] = True
                    st.session_state.uni_result = res
                else:
                    st.info("Run Forecast first, then click Backtest.")

            # Render from session state
            uni_res = st.session_state.get("uni_result")
            if uni_res:
                if "err" in uni_res:
                    st.error(uni_res["err"])
                elif uni_res["kind"] == "stat":
                    train, test, pred, label = uni_res["train"], uni_res["test"], uni_res["pred"], uni_res["label"]
                    st.plotly_chart(make_chart(train, test, pred, unit, label),
                                    use_container_width=True)
                    show_metrics(test.values, pred)
                    with st.expander("Actual vs Forecast"):
                        st.dataframe(pd.DataFrame({
                            "Date":     [d.strftime("%Y-%m-%d") for d in test.index],
                            "Actual":   test.values.round(4),
                            "Forecast": pred.round(4),
                            "Error":    (test.values - pred).round(4),
                        }), use_container_width=True, hide_index=True)
                elif uni_res["kind"] == "lstm":
                    train     = uni_res["train"]
                    test      = uni_res["test"]
                    pred      = uni_res["pred"]
                    label     = uni_res["label"]
                    input_seqs  = uni_res["input_seqs"]
                    input_dates = uni_res["input_dates"]
                    iw        = uni_res["iw"]
                    bt_actual = uni_res["bt_actual"]
                    bt_pred   = uni_res["bt_pred"]

                    if uni_res.get("show_bt"):
                        st.markdown("**Backtest**")
                        st.plotly_chart(
                            make_backtest_chart(train, bt_actual, bt_pred, unit,
                                               f"Backtest — {label}"),
                            use_container_width=True)
                        show_metrics(bt_actual.values, bt_pred)
                        st.markdown("---")

                    st.plotly_chart(make_chart(train, test, pred, unit, label),
                                    use_container_width=True)
                    show_metrics(test.values, pred)

                    st.markdown("#### Input Sequences & Forecast Output")
                    for i in range(5):
                        with st.expander(
                            f"Window {i+1} -> Forecast: **{pred[i]:.4f}** {unit}  |  "
                            f"Actual: **{test.values[i]:.4f}** {unit}  |  "
                            f"Target date: {test.index[i].strftime('%Y-%m-%d')}"
                        ):
                            seq_df = pd.DataFrame({
                                "Date":              [d.strftime("%Y-%m-%d") for d in input_dates[i]],
                                f"Input ({unit})":   input_seqs[i].round(4),
                            })
                            st.dataframe(seq_df, use_container_width=True, hide_index=True)

        with tab_multi:
            if "feat_set" not in st.session_state:
                st.session_state.feat_set = None

            st.markdown("<div class='panel-label'>Feature Set</div>", unsafe_allow_html=True)
            fc1, fc2, fc3 = st.columns(3)
            if fc1.button(" Macro",         use_container_width=True, key="btn_macro"):
                st.session_state.feat_set = "Macro"
            if fc2.button(" News Only",      use_container_width=True, key="btn_news"):
                st.session_state.feat_set = "News Only"
            if fc3.button(" News + Macro",   use_container_width=True, key="btn_newsmacro"):
                st.session_state.feat_set = "News + Macro"

            if st.session_state.feat_set:
                st.markdown(f"<div class='selected-badge'><span>Feature Set</span>"
                            f"{st.session_state.feat_set}</div>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)

            MULTI_MODELS = {
                "Macro":        ["LSTM", "CNN-LSTM", "SARIMAX"],
                "News Only":    ["CNN-LSTM"],
                "News + Macro": ["LSTM", "CNN-LSTM", "SARIMAX"],
            }

            if st.session_state.feat_set:
                feat_set = st.session_state.feat_set
                multi_model = st.selectbox(
                    "Select Model", MULTI_MODELS[feat_set], key="multi_model")
                spec_key = (feat_set, ekey, multi_model)

                if multi_model in ("LSTM", "CNN-LSTM"):
                    mb1, mb2 = st.columns(2)
                    run_multi  = mb1.button("Forecast", key="multi_run",  use_container_width=True)
                    run_multi_bt = mb2.button("Backtest", key="multi_bt", use_container_width=True)
                else:
                    run_multi    = st.button("Forecast", key="multi_run")
                    run_multi_bt = False

                if run_multi:
                    if spec_key in MULTI_KERAS_SPEC:
                        with st.spinner(f"Loading {multi_model} · {feat_set}…"):
                            result = forecast_multi_keras(feat_set, ekey, multi_model, tweaks={})
                        train, test, pred, label, err, tweak_defs, scaler, cols, x_cols, test_X_orig, bt_actual, bt_pred = result
                        if err:
                            st.session_state.multi_result = {"err": err}
                        else:
                            st.session_state.multi_result = {
                                "kind": "keras", "train": train, "test": test,
                                "pred": pred, "label": label,
                                "tweak_defs": tweak_defs, "scaler": scaler,
                                "cols": cols, "x_cols": x_cols,
                                "spec_key": spec_key,
                                "bt_actual": bt_actual, "bt_pred": bt_pred,
                                "show_bt": False,
                            }
                    elif multi_model == "SARIMAX":
                        with st.spinner(f"Loading SARIMAX · {feat_set}…"):
                            train, test, pred, label, err = forecast_multi_sarimax(feat_set, ekey)
                        if err:
                            st.session_state.multi_result = {"err": err}
                        else:
                            st.session_state.multi_result = {
                                "kind": "sarimax", "train": train, "test": test,
                                "pred": pred, "label": label,
                            }
                    else:
                        st.session_state.multi_result = {"kind": "none"}

                if run_multi_bt:
                    res = st.session_state.get("multi_result", {})
                    if res.get("kind") == "keras":
                        res["show_bt"] = True
                        st.session_state.multi_result = res
                    else:
                        st.info("Run Forecast first, then click Backtest.")

                # ── Render from session state ──────────────────────────────────────
                res = st.session_state.get("multi_result")
                if res:
                    if "err" in res:
                        st.error(res["err"])

                    elif res.get("kind") == "keras":
                        train, test, pred, label = res["train"], res["test"], res["pred"], res["label"]
                        spec      = MULTI_KERAS_SPEC.get(res["spec_key"], {})
                        tweak_defs = res["tweak_defs"]
                        scaler    = res["scaler"]
                        cols      = res["cols"]
                        x_cols    = res["x_cols"]

                        # Tweak cols present for this model
                        present_tweak = [(col, cfg) for col, cfg in TWEAK_COLS.items()
                                         if col not in spec.get("drop", [])
                                         and not (col in ("GoldsteinScale","AvgTone")
                                                  and "news_data.csv" in spec.get("exclude", []))]

                        tweaks_ui = {}
                        label_show = label
                        if present_tweak:
                            enable_tweak = st.toggle("Enable Feature Tweaking", value=False,
                                                     key="tweak_toggle")
                            if enable_tweak:
                                st.caption("Default = actual value from first input window · applied to all windows")
                                grid = [present_tweak[i:i+3] for i in range(0, len(present_tweak), 3)]
                                for row in grid:
                                    rcols = st.columns(len(row))
                                    for rc, (col, cfg) in zip(rcols, row):
                                        default = float(round(tweak_defs.get(col, cfg["default"]), 4))
                                        tweaks_ui[col] = rc.slider(
                                            cfg["label"], float(cfg["min"]), float(cfg["max"]),
                                            default, key=f"tweak_{col}")

                                if tweaks_ui:
                                    with st.spinner("Applying tweaks…"):
                                        r2 = forecast_multi_keras(
                                            feat_set, ekey, multi_model, tweaks=tweaks_ui)
                                    _, test, pred, _, err2, _, _, _, _, _, _, _ = r2
                                    if err2:
                                        st.error(err2)
                                    label_show = label + "  *(tweaked)*"

                        st.plotly_chart(make_chart(train, test, pred, unit, label_show),
                                        use_container_width=True)
                        show_metrics(test.values, pred)

                        if res.get("show_bt"):
                            bt_actual = res.get("bt_actual")
                            bt_pred_v = res.get("bt_pred")
                            if bt_actual is not None and bt_pred_v is not None:
                                st.markdown("**Backtest**")
                                st.plotly_chart(
                                    make_backtest_chart(train, bt_actual, bt_pred_v, unit,
                                                       f"Backtest — {label}"),
                                    use_container_width=True)
                                show_metrics(bt_actual.values, bt_pred_v)
                                st.markdown("---")

                        with st.expander("Actual vs Forecast"):
                            st.dataframe(pd.DataFrame({
                                "Date":     [d.strftime("%Y-%m-%d") for d in test.index],
                                "Actual":   test.values.round(4),
                                "Forecast": pred.round(4),
                                "Error":    (test.values - pred).round(4),
                            }), use_container_width=True, hide_index=True)

                    elif res.get("kind") == "sarimax":
                        train, test, pred, label = res["train"], res["test"], res["pred"], res["label"]
                        st.plotly_chart(make_chart(train, test, pred, unit, label),
                                        use_container_width=True)
                        show_metrics(test.values, pred)
                        with st.expander("Actual vs Forecast"):
                            st.dataframe(pd.DataFrame({
                                "Date":     [d.strftime("%Y-%m-%d") for d in test.index],
                                "Actual":   test.values.round(4),
                                "Forecast": pred.round(4),
                                "Error":    (test.values - pred).round(4),
                            }), use_container_width=True, hide_index=True)
            else:
                st.info("Select a feature set to see model options.")