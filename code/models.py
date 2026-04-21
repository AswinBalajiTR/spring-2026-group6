import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


class Models:
    def __init__(self, input_window=None, output_window=None, n_features=None, n_targets=1):
        self.iw, self.ow, self.nf, self.nt = input_window, output_window, n_features, n_targets

    # ── CNN-LSTM ──────────────────────────────────────────────────
    def cnn_lstm(self, train_X, train_Y, filters=64, kernel_size=3, lstm_units=64,
                 dropout=0.2, l2_reg=0.001, lr=1e-3, batch_size=32,
                 epochs=100, val_split=0.1, patience=10):
        inp = Input(shape=(self.iw, self.nf))
        x = Conv1D(filters, kernel_size, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l2(l2_reg))(inp)
        x = MaxPooling1D(pool_size=2)(x)
        x = LSTM(lstm_units, dropout=dropout,
                 kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = Dense(32, activation="relu",
                  kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = Dropout(dropout)(x)
        out = Dense(self.ow * self.nt)(x)
        if self.ow * self.nt > 1:
            out = Reshape((self.ow, self.nt))(out)
        model = Model(inp, out)
        model.compile(optimizer=Adam(lr), loss="huber")
        split_idx = int(len(train_X) * (1 - val_split))
        X_tr, X_val = train_X[:split_idx], train_X[split_idx:]
        y_tr, y_val = train_Y[:split_idx], train_Y[split_idx:]
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
            ], verbose=1
        )
        model._history = history
        return model

    def tune_cnn_lstm(self, train_X, train_Y, n_trials=30, epochs=100, batch_size=32,
                      val_split=0.1, patience=10):
        def objective(trial):
            p = dict(filters=trial.suggest_categorical("filters", [32, 64]),
                     kernel_size=trial.suggest_categorical("kernel_size", [2, 3, 5]),
                     lstm_units=trial.suggest_categorical("lstm_units", [64, 32]),
                     dropout=trial.suggest_float("dropout", 0.1, 0.3),
                     l2_reg=trial.suggest_float("l2_reg", 1e-4, 1e-2, log=True),
                     lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True))
            model = self.cnn_lstm(train_X, train_Y, epochs=epochs, batch_size=batch_size,
                                  val_split=val_split, patience=patience, **p)
            val_loss = min(model._history.history["val_loss"])
            print(f"Trial {trial.number} | val_loss: {val_loss:.6f} | params: {p}")
            return val_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        print(f"\nBest Trial: {study.best_trial.number} | val_loss: {study.best_value:.6f}")
        print("Best Params:", study.best_params)
        return study.best_params

    # ── LSTM ──────────────────────────────────────────────────────
    def lstm(self, train_X, train_Y, lstm_units=64, dropout=0.2, l2_reg=0.001,
             lr=1e-3, batch_size=32, epochs=100, val_split=0.1, patience=10):
        inp = Input(shape=(self.iw, self.nf))
        x = LSTM(lstm_units, return_sequences=True, dropout=dropout,
                 kernel_regularizer=regularizers.l2(l2_reg))(inp)
        x = LSTM(lstm_units // 2, dropout=dropout,
                 kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = Dense(32, activation="relu",
                  kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = Dropout(dropout)(x)
        out = Dense(self.ow * self.nt)(x)
        if self.ow * self.nt > 1:
            out = Reshape((self.ow, self.nt))(out)
        model = Model(inp, out)
        model.compile(optimizer=Adam(lr), loss="huber")
        split_idx = int(len(train_X) * (1 - val_split))
        X_tr, X_val = train_X[:split_idx], train_X[split_idx:]
        y_tr, y_val = train_Y[:split_idx], train_Y[split_idx:]
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
            ], verbose=1
        )
        model._history = history
        return model

    def tune_lstm(self, train_X, train_Y, n_trials=30, epochs=100, batch_size=32,
                  val_split=0.1, patience=10):
        def objective(trial):
            p = dict(lstm_units=trial.suggest_categorical("lstm_units", [32, 64, 128]),
                     dropout=trial.suggest_float("dropout", 0.1, 0.4),
                     l2_reg=trial.suggest_float("l2_reg", 1e-4, 1e-2, log=True),
                     lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True))
            model = self.lstm(train_X, train_Y, epochs=epochs, batch_size=batch_size,
                              val_split=val_split, patience=patience, **p)
            val_loss = min(model._history.history["val_loss"])
            print(f"Trial {trial.number} | val_loss: {val_loss:.6f} | params: {p}")
            return val_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        print(f"\nBest Trial: {study.best_trial.number} | val_loss: {study.best_value:.6f}")
        print("Best Params:", study.best_params)
        return study.best_params

    # ── ARIMA ─────────────────────────────────────────────────────
    def arima(self, train_series, orders=[(1, 0, 1), (1, 1, 1), (2, 2, 1)]):
        results = {}
        rows = []
        for order in orders:
            try:
                fitted = ARIMA(train_series, order=order).fit()
                results[order] = fitted
                rows.append({
                    "order": str(order),
                    "AIC":   round(fitted.aic, 4),
                    "BIC":   round(fitted.bic, 4),
                    "HQIC":  round(fitted.hqic, 4),
                })
                print(f"ARIMA{order} | AIC: {fitted.aic:.4f} | BIC: {fitted.bic:.4f}")
            except Exception as e:
                print(f"ARIMA{order} | FAILED: {e}")

        results_df = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
        print(f"\nARIMA Results (sorted by AIC):")
        print(results_df.to_string(index=False))
        best_order = eval(results_df.iloc[0]["order"])
        print(f"\nBest ARIMA order: {best_order}")
        return results, results_df, best_order

    def predict_arima(self, results, steps=5):
        preds = {}
        for order, fitted in results.items():
            try:
                forecast = fitted.forecast(steps=steps)
                preds[order] = np.array(forecast)
                print(f"ARIMA{order} | forecast: {np.round(preds[order], 4)}")
            except Exception as e:
                print(f"ARIMA{order} | FAILED: {e}")
        return preds

    # ── SARIMA ────────────────────────────────────────────────────
    def sarima(self, train_series, orders=[(1, 0, 1), (1, 1, 1), (2, 2, 1)],
               seasonal_orders=[(1, 0, 1, 12), (1, 1, 1, 12), (0, 1, 1, 12)]):
        results = {}
        rows = []

        # pair each order with seasonal_order — reuse last if shorter
        paired = []
        for i, order in enumerate(orders):
            seas = seasonal_orders[i] if i < len(seasonal_orders) else seasonal_orders[-1]
            paired.append((order, seas))

        for order, seasonal_order in paired:
            key = (order, seasonal_order)
            try:
                fitted = SARIMAX(train_series,
                                 order=order,
                                 seasonal_order=seasonal_order,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False).fit(disp=False)
                results[key] = fitted
                rows.append({
                    "order":          str(order),
                    "seasonal_order": str(seasonal_order),
                    "AIC":            round(fitted.aic, 4),
                    "BIC":            round(fitted.bic, 4),
                    "HQIC":           round(fitted.hqic, 4),
                })
                print(f"SARIMA{order}x{seasonal_order} | AIC: {fitted.aic:.4f} | BIC: {fitted.bic:.4f}")
            except Exception as e:
                print(f"SARIMA{order}x{seasonal_order} | FAILED: {e}")

        results_df = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
        print(f"\nSARIMA Results (sorted by AIC):")
        print(results_df.to_string(index=False))
        best_key = (eval(results_df.iloc[0]["order"]), eval(results_df.iloc[0]["seasonal_order"]))
        print(f"\nBest SARIMA: order={best_key[0]} seasonal={best_key[1]}")
        return results, results_df, best_key

    def predict_sarima(self, results, steps=5):
        preds = {}
        for key, fitted in results.items():
            try:
                forecast = fitted.forecast(steps=steps)
                preds[key] = np.array(forecast)
                print(f"SARIMA{key[0]}x{key[1]} | forecast: {np.round(preds[key], 4)}")
            except Exception as e:
                print(f"SARIMA{key[0]}x{key[1]} | FAILED: {e}")
        return preds