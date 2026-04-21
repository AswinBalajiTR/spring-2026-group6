import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import os
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class EDA:
    def __init__(self, results_dir, target, input_window=None):
        self.results_dir  = results_dir
        self.target       = target
        self.input_window = input_window

    def save(self, name):
        plt.savefig(os.path.join(self.results_dir, f"{name}.svg"), format="svg", bbox_inches="tight")
        plt.close(); print(f"Saved → {name}.svg")

    def metrics(self, a, p, lbl=""):
        a, p = np.array(a).flatten(), np.array(p).flatten()
        mse = np.mean((a - p) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(a - p))
        mape = np.mean(np.abs((a - p) / np.where(np.abs(a) < 1e-8, 1e-8, a))) * 100
        r2 = 1 - np.sum((a - p) ** 2) / np.sum((a - np.mean(a)) ** 2)
        print(f"{lbl:8s}| MSE:{mse:.4f} RMSE:{rmse:.4f} MAE:{mae:.4f} MAPE:{mape:.4f}% R²:{r2:.4f}")
        return mse, rmse, mae, mape, r2

    def loss_plot(self, model):
        hist = model._history.history
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,7),sharex=True)
        ax1.plot(hist["loss"],    color="#1565C0", label="Train"); ax1.set_ylabel("Train Loss"); ax1.legend(); ax1.grid(True,ls="--",alpha=0.4)
        ax1.set_title("Train Loss")
        ax2.plot(hist["val_loss"],color="#C62828", label="Val");   ax2.set_ylabel("Test Loss");   ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True,ls="--",alpha=0.4)
        ax2.set_title("Test Loss")
        plt.tight_layout()
        self.save("train_test_loss")

    def _annotate(self, ax, dates, vals, color, y_offset):
        for i,(d,v) in enumerate(zip(dates,vals)):
            ax.annotate(f"{v:.2f}",(d,v),textcoords="offset points",
                        xytext=(0, y_offset + i*2), fontsize=8, color=color, ha="center")

    def forecast_plot(self, df, dates, act, pred):
        ip_end    = df.index.get_loc(dates[0])
        ip_start  = ip_end - self.input_window
        ctx_start = ip_start - 5
        ctx_dates, input_dates = df.index[ctx_start:ip_start], df.index[ip_start:ip_end]
        ctx_vals,  input_vals  = df[self.target].values[ctx_start:ip_start], df[self.target].values[ip_start:ip_end]

        fig, ax = plt.subplots(figsize=(14,5)); fig.suptitle(f"Forecast | {self.target}", fontsize=11, fontweight="bold")
        ax.plot(ctx_dates,   ctx_vals,   color="gray",    lw=1.5, ls="--", label="Context (5 pts)", alpha=0.6)
        ax.plot(input_dates, input_vals, color="#5C6BC0", lw=1.8, ls="--", label=f"Input ({self.input_window} steps)")
        ax.plot([input_dates[-1], dates[0]], [input_vals[-1], act[0]], color="#5C6BC0", lw=1, ls="--", alpha=0.3)
        ax.plot(dates, act,  color="#1565C0", marker="o", lw=2, label="Actual")
        ax.plot(dates, pred, color="#C62828", marker="^", lw=2, ls="--", label="Predicted")
        self._annotate(ax, dates, pred,  "#C62828",  12)
        self._annotate(ax, dates, act,   "#1565C0", -16)
        ax.legend(); ax.grid(True,ls="--",alpha=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax.tick_params(axis="x",rotation=30)
        plt.tight_layout()
        self.save("forecast_plot")

    def backtest_plot(self, df, dates, act, pred):
        ip_end    = df.index.get_loc(dates[0])
        ip_start  = ip_end - self.input_window
        after_end = df.index.get_loc(dates[-1]) + 6
        input_dates = df.index[ip_start:ip_end]
        after_dates = df.index[df.index.get_loc(dates[-1])+1 : after_end]
        input_vals  = df[self.target].values[ip_start:ip_end]
        after_vals  = df[self.target].values[df.index.get_loc(dates[-1])+1 : after_end]

        fig, ax = plt.subplots(figsize=(14,5)); fig.suptitle(f"Backtest | {self.target}", fontsize=11, fontweight="bold")
        ax.plot(input_dates, input_vals, color="#5C6BC0", lw=1.8, ls="--", label=f"Input ({self.input_window} steps)")
        ax.plot([input_dates[-1], dates[0]], [input_vals[-1], act[0]], color="#5C6BC0", lw=1, ls="--", alpha=0.3)
        ax.plot(dates, act,  color="#1565C0", marker="o", lw=2, label="Actual")
        ax.plot(dates, pred, color="#C62828", marker="^", lw=2, ls="--", label="Predicted")
        ax.plot([dates[-1], after_dates[0]], [act[-1], after_vals[0]], color="gray", lw=1, ls="--", alpha=0.3)
        ax.plot(after_dates, after_vals, color="gray", lw=1.5, ls="--", label="After (5 pts)", alpha=0.6)
        self._annotate(ax, dates, pred,  "#C62828",  12)
        self._annotate(ax, dates, act,   "#1565C0", -16)
        ax.legend(); ax.grid(True,ls="--",alpha=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax.tick_params(axis="x",rotation=30)
        plt.tight_layout()
        self.save("backtest_plot")

    def compute_all_metrics(self, train_act, train_pred, test_act, test_pred, bt_act, bt_pred):
        print("\n" + "=" * 55)
        tr = self.metrics(train_act, train_pred, "Train")
        te = self.metrics(test_act, test_pred, "Test")
        bt = self.metrics(bt_act, bt_pred, "Backtest")
        print("=" * 55)
        return tr, te, bt

    def metrics_table(self, tr, te, bt):
        rows = [["MSE", f"{tr[0]:.4f}", f"{te[0]:.4f}", f"{bt[0]:.4f}"],
                ["RMSE", f"{tr[1]:.4f}", f"{te[1]:.4f}", f"{bt[1]:.4f}"],
                ["MAE", f"{tr[2]:.4f}", f"{te[2]:.4f}", f"{bt[2]:.4f}"],
                ["MAPE%", f"{tr[3]:.4f}", f"{te[3]:.4f}", f"{bt[3]:.4f}"],
                ["R²", f"{tr[4]:.4f}", f"{te[4]:.4f}", f"{bt[4]:.4f}"]]
        fig, ax = plt.subplots(figsize=(7, 2.2));
        ax.axis("off")
        t = ax.table(cellText=rows, colLabels=["Metric", "Train", "Test", "Backtest"], cellLoc="center", loc="center")
        t.auto_set_font_size(False);
        t.set_fontsize(10);
        t.scale(1.2, 1.6)
        for c in range(4): t[0, c].set_facecolor("#1565C0"); t[0, c].set_text_props(color="white", fontweight="bold")
        for r in range(1, 6):
            for c in range(4): t[r, c].set_facecolor("#EEF2FF" if r % 2 else "#FFFFFF")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.save("metrics_table")

    def normalize(self, train_data, test_data, backtest_data, scaler_type="minmax"):
        if scaler_type == "robust":
            scaler = RobustScaler()
        else:
            scaler = MinMaxScaler()

        train_scaled = pd.DataFrame(scaler.fit_transform(train_data), index=train_data.index,
                                    columns=train_data.columns)
        test_scaled = pd.DataFrame(scaler.transform(test_data), index=test_data.index,
                                   columns=test_data.columns)
        backtest_scaled = pd.DataFrame(scaler.transform(backtest_data), index=backtest_data.index,
                                       columns=backtest_data.columns)
        self.scaler = scaler
        self.scaler_type = scaler_type
        return train_scaled, test_scaled, backtest_scaled

    def inverse(self, arr):
        idx = list(self.scaler.feature_names_in_).index(self.target)
        if self.scaler_type == "robust":
            center = self.scaler.center_[idx]
            scale = self.scaler.scale_[idx]
            return np.array(arr).flatten() * scale + center
        else:
            return np.array(arr).flatten() * self.scaler.data_range_[idx] + self.scaler.data_min_[idx]

    def check_multicollinearity(self, df, top_n=20, corr_threshold=0.3, vif_threshold=10):
        features = df.drop(columns=[self.target])
        target_series = df[self.target]

        # ── 1. Pearson Correlation with Target ───────────────────────
        corr = features.corrwith(target_series).sort_values(key=abs, ascending=False)
        top_corr = corr.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#1565C0" if v > 0 else "#C62828" for v in top_corr.values]
        top_corr.plot(kind="bar", ax=ax, color=colors)
        ax.axhline(corr_threshold, color="green", ls="--", lw=1.2, label=f"+{corr_threshold} threshold")
        ax.axhline(-corr_threshold, color="green", ls="--", lw=1.2)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"Pearson Correlation with {self.target} (Top {top_n})")
        ax.set_ylabel("Correlation")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.4)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self.save("pearson_correlation")

        # ── 2. Feature-Feature Correlation Heatmap ───────────────────
        top_features = top_corr.index.tolist()
        corr_matrix = df[top_features].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.zeros_like(corr_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title(f"Feature-Feature Correlation Heatmap (Top {top_n})")
        plt.tight_layout()
        self.save("feature_correlation_heatmap")

        # ── 3. VIF ───────────────────────────────────────────────────
        vif_data = df[top_features].dropna()
        X = add_constant(vif_data)
        vif_df = pd.DataFrame({
            "Feature": vif_data.columns,
            "VIF": [variance_inflation_factor(X.values, i + 1) for i in range(len(vif_data.columns))]
        }).sort_values("VIF", ascending=False).reset_index(drop=True)

        print("\nVIF Results:")
        print(vif_df.to_string(index=False))

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#C62828" if v > vif_threshold else "#1565C0" for v in vif_df["VIF"]]
        ax.barh(vif_df["Feature"], vif_df["VIF"], color=colors)
        ax.axvline(vif_threshold, color="red", ls="--", lw=1.5, label=f"VIF={vif_threshold} (high)")
        ax.axvline(5, color="orange", ls="--", lw=1.5, label="VIF=5 (moderate)")
        ax.set_title("Variance Inflation Factor (VIF)")
        ax.set_xlabel("VIF")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.4, axis="x")
        plt.tight_layout()
        self.save("vif")

        # ── 4. Summary ───────────────────────────────────────────────
        strong = corr[corr.abs() >= corr_threshold].index.tolist()
        low_vif = vif_df[vif_df["VIF"] <= vif_threshold]["Feature"].tolist()
        good = [f for f in strong if f in low_vif]

        print(f"\n{'=' * 55}")
        print(f"Strong correlation (|r|>={corr_threshold}):  {len(strong)} features")
        print(f"Low multicollinearity (VIF<={vif_threshold}): {len(low_vif)} features")
        print(f"Good features (both criteria):      {len(good)} features")
        print(f"Recommended features: {good}")
        print(f"{'=' * 55}")

        return corr, vif_df, good

    def plot_series(self, df, title=None):
        series = df[self.target] if isinstance(df, pd.DataFrame) else df
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(series.index, series.values, color="#1565C0", lw=1.5)
        ax.set_title(title or f"{self.target} — Time Series")
        ax.set_ylabel(self.target)
        ax.grid(True, ls="--", alpha=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        self.save("series_plot")
        return series

    def difference(self, df, periods=1):
        series = df[self.target] if isinstance(df, pd.DataFrame) else df
        diff = series.diff(periods).dropna()
        diff_df = diff.to_frame() if isinstance(diff, pd.Series) else diff
        return diff_df

    def stationarity_check(self, df):
        series = df[self.target] if isinstance(df, pd.DataFrame) else df
        series = series.dropna()

        # ── ADF Test ─────────────────────────────────────────────────
        adf_res = adfuller(series, autolag="AIC")
        adf_stat, adf_p = adf_res[0], adf_res[1]
        adf_result = "Stationary" if adf_p < 0.05 else "Non-Stationary"

        # ── KPSS Test ────────────────────────────────────────────────
        kpss_res = kpss(series, regression="c", nlags="auto")
        kpss_stat, kpss_p = kpss_res[0], kpss_res[1]
        kpss_result = "Stationary" if kpss_p > 0.05 else "Non-Stationary"

        # ── Combined Verdict ─────────────────────────────────────────
        if adf_result == "Stationary" and kpss_result == "Stationary":
            verdict = "✅ Stationary"
        elif adf_result == "Non-Stationary" and kpss_result == "Non-Stationary":
            verdict = "❌ Non-Stationary"
        elif adf_result == "Stationary" and kpss_result == "Non-Stationary":
            verdict = "⚠️ Trend-Stationary"
        else:
            verdict = "⚠️ Difference-Stationary"

        # ── Print Results ────────────────────────────────────────────
        print(f"\n{'=' * 55}")
        print(f"Stationarity Check: {self.target}")
        print(f"{'=' * 55}")
        print(f"ADF  Statistic: {adf_stat:.4f}  p-value: {adf_p:.4f}  → {adf_result}")
        print(f"KPSS Statistic: {kpss_stat:.4f}  p-value: {kpss_p:.4f}  → {kpss_result}")
        print(f"Verdict: {verdict}")
        print(f"{'=' * 55}")

        # ── Table Plot ───────────────────────────────────────────────
        rows = [
            ["ADF", f"{adf_stat:.4f}", f"{adf_p:.4f}", adf_result],
            ["KPSS", f"{kpss_stat:.4f}", f"{kpss_p:.4f}", kpss_result],
            ["Verdict", "", "", verdict],
        ]
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        t = ax.table(
            cellText=rows,
            colLabels=["Test", "Statistic", "p-value", "Result"],
            cellLoc="center", loc="center"
        )
        t.auto_set_font_size(False)
        t.set_fontsize(10)
        t.scale(1.2, 1.8)
        for c in range(4):
            t[0, c].set_facecolor("#1565C0")
            t[0, c].set_text_props(color="white", fontweight="bold")
        for r in range(1, 4):
            for c in range(4):
                t[r, c].set_facecolor("#EEF2FF" if r % 2 else "#FFFFFF")
        # color verdict row
        color = "#C8E6C9" if "✅" in verdict else "#FFCDD2" if "❌" in verdict else "#FFF9C4"
        for c in range(4):
            t[3, c].set_facecolor(color)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.save("stationarity_table")

        return {
            "adf_stat": adf_stat, "adf_p": adf_p, "adf_result": adf_result,
            "kpss_stat": kpss_stat, "kpss_p": kpss_p, "kpss_result": kpss_result,
            "verdict": verdict
        }

    def acf_pacf_plot(self, df, lags=40, title_suffix=""):
        series = df[self.target] if isinstance(df, pd.DataFrame) else df
        series = series.dropna()

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # ── Line Plot ────────────────────────────────────────────────
        axes[0].plot(series.index, series.values, color="#1565C0", lw=1.5)
        axes[0].set_title(f"{self.target} — Series {title_suffix}")
        axes[0].set_ylabel(self.target)
        axes[0].grid(True, ls="--", alpha=0.4)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[0].tick_params(axis="x", rotation=30)

        # ── ACF ──────────────────────────────────────────────────────
        plot_acf(series, lags=lags, ax=axes[1], color="#1565C0",
                 title=f"ACF — {self.target} {title_suffix}")
        axes[1].grid(True, ls="--", alpha=0.4)

        # ── PACF ─────────────────────────────────────────────────────
        plot_pacf(series, lags=lags, ax=axes[2], color="#C62828",
                  title=f"PACF — {self.target} {title_suffix}", method="ywm")
        axes[2].grid(True, ls="--", alpha=0.4)

        plt.tight_layout()
        name = f"acf_pacf{('_' + title_suffix.strip()) if title_suffix else ''}"
        self.save(name)

    def decompose_plot(self, df, model="additive", period=None):
        from statsmodels.tsa.seasonal import seasonal_decompose

        series = df[self.target] if isinstance(df, pd.DataFrame) else df
        series = series.dropna()

        result = seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f"{self.target} — Seasonal Decomposition ({model.capitalize()})", fontsize=12, fontweight="bold")

        axes[0].plot(result.observed, color="#1565C0", lw=1.5);
        axes[0].set_ylabel("Observed");
        axes[0].grid(True, ls="--", alpha=0.4)
        axes[1].plot(result.trend, color="#2E7D32", lw=1.5);
        axes[1].set_ylabel("Trend");
        axes[1].grid(True, ls="--", alpha=0.4)
        axes[2].plot(result.seasonal, color="#F57F17", lw=1.5);
        axes[2].set_ylabel("Seasonal");
        axes[2].grid(True, ls="--", alpha=0.4)
        axes[3].plot(result.resid, color="#C62828", lw=1.5);
        axes[3].set_ylabel("Residual");
        axes[3].grid(True, ls="--", alpha=0.4)
        axes[3].axhline(0, color="black", lw=0.8)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        self.save("decomposition")
        return result

    def forecast_plot_classic(self, train_series, test_act, preds_dict, title_suffix=""):
        """
        train_series : pd.Series — full training target values (unscaled)
        test_act     : array/Series — actual test values (5 points)
        preds_dict   : dict of {label: array} — e.g. {"ARIMA(1,1,1)": [...], "SARIMA": [...]}
        title_suffix : str — added to plot title
        """
        # show last 30 points of train as context
        context = train_series.iloc[-30:]
        test_dates = test_act.index if hasattr(test_act, "index") else pd.RangeIndex(len(test_act))
        test_vals = np.array(test_act).flatten()

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle(f"Forecast | {self.target} {title_suffix}", fontsize=11, fontweight="bold")

        # context
        ax.plot(context.index, context.values, color="#5C6BC0", lw=1.8,
                ls="--", label="Context (last 30)")

        # connector line from last train point to first actual test point
        ax.plot([context.index[-1], test_dates[0]], [context.values[-1], test_vals[0]],
                color="#5C6BC0", lw=1, ls="--", alpha=0.3)

        # actual test
        ax.plot(test_dates, test_vals, color="#1565C0", marker="o", lw=2, label="Actual")
        self._annotate(ax, test_dates, test_vals, "#1565C0", -16)

        # predictions for each model/order
        colors = ["#C62828", "#2E7D32", "#F57F17", "#6A1B9A", "#00838F"]
        markers = ["^", "s", "D", "P", "*"]
        for i, (label, pred) in enumerate(preds_dict.items()):
            pred = np.array(pred).flatten()
            c = colors[i % len(colors)]
            m = markers[i % len(markers)]
            ax.plot(test_dates, pred, color=c, marker=m, lw=2, ls="--", label=label)
            self._annotate(ax, test_dates, pred, c, 12 + i * 10)

        ax.legend()
        ax.grid(True, ls="--", alpha=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        name = f"forecast_plot{('_' + title_suffix.strip()) if title_suffix else ''}"
        self.save(name)

    def suggest_arima_orders(self, series, max_lags=40, significance=0.05):
        from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
        import warnings
        warnings.filterwarnings("ignore")

        series = series.dropna().copy()

        # ── Step 1: Estimate d ───────────────────────────────────────
        d = 0
        temp = series.copy()
        for _ in range(3):
            adf_p = adfuller(temp, autolag="AIC")[1]
            kpss_p = kpss(temp, regression="c", nlags="auto")[1]
            if adf_p < significance and kpss_p > significance:
                break
            temp = temp.diff().dropna()
            d += 1

        # ── Step 2: ACF → q ──────────────────────────────────────────
        diff_series = series.copy()
        for _ in range(d):
            diff_series = diff_series.diff().dropna()

        acf_vals, acf_confint = acf(diff_series, nlags=max_lags, alpha=significance)
        q = 0
        for i in range(1, len(acf_vals)):
            lower = acf_confint[i][0] - acf_vals[i]
            upper = acf_confint[i][1] - acf_vals[i]
            if not (lower <= 0 <= upper):
                q = i
            else:
                break

        # ── Step 3: PACF → p ─────────────────────────────────────────
        pacf_vals, pacf_confint = pacf(
            diff_series,
            nlags=min(max_lags, len(diff_series) // 2 - 1),
            alpha=significance, method="ywm"
        )
        p = 0
        for i in range(1, len(pacf_vals)):
            lower = pacf_confint[i][0] - pacf_vals[i]
            upper = pacf_confint[i][1] - pacf_vals[i]
            if not (lower <= 0 <= upper):
                p = i
            else:
                break

        # ── Step 4: ACF/PACF significance table ──────────────────────
        print(f"\n{'=' * 60}")
        print(f"{'Lag':>5} {'ACF':>10} {'ACF Sig':>10} {'PACF':>10} {'PACF Sig':>10}")
        print(f"{'=' * 60}")
        n_show = min(20, len(acf_vals) - 1, len(pacf_vals) - 1)
        for i in range(1, n_show + 1):
            acf_sig_flag = "✅" if not (
                        acf_confint[i][0] - acf_vals[i] <= 0 <= acf_confint[i][1] - acf_vals[i]) else "  "
            pacf_sig_flag = "✅" if not (
                        pacf_confint[i][0] - pacf_vals[i] <= 0 <= pacf_confint[i][1] - pacf_vals[i]) else "  "
            print(f"{i:>5} {acf_vals[i]:>10.4f} {acf_sig_flag:>10} {pacf_vals[i]:>10.4f} {pacf_sig_flag:>10}")
        print(f"{'=' * 60}")

        # ── Step 5: Suggest 3 orders ──────────────────────────────────
        suggested = [
            (p, d, q),  # base suggestion
            (max(0, p - 1), d, q),  # lower p
            (p, d, max(0, q - 1)),  # lower q
        ]
        # ensure no duplicates
        seen = []
        for o in suggested:
            if o not in seen:
                seen.append(o)
        # pad to 3 if duplicates removed
        extras = [(p + 1, d, q), (p, d, q + 1), (1, d, 1)]
        for o in extras:
            if o not in seen:
                seen.append(o)
            if len(seen) == 3:
                break
        suggested = seen[:3]

        print(f"\n── ARIMA Order Suggestion ──")
        print(f"  d (ADF/KPSS) : {d}")
        print(f"  p (PACF)     : {p}")
        print(f"  q (ACF)      : {q}")
        print(f"\n  Suggested orders:")
        for i, o in enumerate(suggested, 1):
            print(f"    Order {i}: ARIMA{o}")
        print(f"{'=' * 60}")

        return suggested

    def suggest_sarima_orders(self, series, s=12, max_lags=40, significance=0.05):
        from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
        import warnings
        warnings.filterwarnings("ignore")

        series = series.dropna().copy()

        # ── Step 1: Non-seasonal d ───────────────────────────────────
        d = 0
        temp = series.copy()
        for _ in range(3):
            adf_p = adfuller(temp, autolag="AIC")[1]
            kpss_p = kpss(temp, regression="c", nlags="auto")[1]
            if adf_p < significance and kpss_p > significance:
                break
            temp = temp.diff().dropna()
            d += 1

        # ── Step 2: Seasonal D ───────────────────────────────────────
        D = 0
        temp_s = series.copy()
        for _ in range(2):
            adf_p = adfuller(temp_s, autolag="AIC")[1]
            kpss_p = kpss(temp_s, regression="c", nlags="auto")[1]
            if adf_p < significance and kpss_p > significance:
                break
            temp_s = temp_s.diff(s).dropna()
            D += 1

        # ── Step 3: Apply both diffs for ACF/PACF ────────────────────
        diff_series = series.copy()
        for _ in range(d):
            diff_series = diff_series.diff().dropna()
        for _ in range(D):
            diff_series = diff_series.diff(s).dropna()

        print(f"\n── Estimated d={d}, D={D} (s={s}) ──")
        print(f"Series length after differencing: {len(diff_series)}")

        # ── Step 4: ACF → q, Q ───────────────────────────────────────
        acf_vals, acf_confint = acf(diff_series, nlags=max_lags, alpha=significance)

        q = 0
        for i in range(1, min(s, len(acf_vals))):
            lower = acf_confint[i][0] - acf_vals[i]
            upper = acf_confint[i][1] - acf_vals[i]
            if not (lower <= 0 <= upper):
                q = i
            else:
                break

        Q = 0
        if s < len(acf_vals):
            lower = acf_confint[s][0] - acf_vals[s]
            upper = acf_confint[s][1] - acf_vals[s]
            Q = 1 if not (lower <= 0 <= upper) else 0

        # ── Step 5: PACF → p, P ──────────────────────────────────────
        pacf_vals, pacf_confint = pacf(
            diff_series,
            nlags=min(max_lags, len(diff_series) // 2 - 1),
            alpha=significance, method="ywm"
        )

        p = 0
        for i in range(1, min(s, len(pacf_vals))):
            lower = pacf_confint[i][0] - pacf_vals[i]
            upper = pacf_confint[i][1] - pacf_vals[i]
            if not (lower <= 0 <= upper):
                p = i
            else:
                break

        P = 0
        if s < len(pacf_vals):
            lower = pacf_confint[s][0] - pacf_vals[s]
            upper = pacf_confint[s][1] - pacf_vals[s]
            P = 1 if not (lower <= 0 <= upper) else 0

        # ── Step 6: ACF/PACF significance table ──────────────────────
        print(f"\n{'=' * 65}")
        print(f"{'Lag':>5} {'ACF':>10} {'ACF Sig':>10} {'PACF':>10} {'PACF Sig':>10}")
        print(f"{'=' * 65}")
        n_show = min(max_lags, len(acf_vals) - 1, len(pacf_vals) - 1)
        for i in range(1, n_show + 1):
            acf_sig = "✅" if not (acf_confint[i][0] - acf_vals[i] <= 0 <= acf_confint[i][1] - acf_vals[i]) else "  "
            pacf_sig = "✅" if not (
                        pacf_confint[i][0] - pacf_vals[i] <= 0 <= pacf_confint[i][1] - pacf_vals[i]) else "  "
            # mark seasonal lags
            lag_str = f"{i}*" if i % s == 0 else str(i)
            print(f"{lag_str:>5} {acf_vals[i]:>10.4f} {acf_sig:>10} {pacf_vals[i]:>10.4f} {pacf_sig:>10}")
        print(f"{'=' * 65}")
        print(f"  * = seasonal lag (multiple of s={s})")

        # ── Step 7: Suggest 3 SARIMA orders ──────────────────────────
        base_order = (p, d, q)
        base_seasonal = (P, D, Q, s)

        suggested = [
            ((p, d, q), (P, D, Q, s)),
            ((max(0, p - 1), d, q), (P, D, Q, s)),
            ((p, d, max(0, q - 1)), (P, D, max(0, Q - 1), s)),
        ]

        # deduplicate
        seen = []
        for o in suggested:
            if o not in seen:
                seen.append(o)
        extras = [
            ((p + 1, d, q), (P, D, Q, s)),
            ((p, d, q + 1), (P, D, Q, s)),
            ((1, d, 1), (1, D, 1, s)),
        ]
        for o in extras:
            if o not in seen:
                seen.append(o)
            if len(seen) == 3:
                break
        suggested = seen[:3]

        print(f"\n── SARIMA Order Suggestion ──")
        print(f"  d  (non-seasonal diff) : {d}")
        print(f"  D  (seasonal diff s={s}): {D}")
        print(f"  p  (PACF non-seasonal) : {p}")
        print(f"  q  (ACF  non-seasonal) : {q}")
        print(f"  P  (PACF seasonal)     : {P}")
        print(f"  Q  (ACF  seasonal)     : {Q}")
        print(f"\n  Suggested orders:")
        for i, (o, so) in enumerate(suggested, 1):
            print(f"    Order {i}: SARIMA{o}x{so}")
        print(f"{'=' * 65}")

        orders = [o for o, so in suggested]
        seasonal_orders = [so for o, so in suggested]
        return orders, seasonal_orders