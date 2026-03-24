import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_DIR     = os.path.join(PROJECT_ROOT, "Code")

sys.path.insert(0, CODE_DIR)

from eda import TimeSeriesEDA

DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "natural_gas.csv")
df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")

s = df["value"]

eda = TimeSeriesEDA()

s = eda.forward_fill(s)
eda.plot_series(s, name="CNG")
eda.stationarity_tests(s, name="CNG")

s_diff = eda.take_difference(s, periods=1)
eda.plot_series(s_diff, name="CNG_diff")
eda.stationarity_tests(s_diff, name="CNG_diff")
eda.plot_acf_pacf(s_diff, name="CNG_diff", lags=40)
eda.seasonality_test(s_diff, name="CNG_diff", period=12)
eda.decompose(s_diff, name="CNG_diff", period=12, model="additive")


# ── Train / Test Split (last 5 points as test) ────────────────────────────────
train = s.iloc[:-5]
test  = s.iloc[-5:]

print(f"Train: {train.index.min().date()} → {train.index.max().date()}  ({len(train)} obs)")
print(f"Test : {test.index.min().date()}  → {test.index.max().date()}  ({len(test)} obs)")


# ── Fit ARIMA models ──────────────────────────────────────────────────────────
orders   = [(1,1,1), (0,1,1), (1,1,0),(0,1,0)]
fitted_models = {}
forecasts     = {}

for order in orders:
    name = f"ARIMA{order}"
    print(f"Fitting {name}...")
    model            = ARIMA(train, order=order).fit()
    fitted_models[name] = model
    fc               = model.forecast(steps=len(test))
    fc.index         = test.index
    forecasts[name]  = fc


# ── Metrics ───────────────────────────────────────────────────────────────────
def get_metrics(actual, predicted):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

rows = []
for name, fc in forecasts.items():
    mae, rmse, mape = get_metrics(test.values, fc.values)
    rows.append({"Model": name, "MAE": f"{mae:.4f}", "RMSE": f"{rmse:.4f}", "MAPE (%)": f"{mape:.4f}"})

results_df = pd.DataFrame(rows)


# ── Ensemble (best 2 models by RMSE) ─────────────────────────────────────────
results_df["RMSE_float"] = results_df["RMSE"].astype(float)
best_two = results_df.nsmallest(2, "RMSE_float")["Model"].tolist()
print(f"\nEnsemble using: {best_two}")

ensemble_forecast        = pd.concat([forecasts[m] for m in best_two], axis=1).mean(axis=1)
ensemble_forecast.index  = test.index
forecasts["Ensemble"]    = ensemble_forecast

# add ensemble to results table
mae, rmse, mape = get_metrics(test.values, ensemble_forecast.values)
ensemble_row    = pd.DataFrame([{"Model": "Ensemble", "MAE": f"{mae:.4f}",
                                  "RMSE": f"{rmse:.4f}", "MAPE (%)": f"{mape:.4f}",
                                  "RMSE_float": rmse}])
results_df = pd.concat([results_df, ensemble_row], ignore_index=True)

print("\n── Model Comparison ──────────────────────────────────────")
print(results_df[["Model", "MAE", "RMSE", "MAPE (%)"]].to_string(index=False))


# ── Save results table as SVG ─────────────────────────────────────────────────
display_df = results_df[["Model", "MAE", "RMSE", "MAPE (%)"]].copy()

fig, ax = plt.subplots(figsize=(9, 2.8))
ax.axis("off")
tbl = ax.table(cellText=display_df.values, colLabels=display_df.columns,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2)

# highlight best RMSE row
best_idx = results_df["RMSE_float"].idxmin()
for c in range(len(display_df.columns)):
    tbl[0, c].set_facecolor("#1E3A5F")
    tbl[0, c].set_text_props(color="white", fontweight="bold")
    tbl[best_idx + 1, c].set_facecolor("#D1FAE5")

fig.suptitle("ARIMA Model Comparison", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PROJECT_ROOT, "Results", "model_comparison.svg"))
plt.show()
plt.close()


# ── Forecast plot (10 points before test + test window) ───────────────────────
pre_forecast = s.iloc[-(5 + 10):-5]

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(pre_forecast.index, pre_forecast.values, color="grey",  label="Recent Train", linewidth=1.5)
ax.plot(test.index,         test.values,         color="black", label="Actual",       linewidth=2)

colors = ["#2563EB", "#DC2626", "#D97706", "#16A34A"]
for (name, fc), color in zip(forecasts.items(), colors):
    ax.plot(test.index, fc.values, label=name, linestyle="--", color=color, marker="o", markersize=5)

ax.axvline(test.index[0], color="black", linestyle=":", linewidth=1)
ax.set_title("CNG – ARIMA Forecasts vs Actual")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(PROJECT_ROOT, "Results", "forecast_plot.svg"))
plt.show()
plt.close()