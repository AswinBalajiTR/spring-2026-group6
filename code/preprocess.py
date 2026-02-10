#%%
import pandas as pd

folder_path = "Data"

gdp = pd.read_csv(f"{folder_path}/gdp.csv", index_col=0, parse_dates=True)
cpi = pd.read_csv(f"{folder_path}/cpi.csv", index_col=0, parse_dates=True)
cng = pd.read_csv(f"{folder_path}/cng.csv", index_col=0, parse_dates=True)
crudeoil = pd.read_csv(f"{folder_path}/crudeoil.csv", index_col=0, parse_dates=True)

print(gdp.head())
print(cpi.head())
print(cng.head())
print(crudeoil.head())

#%%
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

series_dict = {
    "GDP": gdp["gdp"],
    "CPI": cpi["cpi"],
    "CNG": cng["cng"],
    "Crude Oil": crudeoil["crudeoil"]
}

for name, series in series_dict.items():

    series = series.dropna()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Original plot
    axes[0].plot(series)
    axes[0].set_title(f"{name} - Original Time Series")
    axes[0].grid(True)

    # ACF plot
    plot_acf(series, ax=axes[1], lags=40)
    axes[1].set_title(f"{name} - ACF")

    # PACF plot
    plot_pacf(series, ax=axes[2], lags=40, method="ywm")
    axes[2].set_title(f"{name} - PACF")

    plt.tight_layout()
    plt.show()

#%%
from statsmodels.tsa.stattools import adfuller, kpss
from prettytable import PrettyTable

table = PrettyTable()
table.field_names = [
    "Series",
    "ADF Stat",
    "ADF p-val",
    "ADF Stationary?",
    "KPSS Stat",
    "KPSS p-val",
    "KPSS Stationary?"
]

for name, series in series_dict.items():
    series = series.dropna()

    # ADF Test
    adf_stat, adf_p = adfuller(series)[0], adfuller(series)[1]

    # KPSS Test
    kpss_stat, kpss_p = kpss(series, regression="c", nlags="auto")[0], kpss(series, regression="c", nlags="auto")[1]

    table.add_row([
        name,
        round(adf_stat, 4),
        round(adf_p, 4),
        "Yes" if adf_p < 0.05 else "No",
        round(kpss_stat, 4),
        round(kpss_p, 4),
        "Yes" if kpss_p > 0.05 else "No"
    ])

print(table)

#%%
diff_series = ["GDP", "CNG", "Crude Oil"]

for name, series in series_dict.items():

    series = series.dropna()

    # Apply differencing only for selected series
    if name in diff_series:
        series = series.diff().dropna()
        title_text = f"{name} (Differenced)"
    else:
        title_text = f"{name} (Original)"

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Original plot
    axes[0].plot(series)
    axes[0].set_title(f"{title_text} - Time Series")
    axes[0].grid(True)

    # ACF plot
    plot_acf(series, ax=axes[1], lags=40)
    axes[1].set_title(f"{title_text} - ACF")

    # PACF plot
    plot_pacf(series, ax=axes[2], lags=40, method="ywm")
    axes[2].set_title(f"{title_text} - PACF")

    plt.tight_layout()
    plt.show()

#%%
from statsmodels.tsa.stattools import adfuller, kpss
from prettytable import PrettyTable

table = PrettyTable()
table.field_names = [
    "Series",
    "ADF Stat",
    "ADF p-val",
    "ADF Stationary?",
    "KPSS Stat",
    "KPSS p-val",
    "KPSS Stationary?"
]

for name, series in series_dict.items():
    series = series.diff().dropna()

    # ADF Test
    adf_stat, adf_p = adfuller(series)[0], adfuller(series)[1]

    # KPSS Test
    kpss_stat, kpss_p = kpss(series, regression="c", nlags="auto")[0], kpss(series, regression="c", nlags="auto")[1]

    table.add_row([
        name,
        round(adf_stat, 4),
        round(adf_p, 4),
        "Yes" if adf_p < 0.05 else "No",
        round(kpss_stat, 4),
        round(kpss_p, 4),
        "Yes" if kpss_p > 0.05 else "No"
    ])

print(table)