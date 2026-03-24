# Energy Price Forecasting using Macro Indicators and News Data

Time series analysis and forecasting of Energy prices (CNG & Crude Oil) with help of macro economic variables and global News data.

---

## Team

| Name | Email |
|---|---|
| Aswin Balaji Thippa Ramesh | at119@gwu.edu |
| Vishal Fulsundar | vishal.fulsundar@gwu.edu |

**Mentored by:** Dr. Amir Hossein Jafari — ajafari@gwu.edu

---

## Project Structure

```
spring-2026-group6/
├── Code/
│   ├── eda.py                        # Time Series EDA toolkit (class-based)
│   └── Univariate_model/
│       └── crude_oil.py              # Main modelling script
├── Data/
│   └── wti_crude_oil.csv             # Raw WTI price data
└── Results/                          # Auto-generated SVG outputs
    ├── WTI_series.svg
    ├── WTI_stationarity.svg
    ├── WTI_diff_acf_pacf.svg
    ├── WTI_diff_seasonality.svg
    ├── WTI_diff_decomposition.svg
    ├── model_comparison.svg
    └── forecast_plot.svg
```

---

## Setup

```bash
git clone https://github.com/your-username/spring-2026-group6.git
cd spring-2026-group6
pip install -r requirements.txt
```

## Data

Navigate to the `Code/` folder and run the download script to fetch the dataset from Google Drive and FRED:

```bash
cd Code
python3 download.py \
  --gdrive "https://drive.google.com/file/d/1DcuMzNjG_sDA5C2E1VI-6xenuQOQ1-T9/view?usp=drive_link" \
  --fred-key "f88290f199cb230c738e1e3c6e12562c"
```

This will populate the `Data/` folder with the required CSV files.

---

## Usage

Always run from the project root:

```bash
cd /path/to/spring-2026-group
python Code/Univariate_model/crude_oil.py
```

---

## EDA Toolkit — `eda.py`

The `TimeSeriesEDA` class provides the following methods. All plots are saved as SVG to the `Results/` folder automatically.

| Method | Description |
|---|---|
| `plot_series(s, name)` | Plot raw time series |
| `take_difference(s, periods)` | Return differenced series |
| `stationarity_tests(s, name)` | ADF + KPSS tests with Stationary YES/NO |
| `forward_fill(s, freq)` | Fill date gaps via forward-fill |
| `to_monthly(s, agg)` | Resample daily → monthly |
| `plot_acf_pacf(s, name, lags)` | ACF and PACF subplots |
| `seasonality_test(s, name, period)` | ACF + STL-based seasonality strength |
| `decompose(s, name, period, model)` | Seasonal decomposition plot |

---

## Results

- `model_comparison.svg` — MAE / RMSE / MAPE table for all models + ensemble (best row highlighted in green)
- `forecast_plot.svg` — Forecast vs actual with 10 pre-forecast context points and a vertical cutoff line

---

## Notes

- Data is forward-filled to handle missing dates before modelling
- Daily data is resampled to **monthly frequency** (mean aggregation)
- First differencing is applied to achieve stationarity before model selection
- All plots saved as `.svg` for high-quality, scalable output
