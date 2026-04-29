# Forecasting of Energy Prices Using Economic Indicators and GDELT Derived News Signals

Time series analysis and forecasting of Energy prices (CNG & Crude Oil) using macroeconomic variables and GDELT-derived news signals.

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
├── code/
│   ├── dataloader.py
│   ├── download.py
│   ├── eda.py
│   ├── models.py
│   ├── univariate/        # cng/, oil/
│   └── multivariate/      # macro/, news/, news_macro/ — each with cng/ and oil/
├── data/
├── demo/
│   └── app.py
└── results/
```

---

## Setup

```bash
git clone https://github.com/your-username/spring-2026-group6.git
cd spring-2026-group6
pip install -r requirements.txt
```

---

## Data

> **FRED API Key required.** Get your free key at https://fred.stlouisfed.org/docs/api/api_key.html

```bash
python code/download.py \
  --gdrive "https://drive.google.com/file/d/1DcuMzNjG_sDA5C2E1VI-6xenuQOQ1-T9/view?usp=drive_link" \
  --fred-key "YOUR_FRED_API_KEY"
```

---

## Usage

Run all scripts from the project root:

```bash
python code/univariate/<cng|oil>/<script>.py
python code/multivariate/<macro|news|news_macro>/<cng|oil>/<script>.py
```

Results are saved automatically to `results/`.

---

## Demo

```bash
streamlit run demo/app.py
```

---

## Results

- `model_comparison.svg` — MAE / RMSE / MAPE across models (best highlighted in green)
- `forecast_plot.svg` — Forecast vs. actual with cutoff line

---

## Methodology Notes

- Missing dates forward-filled; daily data resampled to monthly (mean)
- First differencing applied for stationarity
- News signals from **GDELT**; macro indicators from **FRED**
- Outputs saved as `.svg`
