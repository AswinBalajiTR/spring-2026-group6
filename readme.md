# Forecasting of Energy Prices Using Economic Indicators and GDELT Derived News Signals

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
├── code/
│   ├── dataloader.py               # Data loading utilities
│   ├── download.py                 # Downloads data from Google Drive & FRED
│   ├── eda.py                      # Time Series EDA toolkit (class-based)
│   ├── models.py                   # Shared model definitions
│   ├── univariate/
│   │   ├── cng/                    # Univariate models for CNG prices
│   │   └── oil/                    # Univariate models for Crude Oil prices
│   └── multivariate/
│       ├── macro/
│       │   ├── cng/                # Macro-driven multivariate models for CNG
│       │   └── oil/                # Macro-driven multivariate models for Oil
│       ├── news/
│       │   ├── cng/monthly/        # News-driven CNG models (monthly)
│       │   └── oil/
│       │       ├── daily/          # News-driven Oil models (daily)
│       │       └── monthly/        # News-driven Oil models (monthly)
│       └── news_macro/
│           ├── cng/                # Combined news + macro models for CNG
│           └── oil/                # Combined news + macro models for Oil
├── data/                           # Raw CSV files (auto-populated by download.py)
├── demo/
│   └── app.py                      # Streamlit demo application
└── results/
    └── multivariate/
        ├── macro/                  # Macro model result SVGs
        └── news/                   # News model result SVGs
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

From the project root, run the download script to fetch datasets from Google Drive and FRED:

```bash
python code/download.py \
  --gdrive "https://drive.google.com/file/d/1DcuMzNjG_sDA5C2E1VI-6xenuQOQ1-T9/view?usp=drive_link" \
  --fred-key "f88290f199cb230c738e1e3c6e12562c"
```

This will populate the `data/` folder with the required CSV files.

---

## Usage

Always run scripts from the **project root**:

```bash
cd /path/to/spring-2026-group6
```

### Univariate Models

```bash
# Crude Oil
python code/univariate/oil/<script>.py

# CNG
python code/univariate/cng/<script>.py
```

### Multivariate Models

```bash
# Macro variables — Oil
python code/multivariate/macro/oil/<script>.py

# Macro variables — CNG
python code/multivariate/macro/cng/<script>.py

# News signals — Oil (monthly)
python code/multivariate/news/oil/monthly/<script>.py

# News signals — Oil (daily)
python code/multivariate/news/oil/daily/<script>.py

# News signals — CNG (monthly)
python code/multivariate/news/cng/monthly/<script>.py

# News + Macro combined — Oil
python code/multivariate/news_macro/oil/<script>.py

# News + Macro combined — CNG
python code/multivariate/news_macro/cng/<script>.py
```

Results (SVG plots and comparison tables) are saved automatically to the `results/` directory.

---

## Demo

Run the interactive Streamlit dashboard from the project root:

```bash
streamlit run demo/app.py
```

---

## Results

Each model script auto-generates the following outputs in the corresponding `results/` subfolder:

- `model_comparison.svg` — MAE / RMSE / MAPE table for all models + ensemble (best row highlighted in green)
- `forecast_plot.svg` — Forecast vs. actual with 10 pre-forecast context points and a vertical cutoff line

---

## Methodology Notes

- Data is **forward-filled** to handle missing dates before modelling
- Daily data is resampled to **monthly frequency** (mean aggregation)
- **First differencing** is applied to achieve stationarity before model selection
- All plots are saved as `.svg` for high-quality, scalable output
- News sentiment signals are derived from the **GDELT Project** dataset
- Macroeconomic indicators are sourced from the **FRED API**
