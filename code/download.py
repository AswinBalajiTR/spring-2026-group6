import os
import argparse
import pandas as pd
from fredapi import Fred
import gdown

SERIES_MAP = {
    "wti_crude_oil":        "DCOILWTICO",       # WTI Crude Oil Spot Price (daily)
    "brent_crude_oil":      "DCOILBRENTEU",     # Brent Crude Oil Spot Price (daily)
    "natural_gas":          "DHHNGSP",          # Henry Hub Natural Gas Spot Price (daily)
    "gasoline_price":       "GASREGCOVW",       # US Regular Gasoline Price (weekly)
    "usd_index":            "DTWEXBGS",         # USD Broad Trade Weighted Index (daily)
    "real_usd_index":       "RTWEXBGS",         # Real USD Broad Trade Weighted Index (monthly)
    "usd_eur":              "DEXUSEU",          # USD/EUR Exchange Rate (daily)
    "volatility_index":     "OVXCLS",           # CBOE Crude Oil Volatility Index (daily)
    "ind_production_overall": "INDPRO",         # Industrial Production Index (monthly)
    "ind_prod_drill":        "IPN213111S",      # Industrial Production: Drilling Oil & Gas (monthly)
    "cap_util":              "TCU",             # Total Capacity Utilization (monthly)
    "cap_util_oil_gas":      "CAPUTLG211S",     # Capacity Utilization: Oil & Gas (monthly)
    "natgas_electric":       "GASDESW",         # Natural Gas to Electric Power (monthly)
    "cpi":                   "CPIAUCSL",        # CPI All Items (monthly)
    "ppi":                   "PPIENG",          # PPI Energy (monthly)
    "ppidp":                "WPU0561",         # PPI Crude Petroleum (monthly)
    "pce":                   "PCEPI",           # PCE Price Index (monthly)
}

def get_data_dir(folder="data"):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), folder)
    os.makedirs(path, exist_ok=True)
    return path

def download_fred(api_key, data_dir):
    fred = Fred(api_key=api_key)
    for name, sid in SERIES_MAP.items():
        try:
            df = fred.get_series(sid).dropna().rename("value").to_frame()
            df.index.name = "date"
            df.reset_index().to_csv(os.path.join(data_dir, f"{name}.csv"), index=False)
            print(f"[FRED] {name} -> saved")
        except Exception as e:
            print(f"[FRED] {name} -> ERROR: {e}")

def download_gdrive(url, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    if "/folders/" in url:
        gdown.download_folder(url=url, output=data_dir, quiet=False, use_cookies=False)
    else:
        out = os.path.join(data_dir, "news_data.csv")
        gdown.download(url=url, output=out, quiet=False)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdrive", type=str)
    parser.add_argument("--fred-key", type=str)
    parser.add_argument("--data-folder", default="data")
    args = parser.parse_args()

    data_dir = get_data_dir(args.data_folder)

    if args.gdrive:
        download_gdrive(args.gdrive, data_dir)

    if args.fred_key:
        download_fred(args.fred_key, data_dir)


# python3 download.py --gdrive "https://drive.google.com/file/d/1i1HLSoDAAS1TJ_6-AwooQ0c-RfHEqGY3/view?usp=drive_link" --fred-key "f88290f199cb230c738e1e3c6e12562c"
if __name__ == "__main__":
    main()

