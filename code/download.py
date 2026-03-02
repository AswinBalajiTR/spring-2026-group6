import os
import sys
import argparse
from typing import Dict, Optional
import pandas as pd
from fredapi import Fred


class DataLoader:
    def __init__(self, api_key: str, series_map: Dict[str, str], data_folder_name: str = "data"):
        if not api_key or not api_key.strip():
            raise ValueError("FRED API key is required.")
        if not isinstance(series_map, dict) or not series_map:
            raise ValueError("series_map must be a non-empty dict like {'gdp':'GDP'}")

        self.api_key = api_key.strip()
        self.series_map = series_map

        # Detect project root (parent of code/)
        code_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(code_dir)

        # Create ../data folder
        self.data_dir = os.path.join(project_root, data_folder_name)
        os.makedirs(self.data_dir, exist_ok=True)

        self.fred = Fred(api_key=self.api_key)

    def fetch_series(self, series_id: str) -> pd.Series:
        return self.fred.get_series(series_id)

    def save_series(self, local_name: str, series: pd.Series) -> str:
        df = series.rename("value").to_frame()
        df.index.name = "date"
        df = df.reset_index()

        file_path = os.path.join(self.data_dir, f"{local_name}.csv")
        df.to_csv(file_path, index=False)
        return file_path

    def download_all(self) -> Dict[str, str]:
        results = {}
        for local_name, series_id in self.series_map.items():
            try:
                series = self.fetch_series(series_id).dropna()
                saved_path = self.save_series(local_name, series)
                results[local_name] = saved_path
            except Exception as e:
                results[local_name] = f"ERROR: {str(e)}"
        return results

def is_gdrive_folder(url: str) -> bool:
    return "/folders/" in url or "drive.google.com/drive/folders/" in url

def download_gdrive(url: str, data_dir: str, save_as: str = "news_data.csv") -> str:
    try:
        import gdown
    except ImportError:
        raise RuntimeError("Missing dependency: gdown. Install it with: pip install gdown")
    os.makedirs(data_dir, exist_ok=True)

    if is_gdrive_folder(url):
        gdown.download_folder(url=url, output=data_dir, quiet=False, use_cookies=False)
        return f"FOLDER_DOWNLOADED_TO: {data_dir}"

    # Single file
    out_path = os.path.join(data_dir, save_as)
    res = gdown.download(url=url, output=out_path, quiet=False, fuzzy=True)
    if not res or not os.path.exists(out_path):
        raise RuntimeError("Google Drive download failed. Check sharing permissions (Anyone with link: Viewer).")
    return out_path


def default_series_map() -> Dict[str, str]:
    return {
        "gdp": "GDP",
        "cpi": "CPIAUCSL",
        "wti_crude_oil": "DCOILWTICO",
        "natural_gas": "DHHNGSP",
    }

def get_project_data_dir(data_folder_name: str = "data") -> str:
    code_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(code_dir)
    data_dir = os.path.join(project_root, data_folder_name)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download Google Drive news data + FRED time series into ../data/"
    )
    parser.add_argument(
        "--gdrive",
        type=str,
        default=None,
        help="Google Drive file or folder link (downloads into ../data/). File is saved as news_data.csv by default."
    )
    parser.add_argument(
        "--fred-key",
        type=str,
        default=None,
        help="FRED API key (required if you want to download FRED series)."
    )
    parser.add_argument(
        "--no-fred",
        action="store_true",
        help="Skip downloading FRED series."
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        default="data",
        help="Data folder name at project root. Default: data"
    )

    args = parser.parse_args()

    data_dir = get_project_data_dir(args.data_folder)

    # 1) Google Drive
    if args.gdrive:
        print("\n[Google Drive] Starting download...")
        try:
            out = download_gdrive(args.gdrive, data_dir=data_dir, save_as="news_data.csv")
            print(f"[Google Drive] Done -> {out}")
        except Exception as e:
            print(f"[Google Drive] ERROR -> {e}")

    # 2) FRED
    if not args.no_fred:
        if not args.fred_key:
            print("\n[FRED] Skipped (no --fred-key provided).")
        else:
            print("\n[FRED] Starting download...")
            series_map = default_series_map()  # <--- edit defaults above
            try:
                dl = DataLoader(api_key=args.fred_key, series_map=series_map, data_folder_name=args.data_folder)
                results = dl.download_all()
                for name, status in results.items():
                    print(f"[FRED] {name} -> {status}")
                print("[FRED] Done.")
            except Exception as e:
                print(f"[FRED] ERxsROR -> {e}")

    if not args.gdrive and (args.no_fred or not args.fred_key):
        print("\nNothing to do. Provide --gdrive and/or --fred-key (unless using --no-fred).")
        sys.exit(1)

# python3 download.py --gdrive "https://drive.google.com/file/d/1DcuMzNjG_sDA5C2E1VI-6xenuQOQ1-T9/view?usp=drive_link" --fred-key "f88290f199cb230c738e1e3c6e12562c"
if __name__ == "__main__":
    main()

