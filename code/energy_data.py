import os
import pandas as pd
from fredapi import Fred


def download_series(api_key, series_dict, start_date="2000-01-01", end_date=None):
    fred = Fred(api_key=api_key)
    dataframes = {}

    for name, code in series_dict.items():
        df = pd.DataFrame({name: fred.get_series(code)})
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        if start_date:
            df = df.loc[start_date:]
        if end_date:
            df = df.loc[:end_date]

        dataframes[name] = df.dropna()

    return dataframes


def save_series(dataframes, save_path):
    os.makedirs(save_path, exist_ok=True)

    for name, df in dataframes.items():
        file_path = os.path.join(save_path, f"{name}.csv")
        df.to_csv(file_path)
        print(f"Saved: {file_path}")


if __name__ == "__main__":

    fred_api_key = "f88290f199cb230c738e1e3c6e12562c"

    series = {
        "gdp": "GDP",
        "cpi": "CPALTT01USM657N",
        "cng": "DHHNGSP",
        "crudeoil": "DCOILWTICO"
    }

    save_folder = "../Data"

    data = download_series(
        api_key=fred_api_key,
        series_dict=series,
        start_date="2000-01-01",
        end_date=None
    )

    save_series(data, save_folder)

#%%
