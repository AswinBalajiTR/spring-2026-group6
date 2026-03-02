import os
import pandas as pd

class DataLoader:
    def __init__(self, data_folder="data"):
        code_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(code_dir)
        self.data_dir = os.path.join(project_root, data_folder)
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data folder not found: {self.data_dir}")

    def load(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.data_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing file: {path}")
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError(f"{filename} must have a 'date' column.")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
        # if file has multiple columns, keep them; otherwise fine
        return df

    def load_cpi(self):
        return self.load("cpi.csv")

    def load_crude_oil(self):
        return self.load("wti_crude_oil.csv")

    def load_natural_gas(self):
        return self.load("natural_gas.csv")

    def load_news(self):
        return self.load("news_data.csv")

    def load_three_data(self):
        # Load
        cpi = self.load_cpi()
        crude = self.load_crude_oil()
        gas = self.load_natural_gas()
        news = self.load_news()

        # Rename columns (assumes each has one value col besides date)
        cpi.columns = ["cpi"]
        crude.columns = ["crudeoil"]
        gas.columns = ["cng"]
        news.columns = ["news"]

        # Build group dataframes
        macro_df = cpi.sort_index()
        energy_df = crude.join(gas, how="outer").sort_index()
        news_df = news.sort_index()

        return macro_df, energy_df, news_df

    def split_last_10(self, df, last_n=10, test_n=5, val_n=5):
        if test_n + val_n != last_n:
            raise ValueError("test_n + val_n must equal last_n.")
        if len(df) < last_n:
            raise ValueError(f"Not enough rows. Need at least {last_n}, got {len(df)}.")

        tail = df.iloc[-last_n:]
        train = df.iloc[:-last_n]
        test = tail.iloc[:test_n]
        val = tail.iloc[test_n:]
        return train, test, val