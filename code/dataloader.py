import os
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, data_folder="data"):
        code_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(code_dir)
        self.data_dir = os.path.join(project_root, data_folder)

    def read_data(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.data_dir, filename)
        return pd.read_csv(path, parse_dates=["date"], index_col="date", dayfirst=False)

    def _load_all(self, exclude: list = []) -> dict:
        dfs = {}
        for f in os.listdir(self.data_dir):
            if not f.endswith(".csv"): continue
            name = f.replace(".csv", "")
            if name in exclude or f in exclude: continue  # exclude by filename or name
            df = self.read_data(f)
            freq = "MS" if self._is_monthly(df) else "D"
            df = df.resample(freq).ffill()
            if "value" in df.columns:
                df = df[["value"]].rename(columns={"value": name})
            else:
                df.columns = [f"{c}" if len(df.columns) > 1 else name for c in df.columns]
            # exclude specific columns
            drop_cols = [c for c in df.columns if c in exclude]
            if drop_cols: df = df.drop(columns=drop_cols)
            if df.empty or len(df.columns) == 0: continue
            dfs[name] = df
        return dfs

    def _is_monthly(self, df: pd.DataFrame) -> bool:
        return df.index.to_series().diff().dt.days.median() > 20

    def combine_monthly(self, exclude: list = []) -> pd.DataFrame:
        dfs = [df for df in self._load_all(exclude).values() if self._is_monthly(df)]
        combined = pd.concat(dfs, axis=1)
        start, end = combined.index.min(), combined.index.max()
        mask = (combined.index >= start) & (combined.index <= end)
        return combined.loc[mask].dropna()

    def combine_daily(self, exclude: list = []) -> pd.DataFrame:
        dfs = [df for df in self._load_all(exclude).values() if not self._is_monthly(df)]
        combined = pd.concat(dfs, axis=1)
        start, end = combined.index.min(), combined.index.max()
        mask = (combined.index >= start) & (combined.index <= end)
        return combined.loc[mask].dropna()

    def combine_all_as_monthly(self, exclude: list = []) -> pd.DataFrame:
        monthly = self.combine_monthly(exclude)
        daily = self.combine_daily(exclude).resample("MS").mean()
        combined = pd.concat([monthly, daily], axis=1)
        start = max(monthly.index.min(), daily.index.min())
        end = min(monthly.index.max(), daily.index.max())
        return combined.loc[start:end].dropna()

    def split_classic(self, df: pd.DataFrame, target: str = None,tp=5):
        train, test = df.iloc[:-tp], df.iloc[-tp:]
        if target is None:
            return train, test
        return train.drop(columns=[target]), train[[target]], test.drop(columns=[target]), test[[target]]

    def split_time_series(self, data, input_window: int, output_window: int):
        assert output_window in (1, 5), "output_window must be 1 or 5"
        block = input_window + 5
        assert len(data) > 2 * block, f"Dataset too short. Need > {2 * block} rows, got {len(data)}"
        return data.iloc[:block], data.iloc[block: len(data) - block], data.iloc[len(data) - block:]

    def build_sequences(self, backtest_data, train_data, test_data, input_window: int, output_window: int,
                        target: str = None, include_target: bool = False):
        def to_df(d): return d.to_frame() if isinstance(d, pd.Series) else d

        def get_cols(df):
            if target is None: return df.columns.tolist(), df.columns.tolist()
            return (df.columns.tolist() if include_target else [c for c in df.columns if c != target]), [target]

        def squeeze(y): return y.squeeze(-1) if y.shape[-1] == 1 else y

        def make(block, indices):
            arr, x_cols, y_cols = block.values, *get_cols(block)
            xi = [block.columns.get_loc(c) for c in x_cols]
            yi = [block.columns.get_loc(c) for c in y_cols]
            Xs = np.array([arr[i: i + input_window][:, xi] for i in indices])
            ys = squeeze(np.array([arr[i + input_window: i + input_window + output_window][:, yi] for i in indices]))
            return Xs, ys

        backtest_data, train_data, test_data = map(to_df, [backtest_data, train_data, test_data])
        assert target is None or target in train_data.columns, f"Target '{target}' not found."

        rolling = output_window == 1
        bt_idx = list(range(5)) if rolling else [0]
        tst_idx = list(range(5)) if rolling else [0]
        trn_idx = list(range(len(train_data) - input_window - output_window + 1))
        assert len(trn_idx) > 0, "Train block too short."

        btest_X, btest_Y = make(backtest_data, bt_idx)
        test_X, test_Y = make(test_data, tst_idx)
        train_X, train_Y = make(train_data, trn_idx)
        return train_X, train_Y, test_X, test_Y, btest_X, btest_Y