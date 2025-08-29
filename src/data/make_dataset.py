import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw(path):
    df = pd.read_csv(path)
    return df

def split_data(df, cfg):
    target = cfg["target_col"]
    drop_cols = cfg.get("drop_columns", [])
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = X.pop(target)
    return train_test_split(X, y, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=y), X.columns.tolist()
