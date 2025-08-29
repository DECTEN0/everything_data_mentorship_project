#!/usr/bin/env python
import argparse, yaml, os, sys, json
from pathlib import Path
from src.utils.io import ensure_dirs
from src.data.make_dataset import load_raw, split_data
from src.features.build_features import make_preprocessor
from src.models.train_model import train_and_select
from src.models.evaluate import evaluate_and_save

def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ensure_dirs(["data/interim","data/processed","reports/tables","reports/figures","reports/export"])

    # 1) Load
    df = load_raw("data/raw/cohort.csv")

    # 2) Split
    X_train, X_test, y_train, y_test, feature_names = split_data(df, cfg)

    # 3) Features
    preprocessor = make_preprocessor(cfg, feature_names)

    # 4) Train & select best
    best_name, best_model, results = train_and_select(preprocessor, X_train, y_train, cfg)

    # 5) Evaluate
    evaluate_and_save(best_name, best_model, X_test, y_test, cfg)

    # Save leaderboard
    with open("reports/tables/model_leaderboard.json","w",encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Done. Best model:", best_name)
    print("Leaderboard saved to reports/tables/model_leaderboard.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params.yaml")
    args = ap.parse_args()
    main(args.config)
