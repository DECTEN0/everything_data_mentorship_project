import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib

def build_model(model_cfg):
    t = model_cfg["type"]
    p = model_cfg.get("params", {})
    if t == "logistic_regression":
        return LogisticRegression(**p)
    elif t == "random_forest":
        return RandomForestClassifier(**p)
    else:
        raise ValueError(f"Unsupported model type: {t}")

def train_and_select(preprocessor, X_train, y_train, cfg):
    models_cfg = cfg["models"]
    cv = cfg["cv_folds"]
    results = {}

    best_name, best_score, best_pipe = None, -np.inf, None
    for m in models_cfg:
        model = build_model(m)
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        results[m["name"]] = {
            "cv_f1_mean": float(scores.mean()),
            "cv_f1_std": float(scores.std())
        }
        if scores.mean() > best_score:
            best_name, best_score, best_pipe = m["name"], scores.mean(), pipe

    best_pipe.fit(X_train, y_train)
    joblib.dump(best_pipe, f"reports/export/{best_name}_pipeline.joblib")
    return best_name, best_pipe, results
