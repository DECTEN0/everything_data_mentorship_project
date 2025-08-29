from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import json

def evaluate_and_save(name, model, X_test, y_test, cfg):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0))
    }
    with open(f"reports/tables/{name}_test_metrics.json","w",encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(f"reports/tables/{name}_classification_report.txt","w",encoding="utf-8") as f:
        f.write(classification_report(y_test, model.predict(X_test), zero_division=0))
