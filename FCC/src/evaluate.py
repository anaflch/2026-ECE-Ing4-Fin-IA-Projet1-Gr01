import numpy as np
from FCC.src.fairness import compute_performance, compute_fairness

def predict_any(model, X):
    # Fairlearn mitigators can have predict_proba depending on base estimator, but safe handling:
    y_pred = model.predict(X)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)[:, 1]
        except Exception:
            y_proba = None
    return y_pred, y_proba

def evaluate_model(name, model, X_test, y_test, A_test):
    y_pred, y_proba = predict_any(model, X_test)
    perf = compute_performance(y_test, y_pred, y_proba)
    mf, fair = compute_fairness(y_test, y_pred, A_test)

    return {
        "name": name,
        **perf,
        **fair,
        "by_group": mf.by_group.to_dict(),
        "overall": mf.overall.to_dict(),
    }