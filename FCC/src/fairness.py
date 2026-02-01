from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)

from sklearn.metrics import accuracy_score, roc_auc_score

def compute_performance(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_proba is not None:
        out["auc"] = float(roc_auc_score(y_true, y_proba))
    return out

def compute_fairness(y_true, y_pred, sensitive_features):
    # group metrics table
    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "FPR": false_positive_rate,
            "FNR": false_negative_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    # scalar diffs (overall fairness gaps)
    dp_diff = float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features))
    eo_diff = float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features))
    return mf, {"dp_diff": dp_diff, "eo_diff": eo_diff} 
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

def train_fair_model(X_train, y_train, A_train, constraint: str = "dp", eps: float = 0.02):
    """
    Train a fair classifier using in-processing fairness constraints.

    Parameters
    ----------
    X_train : array-like
        Feature matrix.
    y_train : array-like
        Target labels.
    A_train : array-like
        Sensitive attribute.
    constraint : str
        'dp' for Demographic Parity or 'eo' for Equalized Odds.
    eps : float
        Allowed fairness violation (smaller = stricter fairness).

    Returns
    -------
    mitigator : ExponentiatedGradient
        Trained fair model.
    """

    if constraint == "dp":
        moment = DemographicParity()
    elif constraint == "eo":
        moment = EqualizedOdds()
    else:
        raise ValueError("constraint must be 'dp' or 'eo'")

    from sklearn.linear_model import LogisticRegression
    base_estimator = LogisticRegression(max_iter=2000)

    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=moment,
        eps=eps
    )

    mitigator.fit(
        X_train,
        y_train,
        sensitive_features=A_train
    )

    return mitigator