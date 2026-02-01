from sklearn.metrics import accuracy_score, roc_auc_score

from src.model import make_model
from src.fairness import fairness_metrics
from src.data import train_test_from_csv


def main():
    # 1) Charger + split depuis le CSV
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_from_csv(test_size=0.3)

    # 2) Entraîner
    model = make_model()
    model.fit(X_train, y_train)

    # 3) Prédire
    y_pred = model.predict(X_test)

    # AUC nécessite des proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = None

    # 4) Afficher métriques
    print("ACC :", accuracy_score(y_test, y_pred))
    if auc is not None:
        print("AUC :", auc)
    print("FAIR :", fairness_metrics(y_test, y_pred, s_test))


if __name__ == "__main__":
    main()