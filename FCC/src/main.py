import json
from pathlib import Path

from FCC.src.preprocessing import load_dataframe, prepare_splits
from FCC.src.models import train_baseline_logreg
from FCC.src.fairness import train_fair_model
from FCC.src.evaluate import evaluate_model
from FCC.src.config import Paths, Columns, Split
def main():
    Paths.ARTIFACTS.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(str(Paths.DATA_RAW))
    X_train, X_test, y_train, y_test, A_train, A_test, feature_names = prepare_splits(
        df=df,
        target=Columns.TARGET,
        sensitive=Columns.SENSITIVE,
        test_size=Split.TEST_SIZE,
        random_state=Split.RANDOM_STATE,
    )

    # 1) Baseline
    base = train_baseline_logreg(X_train, y_train)
    res_base = evaluate_model("baseline_logreg", base, X_test, y_test, A_test)

    # 2) Fair models (DP / EO)
    fair_dp = train_fair_model(X_train, y_train, A_train, constraint="dp", eps=0.02)
    res_dp = evaluate_model("fair_dp_eps0.02", fair_dp, X_test, y_test, A_test)

    fair_eo = train_fair_model(X_train, y_train, A_train, constraint="eo", eps=0.02)
    res_eo = evaluate_model("fair_eo_eps0.02", fair_eo, X_test, y_test, A_test)

    # 3) Trade-off sweep (eps)
    eps_grid = [0.005, 0.01, 0.02, 0.05, 0.1]
    sweep = []
    for eps in eps_grid:
        m = train_fair_model(X_train, y_train, A_train, constraint="dp", eps=eps)
        sweep.append(evaluate_model(f"fair_dp_eps{eps}", m, X_test, y_test, A_test))

    out = {
        "baseline": res_base,
        "fair_dp": res_dp,
        "fair_eo": res_eo,
        "sweep_dp": sweep,
    }

    out_path = Paths.ARTIFACTS / "results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()