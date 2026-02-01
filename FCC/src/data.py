import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, TARGET_COL, SENSITIVE_COL, DROP_COLS, RANDOM_STATE


def load_data_csv():
    df = pd.read_csv(DATA_PATH)

    # 1) Drop colonnes interdites (ex: name)
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=col)

    # 2) Vérification minimale
    assert TARGET_COL in df.columns, f"Missing target: {TARGET_COL}"
    assert SENSITIVE_COL in df.columns, f"Missing sensitive: {SENSITIVE_COL}"

    # 3) Nettoyage NA
    df = df.dropna(subset=[TARGET_COL, SENSITIVE_COL])

    return df


def train_test_from_csv(test_size=0.3):
    df = load_data_csv()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int).values
    s = df[SENSITIVE_COL].values

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, s_train, s_test