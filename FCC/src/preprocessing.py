import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def prepare_splits(df: pd.DataFrame, target: str, sensitive: str, test_size: float, random_state: int):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    if sensitive not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive}' not found.")

    y = df[target].astype(int)
    A = df[sensitive]
    X = df.drop(columns=[target])

    # one-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, A, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler(with_mean=False) if hasattr(X_train, "sparse") else StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy(), A_train.to_numpy(), A_test.to_numpy(), list(X.columns)