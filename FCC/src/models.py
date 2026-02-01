from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_baseline_logreg(X_train, y_train):
    model = LogisticRegression(max_iter=2000, n_jobs=None)
    model.fit(X_train, y_train)
    return model

def train_baseline_rf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model