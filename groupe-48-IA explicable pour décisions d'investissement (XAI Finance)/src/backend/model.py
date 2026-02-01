import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score

class XGBoostModel:
    def __init__(self, params=None):
        self.params = params if params else {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 2,            # Reduced from 4 to 2 (drastically limits complexity)
            'learning_rate': 0.01,     # Reduced from 0.05 to 0.01 (slower learning)
            'n_estimators': 1000,      # Increased estimators (compensated by early stopping)
            'n_jobs': -1,
            'random_state': 42,
            'early_stopping_rounds': 50,
            # Regularization
            'subsample': 0.7,          # Use only 70% of data per tree
            'colsample_bytree': 0.7,   # Use only 70% of features per tree
            'gamma': 1.0,              # Minimum loss reduction required for split
            'reg_alpha': 0.5,          # L1 Regularization
            'reg_lambda': 1.0          # L2 Regularization
        }
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        """
        Train the model with early stopping.
        """
        # Remove early_stopping_rounds from params if passed to constructor to avoid duplication if user did weird things, 
        # but here we control params.
        
        self.model = xgb.XGBClassifier(**self.params)
        
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            verbose=50
        )
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test):
        probs = self.predict_proba(X_test)
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, (probs > 0.5).astype(int))
        return {"AUC": auc, "Accuracy": acc}

    def get_booster(self):
        return self.model.get_booster()
