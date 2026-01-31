import shap
import numpy as np

def shap_explain_logreg(model, X_background, X_explain, feature_names):
    # For linear models
    explainer = shap.LinearExplainer(model, X_background, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_explain)
    return shap_values, feature_names