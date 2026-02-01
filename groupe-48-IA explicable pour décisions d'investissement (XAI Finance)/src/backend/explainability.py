
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def compute_shap_values(model, X):
    """
    Compute SHAP values for a given model and dataset.
    Returns the shap_values array and the explainer.
    """
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer

def get_mean_abs_shap(model, X):
    """
    Return a Series of mean(|SHAP value|) for each feature.
    Useful for tracking importance over time.
    """
    shap_values, _ = compute_shap_values(model, X)
    
    # Check shape - for binary class, shap_values might be a list or array
    if isinstance(shap_values, list):
        # For binary classification, TreeExplainer usually returns just the values for the log-odds change
        # But sometimes for older versions it returns a list. 
        # XGBoost TreeExplainer typically returns a single matrix for binary
        vals = shap_values[1] if len(shap_values) == 2 else shap_values[0] # Try to get positive class
    else:
        vals = shap_values
        
    mean_abs_shap = pd.DataFrame(vals, columns=X.columns).abs().mean()
    return mean_abs_shap

def plot_shap_summary(model, X, output_path="output/shap_summary.png"):
    """
    Generate SHAP summary plot.
    """
    shap_values, explainer = compute_shap_values(model, X)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved SHAP summary to {output_path}")
    plt.close()

def plot_temporal_importance(importance_df, output_path="output/temporal_importance.png"):
    """
    Plot the evolution of feature importance over time.
    importance_df: DataFrame where Index=Date/Fold, Columns=Features, Values=Importance
    """
    # Filter top N features by total importance to avoid clutter
    top_features = importance_df.sum().sort_values(ascending=False).head(10).index
    plot_df = importance_df[top_features]
    
    plt.figure(figsize=(14, 8))
    
    # Heatmap
    sns.heatmap(plot_df.T, cmap='viridis', annot=False)
    plt.title("Evolution of Top 10 Feature Importance (SHAP) Over Time")
    plt.xlabel("Walk-Forward Fold")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved Temporal Importance to {output_path}")
    plt.close()
    

    # Stacked Area Plot alternative (optional)
    plt.figure(figsize=(14, 8))
    plot_df.plot(kind='area', stacked=True, alpha=0.9, figsize=(14, 8), cmap='tab20')
    plt.title("Feature Importance Composition Over Time")
    plt.ylabel("Total Mean |SHAP|")
    plt.xlabel("Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path.replace(".png", "_area.png"))
    plt.close()

# -----------------------------------------------------------------------------
# PHASE 1: FORMAL ARGUMENTATION LOGIC
# -----------------------------------------------------------------------------

def format_feature_impact(name, value, shap_value):
    """
    Generate a human-readable argument string based on feature name, value, and SHAP impact.
    
    Args:
        name (str): Feature name (e.g., 'rsi_14', 'treasury_10y')
        value (float): Actual value of the feature
        shap_value (float): SHAP value (impact on log-odds)
    
    Returns:
        str: Formatted argument string
    """
    impact_dir = "↑" if shap_value > 0 else "↓" # Impact on prediction (Bullish/Bearish)
    val_str = f"{value:.2f}"
    
    # Heuristics for specific known features for better biological/finance readability
    
    # Macro
    if 'treasury' in name:
        return f"Yields ({val_str}%) {'Pushing Prices Up' if shap_value > 0 else 'Pressuring Prices'}"
    if 'vix' in name:
        return f"Volatility ({val_str}) {'Supportive' if shap_value > 0 else 'Indicates Fear'}"
    if 'cpi' in name:
        return f"Inflation Trend ({val_str}) {'Favorable' if shap_value > 0 else 'Negative Headwind'}"
    
    # Technicals
    if 'rsi' in name:
        state = "Oversold" if value < 30 else ("Overbought" if value > 70 else "Neutral")
        return f"RSI is {state} ({val_str}) -> {impact_dir}"
    
    if 'ma_50' in name or 'dist_ma' in name:
        return f"Trend Deviation ({val_str}) -> {impact_dir}"
        
    if 'volatility' in name:
        return f"Price Stability ({val_str}) -> {impact_dir}"
        
    # Default fallback
    return f"{name} ({val_str}) -> {impact_dir} (Impact: {shap_value:.2f})"

def explain_latest(model, X_latest):
    """
    Generate a structured explanation for a single prediction (the latest one).
    
    Args:
        model: Trained XGBoost model
        X_latest: DataFrame containing a single row (the most recent data point)
        
    Returns:
        dict: Structure containing prediction, confidence, and Bullish/Bearish arguments
    """
    # 1. Get Prediction
    prob = model.predict_proba(X_latest)[0, 1]
    prediction = "LONG" if prob > 0.5 else "SHORT"
    confidence = prob if prediction == "LONG" else 1 - prob
    
    # 2. Compute SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_latest)
    
    # Handle array shape issues (binary classification sometimes returns list or matrix)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0] if len(shap_values) == 2 else shap_values[0][0]
    else:
        # If it's a matrix (1, n_features), take first row
        shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
    base_value = explainer.expected_value
    if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
         base_value = base_value[-1] # scalar
         
    # 3. Separate Arguments
    feature_names = X_latest.columns
    bullish_args = []
    bearish_args = []
    
    for name, val, shap_val in zip(feature_names, X_latest.iloc[0], shap_vals):
        # Contribution > 0 => Pushes probability UP (Bullish)
        # Contribution < 0 => Pushes probability DOWN (Bearish)
        
        arg_text = format_feature_impact(name, val, shap_val)
        item = {
            "feature": name,
            "value": float(val),
            "shap": float(shap_val),
            "text": arg_text
        }
        
        if shap_val > 0:
            bullish_args.append(item)
        else:
            bearish_args.append(item)
            
    # 4. Sort by absolute impact (magnitude)
    bullish_args.sort(key=lambda x: abs(x['shap']), reverse=True)
    bearish_args.sort(key=lambda x: abs(x['shap']), reverse=True)
    
    return {
        "prediction": prediction,
        "probability_score": float(prob),
        "confidence": float(confidence),
        "base_value": float(base_value),
        "bullish_args": bullish_args, # List of dicts
        "bearish_args": bearish_args  # List of dicts
    }

