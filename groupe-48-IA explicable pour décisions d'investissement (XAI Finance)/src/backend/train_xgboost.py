import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

import data_loader as dl
import preprocessing as pp
import model as md

def plot_learning_curve(results):
    """
    Plot training vs validation AUC over boosting rounds.
    """
    epochs = len(results['validation_0']['auc'])
    x_axis = range(0, epochs)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_axis, results['validation_0']['auc'], label='Train')
    ax.plot(x_axis, results['validation_1']['auc'], label='Validation')
    ax.legend()
    ax.set_ylabel('AUC')
    ax.set_title('XGBoost Learning Curve')
    plt.grid(True)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "learning_curve.png")
    print(f"Saved learning curve to {output_dir / 'learning_curve.png'}")
    plt.close()

def plot_feature_importance(model, feature_names):
    """
    Plot Top 20 Feature Importance by Gain.
    """
    importance = model.get_booster().get_score(importance_type='gain')
    # importance is dict {feature: score}
    
    # Map back to names if needed, but XGBoost usually handles names if dataframe passed
    # Sort
    sorted_idx = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    
    features = [k for k, v in sorted_idx]
    scores = [v for k, v in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(features, scores)
    ax.set_xlabel('Gain (Information provided)')
    ax.set_title('Top 20 Features Importance (XGBoost Gain)')
    ax.invert_yaxis() # Highest on top
    
    output_dir = Path("output")
    plt.savefig(output_dir / "feature_importance.png")
    print(f"Saved feature importance to {output_dir / 'feature_importance.png'}")
    plt.close()

def plot_probability_distribution(y_true, y_probs, title="Probability Distribution"):
    """
    Plot histogram of predicted probabilities for positive and negative classes.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(y_probs[y_true == 0], bins=50, alpha=0.5, label='Class 0 (Down)', color='red', density=True)
    plt.hist(y_probs[y_true == 1], bins=50, alpha=0.5, label='Class 1 (Up)', color='green', density=True)
    plt.axvline(0.5, color='black', linestyle='--')
    plt.title(title)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output/prob_distribution.png')
    print("Saved output/prob_distribution.png")
    plt.close()

def plot_cumulative_returns(y_true, y_probs, dates=None, title="Strategy Equity Curve"):
    """
    Simulate a simple strategy: Long if prob > 0.5.
    Assume y_true sign represents the direction of return (approximation).
    NOTE: y_true is binary (0/1), we need actual returns for a real equity curve.
    Here we define a 'Signal Quality' curve: +1 if correct, -1 if wrong.
    """
    signals = (y_probs > 0.5).astype(int)
    correct_preds = (signals == y_true).astype(int)
    # Map 0 -> -1 (Loss), 1 -> +1 (Win) ??? No, that's accuracy.
    # Let's just plot Cumulative Accuracy Drift
    # or better: virtual PnL assuming equal sized bets +1/-1
    
    # Virtual Return: +1% if correct, -1% if wrong (simplified)
    # If y_true=1 (Up) and Signal=1 (Up) -> Win
    # If y_true=0 (Down) and Signal=0 (Down) -> Win
    # If y_true=1 and Signal=0 -> Missed opp (or loss if we short? let's assume we predict Direction)
    # Simple: +1 if prediction matches target, -1 if not.
    
    pnl = np.where(signals == y_true, 1, -1)
    cum_pnl = np.cumsum(pnl)
    
    plt.figure(figsize=(12, 6))
    plt.plot(cum_pnl, label='Strategy (Win-Loss Count)')
    plt.title(title)
    plt.xlabel('Trades (Test Set Time)')
    plt.ylabel('Net Correct Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output/equity_curve.png')
    print("Saved output/equity_curve.png")
    plt.close()

import explainability as xai

def run_xgboost_analysis():
    print("🧠 Starting Dedicated XGBoost Analysis...")
    
    # Load
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Load (Optimized for demonstration/verification - limit history)
    print("⏳ Loading Recent Data (from 2020)...")
    prices = dl.load_prices(start_date="2020-01-01")
    macro = dl.load_macro()
    technicals = dl.load_technicals()
    fundamentals = dl.load_fundamentals()
    
    if prices is None or prices.empty:
        print("❌ Failed to load data.")
        return

    # Process
    print("PREPROCESSING...")
    prices = pp.clean_data(prices)
    full_df = pp.merge_data(prices, technicals, macro, fundamentals)
    
    # Standard horizon
    labeled_df = pp.create_target(full_df, horizon=20)
    
    # ---------------------------------------------------------
    # OPTION C: MONTHLY RETRAINING (WALK-FORWARD)
    # ---------------------------------------------------------
    print("🔄 OPTION C: Walk-Forward Validation (Monthly Retraining)...")
    
    # Filter features
    drop_cols = ['id', 'symbol', 'ticker', 'date', 'trade_date', 'fetched_at', 
                 'close_future', 'fwd_return', 'fwd_return_20d', 'target', 'created_at', 'updated_at',
                 'sample_weight'] 
    
    feature_cols = [c for c in labeled_df.columns if c not in drop_cols and not str(c).endswith('_at') and not str(c).endswith('_id') and c != 'hawkish_stance']
    
    print(f"Features ({len(feature_cols)}): {feature_cols[:5]} ...")
    
    # We want to test on the "Problematic Period": mid-2025 to end-2025
    # Start simulating from 2025-01-01
    
    TEST_START_DATE = pd.to_datetime("2025-01-01")
    TEST_END_DATE = labeled_df['date'].max()
    
    # Sliding Window Size for Training (e.g. 18 months lookback)
    TRAIN_WINDOW_DAYS = 365 * 1.5 
    
    current_date = TEST_START_DATE
    all_predictions = []
    all_targets = []
    
    shap_history = [] # For Temporal Importance
    dates_history = []
    
    print(f"Starting Walk-Forward Loop from {TEST_START_DATE} to {TEST_END_DATE}")
    
    iteration = 0
    while current_date < TEST_END_DATE:
        iteration += 1
        
        # Define Train Window [current - lookback : current]
        train_start = current_date - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
        train_end = current_date
        
        # Define Test Window [current : current + 1 month]
        test_end = current_date + pd.Timedelta(days=30)
        
        print(f"Round {iteration}: Train [{train_start.date()} -> {train_end.date()}] | Test [{train_end.date()} -> {test_end.date()}]")
        
        # Slicing
        train_mask = (labeled_df['date'] >= train_start) & (labeled_df['date'] < train_end)
        test_mask = (labeled_df['date'] >= train_end) & (labeled_df['date'] < test_end)
        
        if not test_mask.any():
            print("  ⚠️ No test data for this month, skipping.")
            current_date = test_end
            continue
            
        X_train = labeled_df.loc[train_mask, feature_cols].select_dtypes(include=[np.number])
        y_train = labeled_df.loc[train_mask, 'target']
        
        X_test_chunk = labeled_df.loc[test_mask, feature_cols].select_dtypes(include=[np.number])
        y_test_chunk = labeled_df.loc[test_mask, 'target']
        
        if len(X_train) < 1000:
            print("  ⚠️ Not enough training data, skipping.")
            current_date = test_end
            continue
            
        # Train Model (No Val set needed, we want best fit on recent data, explicit regularization via params)
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100, # Lightweight trees
            max_depth=3,      # Simple trees to avoid overfitting small recent sample
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        # Predict
        preds = model.predict_proba(X_test_chunk)[:, 1]
        
        # Explain (SHAP) - Capture importance for this specific regime
        # computing on smaller sample of test set to save time if needed, but X_test_chunk is small (1 month)
        print("  🧠 Computing SHAP...")
        monthly_importances = xai.get_mean_abs_shap(model, X_test_chunk)
        shap_history.append(monthly_importances)
        dates_history.append(train_end.date())
        
        # Store
        all_predictions.extend(preds)
        all_targets.extend(y_test_chunk)
        
        # Move forward
        current_date = test_end
        
    # Final Evaluation
    print("📊 Evaluating Walk-Forward Results...")
    
    y_true = np.array(all_targets)
    y_scores = np.array(all_predictions)
    
    if len(y_true) == 0:
        print("❌ No predictions made.")
        return

    test_auc = roc_auc_score(y_true, y_scores)
    test_acc = accuracy_score(y_true, (y_scores > 0.5).astype(int))
    
    print(f"\n🏆 TEST RESULTS (Option C - Monthly Retraining):\nAUC: {test_auc:.4f}\nAccuracy: {test_acc:.4f}")
    
    # Plots based on aggregated walk-forward predictions
    plot_probability_distribution(y_true, y_scores, title="Prob Dist (Walk-Forward)")
    plot_cumulative_returns(y_true, y_scores, title="Equity Curve (Walk-Forward)")
    
    # --- TEMPORAL FEATURE IMPORTANCE PLOT ---
    if shap_history:
        print("📈 Generating Temporal Feature Importance Plot...")
        importance_df = pd.DataFrame(shap_history, index=dates_history)
        xai.plot_temporal_importance(importance_df, output_path="output/temporal_importance.png")
    
    # Also generate a global summary for the LAST model instance
    print("📝 Generating Global SHAP Summary for last month...")
    xai.plot_shap_summary(model, X_test_chunk, output_path="output/shap_summary_last.png")

    # ---------------------------------------------------------
    # SAVE ARTIFACTS FOR DASHBOARD (Phase 2 Requirement)
    # ---------------------------------------------------------
    print("💾 Saving Model & Features for Dashboard...")
    
    # Save Model
    model_path = output_dir / "latest_model.json"
    model.save_model(model_path)
    print(f"  Saved model to {model_path}")
    
    # Save Feature Columns (Critical for alignment)
    features_path = output_dir / "feature_cols.json"
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f)
    print(f"  Saved feature columns to {features_path}")

    
    print("Done.")

if __name__ == "__main__":
    run_xgboost_analysis()
