import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

import data_loader as dl
import preprocessing as pp
import model as md
import explainability as xai

def run_pipeline():
    print("🚀 Starting XAI Finance Pipeline...")
    
    # 1. Load Data
    print("📥 Loading Data...")
    prices = dl.load_prices() 
    macro = dl.load_macro()
    technicals = dl.load_technicals()
    
    if prices is None or prices.empty:
        print("❌ No price data found. Check connections.")
        return

    print(f"Loaded Prices: {prices.shape}")
    
    # 2. Preprocessing
    print("🔄 Preprocessing & Merging...")
    prices = pp.clean_data(prices)
    full_df = pp.merge_data(prices, technicals, macro)
    
    print(f"Merged Data: {full_df.shape}")
    
    # 3. Target Creation
    print("🎯 Creating Targets (20d forward)...")
    labeled_df = pp.create_target(full_df, horizon=20)
    print(f"Labeled Data: {labeled_df.shape}")
    
    if labeled_df.empty:
        print("❌ No data left after labeling.")
        return

    # Features selection: Drop identifiers and target cols
    drop_cols = ['id', 'symbol', 'ticker', 'date', 'trade_date', 'fetched_at', 
                 'close_future', 'fwd_return', 'target', 'created_at', 'updated_at',
                 'provider_code', 'series_id', 'currency', 'exchange']
    
    feature_cols = [c for c in labeled_df.columns if c not in drop_cols and not str(c).endswith('_at') and not str(c).endswith('_id')]
    
    # Ensure numeric
    X = labeled_df[feature_cols].copy()
    # Filter only numeric columns
    X = X.select_dtypes(include=[np.number])
    y = labeled_df['target']
    
    # Add back date for splitting
    X['date'] = labeled_df['date']
    
    print(f"Features ({len(X.columns)-1}): {list(X.columns)}")
    
    if len(X.columns) <= 1:
        print("❌ No features found.")
        return

    # 4. Split
    print("✂️ Temporal Split...")
    X_train, X_val, X_test = pp.temporal_split(X)
    
    if X_train.empty:
        print("❌ Train set empty.")
        return

    y_train = y.loc[X_train.index]
    y_val = y.loc[X_val.index]
    y_test = y.loc[X_test.index]
    
    # Remove date from X for training
    X_train = X_train.drop(columns=['date'])
    X_val = X_val.drop(columns=['date'])
    X_test = X_test.drop(columns=['date'])
    
    # 5. Training
    print("🤖 Training XGBoost...")
    model = md.XGBoostModel()
    try:
        model.train(X_train, y_train, X_val, y_val)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # 6. Evaluation
    print("📊 Evaluating...")
    try:
        metrics = model.evaluate(X_test, y_test)
        print(f"Test Metrics: {metrics}")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

    # 7. Explainability
    print("💡 Calculating SHAP Values...")
    try:
        # Use a sample of test set for speed
        sample_size = min(100, len(X_test))
        if sample_size > 0:
            X_sample = X_test.sample(sample_size, random_state=42)
            explainer, shap_values = xai.compute_shap_values(model.get_booster(), X_sample)
            
            print("Checking Stability...")
            stability_score = xai.check_stability(model.get_booster(), X_sample)
            print(f"Stability Score (Mean Rank Var): {stability_score:.4f}")
            
            # Save plot
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            xai.plot_shap_summary(shap_values, X_sample, show=False, save_path=output_dir / "shap_summary.png")
            
        else:
            print("⚠️ Test set too small for SHAP.")
    except Exception as e:
        print(f"❌ XAI failed: {e}")
    
    print("✅ Pipeline Completed Successfully.")

if __name__ == "__main__":
    run_pipeline()
